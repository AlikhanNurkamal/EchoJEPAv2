# src/datasets/webdataset_video_dataset.py
#
# Custom IterableDataset for EchoJEPAv2 pretraining.
# Reads from .tar shards produced by preprocessing_alikhan/create_webdataset.py.
# Each shard contains samples keyed as:
#   <uuid>.frames.npy    ->  (T, 336, 336, 3) uint8 RGB numpy array
#   <uuid>.metadata.json ->  dict with dicom metadata
#
# Returns samples in the same format as VideoDataset:
#   (clips, label, clip_indices)
#   clips:       list of num_clips tensors, each (C, T, H, W) float32
#   label:       0  (unsupervised)
#   clip_indices: list of num_clips numpy arrays, each of length frames_per_clip

import glob
import io
import os
import tarfile
from collections import defaultdict
from logging import getLogger

import numpy as np
import torch
import torch.distributed as dist

logger = getLogger()


class DummySampler:
    """
    Returned in place of DistributedSampler for IterableDataset.
    The training loop calls sampler.set_epoch(epoch) – this is a no-op.
    """

    def set_epoch(self, epoch: int):
        pass


def make_webdatasetvideodataset(
    shard_dir,
    batch_size,
    frames_per_clip=16,
    fps_stored=24,
    fps_sample=24,
    num_clips=1,
    random_clip_sampling=True,
    transform=None,
    shared_transform=None,
    collator=None,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    world_size=1,
    rank=0,
    shuffle_buffer=500,
    drop_last=True,
    **kwargs,
):
    dataset = WebDatasetVideoDataset(
        shard_dir=shard_dir,
        frames_per_clip=frames_per_clip,
        fps_stored=fps_stored,
        fps_sample=fps_sample,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        transform=transform,
        shared_transform=shared_transform,
        world_size=world_size,
        rank=rank,
        shuffle_buffer=shuffle_buffer,
    )

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0) and persistent_workers,
        # No sampler: IterableDataset manages its own distributed splitting
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    data_loader = torch.utils.data.DataLoader(**dl_kwargs)

    # Estimate batches per epoch so the training loop knows ipe
    num_shards = len(dataset.shard_paths)
    estimated_samples = num_shards * 1000  # default shard_size in create_webdataset.py
    per_rank_samples = estimated_samples // max(world_size, 1)
    data_loader.num_batches = max(1, per_rank_samples // batch_size)

    dummy_sampler = DummySampler()

    logger.info(
        f"WebDatasetVideoDataset: {num_shards} shards, "
        f"~{estimated_samples} total samples, "
        f"~{data_loader.num_batches} batches/epoch @ batch_size={batch_size}"
    )

    return dataset, data_loader, dummy_sampler


class WebDatasetVideoDataset(torch.utils.data.IterableDataset):
    """
    Streams echo DICOM clips from .tar WebDataset shards.

    Shard layout (per sample):
        <uuid>.frames.npy    — (T, 336, 336, 3) uint8 RGB
        <uuid>.metadata.json — metadata dict (not used during pretraining)

    Temporal sampling:
        stride = fps_stored // fps_sample
        A contiguous window of frames_per_clip * stride frames is randomly
        selected from the stored video and sub-sampled at the given stride.
    """

    def __init__(
        self,
        shard_dir,
        frames_per_clip: int = 16,
        fps_stored: int = 24,
        fps_sample: int = 24,
        num_clips: int = 1,
        random_clip_sampling: bool = True,
        transform=None,
        shared_transform=None,
        world_size: int = 1,
        rank: int = 0,
        shuffle_buffer: int = 500,
    ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.fps_stored = fps_stored
        self.fps_sample = fps_sample
        self.stride = max(1, fps_stored // fps_sample)
        self.num_clips = num_clips
        self.random_clip_sampling = random_clip_sampling
        self.transform = transform
        self.shared_transform = shared_transform
        self.world_size = world_size
        self.rank = rank
        self.shuffle_buffer = shuffle_buffer

        if isinstance(shard_dir, (list, tuple)):
            # Accept either a directory or a list of directories
            paths = []
            for d in shard_dir:
                paths.extend(sorted(glob.glob(os.path.join(d, "shard-*.tar"))))
            self.shard_paths = paths
        else:
            self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))

        if not self.shard_paths:
            raise ValueError(f"No shard-*.tar files found in: {shard_dir}")

        # Filter out empty / corrupt shard files (e.g. from disk-full during preprocessing)
        min_shard_bytes = 10_000  # a valid shard with at least 1 sample is >> 10KB
        before = len(self.shard_paths)
        self.shard_paths = [p for p in self.shard_paths if os.path.getsize(p) > min_shard_bytes]
        skipped = before - len(self.shard_paths)
        if skipped:
            logger.warning(f"Filtered out {skipped} empty/corrupt shards (< {min_shard_bytes} bytes)")

        logger.info(f"WebDatasetVideoDataset: {len(self.shard_paths)} valid shards in {shard_dir}")

        # Optional dicom_uuid denylist for held-out site evaluation.
        # Activated by env var ECHOJEPA_HOLDOUT_DICOMS=/abs/path/to/file (one uuid per line).
        self.holdout_dicoms = set()
        holdout_path = os.environ.get("ECHOJEPA_HOLDOUT_DICOMS")
        if holdout_path and os.path.isfile(holdout_path):
            with open(holdout_path) as f:
                self.holdout_dicoms = {line.strip() for line in f if line.strip()}
            logger.info(
                f"WebDatasetVideoDataset: loaded {len(self.holdout_dicoms)} held-out dicom_uuids from {holdout_path}"
            )

        # Optional allowlist to restrict to a specific split (e.g. TRAIN only).
        # Activated by env var ECHOJEPA_ALLOWED_DICOMS=/abs/path/to/file (one uuid per line).
        self.allowed_dicoms = None
        allowed_path = os.environ.get("ECHOJEPA_ALLOWED_DICOMS")
        if allowed_path and os.path.isfile(allowed_path):
            with open(allowed_path) as f:
                self.allowed_dicoms = {line.strip() for line in f if line.strip()}
            logger.info(
                f"WebDatasetVideoDataset: loaded {len(self.allowed_dicoms)} allowed dicom_uuids from {allowed_path}"
            )

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        # Determine global rank / world_size from distributed context at
        # runtime (more reliable than constructor-time values).
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = self.rank
            world_size = self.world_size

        worker_info = torch.utils.data.get_worker_info()

        # 1. Split shards across ranks
        rank_shards = self.shard_paths[rank::world_size]

        # 2. Split rank's shards across dataloader workers
        if worker_info is not None:
            worker_shards = rank_shards[worker_info.id :: worker_info.num_workers]
        else:
            worker_shards = rank_shards

        # 3. Shuffle shards (uses PyTorch's per-worker seed which changes
        #    each epoch when the DataLoader is re-iterated)
        rng = np.random.default_rng(seed=torch.initial_seed() % (2**31))
        worker_shards = list(worker_shards)
        rng.shuffle(worker_shards)

        for shard_path in worker_shards:
            yield from self._iter_shard(shard_path, rng)

    # ------------------------------------------------------------------
    # Shard iteration
    # ------------------------------------------------------------------

    def _iter_shard(self, shard_path: str, rng: np.random.Generator):
        """Stream through a tar shard, yielding samples without pre-scanning."""
        try:
            with tarfile.open(shard_path, "r:") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if name.endswith(".frames.npy"):
                        dicom_uuid = name[: -len(".frames.npy")]
                        fmt = "npy"
                    elif name.endswith(".frames.npz"):
                        dicom_uuid = name[: -len(".frames.npz")]
                        fmt = "npz"
                    else:
                        continue
                    if self.holdout_dicoms and dicom_uuid in self.holdout_dicoms:
                        continue
                    if self.allowed_dicoms is not None and dicom_uuid not in self.allowed_dicoms:
                        continue
                    result = self._process_member(tar, member, fmt)
                    if result is not None:
                        yield result
        except Exception as e:
            logger.warning(f"Error reading shard {shard_path}: {e}")

    # ------------------------------------------------------------------
    # Sample processing
    # ------------------------------------------------------------------

    def _process_member(self, tar: tarfile.TarFile, member: tarfile.TarInfo, fmt: str = "npy"):
        """Load one .npy/.npz member and return a (clips, label, clip_indices) tuple."""
        try:
            raw = tar.extractfile(member)
            if raw is None:
                return None
            data = np.load(io.BytesIO(raw.read()))
            frames = data["frames"] if fmt == "npz" else data  # (T, H, W, 3) uint8 RGB

            if frames.ndim != 4 or frames.shape[-1] != 3:
                return None

            T = frames.shape[0]
            if T == 0:
                return None

            if self.shared_transform is not None:
                frames = self.shared_transform(frames)

            clips, clip_indices = [], []
            for _ in range(self.num_clips):
                clip, indices = self._sample_clip(frames, T)
                if clip is None:
                    return None
                if self.transform is not None:
                    clip = self.transform(clip)
                clips.append(clip)
                clip_indices.append(indices)

            return clips, 0, clip_indices

        except Exception as e:
            logger.debug(f"Failed to process member {member.name}: {e}")
            return None

    def _sample_clip(self, frames: np.ndarray, T: int):
        """
        Sample frames_per_clip frames at stride self.stride.

        Returns (clip_frames, indices) where:
            clip_frames: (frames_per_clip, H, W, 3) uint8
            indices:     np.ndarray of shape (frames_per_clip,) — frame indices used
        """
        fpc = self.frames_per_clip
        stride = self.stride
        clip_len = fpc * stride  # number of stored frames spanned by the clip

        if T < fpc:
            # Video shorter than clip: repeat-pad via linspace
            idx = np.linspace(0, T - 1, num=fpc).astype(np.int64)
        elif T < clip_len:
            # Video long enough for fpc frames but stride would exceed bounds:
            # sample evenly across the full video
            idx = np.linspace(0, T - 1, num=fpc).astype(np.int64)
        else:
            # Normal case
            if self.random_clip_sampling:
                start = np.random.randint(0, T - clip_len + 1)
            else:
                start = 0
            idx = np.arange(start, start + clip_len, stride)[:fpc]

        return frames[idx], idx.astype(np.int32)
