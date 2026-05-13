# src/datasets/webdataset_labeled_dataset.py
#
# IterableDataset that streams iCardio WebDataset shards for labeled
# downstream evaluation (LVEF regression probe training / inference).
#
# CSV format (space-separated, no header):
#   <dicom_uuid>  <normalised_label>
#
# Holdout filtering: set ECHOJEPA_HOLDOUT_DICOMS env var (same as pretraining).

import glob
import io
import os
import tarfile
from logging import getLogger

import numpy as np
import torch
import torch.distributed as dist

logger = getLogger()


class DummySampler:
    def set_epoch(self, epoch: int):
        pass


def make_webdatasetlabeleddataset(
    shard_dir,
    batch_size,
    label_csv,
    frames_per_clip=16,
    frame_step=2,
    num_segments=2,
    resolution=224,
    transform=None,
    shared_transform=None,
    collator=None,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    world_size=1,
    rank=0,
    drop_last=True,
    steps_per_epoch=2000,
    **kwargs,
):
    samples_per_epoch = steps_per_epoch * batch_size

    dataset = WebDatasetLabeledDataset(
        shard_dir=shard_dir,
        label_csv=label_csv,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        resolution=resolution,
        transform=transform,
        shared_transform=shared_transform,
        world_size=world_size,
        rank=rank,
        samples_per_epoch=samples_per_epoch,
    )

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    data_loader = torch.utils.data.DataLoader(**dl_kwargs)
    num_shards = len(dataset.shard_paths)
    labeled_samples = len(dataset.label_dict)
    data_loader.num_batches = steps_per_epoch

    dummy_sampler = DummySampler()
    logger.info(
        f"WebDatasetLabeledDataset: {num_shards} shards, "
        f"{labeled_samples} labeled samples, "
        f"{steps_per_epoch} steps/epoch ({samples_per_epoch} samples)"
    )
    return dataset, data_loader, dummy_sampler


class WebDatasetLabeledDataset(torch.utils.data.IterableDataset):
    """
    Streams iCardio .npy shards and yields labeled samples for probe training.

    Only DICOMs whose uuid appears in label_csv are yielded.
    Respects ECHOJEPA_HOLDOUT_DICOMS env var (same logic as pretraining).

    Returns: (clips, label, clip_indices)
        clips       : list of num_segments tensors, each (C, T, H, W) float32
        label       : float32 scalar (normalised EF)
        clip_indices: list of num_segments arrays
    """

    def __init__(
        self,
        shard_dir,
        label_csv,
        frames_per_clip: int = 16,
        frame_step: int = 2,
        num_segments: int = 2,
        resolution: int = 224,
        transform=None,
        shared_transform=None,
        world_size: int = 1,
        rank: int = 0,
        min_shard_bytes: int = 10_000,
        samples_per_epoch: int = 32_000,
    ):
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_segments = num_segments
        self.resolution = resolution
        self.transform = transform
        self.shared_transform = shared_transform
        self.world_size = world_size
        self.rank = rank
        self.samples_per_epoch = samples_per_epoch

        # Load label dict
        self.label_dict = {}
        with open(label_csv) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uuid, label = line.split()
                self.label_dict[uuid] = float(label)
        logger.info(f"WebDatasetLabeledDataset: loaded {len(self.label_dict)} labels from {label_csv}")

        # Holdout denylist
        self.holdout_dicoms: set = set()
        holdout_path = os.environ.get("ECHOJEPA_HOLDOUT_DICOMS")
        if holdout_path and os.path.isfile(holdout_path):
            with open(holdout_path) as f:
                self.holdout_dicoms = {l.strip() for l in f if l.strip()}
            logger.info(f"WebDatasetLabeledDataset: {len(self.holdout_dicoms)} held-out dicoms excluded")

        # Collect shard paths
        if isinstance(shard_dir, (list, tuple)):
            paths = []
            for d in shard_dir:
                paths.extend(sorted(glob.glob(os.path.join(d, "shard-*.tar"))))
            self.shard_paths = paths
        else:
            self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))

        self.shard_paths = [
            p for p in self.shard_paths if os.path.getsize(p) > min_shard_bytes
        ]
        logger.info(f"WebDatasetLabeledDataset: {len(self.shard_paths)} valid shards")

    def __len__(self):
        """Per-rank sample count — drives DataLoader length and scheduler ipe."""
        return max(1, self.samples_per_epoch // max(self.world_size, 1))

    def __iter__(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = self.rank
            world_size = self.world_size

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Each worker yields its share of samples_per_epoch
        per_rank = self.samples_per_epoch // max(world_size, 1)
        per_worker = max(1, per_rank // num_workers)

        rank_shards = self.shard_paths[rank::world_size]
        worker_shards = rank_shards[worker_id::num_workers]

        rng = np.random.default_rng(seed=torch.initial_seed() % (2**31))
        worker_shards = list(worker_shards)
        rng.shuffle(worker_shards)

        yielded = 0
        for shard_path in worker_shards:
            for sample in self._iter_shard(shard_path, rng):
                yield sample
                yielded += 1
                if yielded >= per_worker:
                    return

    def _iter_shard(self, shard_path: str, rng: np.random.Generator):
        try:
            with tarfile.open(shard_path, "r:") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if not name.endswith(".frames.npy"):
                        continue
                    dicom_uuid = name[: -len(".frames.npy")]
                    if dicom_uuid in self.holdout_dicoms:
                        continue
                    if dicom_uuid not in self.label_dict:
                        continue
                    label = self.label_dict[dicom_uuid]
                    result = self._process_member(tar, member, label, rng)
                    if result is not None:
                        yield result
        except Exception as e:
            logger.warning(f"Error reading shard {shard_path}: {e}")

    def _process_member(self, tar, member, label: float, rng: np.random.Generator):
        try:
            raw = tar.extractfile(member)
            if raw is None:
                return None
            frames = np.load(io.BytesIO(raw.read()))   # (T, H, W, 3) uint8
            if frames.ndim != 4 or frames.shape[-1] != 3:
                return None
            T = frames.shape[0]
            if T == 0:
                return None

            if self.shared_transform is not None:
                frames = self.shared_transform(frames)

            clips, clip_indices = [], []
            for _ in range(self.num_segments):
                clip, indices = self._sample_clip(frames, T, rng)
                if clip is None:
                    return None
                if self.transform is not None:
                    clip = self.transform(clip)
                clips.append(clip)
                clip_indices.append(indices)

            return clips, label, clip_indices

        except Exception as e:
            logger.debug(f"Failed to process {member.name}: {e}")
            return None

    def _sample_clip(self, frames: np.ndarray, T: int, rng: np.random.Generator):
        fpc = self.frames_per_clip
        stride = self.frame_step
        clip_len = fpc * stride

        if T < fpc:
            idx = np.linspace(0, T - 1, fpc).astype(np.int64)
        elif T < clip_len:
            idx = np.linspace(0, T - 1, fpc).astype(np.int64)
        else:
            start = int(rng.integers(0, T - clip_len + 1))
            idx = np.arange(start, start + clip_len, stride)[:fpc]

        return frames[idx], idx.astype(np.int32)
