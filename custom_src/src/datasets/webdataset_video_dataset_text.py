# src/datasets/webdataset_video_dataset_text.py
#
# Text-aware variant of WebDatasetVideoDataset for EchoJEPAv2 pretraining.
#
# Identical to webdataset_video_dataset.py except each sample also returns
# a clinical text string built from the .metadata.json stored alongside the
# video frames in every tar shard.
#
# Return format:
#   (clips, label, clip_indices, text_str)
#   clips:        list[Tensor(C,T,H,W)] float32 — one tensor per num_clips
#   label:        0  (unsupervised)
#   clip_indices: list[ndarray(frames_per_clip,)] — sampled frame indices
#   text_str:     str — clinical text built from study_info fields
#
# The text string is tokenised downstream in the MaskCollatorText (main
# process), so no tokeniser lives inside this worker-safe dataset class.

import glob
import io
import json
import os
import tarfile
from logging import getLogger

import numpy as np
import torch
import torch.distributed as dist

logger = getLogger()

# ── Study-info fields used to build the clinical text string ──────────────────

_FREE_TEXT_FIELDS = [
    "left_ventricle",
    "right_ventricle",
    "left_atrium",
    "right_atrium",
    "aortic_valve",
    "mitral_valve",
    "tricuspid_valve",
    "pulmonic_valve",
    "pericardium",
    "aortic_root",
    "aortic_arch",
    "pulmonary_artery",
    "conclusions",
]
_STRUCTURED_FIELDS = [
    "conditions",
    "characterizations",
    "stratifications",
]
_FALLBACK_TEXT = "No clinical information available."


def _build_text(metadata: dict) -> str:
    """Build a single clinical string from a sample's metadata dict."""
    parts = []

    age = metadata.get("age_at_visit")
    if age is not None:
        try:
            parts.append(f"Age: {int(float(age))} years.")
        except (ValueError, TypeError):
            pass

    ef = metadata.get("ejection_fraction")
    if ef is not None:
        try:
            parts.append(f"Ejection fraction: {ef}.")
        except Exception:
            pass

    for field in _FREE_TEXT_FIELDS + _STRUCTURED_FIELDS:
        val = metadata.get(field)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())

    return " ".join(parts) if parts else _FALLBACK_TEXT


# ── Sampler ───────────────────────────────────────────────────────────────────


class DummySampler:
    """No-op sampler returned in place of DistributedSampler for IterableDataset."""

    def set_epoch(self, epoch: int):
        pass


# ── Factory ───────────────────────────────────────────────────────────────────


def make_webdatasetvideodataset_text(
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
    dataset = WebDatasetVideoDatasetText(
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
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    data_loader = torch.utils.data.DataLoader(**dl_kwargs)

    num_shards = len(dataset.shard_paths)
    estimated_samples = num_shards * 1000
    per_rank_samples = estimated_samples // max(world_size, 1)
    data_loader.num_batches = max(1, per_rank_samples // batch_size)

    dummy_sampler = DummySampler()

    logger.info(
        f"WebDatasetVideoDatasetText: {num_shards} shards, "
        f"~{estimated_samples} total samples, "
        f"~{data_loader.num_batches} batches/epoch @ batch_size={batch_size}"
    )

    return dataset, data_loader, dummy_sampler


# ── Dataset ───────────────────────────────────────────────────────────────────


class WebDatasetVideoDatasetText(torch.utils.data.IterableDataset):
    """
    Text-aware streaming echo dataset.

    Each sample yields (clips, label, clip_indices, text_str) where text_str
    is a clinical string built from .metadata.json stored next to .frames.npy
    in the tar shard.  If metadata is missing or has no text fields, the
    fallback string "No clinical information available." is used.
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
            paths = []
            for d in shard_dir:
                paths.extend(sorted(glob.glob(os.path.join(d, "shard-*.tar"))))
            self.shard_paths = paths
        else:
            self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))

        if not self.shard_paths:
            raise ValueError(f"No shard-*.tar files found in: {shard_dir}")

        min_shard_bytes = 10_000
        before = len(self.shard_paths)
        self.shard_paths = [p for p in self.shard_paths if os.path.getsize(p) > min_shard_bytes]
        skipped = before - len(self.shard_paths)
        if skipped:
            logger.warning(f"Filtered out {skipped} empty/corrupt shards (< {min_shard_bytes} bytes)")

        logger.info(f"WebDatasetVideoDatasetText: {len(self.shard_paths)} valid shards in {shard_dir}")

        self.holdout_dicoms = set()
        holdout_path = os.environ.get("ECHOJEPA_HOLDOUT_DICOMS")
        if holdout_path and os.path.isfile(holdout_path):
            with open(holdout_path) as f:
                self.holdout_dicoms = {line.strip() for line in f if line.strip()}
            logger.info(
                f"WebDatasetVideoDatasetText: loaded {len(self.holdout_dicoms)} held-out dicom_uuids from {holdout_path}"
            )

        self.allowed_dicoms = None
        allowed_path = os.environ.get("ECHOJEPA_ALLOWED_DICOMS")
        if allowed_path and os.path.isfile(allowed_path):
            with open(allowed_path) as f:
                self.allowed_dicoms = {line.strip() for line in f if line.strip()}
            logger.info(
                f"WebDatasetVideoDatasetText: loaded {len(self.allowed_dicoms)} allowed dicom_uuids from {allowed_path}"
            )

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = self.rank
            world_size = self.world_size

        worker_info = torch.utils.data.get_worker_info()

        rank_shards = self.shard_paths[rank::world_size]

        if worker_info is not None:
            worker_shards = rank_shards[worker_info.id :: worker_info.num_workers]
        else:
            worker_shards = rank_shards

        rng = np.random.default_rng(seed=torch.initial_seed() % (2**31))
        worker_shards = list(worker_shards)
        rng.shuffle(worker_shards)

        for shard_path in worker_shards:
            yield from self._iter_shard(shard_path, rng)

    # ------------------------------------------------------------------
    # Shard iteration — two-pass: collect metadata alongside frames
    # ------------------------------------------------------------------

    def _iter_shard(self, shard_path: str, rng: np.random.Generator):
        """
        Stream through a tar shard.

        We do a single sequential pass: when we see a .frames.npy we check
        if we already buffered the matching .metadata.json (arrived earlier),
        or we buffer the .npy and wait for the metadata that comes next.
        Because WebDataset convention stores members in key order
        (frames before metadata), we pair them as we go.
        """
        try:
            pending: dict[str, dict] = {}  # uuid -> {frames, metadata}
            with tarfile.open(shard_path, "r:") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)

                    if name.endswith(".frames.npy"):
                        uuid = name[: -len(".frames.npy")]
                        fmt = "npy"
                    elif name.endswith(".frames.npz"):
                        uuid = name[: -len(".frames.npz")]
                        fmt = "npz"
                    elif name.endswith(".metadata.json"):
                        uuid = name[: -len(".metadata.json")]
                        fmt = None
                    else:
                        continue

                    if fmt is not None:  # frames member
                        if self.holdout_dicoms and uuid in self.holdout_dicoms:
                            pending.pop(uuid, None)
                            continue
                        if self.allowed_dicoms is not None and uuid not in self.allowed_dicoms:
                            pending.pop(uuid, None)
                            continue
                        raw = tar.extractfile(member)
                        if raw is None:
                            continue
                        data = np.load(io.BytesIO(raw.read()))
                        frames = data["frames"] if fmt == "npz" else data
                        entry = pending.setdefault(uuid, {})
                        entry["frames"] = frames
                        if "metadata" in entry:
                            result = self._make_sample(entry, rng)
                            if result is not None:
                                yield result
                            del pending[uuid]

                    else:  # metadata member
                        raw = tar.extractfile(member)
                        meta = {}
                        if raw is not None:
                            try:
                                meta = json.loads(raw.read().decode("utf-8"))
                            except Exception:
                                pass
                        entry = pending.setdefault(uuid, {})
                        entry["metadata"] = meta
                        if "frames" in entry:
                            result = self._make_sample(entry, rng)
                            if result is not None:
                                yield result
                            del pending[uuid]

                # Yield any buffered frames that never got metadata
                for uuid, entry in pending.items():
                    if "frames" in entry and "metadata" not in entry:
                        entry["metadata"] = {}
                        result = self._make_sample(entry, rng)
                        if result is not None:
                            yield result

        except Exception as e:
            logger.warning(f"Error reading shard {shard_path}: {e}")

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _make_sample(self, entry: dict, rng: np.random.Generator):
        """
        Build one (clips, label, clip_indices, text_str) sample from a
        buffered {frames, metadata} entry.
        """
        frames = entry["frames"]
        metadata = entry.get("metadata", {})

        try:
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

            text_str = _build_text(metadata)
            return clips, 0, clip_indices, text_str

        except Exception as e:
            logger.debug(f"Failed to build sample: {e}")
            return None

    def _sample_clip(self, frames: np.ndarray, T: int):
        fpc = self.frames_per_clip
        stride = self.stride
        clip_len = fpc * stride

        if T < fpc:
            idx = np.linspace(0, T - 1, num=fpc).astype(np.int64)
        elif T < clip_len:
            idx = np.linspace(0, T - 1, num=fpc).astype(np.int64)
        else:
            if self.random_clip_sampling:
                start = np.random.randint(0, T - clip_len + 1)
            else:
                start = 0
            idx = np.arange(start, start + clip_len, stride)[:fpc]

        return frames[idx], idx.astype(np.int32)
