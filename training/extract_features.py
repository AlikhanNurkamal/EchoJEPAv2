#!/usr/bin/env python3
"""
Parallel sliding-window feature extraction from iCardio WebDataset shards.

Uses DataLoader with num_workers to pipeline HDD I/O with GPU compute,
matching the throughput of pretraining.

For each DICOM video:
  - Slide a 16-frame window with `--stride` across the full frame sequence
  - Run frozen ViT-Large encoder on each window  →  (N_tokens, 1024)
  - Mean-pool tokens                             →  (1024,) per window
  - Mean-pool across all windows                 →  (1024,) per DICOM

Output: <output_dir>/embeddings.pt
        dict[dicom_uuid -> np.float32 array shape (1024,)]

Resumable: already-extracted UUIDs are skipped on restart.
"""

import argparse
import glob
import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional as TF
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "EchoJEPA"))

import src.models.vision_transformer as vit  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dirs", nargs="+",
                   default=["/hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa",
                            "/hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa"])
    p.add_argument("--checkpoint",
                   default="/home/ahmedaly/iCardio/checkpoints/pretrain/"
                           "icardio_vitl16_224px_16f/latest.pt")
    p.add_argument("--checkpoint_key", default="target_encoder")
    p.add_argument("--output_dir",
                   default="/data/ahmedaly/icardio_embeddings")
    p.add_argument("--holdout", default="",
                   help="Path to holdout_dicoms.txt to skip. Leave empty to extract all.")
    p.add_argument("--window",       type=int, default=16)
    p.add_argument("--stride",       type=int, default=16, help="Sliding window stride")
    p.add_argument("--resolution",   type=int, default=224)
    p.add_argument("--num_workers",  type=int, default=8,
                   help="DataLoader workers for parallel HDD I/O")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Windows per GPU forward pass")
    p.add_argument("--save_every",   type=int, default=10000)
    p.add_argument("--device",       default="cuda:1")
    p.add_argument("--min_shard_bytes", type=int, default=10_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path, checkpoint_key, resolution, window, device):
    print(f"Loading encoder from {checkpoint_path} (key={checkpoint_key})")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt[checkpoint_key]
    state = {k.replace("module.", "").replace("backbone.", ""): v
             for k, v in state.items()}
    encoder = vit.__dict__["vit_large"](
        img_size=resolution, num_frames=window,
        patch_size=16, tubelet_size=2,
        uniform_power=True, use_rope=True,
    )
    msg = encoder.load_state_dict(state, strict=False)
    print(f"Encoder loaded: {msg}")
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ---------------------------------------------------------------------------
# Dataset — IterableDataset that splits shards across workers
# ---------------------------------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def _preprocess(frames: np.ndarray, resolution: int) -> torch.Tensor:
    t = torch.from_numpy(frames).float() / 255.0   # (T, H, W, 3)
    t = t.permute(0, 3, 1, 2)                       # (T, 3, H, W)
    if t.shape[-1] != resolution or t.shape[-2] != resolution:
        t = torch.stack([
            TF.resize(f, [resolution, resolution],
                      interpolation=TF.InterpolationMode.BILINEAR,
                      antialias=True)
            for f in t
        ])
    t = (t - _MEAN.view(1, 3, 1, 1)) / _STD.view(1, 3, 1, 1)
    return t


def _make_clips(frames: np.ndarray, window: int, stride: int,
                resolution: int) -> torch.Tensor:
    """Returns (N_windows, 3, window, H, W)."""
    T = frames.shape[0]
    t = _preprocess(frames, resolution)   # (T, 3, H, W)

    if T < window:
        idx = np.linspace(0, T - 1, window).astype(np.int64)
        t = t[idx]
        T = window

    if T == window:
        starts = [0]
    else:
        starts = list(range(0, T - window + 1, stride))
        if starts[-1] != T - window:
            starts.append(T - window)

    return torch.stack([t[s:s + window].permute(1, 0, 2, 3) for s in starts])


class ShardDataset(torch.utils.data.IterableDataset):
    """
    Streams (clips, uuid) pairs from WebDataset tar shards.
    Splits shards evenly across DataLoader workers so I/O is parallel.
    clips: (N_windows, 3, window, H, W) float32 CPU tensor
    """

    def __init__(self, shard_paths, window, stride, resolution, skip_uuids):
        self.shard_paths = list(shard_paths)
        self.window = window
        self.stride = stride
        self.resolution = resolution
        self.skip_uuids = frozenset(skip_uuids)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        shards = (self.shard_paths[wi.id::wi.num_workers]
                  if wi is not None else self.shard_paths)

        for shard in shards:
            try:
                with tarfile.open(shard, "r:") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        name = os.path.basename(member.name)
                        if not name.endswith(".frames.npy"):
                            continue
                        uuid = name[:-len(".frames.npy")]
                        if uuid in self.skip_uuids:
                            continue
                        try:
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            frames = np.load(io.BytesIO(f.read()))
                            if (frames.ndim != 4 or frames.shape[-1] != 3
                                    or frames.shape[0] == 0):
                                continue
                            clips = _make_clips(frames, self.window,
                                                self.stride, self.resolution)
                            yield clips, uuid
                        except Exception:
                            continue
            except Exception:
                continue


def collate_fn(batch):
    # batch is a list of 1 item: [(clips, uuid)]
    return batch[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "embeddings.pt")

    # Resume
    if os.path.isfile(out_path):
        print(f"Resuming from {out_path}")
        embeddings: dict = torch.load(out_path, map_location="cpu", weights_only=False)
        print(f"  Already extracted: {len(embeddings)} DICOMs")
    else:
        embeddings = {}

    # Holdout
    holdout: set = set()
    if args.holdout and os.path.isfile(args.holdout):
        with open(args.holdout) as f:
            holdout = {l.strip() for l in f if l.strip()}
        print(f"Holdout: {len(holdout)} DICOMs skipped")

    skip_uuids = set(embeddings.keys()) | holdout

    # Shards
    shard_paths = []
    for d in args.shard_dirs:
        shard_paths.extend(sorted(glob.glob(os.path.join(d, "shard-*.tar"))))
    shard_paths = [p for p in shard_paths
                   if os.path.getsize(p) > args.min_shard_bytes]
    print(f"Found {len(shard_paths)} shards")

    encoder = load_encoder(args.checkpoint, args.checkpoint_key,
                           args.resolution, args.window, device)

    dataset = ShardDataset(shard_paths, args.window, args.stride,
                           args.resolution, skip_uuids)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    total = 0
    errors = 0

    pbar = tqdm(loader, unit="dicom", dynamic_ncols=True)
    with torch.no_grad():
        for clips, uuid in pbar:
            try:
                # clips: (N_windows, 3, window, H, W)
                # Batch through encoder in chunks of batch_size
                window_embeds = []
                for i in range(0, len(clips), args.batch_size):
                    batch = clips[i:i + args.batch_size].to(device,
                                                             non_blocking=True)
                    tokens = encoder(batch)             # (B, N_tokens, D)
                    window_embeds.append(tokens.mean(1).cpu().float())

                emb = torch.cat(window_embeds, 0).mean(0).numpy()
                embeddings[uuid] = emb
                total += 1

                if total % args.save_every == 0:
                    pbar.write(f"Saving checkpoint ({len(embeddings)} total)...")
                    torch.save(embeddings, out_path)

            except Exception as e:
                errors += 1
                if errors < 20:
                    pbar.write(f"  Error on {uuid}: {e}")

            pbar.set_postfix(extracted=total, errors=errors)

    print(f"\nDone. Extracted {total} new. Total: {len(embeddings)}. Errors: {errors}")
    print(f"Saving to {out_path} ...")
    torch.save(embeddings, out_path)
    print("Saved.")


if __name__ == "__main__":
    main()
