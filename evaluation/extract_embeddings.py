#!/usr/bin/env python3
"""
Extract and cache encoder embeddings for a single task (AS / MR / TR).

Runs the frozen ViT-L encoder once over all splits and saves mean-pooled
embeddings to disk. Only needs to be run once per task; train_probe.py
then trains purely on the cached tensors.

Usage:
    PYTHONPATH=EchoJEPA python evaluation/extract_embeddings.py --task AS --device cuda:0
    PYTHONPATH=EchoJEPA python evaluation/extract_embeddings.py --task MR --device cuda:1
    PYTHONPATH=EchoJEPA python evaluation/extract_embeddings.py --task TR --device cuda:2

Output:
    evaluation/embeddings/{task}/train.pt
    evaluation/embeddings/{task}/valid.pt
    evaluation/embeddings/{task}/test.pt

Each .pt file is a dict:
    embeddings: FloatTensor (N, 1024)
    labels:     LongTensor  (N,)
    uuids:      list[str]   length N
"""

import argparse
import gc
import io
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_repo = Path(__file__).resolve().parent.parent / "EchoJEPA"
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import src.models.vision_transformer as vit

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SHARD_DIRS = [
    "/hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa",
    "/hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa",
    "/hdd1/ahmedaly/preprocessed_valve_eval",
    "/hdd2/ahmedaly/preprocessed_missing_labels",
]

TASK_CSV = {"AS": "AS.csv", "MR": "MR.csv", "TR": "TR.csv"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EchoDataset(Dataset):
    def __init__(self, records, shard_index, frames_per_clip=16):
        self.records = records
        self.shard_index = shard_index
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        dicom_uuid, label = self.records[idx]
        shard_path, offset, size, fmt = self.shard_index[dicom_uuid]

        try:
            with open(shard_path, "rb") as f:
                f.seek(offset)
                raw = f.read(size)
            frames = np.load(io.BytesIO(raw))["frames"] if fmt == "npz" else np.load(io.BytesIO(raw))
        except Exception:
            frames = np.zeros((self.frames_per_clip, 336, 336, 3), dtype=np.uint8)

        T, N = len(frames), self.frames_per_clip
        if T == 0:
            frames = np.zeros((N, 336, 336, 3), dtype=np.uint8)
        elif T < N:
            frames = np.concatenate([frames] + [frames[-1:]] * (N - T), axis=0)
        else:
            start = (T - N) // 2
            frames = frames[start:start + N]

        x = frames.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = torch.from_numpy(x).permute(3, 0, 1, 2)  # (3, N, H, W)
        return x, label, dicom_uuid


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path, device):
    print(f"Loading encoder from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    encoder = vit.__dict__["vit_large"](
        img_size=336, patch_size=16, num_frames=16, tubelet_size=2,
        uniform_power=True, use_sdpa=True, use_silu=False, wide_silu=False,
        use_activation_checkpointing=False, use_rope=True,
    )

    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[key].items()}

    epoch = ckpt.get("epoch", "?")
    del ckpt; gc.collect()

    msg = encoder.load_state_dict(state, strict=False)
    print(f"  Load msg: {msg}  |  Checkpoint epoch: {epoch}")
    del state; gc.collect()

    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_shard_index(eval_dir):
    index_path = eval_dir / "shard_index.pkl"
    print(f"Loading shard index from {index_path} ...")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    print(f"  {len(index):,} entries loaded.")
    return index


def load_split(df, split, label_map, shard_index):
    subset = df[df["designation"] == split]
    records, skipped = [], 0
    for _, row in subset.iterrows():
        uuid, label_str = row["dicom_uuid"], row["stratification"]
        if pd.isna(label_str) or label_str not in label_map or uuid not in shard_index:
            skipped += 1
            continue
        records.append((uuid, label_map[label_str]))
    print(f"  {split}: {len(records)} samples ({skipped} skipped)")
    return records


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_split(encoder, records, shard_index, device, batch_size, num_workers, frames_per_clip):
    ds = EchoDataset(records, shard_index, frames_per_clip)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory="cuda" in device,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_embs, all_labels, all_uuids = [], [], []
    autocast_device = "cuda" if "cuda" in device else "cpu"

    for clips, labels, uuids in tqdm(loader, desc="  extracting", leave=False):
        clips = clips.to(device)
        with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
            tokens = encoder(clips)          # (B, N, D)
        embs = tokens.float().mean(dim=1)    # (B, D) — mean pool
        all_embs.append(embs.cpu())
        all_labels.append(labels)
        all_uuids.extend(uuids)

    return {
        "embeddings": torch.cat(all_embs, dim=0),   # (N, D)
        "labels":     torch.cat(all_labels, dim=0),  # (N,)
        "uuids":      all_uuids,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["AS", "MR", "TR"])
    parser.add_argument("--checkpoint", default=(
        "/home/ahmedaly/iCardio/checkpoints/pretrain/"
        "icardio_vitl16_336px_16f/target_encoder_e200.pt"
    ))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent
    out_dir = eval_dir / "embeddings" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_index = load_shard_index(eval_dir)

    df = pd.read_csv(eval_dir.parent / "labels" / TASK_CSV[args.task])
    classes = sorted(df["stratification"].dropna().unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    print(f"\nTask: {args.task}  |  Classes ({len(classes)}): {classes}")

    print("\nBuilding split records ...")
    splits = {
        "train": load_split(df, "train", label_map, shard_index),
        "valid": load_split(df, "valid", label_map, shard_index),
        "test":  load_split(df, "test",  label_map, shard_index),
    }

    encoder = load_encoder(args.checkpoint, args.device)

    for split_name, records in splits.items():
        out_path = out_dir / f"{split_name}.pt"
        if out_path.exists():
            print(f"\n[skip] {split_name} embeddings already exist at {out_path}")
            continue
        print(f"\nExtracting {split_name} ({len(records)} samples) ...")
        data = extract_split(encoder, records, shard_index, args.device,
                             args.batch_size, args.num_workers, args.frames_per_clip)
        torch.save(data, out_path)
        print(f"  Saved {data['embeddings'].shape} → {out_path}")

    print(f"\nDone. Embeddings saved to {out_dir}")


if __name__ == "__main__":
    main()
