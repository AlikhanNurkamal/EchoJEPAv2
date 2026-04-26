#!/usr/bin/env python3
"""
Linear probing for valvular disease severity classification (AS / MR / TR).

Trains a frozen EchoJEPA ViT-Large encoder + linear head on multi-class
stratification labels (Normal / Trace / Mild / Moderate / Severe).

Data is loaded from preprocessed WebDataset shards at:
  /hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa/
  /hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa/

Each shard contains {uuid}.frames.npy (T,336,336,3 uint8 RGB) and
{uuid}.metadata.json. A UUID→shard index is built on first run and
cached to evaluation/shard_index.pkl.

Usage:
    PYTHONPATH=/home/ahmedaly/iCardio/EchoJEPAv2/EchoJEPA \
    /home/ahmedaly/.conda/envs/echofm/bin/python evaluation/linear_probe.py \
        --task AS \
        --checkpoint /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/target_encoder_e200.pt \
        --output-dir evaluation/results \
        --device cuda:0 \
        --epochs 50 \
        --batch-size 32
"""

import argparse
import gc
import io
import json
import os
import pickle
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add EchoJEPA to path if not already present
_repo = Path(__file__).resolve().parent.parent / "EchoJEPA"
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import src.models.vision_transformer as vit
from src.models.linear_pooler import LinearClassifier

# Import metrics from the same evaluation/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_metrics_with_ci, save_metrics_report, save_predictions

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SHARD_DIRS = [
    "/hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa",
    "/hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa",
]

TASK_CSV = {
    "AS": "AS.csv",
    "MR": "MR.csv",
    "TR": "TR.csv",
}


# ---------------------------------------------------------------------------
# Shard index
# ---------------------------------------------------------------------------

def _fast_tar_uuids(tar_path):
    """Read only 512-byte tar headers (skipping file data via seek) to extract UUIDs."""
    uuids = []
    with open(tar_path, "rb") as f:
        while True:
            hdr = f.read(512)
            if len(hdr) < 512 or hdr == b"\x00" * 512:
                break
            name = hdr[:100].rstrip(b"\x00").decode("utf-8", errors="replace")
            size_str = hdr[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
            try:
                size = int(size_str, 8) if size_str else 0
            except ValueError:
                break
            if name.endswith(".frames.npy"):
                uuids.append(name.replace(".frames.npy", ""))
            f.seek(((size + 511) // 512) * 512, 1)
    return uuids


def build_shard_index(shard_dirs, index_path):
    """Scan all tar shards and build a {dicom_uuid: shard_path} dict."""
    index = {}
    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir)
        shards = sorted(shard_dir.glob("shard-*.tar"))
        print(f"  Scanning {len(shards)} shards in {shard_dir} ...")
        for shard_path in shards:
            try:
                for uuid in _fast_tar_uuids(shard_path):
                    index[uuid] = str(shard_path)
            except Exception as e:
                print(f"  Warning: could not read {shard_path}: {e}")
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    print(f"  Index saved: {len(index)} entries → {index_path}")
    return index


def load_or_build_index(eval_dir, shard_dirs):
    index_path = eval_dir / "shard_index.pkl"
    if index_path.exists():
        print(f"Loading shard index from {index_path} ...")
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        print(f"  {len(index)} entries loaded.")
        return index
    print("Building shard index (one-time scan, will be cached) ...")
    return build_shard_index(shard_dirs, index_path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EchoDataset(Dataset):
    def __init__(self, records, shard_index, frames_per_clip=16, training=False):
        self.records = records          # list of (dicom_uuid, label_int)
        self.shard_index = shard_index  # {uuid: shard_path}
        self.frames_per_clip = frames_per_clip
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        dicom_uuid, label = self.records[idx]
        shard_path = self.shard_index[dicom_uuid]

        with tarfile.open(shard_path, "r") as tf:
            raw = tf.extractfile(f"{dicom_uuid}.frames.npy").read()

        frames = np.load(io.BytesIO(raw))  # (T, H, W, 3) uint8 RGB

        T = len(frames)
        N = self.frames_per_clip

        if T == 0:
            frames = np.zeros((N, 336, 336, 3), dtype=np.uint8)
        elif T < N:
            pad = [frames[-1:]] * (N - T)
            frames = np.concatenate([frames] + pad, axis=0)
        else:
            if self.training:
                start = random.randint(0, T - N)
            else:
                start = (T - N) // 2
            frames = frames[start:start + N]

        # Normalize and convert to (C, T, H, W)
        x = frames.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD                         # (N, H, W, 3)
        x = torch.from_numpy(x).permute(3, 0, 1, 2)   # (3, N, H, W)

        return x, label, dicom_uuid


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, device: str):
    print(f"Loading encoder from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    encoder = vit.__dict__["vit_large"](
        img_size=336,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=False,
        use_activation_checkpointing=False,
        use_rope=True,
    )

    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        print(f"  Using '{key}' weights")
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[key].items()}

    epoch = ckpt.get("epoch", "?")
    del ckpt; gc.collect()

    msg = encoder.load_state_dict(state, strict=False)
    print(f"  Load msg: {msg}")
    del state; gc.collect()

    encoder.eval().to(device)
    print(f"  Checkpoint epoch: {epoch}")
    return encoder


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_label_map(df):
    classes = sorted(df["stratification"].dropna().unique().tolist())
    return {c: i for i, c in enumerate(classes)}, classes


def load_split(df, split, label_map, shard_index):
    subset = df[df["designation"] == split].copy()
    records = []
    skipped = 0
    for _, row in subset.iterrows():
        uuid = row["dicom_uuid"]
        label_str = row["stratification"]
        if pd.isna(label_str) or label_str not in label_map:
            skipped += 1
            continue
        if uuid not in shard_index:
            skipped += 1
            continue
        records.append((uuid, label_map[label_str]))
    print(f"  {split}: {len(records)} samples ({skipped} skipped)")
    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(encoder, probe, loader, device):
    probe.eval()
    all_logits, all_labels, all_uuids = [], [], []

    autocast_device = "cuda" if "cuda" in device else "cpu"
    for clips, labels, uuids in loader:
        clips = clips.to(device)
        with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
            tokens = encoder(clips)
        logits = probe(tokens.float())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_uuids.extend(uuids)

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = logits.argmax(axis=1)
    acc = (y_pred == y_true).mean()
    return acc, y_true, y_pred, logits, all_uuids


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
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shard index
    # ------------------------------------------------------------------
    shard_index = load_or_build_index(eval_dir, SHARD_DIRS)

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    csv_path = eval_dir / TASK_CSV[args.task]
    df = pd.read_csv(csv_path)
    label_map, classes = build_label_map(df)
    num_classes = len(classes)
    print(f"\nTask: {args.task}  |  Classes ({num_classes}): {classes}")
    print(f"Label map: {label_map}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\nBuilding datasets ...")
    train_records = load_split(df, "train", label_map, shard_index)
    val_records   = load_split(df, "valid", label_map, shard_index)
    test_records  = load_split(df, "test",  label_map, shard_index)

    def make_loader(records, training):
        ds = EchoDataset(records, shard_index, args.frames_per_clip, training=training)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=training,
            num_workers=args.num_workers,
            pin_memory="cuda" in args.device,
            drop_last=False,
        )

    train_loader = make_loader(train_records, training=True)
    val_loader   = make_loader(val_records,   training=False)
    test_loader  = make_loader(test_records,  training=False)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    encoder = load_encoder(args.checkpoint, args.device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    probe = LinearClassifier(embed_dim=1024, num_classes=num_classes, use_layernorm=True).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    probe_ckpt_path = out_dir / "probe_best.pt"
    best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    autocast_device = "cuda" if "cuda" in args.device else "cpu"
    print(f"\nTraining for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss = 0.0
        n_batches = 0

        for clips, labels, _ in train_loader:
            clips  = clips.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
                    tokens = encoder(clips)

            logits = probe(tokens.float())
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        val_acc, _, _, _, _ = evaluate(encoder, probe, val_loader, args.device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"probe": probe.state_dict(), "epoch": epoch, "val_acc": val_acc}, probe_ckpt_path)

        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={total_loss/n_batches:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation on val + test using best checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading best probe (val_acc={best_val_acc:.4f}) ...")
    ckpt = torch.load(probe_ckpt_path, map_location=args.device, weights_only=False)
    probe.load_state_dict(ckpt["probe"])

    int_to_label = {v: k for k, v in label_map.items()}

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        print(f"\nEvaluating on {split_name} ...")
        acc, y_true, y_pred, logits, uuids = evaluate(encoder, probe, loader, args.device)

        preds_df = pd.DataFrame({
            "dicom_uuid":   uuids,
            "y_true":       y_true,
            "y_pred":       y_pred,
            "y_true_label": [int_to_label[i] for i in y_true],
            "y_pred_label": [int_to_label[i] for i in y_pred],
            "y_prob":       [json.dumps(logits[i].tolist()) for i in range(len(y_true))],
        })
        save_predictions(preds_df, str(out_dir / f"predictions_{split_name}.csv"))

        metrics = compute_metrics_with_ci(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=logits,
            classification_mode="max",
            n_bootstrap=1000,
            confidence_level=0.95,
        )

        from metrics import format_metrics_table
        print(format_metrics_table(metrics))

        save_metrics_report(
            metrics,
            output_path=str(out_dir / "metrics.csv"),
            task_name=args.task,
            mode=f"linear_probe_{split_name}",
            model_name="icardio_vitl16_336px_e200",
            label_mapping=label_map,
        )

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
