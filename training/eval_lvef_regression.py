#!/usr/bin/env python3
"""
EchoJEPA LVEF Regression Evaluation (Linear Probe)

Trains a linear regressor on frozen EchoJEPA encoder features to predict
Left Ventricular Ejection Fraction (LVEF) from EchoNet-Dynamic videos.

Usage:
    PYTHONPATH=/home/ahmedaly/iCardio/EchoJEPAv2/EchoJEPA \
    python training/eval_lvef_regression.py \
        --checkpoint /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/latest.pt \
        --echonet-dir /data/ahmedaly/public/EchoNet_Dynamic/EchoNet-Dynamic \
        --device cuda:1 \
        --output-dir /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/eval_lvef_latest
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.vision_transformer as vit


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, device: str):
    """Load the target encoder (EMA) from a training checkpoint."""
    import gc
    print(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

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

    # Support both full checkpoint and extracted encoder-only checkpoint
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        print(f"Using '{key}' weights")
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[key].items()}

    epoch = ckpt.get("epoch", "?")
    del ckpt; gc.collect()

    msg = encoder.load_state_dict(state, strict=False)
    print(f"Load msg: {msg}")
    del state; gc.collect()

    encoder.eval().to(device)
    print(f"Checkpoint epoch: {epoch}")
    torch.cuda.empty_cache()
    return encoder


# ---------------------------------------------------------------------------
# Data loading — EchoNet-Dynamic
# ---------------------------------------------------------------------------

def load_echonet_split(echonet_dir, split, frames_per_clip=16, target_size=336):
    """Load EchoNet-Dynamic videos for a given split (TRAIN/VAL/TEST)."""
    filelist = pd.read_csv(os.path.join(echonet_dir, "FileList.csv"))
    filelist = filelist[filelist["Split"] == split].reset_index(drop=True)

    videos_dir = os.path.join(echonet_dir, "Videos")
    samples = []
    skipped = 0

    for i, row in filelist.iterrows():
        video_path = os.path.join(videos_dir, f"{row['FileName']}.avi")
        if not os.path.exists(video_path):
            skipped += 1
            continue

        ef = row["EF"]
        if pd.isna(ef):
            skipped += 1
            continue

        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < frames_per_clip:
            skipped += 1
            continue

        # Sample clip (center crop temporally for deterministic eval)
        T = len(frames)
        start = max(0, (T - frames_per_clip) // 2)
        clip_frames = frames[start:start + frames_per_clip]

        # Resize to target_size
        clip = np.stack([
            cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            for f in clip_frames
        ])  # (16, 336, 336, 3)

        # Convert to tensor: (C, T, H, W) float32 normalized to [0, 1]
        clip_tensor = torch.from_numpy(clip).float().permute(3, 0, 1, 2) / 255.0

        samples.append((clip_tensor, ef))

        if (i + 1) % 500 == 0:
            print(f"  Loaded {i + 1}/{len(filelist)} videos ({len(samples)} valid) ...")

    print(f"  {split}: {len(samples)} valid, {skipped} skipped")
    return samples


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(encoder, samples, device, batch_size=8):
    """Extract mean-pooled, L2-normalized features."""
    all_features = []
    all_labels = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        clips = torch.stack([s[0] for s in batch]).to(device)
        labels = [s[1] for s in batch]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(clips)

        feat = out.float().mean(dim=1)  # mean pool over tokens
        feat = F.normalize(feat, dim=-1)

        all_features.append(feat.cpu())
        all_labels.extend(labels)

        if (i // batch_size) % 25 == 0:
            print(f"  Extracted {min(i + batch_size, len(samples))}/{len(samples)} ...")

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.float32)
    return features, labels


# ---------------------------------------------------------------------------
# Linear regression probe
# ---------------------------------------------------------------------------

class LinearRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def train_regression_probe(
    train_feats, train_labels, val_feats, val_labels,
    device, epochs=30, lr=1e-3, wd=1e-4,
):
    in_dim = train_feats.shape[1]
    probe = LinearRegressor(in_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_feats_d = train_feats.to(device)
    train_labels_d = train_labels.to(device)
    val_feats_d = val_feats.to(device)
    val_labels_d = val_labels.to(device)

    best_val_mae = float("inf")
    best_epoch = 0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING LINEAR REGRESSOR — {train_feats.shape[0]} train, {val_feats.shape[0]} val")
    print(f"  lr={lr}, wd={wd}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # Train
        probe.train()
        preds = probe(train_feats_d)
        loss = F.mse_loss(preds, train_labels_d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_mae = (preds - train_labels_d).abs().mean().item()

        # Val
        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_feats_d)
            val_mse = F.mse_loss(val_preds, val_labels_d).item()
            val_mae = (val_preds - val_labels_d).abs().mean().item()
            val_rmse = val_mse ** 0.5

            # Pearson correlation
            vp = val_preds - val_preds.mean()
            vl = val_labels_d - val_labels_d.mean()
            pearson_r = (vp * vl).sum() / (vp.norm() * vl.norm() + 1e-8)
            pearson_r = pearson_r.item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            best_metrics = {
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_mse": val_mse,
                "pearson_r": pearson_r,
            }

        history.append({
            "epoch": epoch,
            "train_mse": loss.item(),
            "train_mae": train_mae,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "pearson_r": pearson_r,
        })

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>3d}: train_mae={train_mae:.3f} | val_mae={val_mae:.3f} val_rmse={val_rmse:.3f} r={pearson_r:.4f}")

    print(f"\nBest val MAE: {best_val_mae:.3f} at epoch {best_epoch}")
    print(f"  RMSE: {best_metrics['val_rmse']:.3f}, Pearson r: {best_metrics['pearson_r']:.4f}")

    # Restore best model for final predictions
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        final_preds = probe(val_feats_d).cpu()

    return probe, best_metrics, history, final_preds


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualizations(val_labels, val_preds, history, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Training curves ---
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [h["train_mae"] for h in history], label="Train MAE")
    ax1.plot(epochs, [h["val_mae"] for h in history], label="Val MAE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (EF %)")
    ax1.set_title("Mean Absolute Error")
    ax1.legend()
    ax2.plot(epochs, [h["pearson_r"] for h in history], label="Pearson r", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pearson r")
    ax2.set_title("Correlation")
    ax2.legend()
    fig.suptitle("LVEF Regression Training")
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/training_curves.png")

    # --- Scatter: predicted vs actual ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(val_labels.numpy(), val_preds.numpy(), alpha=0.3, s=10, color="steelblue")
    lims = [min(val_labels.min(), val_preds.min()) - 5, max(val_labels.max(), val_preds.max()) + 5]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True EF (%)")
    ax.set_ylabel("Predicted EF (%)")
    ax.set_title("LVEF: Predicted vs Actual")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(os.path.join(output_dir, "scatter_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/scatter_pred_vs_actual.png")

    # --- Error distribution ---
    errors = (val_preds - val_labels).numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Prediction Error (EF %)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (mean={errors.mean():.2f}, std={errors.std():.2f})")
    fig.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/error_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EchoJEPA LVEF regression eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--echonet-dir", type=str, required=True,
                        help="Path to EchoNet-Dynamic root (containing FileList.csv and Videos/)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=336)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(ckpt_dir, "eval_lvef_latest")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load encoder
    t0 = time.time()
    encoder = load_encoder(args.checkpoint, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # 2) Load EchoNet-Dynamic data
    print("Loading TRAIN videos ...")
    t0 = time.time()
    train_samples = load_echonet_split(args.echonet_dir, "TRAIN", args.frames_per_clip, args.target_size)
    print("Loading VAL videos ...")
    val_samples = load_echonet_split(args.echonet_dir, "VAL", args.frames_per_clip, args.target_size)
    print("Loading TEST videos ...")
    test_samples = load_echonet_split(args.echonet_dir, "TEST", args.frames_per_clip, args.target_size)
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    # EF statistics
    train_efs = [s[1] for s in train_samples]
    print(f"Train EF stats: mean={np.mean(train_efs):.2f}, std={np.std(train_efs):.2f}, "
          f"min={np.min(train_efs):.2f}, max={np.max(train_efs):.2f}")

    # 3) Extract features
    print("\nExtracting TRAIN features ...")
    t0 = time.time()
    train_feats, train_labels = extract_features(encoder, train_samples, args.device, args.batch_size)
    print("Extracting VAL features ...")
    val_feats, val_labels = extract_features(encoder, val_samples, args.device, args.batch_size)
    print("Extracting TEST features ...")
    test_feats, test_labels = extract_features(encoder, test_samples, args.device, args.batch_size)
    print(f"Features extracted in {time.time() - t0:.1f}s\n")

    del encoder
    torch.cuda.empty_cache()

    # 4) Train with hyperparameter search
    best_overall_mae = float("inf")
    best_config = None
    all_results = []

    for lr in [1e-2, 1e-3, 5e-4, 1e-4]:
        for wd in [1e-4, 1e-2, 1e-1]:
            probe, metrics, history, val_preds = train_regression_probe(
                train_feats, train_labels, val_feats, val_labels,
                args.device, args.epochs, lr, wd,
            )
            all_results.append({"lr": lr, "wd": wd, **metrics})

            if metrics["val_mae"] < best_overall_mae:
                best_overall_mae = metrics["val_mae"]
                best_config = (lr, wd)
                best_metrics = metrics
                best_history = history
                best_probe_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                best_val_preds = val_preds

    print(f"\n{'='*60}")
    print(f"BEST CONFIG: lr={best_config[0]}, wd={best_config[1]}")
    print(f"  Val MAE:  {best_metrics['val_mae']:.3f}")
    print(f"  Val RMSE: {best_metrics['val_rmse']:.3f}")
    print(f"  Pearson r: {best_metrics['pearson_r']:.4f}")
    print(f"{'='*60}")

    # 5) Evaluate best model on TEST set
    print("\nEvaluating best model on TEST set ...")
    probe = LinearRegressor(train_feats.shape[1]).to(args.device)
    probe.load_state_dict(best_probe_state)
    probe.eval()
    with torch.no_grad():
        test_feats_d = test_feats.to(args.device)
        test_labels_d = test_labels.to(args.device)
        test_preds = probe(test_feats_d)
        test_mae = (test_preds - test_labels_d).abs().mean().item()
        test_mse = F.mse_loss(test_preds, test_labels_d).item()
        test_rmse = test_mse ** 0.5
        tp = test_preds - test_preds.mean()
        tl = test_labels_d - test_labels_d.mean()
        test_pearson = (tp * tl).sum() / (tp.norm() * tl.norm() + 1e-8)
        test_pearson = test_pearson.item()
        test_preds_cpu = test_preds.cpu()

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"  MAE:       {test_mae:.3f}")
    print(f"  RMSE:      {test_rmse:.3f}")
    print(f"  Pearson r: {test_pearson:.4f}")
    print(f"{'='*60}")

    # 6) Visualizations
    print("\nSaving visualizations ...")
    save_visualizations(test_labels, test_preds_cpu, best_history, args.output_dir)

    # 7) Save results
    results = {
        "checkpoint": args.checkpoint,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "best_lr": best_config[0],
        "best_wd": best_config[1],
        "val_mae": best_metrics["val_mae"],
        "val_rmse": best_metrics["val_rmse"],
        "val_pearson_r": best_metrics["pearson_r"],
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_pearson_r": test_pearson,
        "all_runs": all_results,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")
    print("Done!")


if __name__ == "__main__":
    main()
