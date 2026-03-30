#!/usr/bin/env python3
"""
EchoJEPA View Classification Evaluation (Linear Probe)

Trains a linear classifier on frozen EchoJEPA encoder features to predict
the echocardiogram view type (A4C, PLAX, PSAX, etc.).

Reads data directly from WebDataset .tar shards, using the 'view' field
in metadata as labels.

Usage:
    PYTHONPATH=/home/ahmedaly/iCardio/EchoJEPAv2/EchoJEPA \
    python training/eval_view_classification.py \
        --checkpoint /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/latest.pt \
        --shard-dirs /hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa \
                     /hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa \
        --num-train 5000 --num-val 1000 \
        --device cuda:1 \
        --epochs 20 \
        --output-dir /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/eval_view_cls
"""

import argparse
import io
import json
import os
import tarfile
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

import src.models.vision_transformer as vit


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, device: str):
    """Load the target encoder (EMA) from a training checkpoint."""
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

    key = "target_encoder" if "target_encoder" in ckpt else "encoder"
    print(f"Using '{key}' weights")
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[key].items()}
    msg = encoder.load_state_dict(state, strict=False)
    print(f"Load msg: {msg}")

    encoder.eval().to(device)
    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint epoch: {epoch}")
    del ckpt
    torch.cuda.empty_cache()
    return encoder


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Exclude non-informative views
EXCLUDE_VIEWS = {"nan", "Line Graph", "Other", "Unclear Noisy", "Unclear Dark"}


def gather_shards(shard_dirs, min_size=10_000):
    shards = []
    for d in shard_dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith("shard-") and f.endswith(".tar"):
                p = os.path.join(d, f)
                if os.path.getsize(p) >= min_size:
                    shards.append(p)
    return shards


def load_labeled_samples(shards, num_samples, frames_per_clip=16, seed=42):
    """Load samples that have valid view labels."""
    stride = 1  # fps_stored == fps_sample == 24
    samples = []
    rng = np.random.default_rng(seed)
    shuffled = list(shards)
    rng.shuffle(shuffled)

    for shard_path in shuffled:
        if len(samples) >= num_samples:
            break
        try:
            with tarfile.open(shard_path, "r:") as tar:
                pending_frames = {}
                pending_meta = {}
                for member in tar:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if name.endswith(".frames.npy"):
                        uuid = name.replace(".frames.npy", "")
                        pending_frames[uuid] = member
                    elif name.endswith(".metadata.json"):
                        uuid = name.replace(".metadata.json", "")
                        pending_meta[uuid] = member

                    for uuid in list(pending_frames.keys()):
                        if uuid in pending_meta:
                            try:
                                frames_member = pending_frames.pop(uuid)
                                meta_member = pending_meta.pop(uuid)

                                mf = tar.extractfile(meta_member)
                                meta = json.loads(mf.read().decode())

                                view = meta.get("view", "nan")
                                if view in EXCLUDE_VIEWS or view is None:
                                    continue

                                f = tar.extractfile(frames_member)
                                frames = np.load(io.BytesIO(f.read()))
                                T_total = frames.shape[0]

                                available = T_total // stride
                                if available < frames_per_clip:
                                    continue

                                max_start = available - frames_per_clip
                                start = rng.integers(0, max_start + 1) if max_start > 0 else 0
                                indices = np.arange(start, start + frames_per_clip) * stride

                                clip = frames[indices]
                                clip = torch.from_numpy(clip).float().permute(3, 0, 1, 2) / 255.0

                                samples.append((clip, view))

                                if len(samples) >= num_samples:
                                    break
                            except Exception:
                                continue

                    if len(samples) >= num_samples:
                        break
        except Exception as e:
            print(f"  Skipping {shard_path}: {e}")
            continue

        if len(samples) % 500 < 50:
            print(f"  Loaded {len(samples)}/{num_samples} labeled samples ...")

    print(f"Loaded {len(samples)} labeled samples")
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
        batch = samples[i : i + batch_size]
        clips = torch.stack([s[0] for s in batch]).to(device)
        labels = [s[1] for s in batch]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(clips)

        feat = out.float().mean(dim=1)
        feat = F.normalize(feat, dim=-1)

        all_features.append(feat.cpu())
        all_labels.extend(labels)

        if (i // batch_size) % 25 == 0:
            print(f"  Extracted {min(i + batch_size, len(samples))}/{len(samples)} ...")

    features = torch.cat(all_features, dim=0)
    return features, all_labels


# ---------------------------------------------------------------------------
# Linear probe training
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(
    train_feats, train_labels_idx, val_feats, val_labels_idx,
    num_classes, label_names, device, epochs=20, lr=1e-3, wd=1e-4, output_dir=None,
):
    in_dim = train_feats.shape[1]
    probe = LinearProbe(in_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights for imbalanced data
    class_counts = torch.bincount(train_labels_idx, minlength=num_classes).float()
    class_weights = (1.0 / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_feats_d = train_feats.to(device)
    train_labels_d = train_labels_idx.to(device)
    val_feats_d = val_feats.to(device)
    val_labels_d = val_labels_idx.to(device)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING LINEAR PROBE — {num_classes} classes, {train_feats.shape[0]} train, {val_feats.shape[0]} val")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # Train (full batch — features are small enough)
        probe.train()
        logits = probe(train_feats_d)
        loss = criterion(logits, train_labels_d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_acc = (logits.argmax(1) == train_labels_d).float().mean().item()

        # Val
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats_d)
            val_loss = F.cross_entropy(val_logits, val_labels_d).item()
            val_preds = val_logits.argmax(1)
            val_acc = (val_preds == val_labels_d).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": loss.item(),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>3d}: train_loss={loss.item():.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Load best model and compute detailed metrics
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_feats_d)
        val_preds = val_logits.argmax(1).cpu().numpy()
        val_true = val_labels_idx.numpy()

    # Per-class report
    report = classification_report(val_true, val_preds, target_names=label_names, digits=4, zero_division=0)
    print(f"\nClassification Report (best epoch {best_epoch}):")
    print(report)

    # Top-5 accuracy
    val_logits_cpu = val_logits.cpu()
    top5_preds = val_logits_cpu.topk(min(5, num_classes), dim=1).indices
    top5_correct = (top5_preds == val_labels_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    print(f"Top-1 Accuracy: {best_val_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_correct:.4f}")

    return probe, best_val_acc, top5_correct, history, report, val_preds, val_true


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualizations(
    train_feats, train_labels, val_feats, val_labels,
    val_preds, val_true, label_names, history, output_dir
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib/sklearn not available, skipping visualizations")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Training curves ---
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="Train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    fig.suptitle("Linear Probe Training")
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/training_curves.png")

    # --- Confusion matrix ---
    cm = confusion_matrix(val_true, val_preds)
    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/confusion_matrix.png")

    # --- t-SNE colored by view ---
    feats_all = torch.cat([train_feats, val_feats], dim=0).numpy()
    labels_all = train_labels + val_labels
    unique_labels = sorted(set(labels_all))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    color_idx = np.array([label_to_idx[l] for l in labels_all])

    N = feats_all.shape[0]
    # Subsample for t-SNE if too large
    if N > 3000:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, 3000, replace=False)
        feats_sub = feats_all[idx]
        color_sub = color_idx[idx]
    else:
        feats_sub = feats_all
        color_sub = color_idx

    print("  Computing t-SNE (colored by view) ...")
    perplexity = min(30, len(feats_sub) // 4)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(feats_sub)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=color_sub, cmap="tab20", alpha=0.6, s=10)

    # Legend
    handles = []
    for i, lbl in enumerate(unique_labels):
        mask = color_sub == i
        if mask.any():
            h = ax.scatter([], [], c=[plt.cm.tab20(i / max(len(unique_labels) - 1, 1))], s=30, label=lbl)
            handles.append(h)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7, ncol=1)
    ax.set_title(f"t-SNE of EchoJEPA Features Colored by View (N={len(feats_sub)})")
    fig.savefig(os.path.join(output_dir, "tsne_views.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/tsne_views.png")

    # --- Per-class accuracy bar chart ---
    cm_diag = np.diag(cm_norm)
    sorted_idx = np.argsort(cm_diag)[::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(label_names)), cm_diag[sorted_idx], color="steelblue")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels([label_names[i] for i in sorted_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy (View Classification)")
    ax.axhline(y=cm_diag.mean(), color="red", linestyle="--", label=f"Mean: {cm_diag.mean():.3f}")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/per_class_accuracy.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EchoJEPA view classification eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shard-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join(ckpt_dir, f"eval_view_cls_{ckpt_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load encoder
    t0 = time.time()
    encoder = load_encoder(args.checkpoint, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # 2) Load labeled data (different seeds for train/val split)
    print("Loading TRAIN samples ...")
    t0 = time.time()
    shards = gather_shards(args.shard_dirs)
    train_samples = load_labeled_samples(shards, args.num_train, args.frames_per_clip, seed=42)

    print("\nLoading VAL samples ...")
    val_samples = load_labeled_samples(shards, args.num_val, args.frames_per_clip, seed=9999)
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    # Build label mapping
    all_views = [s[1] for s in train_samples] + [s[1] for s in val_samples]
    unique_views = sorted(set(all_views))
    view_to_idx = {v: i for i, v in enumerate(unique_views)}
    num_classes = len(unique_views)

    print(f"Classes ({num_classes}): {unique_views}")
    train_view_counts = Counter([s[1] for s in train_samples])
    print(f"\nTrain distribution:")
    for v in unique_views:
        print(f"  {v}: {train_view_counts.get(v, 0)}")

    # 3) Extract features
    print("\nExtracting TRAIN features ...")
    t0 = time.time()
    train_feats, train_labels = extract_features(encoder, train_samples, args.device, args.batch_size)
    print("Extracting VAL features ...")
    val_feats, val_labels = extract_features(encoder, val_samples, args.device, args.batch_size)
    print(f"Features extracted in {time.time() - t0:.1f}s\n")

    del encoder
    torch.cuda.empty_cache()

    # Convert labels to indices
    train_labels_idx = torch.tensor([view_to_idx[l] for l in train_labels])
    val_labels_idx = torch.tensor([view_to_idx[l] for l in val_labels])

    # 4) Train linear probe with hyperparameter search
    best_overall = 0.0
    best_config = None
    all_results = []

    for lr in [1e-2, 1e-3, 5e-4]:
        for wd in [1e-4, 1e-2]:
            print(f"\n--- lr={lr}, wd={wd} ---")
            probe, val_acc, top5_acc, history, report, val_preds, val_true = train_linear_probe(
                train_feats, train_labels_idx, val_feats, val_labels_idx,
                num_classes, unique_views, args.device, args.epochs, lr, wd, args.output_dir,
            )
            all_results.append({
                "lr": lr, "wd": wd,
                "val_acc": val_acc, "top5_acc": top5_acc,
                "history": history,
            })
            if val_acc > best_overall:
                best_overall = val_acc
                best_config = (lr, wd)
                best_report = report
                best_history = history
                best_preds = val_preds
                best_true = val_true
                best_top5 = top5_acc

    print(f"\n{'='*60}")
    print(f"BEST CONFIG: lr={best_config[0]}, wd={best_config[1]}")
    print(f"Top-1 Accuracy: {best_overall:.4f}")
    print(f"Top-5 Accuracy: {best_top5:.4f}")
    print(f"{'='*60}")

    # 5) Visualizations
    print("\nSaving visualizations ...")
    save_visualizations(
        train_feats, train_labels, val_feats, val_labels,
        best_preds, best_true, unique_views, best_history, args.output_dir,
    )

    # 6) Save results
    results = {
        "checkpoint": args.checkpoint,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_classes": num_classes,
        "classes": unique_views,
        "best_lr": best_config[0],
        "best_wd": best_config[1],
        "best_top1_acc": best_overall,
        "best_top5_acc": best_top5,
        "all_runs": [{k: v for k, v in r.items() if k != "history"} for r in all_results],
        "classification_report": best_report,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")
    print("Done!")


if __name__ == "__main__":
    main()
