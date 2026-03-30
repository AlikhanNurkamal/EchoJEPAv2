#!/usr/bin/env python3
"""
EchoJEPA Feature Extraction + Retrieval Evaluation

Extracts features from the frozen (target) encoder on a subset of WebDataset
shards, then evaluates:
  1. kNN retrieval accuracy (using cosine similarity)
  2. Cosine similarity distribution statistics
  3. t-SNE / PCA visualizations saved as images

Usage:
    PYTHONPATH=/home/ahmedaly/iCardio/EchoJEPAv2/EchoJEPA \
    python training/eval_retrieval.py \
        --checkpoint /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/latest.pt \
        --shard-dirs /hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa \
                     /hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa \
        --num-samples 2000 \
        --device cuda:1 \
        --output-dir /home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/eval_retrieval
"""

import argparse
import io
import json
import os
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import src.models.vision_transformer as vit


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, device: str):
    """Load the target encoder (EMA) from a training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Model config — match our pretrain yaml
    model_name = "vit_large"
    img_size = 336
    patch_size = 16
    num_frames = 16
    tubelet_size = 2

    encoder = vit.__dict__[model_name](
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=False,
        use_activation_checkpointing=False,
        use_rope=True,
    )

    # Prefer target_encoder (EMA), fall back to encoder
    key = "target_encoder" if "target_encoder" in ckpt else "encoder"
    print(f"Using '{key}' weights from checkpoint")
    state = ckpt[key]
    # Strip DDP / wrapper prefixes
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

    msg = encoder.load_state_dict(state, strict=False)
    print(f"Loaded with msg: {msg}")

    encoder.eval().to(device)
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", "?")
    print(f"Checkpoint epoch: {epoch}, loss: {loss}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    del ckpt
    torch.cuda.empty_cache()
    return encoder


# ---------------------------------------------------------------------------
# Data loading from tar shards
# ---------------------------------------------------------------------------

def gather_shards(shard_dirs, min_size=10_000):
    """Collect all valid shard paths."""
    shards = []
    for d in shard_dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith("shard-") and f.endswith(".tar"):
                p = os.path.join(d, f)
                if os.path.getsize(p) >= min_size:
                    shards.append(p)
    print(f"Found {len(shards)} valid shards across {len(shard_dirs)} directories")
    return shards


def load_samples_from_shards(shards, num_samples, frames_per_clip=16, fps_stored=24, fps_sample=24):
    """
    Stream through shards and collect processed video clips + metadata.
    Returns list of (clip_tensor, metadata_dict).
    clip_tensor: (C, T, H, W) float32, normalized to [0, 1].
    """
    stride = max(1, fps_stored // fps_sample)
    samples = []
    rng = np.random.default_rng(42)
    shuffled = list(shards)
    rng.shuffle(shuffled)

    for shard_path in shuffled:
        if len(samples) >= num_samples:
            break
        try:
            with tarfile.open(shard_path, "r:") as tar:
                # Collect .frames.npy members
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

                    # Process completed pairs
                    for uuid in list(pending_frames.keys()):
                        if uuid in pending_meta:
                            try:
                                frames_member = pending_frames.pop(uuid)
                                meta_member = pending_meta.pop(uuid)

                                # Load frames
                                f = tar.extractfile(frames_member)
                                frames = np.load(io.BytesIO(f.read()))  # (T, H, W, 3)
                                T_total = frames.shape[0]

                                # Load metadata
                                mf = tar.extractfile(meta_member)
                                meta = json.loads(mf.read().decode())

                                # Temporal sampling
                                available = T_total // stride
                                if available < frames_per_clip:
                                    continue

                                # Random start
                                max_start = available - frames_per_clip
                                start = rng.integers(0, max_start + 1) if max_start > 0 else 0
                                indices = np.arange(start, start + frames_per_clip) * stride

                                clip = frames[indices]  # (T, H, W, 3)
                                # Convert to (C, T, H, W) float32 [0, 1]
                                clip = torch.from_numpy(clip).float().permute(3, 0, 1, 2) / 255.0

                                meta["_shard"] = os.path.basename(shard_path)
                                meta["_uuid"] = uuid
                                samples.append((clip, meta))

                                if len(samples) >= num_samples:
                                    break
                            except Exception:
                                continue

                    if len(samples) >= num_samples:
                        break
        except Exception as e:
            print(f"  Skipping shard {shard_path}: {e}")
            continue

        if len(samples) % 200 < 50:
            print(f"  Loaded {len(samples)}/{num_samples} samples ...")

    print(f"Loaded {len(samples)} samples total")
    return samples


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(encoder, samples, device, batch_size=8):
    """
    Extract mean-pooled features from the encoder.
    Returns: features (N, D), metadata list.
    """
    all_features = []
    all_meta = []

    for i in range(0, len(samples), batch_size):
        batch_clips = []
        batch_meta = []
        for clip, meta in samples[i : i + batch_size]:
            batch_clips.append(clip)
            batch_meta.append(meta)

        x = torch.stack(batch_clips).to(device)  # (B, C, T, H, W)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = encoder(x)  # (B, N, D)

        # Mean pool over all tokens
        feat = out.float().mean(dim=1)  # (B, D)
        feat = F.normalize(feat, dim=-1)  # L2 normalize

        all_features.append(feat.cpu())
        all_meta.extend(batch_meta)

        if (i // batch_size) % 20 == 0:
            print(f"  Extracted {min(i + batch_size, len(samples))}/{len(samples)} ...")

    features = torch.cat(all_features, dim=0)  # (N, D)
    print(f"Feature matrix: {features.shape}")
    return features, all_meta


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(features, metadata, k_values=(1, 5, 10, 20)):
    """
    Compute kNN retrieval metrics using cosine similarity.
    Groups samples by metadata keys (e.g. same study, same view).
    """
    N, D = features.shape
    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVALUATION — {N} samples, {D}-dim features")
    print(f"{'='*60}")

    # Cosine similarity matrix (features already L2-normed)
    sim = features @ features.T  # (N, N)
    # Zero out self-similarity
    sim.fill_diagonal_(-1.0)

    # --- Global similarity stats ---
    upper_tri = sim[torch.triu_indices(N, N, offset=1)[0], torch.triu_indices(N, N, offset=1)[1]]
    print(f"\nCosine Similarity Distribution (all pairs):")
    print(f"  Mean:   {upper_tri.mean().item():.4f}")
    print(f"  Std:    {upper_tri.std().item():.4f}")
    print(f"  Min:    {upper_tri.min().item():.4f}")
    print(f"  Max:    {upper_tri.max().item():.4f}")
    print(f"  Median: {upper_tri.median().item():.4f}")

    # --- Build group labels from metadata ---
    results = {}

    # Group by study (dicom_study_uuid or study_uuid)
    study_labels = []
    for m in metadata:
        study = m.get("dicom_study_uuid", m.get("study_uuid", m.get("_uuid", "unknown")))
        study_labels.append(study)

    unique_studies = list(set(study_labels))
    if len(unique_studies) < N:
        # There are shared studies — we can measure retrieval
        study_to_idx = {s: i for i, s in enumerate(unique_studies)}
        label_tensor = torch.tensor([study_to_idx[s] for s in study_labels])
        results["study"] = _compute_retrieval_at_k(sim, label_tensor, k_values, "Study-level")

    # Group by view type if available
    view_labels = []
    for m in metadata:
        view = m.get("view_label", m.get("view", None))
        view_labels.append(view)

    if any(v is not None for v in view_labels):
        # Replace None with "unknown"
        view_labels = [v if v is not None else "unknown" for v in view_labels]
        unique_views = list(set(view_labels))
        if len(unique_views) > 1:
            view_to_idx = {v: i for i, v in enumerate(unique_views)}
            label_tensor = torch.tensor([view_to_idx[v] for v in view_labels])
            results["view"] = _compute_retrieval_at_k(sim, label_tensor, k_values, "View-level")

    # --- Self-retrieval: query each sample, check if nearest neighbors
    # come from same shard (proxy for temporal/patient similarity) ---
    shard_labels = [m.get("_shard", "unknown") for m in metadata]
    unique_shards = list(set(shard_labels))
    if len(unique_shards) > 1:
        shard_to_idx = {s: i for i, s in enumerate(unique_shards)}
        label_tensor = torch.tensor([shard_to_idx[s] for s in shard_labels])
        results["shard"] = _compute_retrieval_at_k(sim, label_tensor, k_values, "Shard-level")

    # --- Feature space quality metrics ---
    print(f"\nFeature Space Quality:")
    # Uniformity: how uniformly distributed features are on the hypersphere
    # Lower is better (more uniform)
    sq_dist = 2 - 2 * sim  # squared distance on unit sphere
    uniformity = torch.log(torch.exp(-2 * sq_dist).mean()).item()
    print(f"  Uniformity (log): {uniformity:.4f}  (lower = more uniform, target < -1)")

    # Effective rank of feature matrix (via SVD)
    _, s, _ = torch.svd(features)
    p = s / s.sum()
    eff_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()
    print(f"  Effective rank: {eff_rank:.1f} / {D}  (higher = features use more dimensions)")

    # Avg k-NN distance
    topk_sim, _ = sim.topk(10, dim=1)
    print(f"  Avg cosine sim to 10-NN: {topk_sim.mean().item():.4f}")
    print(f"  Avg cosine sim to 1-NN:  {topk_sim[:, 0].mean().item():.4f}")

    return results, {
        "uniformity": uniformity,
        "effective_rank": eff_rank,
        "avg_sim_10nn": topk_sim.mean().item(),
        "avg_sim_1nn": topk_sim[:, 0].mean().item(),
        "sim_mean": upper_tri.mean().item(),
        "sim_std": upper_tri.std().item(),
    }


def _compute_retrieval_at_k(sim, labels, k_values, name):
    """Compute Recall@K: fraction of queries where at least one of the top-K
    neighbors has the same label."""
    N = sim.size(0)
    max_k = max(k_values)

    # Check how many items share each label
    label_counts = torch.bincount(labels)
    # Only evaluate on labels with at least 2 samples
    valid_mask = label_counts[labels] >= 2
    n_valid = valid_mask.sum().item()

    if n_valid < 10:
        print(f"\n{name} Retrieval: Too few samples with shared labels ({n_valid}), skipping.")
        return {}

    topk_indices = sim.topk(max_k, dim=1).indices  # (N, max_k)
    topk_labels = labels[topk_indices]  # (N, max_k)

    query_labels = labels.unsqueeze(1).expand_as(topk_labels)
    matches = (topk_labels == query_labels)  # (N, max_k)

    results = {}
    print(f"\n{name} Retrieval ({n_valid} valid queries, {label_counts[labels[valid_mask]].float().mean():.1f} avg group size):")
    for k in k_values:
        if k > max_k:
            continue
        recall = matches[valid_mask, :k].any(dim=1).float().mean().item()
        precision = matches[valid_mask, :k].float().mean().item()
        results[f"recall@{k}"] = recall
        results[f"precision@{k}"] = precision
        print(f"  Recall@{k:>3d}: {recall:.4f}   Precision@{k:>3d}: {precision:.4f}")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualizations(features, metadata, output_dir):
    """Save t-SNE and PCA visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib or sklearn not available, skipping visualizations")
        return

    os.makedirs(output_dir, exist_ok=True)
    feats_np = features.numpy()
    N = feats_np.shape[0]

    # Color by shard (proxy for patient/study grouping)
    shard_labels = [m.get("_shard", "unknown") for m in metadata]
    unique_shards = sorted(set(shard_labels))
    shard_to_idx = {s: i for i, s in enumerate(unique_shards)}
    colors = np.array([shard_to_idx[s] for s in shard_labels])

    # --- PCA ---
    print("Computing PCA ...")
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(feats_np)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, cmap="tab20", alpha=0.5, s=8)
    ax.set_title(f"PCA of EchoJEPA Features (N={N})\nPC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(os.path.join(output_dir, "pca.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/pca.png")

    # --- t-SNE ---
    if N > 100:
        print("Computing t-SNE (this may take a minute) ...")
        perplexity = min(30, N // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        tsne_2d = tsne.fit_transform(feats_np)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colors, cmap="tab20", alpha=0.5, s=8)
        ax.set_title(f"t-SNE of EchoJEPA Features (N={N}, perplexity={perplexity})")
        fig.savefig(os.path.join(output_dir, "tsne.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {output_dir}/tsne.png")

    # --- Similarity histogram ---
    sim = features @ features.T
    sim.fill_diagonal_(float("nan"))
    sim_vals = sim[~sim.isnan()].numpy()
    # Subsample for histogram if too many
    if len(sim_vals) > 500_000:
        sim_vals = np.random.default_rng(42).choice(sim_vals, 500_000, replace=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_vals, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(sim_vals.mean(), color="red", linestyle="--", label=f"Mean: {sim_vals.mean():.3f}")
    ax.set_title(f"Cosine Similarity Distribution (N={N}, {len(sim_vals):,} pairs)")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "similarity_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/similarity_hist.png")

    # --- Nearest neighbor gallery ---
    save_nn_gallery(features, metadata, output_dir, n_queries=8, k=5)


def save_nn_gallery(features, metadata, output_dir, n_queries=8, k=5):
    """Save a visual gallery of queries and their nearest neighbors."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    N = features.shape[0]
    sim = features @ features.T
    sim.fill_diagonal_(-1.0)

    rng = np.random.default_rng(123)
    query_idxs = rng.choice(N, size=min(n_queries, N), replace=False)

    fig, axes = plt.subplots(n_queries, k + 1, figsize=(3 * (k + 1), 3 * n_queries))
    if n_queries == 1:
        axes = axes.reshape(1, -1)

    for row, qi in enumerate(query_idxs):
        topk = sim[qi].topk(k)
        nn_idxs = topk.indices.numpy()
        nn_sims = topk.values.numpy()

        # Show query info
        qm = metadata[qi]
        axes[row, 0].text(0.5, 0.5,
                          f"Query #{qi}\n{qm.get('_shard', '?')}\n{qm.get('_uuid', '?')[:12]}",
                          ha="center", va="center", fontsize=8, transform=axes[row, 0].transAxes)
        axes[row, 0].set_title("Query", fontsize=9)
        axes[row, 0].axis("off")

        for col, (ni, ns) in enumerate(zip(nn_idxs, nn_sims)):
            nm = metadata[ni]
            same_shard = "SAME" if nm.get("_shard") == qm.get("_shard") else "diff"
            axes[row, col + 1].text(0.5, 0.5,
                                    f"#{ni}\nsim={ns:.3f}\n{same_shard} shard\n{nm.get('_uuid', '?')[:12]}",
                                    ha="center", va="center", fontsize=7, transform=axes[row, col + 1].transAxes)
            axes[row, col + 1].set_title(f"NN-{col+1}", fontsize=9)
            axes[row, col + 1].axis("off")

    fig.suptitle("Nearest Neighbor Retrieval Gallery", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nn_gallery.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir}/nn_gallery.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EchoJEPA retrieval evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--shard-dirs", type=str, nargs="+", required=True, help="Directories containing shard-*.tar")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save results")
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 20])
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join(ckpt_dir, f"eval_retrieval_{ckpt_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    # 1) Load model
    t0 = time.time()
    encoder = load_encoder(args.checkpoint, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # 2) Load data
    t0 = time.time()
    shards = gather_shards(args.shard_dirs)
    samples = load_samples_from_shards(shards, args.num_samples, args.frames_per_clip)
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    # 3) Extract features
    t0 = time.time()
    features, metadata = extract_features(encoder, samples, args.device, args.batch_size)
    print(f"Features extracted in {time.time() - t0:.1f}s\n")

    # Free GPU memory
    del encoder
    torch.cuda.empty_cache()

    # 4) Evaluate retrieval
    retrieval_results, quality_metrics = evaluate_retrieval(features, metadata, args.k_values)

    # 5) Save visualizations
    print(f"\nSaving visualizations to {args.output_dir} ...")
    save_visualizations(features, metadata, args.output_dir)

    # 6) Save results to JSON
    results = {
        "checkpoint": args.checkpoint,
        "num_samples": len(samples),
        "feature_dim": features.shape[1],
        "quality_metrics": quality_metrics,
        "retrieval": {k: v for k, v in retrieval_results.items()},
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("Done!")


if __name__ == "__main__":
    main()
