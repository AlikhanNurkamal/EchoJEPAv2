#!/usr/bin/env python3
"""
EchoJEPAv2 linear probe evaluation on iCardio downstream tasks.

Two-phase pipeline:
  Phase 1 — Extract and cache embeddings for all DICOMs needed across selected tasks.
             Skipped if cache already exists for this checkpoint.
  Phase 2 — For each task: aggregate DICOM embeddings to study level, train sklearn
             probe (logistic regression or ridge), evaluate with bootstrap CIs.

Usage:
  # Run all default tasks on a single checkpoint
  python evaluation/eval_icardio.py \\
      --checkpoint checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu/latest.pt \\
      --run-name 5pct \\
      --device cuda:1

  # Specific tasks only
  python evaluation/eval_icardio.py \\
      --checkpoint checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu_30pct/latest.pt \\
      --run-name 30pct --tasks as_binary ef mr_binary lv_systolic --device cuda:1

  # Skip extraction if embeddings already cached, re-run probes only
  python evaluation/eval_icardio.py \\
      --checkpoint checkpoints/pretrain/.../latest.pt \\
      --run-name 30pct --probe-only

Outputs per run-name:
  {output_dir}/{run_name}/
    embeddings.pt           — {dicom_uuid: (1024,) float32 tensor} embedding cache
    {task}/metrics.json     — metrics with 95% bootstrap CIs
    {task}/predictions.csv  — study_id, y_true, y_pred, [y_prob]
    summary.csv             — one row per task, key metrics side-by-side
"""

import argparse
import gc
import io
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Repo path ─────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "EchoJEPA"))

import src.models.vision_transformer as vit

# ── Paths (defaults; all overridable via CLI args) ─────────────────────────────
EVAL_DIR    = Path(__file__).resolve().parent
REPO_DIR    = EVAL_DIR.parent
_DEFAULT_LABELS_DIR  = Path("/home/mashrafimonon/iCardio/output_with_labels/output")
_DEFAULT_MANIFEST    = _DEFAULT_LABELS_DIR / "manifest_clinical_findings_with_eval_labels.parquet"
_DEFAULT_SHARD_INDEX = EVAL_DIR / "shard_index.pkl"

# ImageNet normalisation (matches extract_embeddings.py)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Task registry ──────────────────────────────────────────────────────────────
# Each entry:
#   csv       — filename under LABELS_DIR
#   type      — "binary", "multiclass", "regression"
#   binary_pos— (classification only) labels mapped to 1; rest → 0
#               None means use all label values as-is (multiclass)

TASKS = {
    # ── Valvular disease ──────────────────────────────────────────────────────
    "as_binary": dict(
        csv="aortic_stenosis_classification.csv",
        type="binary",
        binary_pos={"mild", "moderate", "severe"},
        desc="Aortic Stenosis (any vs normal)",
    ),
    "as_multi": dict(
        csv="aortic_stenosis_classification.csv",
        type="multiclass",
        desc="Aortic Stenosis (4-class)",
    ),
    "mr_binary": dict(
        csv="mitral_regurgitation_classification.csv",
        type="binary",
        binary_pos={"moderate", "severe"},
        desc="Mitral Regurgitation (mod/severe vs rest)",
    ),
    "tr_binary": dict(
        csv="tricuspid_regurgitation_classification.csv",
        type="binary",
        binary_pos={"moderate", "severe"},
        desc="Tricuspid Regurgitation (mod/severe vs rest)",
    ),
    "ar_binary": dict(
        csv="aortic_regurgitation_classification.csv",
        type="binary",
        binary_pos={"moderate", "severe"},
        desc="Aortic Regurgitation (mod/severe vs rest)",
    ),
    "lvh_binary": dict(
        csv="lvh_classification.csv",
        type="binary",
        binary_pos={"mild", "borderline", "moderate", "severe"},
        desc="LV Hypertrophy (any vs normal)",
    ),
    # ── LV function ───────────────────────────────────────────────────────────
    "lv_systolic": dict(
        csv="lv_systolic_function_classification.csv",
        type="binary",
        binary_pos={"mildly_reduced", "moderately_reduced", "severely_reduced", "reduced"},
        desc="LV Systolic Dysfunction (any vs normal)",
    ),
    "lv_diastolic": dict(
        csv="lv_diastolic_function_classification.csv",
        type="binary",
        binary_pos={"grade_i", "grade_ii", "grade_iii", "diastolic_dysfunction"},
        desc="LV Diastolic Dysfunction (any vs normal)",
    ),
    # ── Other cardiac ─────────────────────────────────────────────────────────
    "pericardial": dict(
        csv="pericardial_effusion_classification.csv",
        type="binary",
        binary_pos={"trace", "small", "moderate", "large", "tamponade"},
        desc="Pericardial Effusion (any vs normal)",
    ),
    "heart_failure": dict(
        csv="heart_failure_classification.csv",
        type="binary",
        binary_pos={"heart_failure"},
        desc="Heart Failure",
    ),
    "rv_binary": dict(
        csv="right_ventricle_classification.csv",
        type="binary",
        binary_pos={"mild", "enlarged", "moderate", "severe"},
        desc="RV Dilation (any vs normal)",
    ),
    # ── Regression ────────────────────────────────────────────────────────────
    "ef": dict(
        csv="ejection_fraction_regression.csv",
        type="regression",
        desc="Ejection Fraction (%)",
    ),
    "rvsp": dict(
        csv="rvsp_regression.csv",
        type="regression",
        desc="RVSP (mmHg)",
    ),
    "lavi": dict(
        csv="lavi_regression.csv",
        type="regression",
        desc="LAVI (mL/m²)",
    ),
}

DEFAULT_TASKS = [
    "as_binary", "mr_binary", "tr_binary", "lv_systolic",
    "pericardial", "heart_failure", "ef",
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class DicomDataset(Dataset):
    """Load DICOM frames from shard .tar files via byte-range reads."""

    def __init__(self, uuids, shard_index, frames_per_clip=16):
        self.uuids = uuids
        self.shard_index = shard_index
        self.N = frames_per_clip

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        shard_path, offset, size, fmt = self.shard_index[uuid]
        try:
            with open(shard_path, "rb") as f:
                f.seek(offset)
                raw = f.read(size)
            buf = io.BytesIO(raw)
            frames = np.load(buf)["frames"] if fmt == "npz" else np.load(buf)
        except Exception:
            frames = np.zeros((self.N, 336, 336, 3), dtype=np.uint8)

        T = len(frames)
        N = self.N
        if T == 0:
            frames = np.zeros((N, 336, 336, 3), dtype=np.uint8)
        elif T < N:
            frames = np.concatenate([frames] + [frames[-1:]] * (N - T), axis=0)
        else:
            start = (T - N) // 2
            frames = frames[start:start + N]

        x = frames.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return torch.from_numpy(x).permute(3, 0, 1, 2).float(), uuid  # (3, N, H, W)


# ── Encoder loading ────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path, device):
    print(f"Loading encoder from {checkpoint_path}")
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
        state = {k.replace("module.", "").replace("backbone.", ""): v
                 for k, v in ckpt[key].items()}

    epoch = ckpt.get("epoch", "?")
    del ckpt; gc.collect()

    msg = encoder.load_state_dict(state, strict=False)
    print(f"  epoch={epoch}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    del state; gc.collect()

    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, uuids, shard_index, device, batch_size, num_workers):
    """Return {dicom_uuid: np.ndarray (D,)} for all requested uuids."""
    ds = DicomDataset(uuids, shard_index)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=("cuda" in device),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    cache = {}
    autocast_dev = "cuda" if "cuda" in device else "cpu"
    for clips, batch_uuids in tqdm(loader, desc="  extracting", unit="batch"):
        clips = clips.to(device)
        with torch.amp.autocast(autocast_dev, dtype=torch.bfloat16):
            tokens = encoder(clips)          # (B, N_tokens, D)
        embs = tokens.float().mean(dim=1)    # (B, D) mean-pool over tokens
        embs_np = embs.cpu().numpy()
        for uuid, emb in zip(batch_uuids, embs_np):
            cache[uuid] = emb

    return cache


# ── Manifest + task data loading ───────────────────────────────────────────────

def build_study_dicom_map(manifest_path, needed_study_ids, shard_index, max_per_study=3):
    """Load manifest, filter to needed studies, return study_id -> [dicom_uuid].

    Keeps at most max_per_study DICOMs per study, ranked by quality_score descending
    so that low-quality / off-target DICOMs are de-prioritised.
    """
    print("Loading manifest …")
    cols = ["dicom_uuid", "study_uuid", "n_frames"]
    manifest = pd.read_parquet(manifest_path, columns=cols)
    manifest = manifest[manifest["study_uuid"].isin(needed_study_ids)]
    manifest = manifest[manifest["dicom_uuid"].isin(shard_index)]

    # Sort by n_frames descending (more frames = more temporal signal) then keep top-k per study
    manifest = manifest.sort_values("n_frames", ascending=False, na_position="last")
    manifest = manifest.groupby("study_uuid").head(max_per_study)

    study_map = manifest.groupby("study_uuid")["dicom_uuid"].apply(list).to_dict()
    total_dicoms = sum(len(v) for v in study_map.values())
    print(f"  {len(study_map):,} studies / {total_dicoms:,} DICOMs (max {max_per_study}/study, ranked by n_frames)")
    return study_map


def load_task_records(task_cfg, study_dicom_map, emb_cache, labels_dir):
    """Return (study_embeddings, labels, splits) as numpy arrays, one row per study."""
    df = pd.read_csv(labels_dir / task_cfg["csv"])
    # normalise column names
    df = df.rename(columns={"study designation": "split"})
    df["split"] = df["split"].str.upper()

    records = []
    for _, row in df.iterrows():
        sid = row["study_id"]
        label = row["label"]
        split = row["split"]
        if pd.isna(label):
            continue
        dicoms = study_dicom_map.get(sid, [])
        study_embs = [emb_cache[d] for d in dicoms if d in emb_cache]
        if not study_embs:
            continue
        avg_emb = np.mean(study_embs, axis=0)
        records.append((avg_emb, label, split))

    embeddings = np.stack([r[0] for r in records]).astype(np.float32)
    labels_raw = [r[1] for r in records]
    splits = np.array([r[2] for r in records])
    return embeddings, labels_raw, splits


# ── Probe training ─────────────────────────────────────────────────────────────

def train_classification_probe(X_train, y_train, C=1.0, class_weight="balanced"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        C=C, max_iter=2000, solver="lbfgs",
        multi_class="auto", class_weight=class_weight,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return scaler, clf


def train_regression_probe(X_train, y_train, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    reg = Ridge(alpha=alpha, random_state=42)
    reg.fit(X_train, y_train)
    return scaler, reg


# ── Metrics ────────────────────────────────────────────────────────────────────

def _bootstrap(y_true, y_score, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    pt = metric_fn(y_true, y_score)
    lo, hi = (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (pt, pt)
    return {"value": float(pt), "lower": float(lo), "upper": float(hi)}


def eval_classification_metrics(y_true, y_prob, binary=True, n_bootstrap=1000):
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score

    results = {}
    if binary:
        prob_pos = y_prob[:, 1]
        y_pred = (prob_pos >= 0.5).astype(int)
        results["auc"] = _bootstrap(y_true, prob_pos, roc_auc_score, n_bootstrap)
        results["auprc"] = _bootstrap(y_true, prob_pos, average_precision_score, n_bootstrap)
        def bal_acc(yt, yp): return balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
        results["balanced_acc"] = _bootstrap(y_true, prob_pos, bal_acc, n_bootstrap)
        def f1(yt, yp): return f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
        results["f1"] = _bootstrap(y_true, prob_pos, f1, n_bootstrap)
    else:
        y_pred = y_prob.argmax(axis=1)
        def auc_mc(yt, yp): return roc_auc_score(yt, yp, multi_class="ovr", average="macro")
        try:
            results["auc_macro"] = _bootstrap(y_true, y_prob, auc_mc, n_bootstrap)
        except Exception:
            results["auc_macro"] = {"value": float("nan"), "lower": float("nan"), "upper": float("nan")}
        def f1_macro(yt, yp): return f1_score(yt, yp.argmax(axis=1), average="macro", zero_division=0)
        results["f1_macro"] = _bootstrap(y_true, y_prob, f1_macro, n_bootstrap)
        def bal_acc_mc(yt, yp): return balanced_accuracy_score(yt, yp.argmax(axis=1))
        results["balanced_acc"] = _bootstrap(y_true, y_prob, bal_acc_mc, n_bootstrap)
    return results, y_pred


def eval_regression_metrics(y_true, y_pred, n_bootstrap=1000):
    from sklearn.metrics import mean_absolute_error, r2_score
    import math

    def mae(yt, yp): return mean_absolute_error(yt, yp)
    def rmse(yt, yp): return math.sqrt(np.mean((yt - yp) ** 2))
    def r2(yt, yp): return r2_score(yt, yp)
    def pearson(yt, yp): return float(np.corrcoef(yt, yp)[0, 1]) if len(yt) >= 2 else 0.0

    results = {
        "mae":      _bootstrap(y_true, y_pred, mae, n_bootstrap),
        "rmse":     _bootstrap(y_true, y_pred, rmse, n_bootstrap),
        "r2":       _bootstrap(y_true, y_pred, r2, n_bootstrap),
        "pearson_r": _bootstrap(y_true, y_pred, pearson, n_bootstrap),
    }
    return results


# ── Per-task evaluation ────────────────────────────────────────────────────────

def run_task(task_name, task_cfg, study_dicom_map, emb_cache, out_dir, run_name, labels_dir,
             train_fraction=1.0, seed=42):
    print(f"\n{'─'*60}")
    print(f"Task: {task_cfg['desc']}  [{task_name}]")
    task_out = out_dir / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    embeddings, labels_raw, splits = load_task_records(task_cfg, study_dicom_map, emb_cache, labels_dir)
    print(f"  {len(embeddings):,} studies with embeddings")

    task_type = task_cfg["type"]

    # ── Encode labels ──────────────────────────────────────────────────────────
    if task_type == "regression":
        labels = np.array([float(l) for l in labels_raw], dtype=np.float32)
    elif task_type == "binary":
        pos_set = task_cfg["binary_pos"]
        labels = np.array([1 if str(l) in pos_set else 0 for l in labels_raw], dtype=np.int32)
    else:  # multiclass
        unique = sorted(set(labels_raw))
        lmap = {l: i for i, l in enumerate(unique)}
        labels = np.array([lmap[l] for l in labels_raw], dtype=np.int32)

    # ── Split ──────────────────────────────────────────────────────────────────
    mask_tr = splits == "TRAIN"
    mask_va = splits == "VAL"
    mask_te = splits == "TEST"

    X_tr, y_tr = embeddings[mask_tr], labels[mask_tr]
    X_va, y_va = embeddings[mask_va], labels[mask_va]
    X_te, y_te = embeddings[mask_te], labels[mask_te]

    # Optionally subsample training set to match pretraining data fraction
    if train_fraction < 1.0 and len(X_tr) > 0:
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(len(X_tr) * train_fraction))
        idx = rng.choice(len(X_tr), n_keep, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    print(f"  train={len(X_tr):,}  val={mask_va.sum():,}  test={mask_te.sum():,}"
          + (f"  (sampled {train_fraction*100:.0f}%)" if train_fraction < 1.0 else ""))

    if len(X_tr) == 0 or len(X_te) == 0:
        print("  [skip] insufficient data")
        return None

    # ── Train probe ────────────────────────────────────────────────────────────
    if task_type == "regression":
        scaler, probe = train_regression_probe(X_tr, y_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)
        val_pred = probe.predict(X_va_s)
        test_pred = probe.predict(X_te_s)

        val_metrics  = eval_regression_metrics(y_va.astype(float), val_pred)
        test_metrics = eval_regression_metrics(y_te.astype(float), test_pred)

        preds_df = pd.DataFrame({
            "split":  np.concatenate([["val"] * len(y_va), ["test"] * len(y_te)]),
            "y_true": np.concatenate([y_va, y_te]).tolist(),
            "y_pred": np.concatenate([val_pred, test_pred]).tolist(),
        })

    else:
        is_binary = (task_type == "binary")
        scaler, probe = train_classification_probe(X_tr, y_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        # Use decision_function or predict_proba
        if hasattr(probe, "predict_proba"):
            val_prob  = probe.predict_proba(X_va_s)
            test_prob = probe.predict_proba(X_te_s)
        else:
            # fallback: one-hot from decision_function
            df_va = probe.decision_function(X_va_s)
            df_te = probe.decision_function(X_te_s)
            def softmax(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
            val_prob  = softmax(df_va if df_va.ndim == 2 else df_va[:, None])
            test_prob = softmax(df_te if df_te.ndim == 2 else df_te[:, None])

        val_metrics,  val_pred  = eval_classification_metrics(y_va, val_prob,  is_binary)
        test_metrics, test_pred = eval_classification_metrics(y_te, test_prob, is_binary)

        if task_type == "multiclass":
            unique = sorted(set(labels_raw))
            lmap = {l: i for i, l in enumerate(unique)}
            inv_lmap = {i: l for l, i in lmap.items()}
            prob_cols = {f"prob_{inv_lmap[i]}": test_prob[:, i].tolist() for i in range(len(unique))}
        else:
            prob_cols = {"prob_pos": test_prob[:, 1].tolist()}

        preds_df = pd.DataFrame({
            "split":  np.concatenate([["val"] * len(y_va), ["test"] * len(y_te)]),
            "y_true": np.concatenate([y_va, y_te]).tolist(),
            "y_pred": np.concatenate([val_pred, test_pred]).tolist(),
        })

    # ── Print & save ───────────────────────────────────────────────────────────
    key_metric = list(test_metrics.keys())[0]
    km = test_metrics[key_metric]
    print(f"  test {key_metric}: {km['value']:.4f} [{km['lower']:.4f}–{km['upper']:.4f}]")

    metrics_out = {
        "task": task_name,
        "desc": task_cfg["desc"],
        "run_name": run_name,
        "n_train": int(mask_tr.sum()),
        "n_val":   int(mask_va.sum()),
        "n_test":  int(mask_te.sum()),
        "val":  {k: v for k, v in val_metrics.items()},
        "test": {k: v for k, v in test_metrics.items()},
    }
    with open(task_out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    preds_df.to_csv(task_out / "predictions.csv", index=False)
    return metrics_out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--run-name",   required=True, help="Short name for this run (e.g. 5pct)")
    ap.add_argument("--tasks",  nargs="+", default=DEFAULT_TASKS,
                    choices=list(TASKS), metavar="TASK",
                    help=f"Tasks to evaluate. Available: {list(TASKS)}")
    ap.add_argument("--output-dir", default=str(REPO_DIR / "evaluation" / "results" / "icardio"),
                    help="Base output directory")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",  type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="Fraction of labeled TRAIN studies to use for probe training (e.g. 0.05 for 5pct run)")
    ap.add_argument("--max-dicoms-per-study", type=int, default=3,
                    help="Keep top-k DICOMs per study by n_frames (reduces extraction time)")
    ap.add_argument("--labels-dir",   default=str(_DEFAULT_LABELS_DIR),
                    help="Directory containing task CSVs and the manifest parquet")
    ap.add_argument("--manifest",     default=None,
                    help="Path to manifest parquet (default: {labels_dir}/manifest_clinical_findings_with_eval_labels.parquet)")
    ap.add_argument("--shard-index",  default=str(_DEFAULT_SHARD_INDEX),
                    help="Path to shard_index.pkl (built by evaluation/build_index.py)")
    ap.add_argument("--probe-only",  action="store_true",
                    help="Skip extraction, load cached embeddings only")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    labels_dir  = Path(args.labels_dir)
    manifest    = Path(args.manifest) if args.manifest else labels_dir / "manifest_clinical_findings_with_eval_labels.parquet"
    shard_index_path = Path(args.shard_index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "embeddings.pt"

    print(f"\n{'='*60}")
    print(f"EchoJEPAv2 iCardio Eval  |  run: {args.run_name}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Labels dir  : {labels_dir}")
    print(f"Shard index : {shard_index_path}")
    print(f"Tasks       : {args.tasks}")
    print(f"Output      : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load shard index ───────────────────────────────────────────────────────
    print(f"Loading shard index from {shard_index_path} …")
    with open(shard_index_path, "rb") as f:
        shard_index = pickle.load(f)
    print(f"  {len(shard_index):,} DICOMs indexed")

    # ── Collect all study IDs needed across selected tasks ─────────────────────
    print("\nCollecting needed study IDs …")
    all_study_ids = set()
    for task_name in args.tasks:
        df = pd.read_csv(labels_dir / TASKS[task_name]["csv"])
        all_study_ids.update(df["study_id"].dropna().tolist())
    print(f"  {len(all_study_ids):,} unique studies across {len(args.tasks)} tasks")

    # ── Build study → [dicom_uuid] map from manifest ───────────────────────────
    study_dicom_map = build_study_dicom_map(manifest, all_study_ids, shard_index,
                                            max_per_study=args.max_dicoms_per_study)

    needed_uuids = [uuid for uuids in study_dicom_map.values() for uuid in uuids]
    print(f"  {len(needed_uuids):,} DICOMs to embed")

    # ── Phase 1: Extract / load embeddings ────────────────────────────────────
    if cache_path.exists() and args.probe_only:
        print(f"\nLoading cached embeddings from {cache_path} …")
        emb_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  {len(emb_cache):,} embeddings loaded")
    else:
        if cache_path.exists():
            print(f"\nCache exists at {cache_path}. Checking coverage …")
            existing = torch.load(cache_path, map_location="cpu", weights_only=False)
            missing = [u for u in needed_uuids if u not in existing]
            print(f"  {len(existing):,} cached, {len(missing):,} missing")
            if not missing:
                emb_cache = existing
            else:
                print(f"\nExtracting {len(missing):,} missing DICOMs …")
                encoder = load_encoder(args.checkpoint, args.device)
                new_cache = extract_embeddings(
                    encoder, missing, shard_index,
                    args.device, args.batch_size, args.num_workers,
                )
                del encoder; gc.collect(); torch.cuda.empty_cache()
                emb_cache = {**existing, **new_cache}
                torch.save(emb_cache, cache_path)
                print(f"  Updated cache: {len(emb_cache):,} embeddings → {cache_path}")
        else:
            print(f"\nExtracting {len(needed_uuids):,} DICOM embeddings …")
            encoder = load_encoder(args.checkpoint, args.device)
            emb_cache = extract_embeddings(
                encoder, needed_uuids, shard_index,
                args.device, args.batch_size, args.num_workers,
            )
            del encoder; gc.collect(); torch.cuda.empty_cache()
            torch.save(emb_cache, cache_path)
            print(f"  Saved {len(emb_cache):,} embeddings → {cache_path}")

    # ── Phase 2: Run probes ────────────────────────────────────────────────────
    all_results = []
    for task_name in args.tasks:
        result = run_task(
            task_name, TASKS[task_name],
            study_dicom_map, emb_cache,
            out_dir, args.run_name, labels_dir,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        if result:
            all_results.append(result)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    if all_results:
        rows = []
        for r in all_results:
            row = {"run": r["run_name"], "task": r["task"], "desc": r["desc"],
                   "n_train": r["n_train"], "n_val": r["n_val"], "n_test": r["n_test"]}
            for split in ("val", "test"):
                for metric, vals in r[split].items():
                    row[f"{split}_{metric}"] = round(vals["value"], 4)
                    row[f"{split}_{metric}_lo"] = round(vals["lower"], 4)
                    row[f"{split}_{metric}_hi"] = round(vals["upper"], 4)
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"SUMMARY — {args.run_name}")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
