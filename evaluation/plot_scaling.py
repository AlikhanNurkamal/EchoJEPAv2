#!/usr/bin/env python3
"""
Plot downstream performance vs. pretraining data fraction for EchoJEPAv2 (ViT-L/16).

Shows that linear probe performance scales consistently with pretraining data,
justifying use of a data subset for ablation experiments.

Run from EchoJEPAv2/:
  python evaluation/plot_scaling.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR   = Path(__file__).resolve().parent
EVAL_DIR  = OUT_DIR
FRACS     = [5, 10, 30, 100]
RUN_NAMES = ["5pct", "10pct", "30pct", None]   # None = hardcoded AMD results

# ── 100pct results (AMD cluster, ViT-L 200 ep) ────────────────────────────────
AMD_100 = {
    "mr_binary":     {"test_auc": 0.822, "test_auc_lo": None, "test_auc_hi": None},
    "tr_binary":     {"test_auc": 0.800, "test_auc_lo": None, "test_auc_hi": None},
    "lv_systolic":   {"test_auc": 0.889, "test_auc_lo": None, "test_auc_hi": None},
    "pericardial":   {"test_auc": 0.734, "test_auc_lo": None, "test_auc_hi": None},
    "heart_failure": {"test_auc": 0.759, "test_auc_lo": None, "test_auc_hi": None},
    "ef":            {"test_pearson_r": 0.50, "test_mae": 4.67},  # approx from R²≈0.25
}

TASK_LABELS = {
    "as_binary":     "AS",
    "mr_binary":     "MR",
    "tr_binary":     "TR",
    "lv_systolic":   "LV Systolic",
    "pericardial":   "Pericardial Eff.",
    "heart_failure": "Heart Failure",
    "ef":            "EF (regression)",
}

COLORS = {
    "as_binary":     "#4C72B0",
    "mr_binary":     "#DD8452",
    "tr_binary":     "#55A868",
    "lv_systolic":   "#C44E52",
    "pericardial":   "#8172B2",
    "heart_failure": "#937860",
    "ef":            "#DA8BC3",
}

# ── Load local CSVs ───────────────────────────────────────────────────────────
def load_summary(run_name):
    path = EVAL_DIR / "results" / "icardio" / run_name / "summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

summaries = {}
for frac, run_name in zip(FRACS[:3], RUN_NAMES[:3]):
    df = load_summary(run_name)
    if df is not None:
        summaries[frac] = df

# ── Build per-task scaling data ───────────────────────────────────────────────
clf_tasks = ["as_binary", "mr_binary", "tr_binary", "lv_systolic", "pericardial", "heart_failure"]
reg_task  = "ef"

clf_data = {task: {"fracs": [], "auc": [], "lo": [], "hi": []} for task in clf_tasks}
ef_data  = {"fracs": [], "pearson": [], "mae": []}

for frac, df in summaries.items():
    for task in clf_tasks:
        row = df[df["task"] == task]
        if row.empty:
            continue
        row = row.iloc[0]
        clf_data[task]["fracs"].append(frac)
        clf_data[task]["auc"].append(row["test_auc"])
        clf_data[task]["lo"].append(row["test_auc_lo"])
        clf_data[task]["hi"].append(row["test_auc_hi"])

    ef_row = df[df["task"] == reg_task]
    if not ef_row.empty:
        ef_row = ef_row.iloc[0]
        ef_data["fracs"].append(frac)
        ef_data["pearson"].append(ef_row["test_pearson_r"])
        ef_data["mae"].append(ef_row["test_mae"])

# Add 100pct
for task in clf_tasks:
    if task in AMD_100:
        clf_data[task]["fracs"].append(100)
        clf_data[task]["auc"].append(AMD_100[task]["test_auc"])
        clf_data[task]["lo"].append(None)
        clf_data[task]["hi"].append(None)

ef_data["fracs"].append(100)
ef_data["pearson"].append(AMD_100["ef"]["test_pearson_r"])
ef_data["mae"].append(AMD_100["ef"]["test_mae"])

# ── Compute avg AUC across classification tasks per fraction ──────────────────
avg_auc = {}
for frac in FRACS:
    vals = []
    for task in clf_tasks:
        d = clf_data[task]
        frac_auc = dict(zip(d["fracs"], d["auc"]))
        if frac in frac_auc:
            vals.append(frac_auc[frac])
    if vals:
        avg_auc[frac] = np.mean(vals)

avg_fracs = sorted(avg_auc.keys())
avg_vals  = [avg_auc[f] for f in avg_fracs]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: average AUC across classification tasks
ax1.plot(avg_fracs, avg_vals, "o-", color="#2d6a9f", linewidth=2.5, markersize=8)
for f, v in zip(avg_fracs, avg_vals):
    ax1.annotate(f"{v:.3f}", (f, v), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=10, fontweight="bold", color="#2d6a9f")

ax1.set_xscale("log")
ax1.set_xticks(FRACS)
ax1.set_xticklabels([f"{p}%" for p in FRACS], fontsize=11)
ax1.set_xlabel("Pretraining Data Fraction", fontsize=12)
ax1.set_ylabel("Average Test AUC (6 tasks)", fontsize=12)
ax1.set_title("Classification — Avg. AUC vs. Data Scale\n(ViT-L/16 · 336px · 16f · frozen linear probe)", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.55, 0.85)

# Right: EF Pearson r scaling
color_ef = "#2d6a9f"
ax2.plot(ef_data["fracs"], ef_data["mae"], "o-", color=color_ef,
         linewidth=2.5, markersize=8)

for f, v in zip(ef_data["fracs"], ef_data["mae"]):
    ax2.annotate(f"{v:.2f}", (f, v), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=10, fontweight="bold", color=color_ef)

ax2.set_xscale("log")
ax2.set_xticks(FRACS)
ax2.set_xticklabels([f"{p}%" for p in FRACS], fontsize=11)
ax2.set_xlabel("Pretraining Data Fraction", fontsize=12)
ax2.set_ylabel("Test MAE (%)", fontsize=12)
ax2.set_title("EF Regression — MAE vs. Data Scale\n(ViT-L/16 · 336px · 16f · frozen Ridge probe)", fontsize=11)
ax2.set_ylim(4.0, 6.5)
ax2.grid(True, alpha=0.3)

fig.suptitle("EchoJEPAv2 Downstream Performance Scales with Pretraining Data",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out = OUT_DIR / "scaling_downstream.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Task':<18} {'5%':>8} {'10%':>8} {'30%':>8} {'100%':>8}  Δ(5→100)")
print("-" * 60)
for task in clf_tasks:
    d = clf_data[task]
    vals = dict(zip(d["fracs"], d["auc"]))
    v5   = vals.get(5,   float("nan"))
    v10  = vals.get(10,  float("nan"))
    v30  = vals.get(30,  float("nan"))
    v100 = vals.get(100, float("nan"))
    delta = v100 - v5 if not (np.isnan(v5) or np.isnan(v100)) else float("nan")
    print(f"{TASK_LABELS[task]:<18} {v5:>8.3f} {v10:>8.3f} {v30:>8.3f} {v100:>8.3f}  +{delta:.3f}")

print(f"\n{'EF Pearson r':<18} ", end="")
pvals = dict(zip(ef_data["fracs"], ef_data["pearson"]))
for frac in FRACS:
    print(f"{pvals.get(frac, float('nan')):>8.3f} ", end="")
print()
