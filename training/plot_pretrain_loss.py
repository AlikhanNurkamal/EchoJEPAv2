#!/usr/bin/env python3
"""
Plot pretraining loss curves for all EchoJEPAv2 scaling experiments.

Produces two figures:
  1. scaling_loss_curves.png  — local 5/10/30 pct runs (epoch 1–30) side-by-side
                                with the AMD 100 pct run (epoch 1–200) on right
  2. final_loss_vs_scale.png  — final-epoch loss vs. data fraction (scaling bar)

Run from EchoJEPAv2/:
  python training/plot_pretrain_loss.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = Path(__file__).resolve().parent

RUNS = [
    dict(
        label="5% iCardio (30 ep)",
        log="/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu/log_r0.csv",
        color="#4C72B0",
        linestyle="-",
        pct=5,
    ),
    dict(
        label="10% iCardio (30 ep)",
        log="/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu_10pct/log_r0.csv",
        color="#DD8452",
        linestyle="-",
        pct=10,
    ),
    dict(
        label="30% iCardio (30 ep)",
        log="/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu_30pct/log_r0.csv",
        color="#55A868",
        linestyle="-",
        pct=30,
    ),
    dict(
        label="100% iCardio (200 ep)",
        log="/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/log_r0.csv",
        color="#C44E52",
        linestyle="-",
        pct=100,
    ),
]


def load_log(path):
    """Load log_r0.csv and return per-epoch mean loss."""
    df = pd.read_csv(path, names=["epoch", "itr", "loss", "iter_ms", "gpu_ms", "data_ms"])
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notna()]
    df["epoch"] = df["epoch"].astype(int)
    df["loss"]  = df["loss"].astype(float)
    per_epoch = df.groupby("epoch")["loss"].mean().reset_index()
    return per_epoch


def smooth(values, window=5):
    """Simple rolling mean."""
    s = pd.Series(values)
    return s.rolling(window, min_periods=1, center=True).mean().to_numpy()


# ── Figure 1: loss curves ──────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.35)

ax_local = fig.add_subplot(gs[0])  # local 5/10/30 pct
ax_amd   = fig.add_subplot(gs[1])  # AMD 100 pct

local_runs = [r for r in RUNS if r["pct"] != 100]
amd_run    = next(r for r in RUNS if r["pct"] == 100)

for run in local_runs:
    per_epoch = load_log(run["log"])
    smoothed  = smooth(per_epoch["loss"].to_numpy())
    ax_local.plot(per_epoch["epoch"], smoothed,
                  color=run["color"], linestyle=run["linestyle"],
                  linewidth=2, label=run["label"])

ax_local.set_xlabel("Epoch", fontsize=12)
ax_local.set_ylabel("Pretraining Loss (L1)", fontsize=12)
ax_local.set_title("Scaling: 5 / 10 / 30 pct iCardio\n(ViT-L/16 · 336px · 16f · 2× A6000)", fontsize=11)
ax_local.legend(fontsize=9, loc="upper right")
ax_local.grid(True, alpha=0.3)
ax_local.set_xlim(1, None)

per_epoch_amd = load_log(amd_run["log"])
smoothed_amd  = smooth(per_epoch_amd["loss"].to_numpy(), window=10)
ax_amd.plot(per_epoch_amd["epoch"], smoothed_amd,
            color=amd_run["color"], linewidth=2, label=amd_run["label"])
ax_amd.axhline(per_epoch_amd["loss"].iloc[-10:].mean(), color=amd_run["color"],
               linestyle="--", linewidth=1, alpha=0.6, label="Final 10-ep avg")

ax_amd.set_xlabel("Epoch", fontsize=12)
ax_amd.set_ylabel("Pretraining Loss (L1)", fontsize=12)
ax_amd.set_title("Full scale: 100% iCardio (~3.1M DICOMs)\n(ViT-L/16 · 336px · 16f · 8× MI300X)", fontsize=11)
ax_amd.legend(fontsize=9)
ax_amd.grid(True, alpha=0.3)
ax_amd.set_xlim(1, None)

fig.suptitle("EchoJEPAv2 Pretraining Loss — Data Scaling", fontsize=14, fontweight="bold")
plt.savefig(OUT_DIR / "scaling_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved scaling_loss_curves.png")


# ── Figure 2: final loss vs. data scale ───────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 5))

fracs  = []
losses = []
labels = []
colors = []

for run in RUNS:
    per_epoch = load_log(run["log"])
    final_loss = per_epoch["loss"].iloc[-5:].mean()  # last-5 epoch avg
    fracs.append(run["pct"])
    losses.append(final_loss)
    labels.append(run["label"])
    colors.append(run["color"])

bars = ax2.bar(range(len(fracs)), losses, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

for bar, loss_val in zip(bars, losses):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{loss_val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.set_xticks(range(len(fracs)))
ax2.set_xticklabels([f"{p}%" for p in fracs], fontsize=12)
ax2.set_xlabel("Training Data Fraction", fontsize=12)
ax2.set_ylabel("Final Pretraining Loss (L1, last-5-ep avg)", fontsize=12)
ax2.set_title("EchoJEPAv2 Final Loss vs. Data Scale\n(Higher loss with more data is expected — JEPA training signal increases)", fontsize=11)
ax2.grid(True, axis="y", alpha=0.3)
ax2.set_ylim(0, max(losses) * 1.15)

plt.tight_layout()
plt.savefig(OUT_DIR / "final_loss_vs_scale.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved final_loss_vs_scale.png")


# ── Print summary table ────────────────────────────────────────────────────────
print()
print(f"{'Run':<35} {'Epochs':>7} {'Final Loss (last-5ep)':>22} {'Min Loss':>10}")
print("-" * 78)
for run in RUNS:
    per_epoch = load_log(run["log"])
    final     = per_epoch["loss"].iloc[-5:].mean()
    min_loss  = per_epoch["loss"].min()
    n_epochs  = per_epoch["epoch"].max()
    print(f"{run['label']:<35} {n_epochs:>7} {final:>22.5f} {min_loss:>10.5f}")
