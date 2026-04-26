"""Regenerate fig_retrieval.pdf with only study and view retrieval recall@k."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_path = "/home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f/eval_retrieval_latest/results.json"
output_pdf = "/home/ahmedaly/iCardio/EchoJEPAv2/report/fig_retrieval.pdf"
output_png = "/home/ahmedaly/iCardio/EchoJEPAv2/report/fig_retrieval.png"

with open(results_path) as f:
    data = json.load(f)

retrieval = data["retrieval"]
k_values = [1, 5, 10, 20]

groups = {
    "Study": [retrieval["study"][f"recall@{k}"] for k in k_values],
    "View":  [retrieval["view"][f"recall@{k}"]  for k in k_values],
}

x = np.arange(len(k_values))
width = 0.35
colors = ["#2196F3", "#FF9800"]

fig, ax = plt.subplots(figsize=(6, 4))

for i, (label, values) in enumerate(groups.items()):
    offset = (i - (len(groups) - 1) / 2) * width
    bars = ax.bar(x + offset, [v * 100 for v in values], width,
                  label=label, color=colors[i], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val*100:.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("Recall@k (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f"@{k}" for k in k_values])
ax.set_ylim(0, 105)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(output_pdf, bbox_inches="tight")
fig.savefig(output_png, dpi=150, bbox_inches="tight")
print(f"Saved {output_pdf}")
print(f"Saved {output_png}")
