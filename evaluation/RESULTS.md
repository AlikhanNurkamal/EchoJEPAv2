# EchoJEPAv2 Evaluation Results

All results use frozen linear probes (LogisticRegression / Ridge) on study-level embeddings
(top-3 DICOMs per study by n_frames, mean-pooled ViT tokens → 1024-dim for ViT-L, 768-dim for ViT-B).
Bootstrap CIs: 1000 samples, 95%. Full metrics in `results/icardio/{run_name}/summary.csv`.

---

## ViT-L/16 · 336px · 16f — Data Scaling (5% → 100% iCardio)

Probe trained on same fraction as pretraining (e.g. 5% pretrain → 5% probe TRAIN split).

| Task | 5pct | 10pct | 30pct | 100pct (AMD) |
|------|------|-------|-------|--------------|
| AS (binary) | 0.566 | 0.560 | 0.619 | — |
| MR (binary) | 0.588 | 0.592 | 0.637 | 0.822 |
| TR (binary) | 0.577 | 0.592 | 0.651 | 0.800 |
| LV Systolic | 0.770 | 0.749 | 0.813 | 0.889 |
| Pericardial | 0.562 | 0.594 | 0.611 | 0.734 |
| Heart Failure | 0.610 | 0.630 | 0.642 | 0.759 |
| **Avg AUC** | **0.612** | **0.619** | **0.662** | **0.801** |
| EF MAE (%) | 5.85 | 5.33 | 5.05 | 4.67 |
| EF Pearson r | 0.165 | 0.210 | 0.258 | ~0.500 |

Checkpoints:
- 5pct:  `/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu/latest.pt` (epoch 30)
- 10pct: `/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu_10pct/latest.pt` (epoch 30)
- 30pct: `/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f_local2gpu_30pct/latest.pt` (epoch 30)
- 100pct: `/vast/users/mohammad.yaqub/project/checkpoints/pretrain_icardio_336px_16f/latest.pt` (epoch 200, AMD cluster)

Scaling plot: `evaluation/scaling_downstream.png`

---

## ViT-B/16 · 336px · 16f — 5% iCardio (Architecture Comparison)

Probe trained on 5% TRAIN split. Compare against ViT-L 5pct above.

| Task | ViT-B test AUC | ViT-L test AUC | Δ |
|------|---------------|---------------|---|
| AS | 0.540 | 0.566 | -0.026 |
| MR | 0.580 | 0.588 | -0.008 |
| TR | 0.573 | 0.577 | -0.004 |
| LV Systolic | 0.729 | 0.770 | -0.041 |
| Pericardial | 0.540 | 0.562 | -0.022 |
| Heart Failure | 0.656 | 0.610 | +0.046 |
| **Avg AUC** | **0.603** | **0.612** | **-0.009** |
| EF MAE (%) | 5.71 | 5.85 | +0.14 (↓ better) |
| EF Pearson r | 0.151 | 0.165 | -0.014 |

Checkpoint: `/home/mashrafimonon/iCardio/checkpoints/pretrain/icardio_vitb16_336px_16f_local2gpu_5pct/latest.pt` (epoch 30)

**Takeaway:** ViT-B ≈ ViT-L at 5% data — gap is minimal, bottleneck is data not model capacity.

---

## Planned / In Progress

| Experiment | Status | Config |
|------------|--------|--------|
| ViT-L + text (5pct) | pending | `training/pretrain_icardio_336px_16f_text_5pct.yaml` |
| LeWorldModel (5pct) | investigating | see notes below |

---

## Notes

- **Pericardial effusion** is consistently weak (AUPRC ~0.01) due to extreme class imbalance (~1% positive).
- **EF R²** is negative at 5%/10% — probe predicts near-mean; improves to +0.056 at 30%, ~0.25 at 100%.
- **100pct** used 8× MI300X on AMD cluster, 200 epochs vs 30 epochs locally — not a pure data comparison.
- AS results missing for 100pct AMD run.
