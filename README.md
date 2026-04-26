# EchoJEPAv2

Self-supervised echocardiography pretraining using a Joint-Embedding Predictive Architecture (JEPA), trained on the large-scale iCardio dataset (79,584 studies). This repository contains the preprocessing pipeline, training configuration, and downstream evaluation code.

## Overview

EchoJEPAv2 extends [EchoJEPA](https://arxiv.org/abs/2602.02603) with:
- **Larger resolution**: 336×336 px (vs. 224px in the original)
- **Text-conditioned variant**: BioClinicalBERT embeddings fused into the JEPA predictor via gated cross-attention
- **Full-scale iCardio pretraining**: 79,584 echocardiographic studies, preprocessed into standardized WebDataset shards

### Model Variants

| Variant | Backbone | Resolution | Predictor |
|---|---|---|---|
| EchoJEPAv2 | ViT-L/16 | 336×336, 16f | Standard JEPA predictor |
| EchoJEPAv2+text | ViT-L/16 | 336×336, 16f | Gated cross-attention with BioClinicalBERT |

## Repository Structure

```
EchoJEPAv2/
├── EchoJEPA/               # V-JEPA codebase (git submodule)
├── preprocessing/          # DICOM → WebDataset pipeline
│   ├── create_webdataset.py          # Main parallel pipeline
│   ├── max_workers_create_webdataset.py  # Memory-optimized variant
│   ├── json_to_sqlite.py             # One-time annotation DB setup
│   ├── utils.py                      # FPS resampling, pixel spacing
│   ├── utils_fan.py                  # Fan ROI extraction, padding
│   ├── load_shard_samples.py         # Inspect output shards
│   └── visualize_preprocessed_dicoms.py
├── training/               # Configs, evaluation scripts, data CSVs
│   ├── pretrain_icardio_336px_16f.yaml       # Main pretraining config
│   ├── pretrain_icardio_336px_16f_text.yaml  # Text-conditioned pretraining
│   ├── cooldown_icardio_224px_16f.yaml       # 224px ablation
│   ├── eval_*.yaml                           # Downstream evaluation configs
│   ├── eval_retrieval.py                     # Retrieval evaluation
│   ├── eval_view_classification.py           # View classification
│   ├── eval_lvef_regression.py               # LVEF regression
│   ├── extract_features.py                   # Feature extraction
│   ├── data_csvs/                            # Train/val/test splits
│   │   ├── echonet_dynamic_{train,val,test}.csv
│   │   ├── echonet_pediatric_{train,val,test}.csv
│   │   ├── camus_{train,val,test}.csv
│   │   └── icardio_lvef_{train,val,holdout}.csv
│   └── setup_env.sh
└── evaluation/             # Frozen linear probe evaluation
    ├── linear_probe.py
    └── metrics.py
```

## Preprocessing Pipeline

Converts raw DICOM PNG sequences into standardized WebDataset `.tar` shards. Each sample undergoes:

1. Load PNG frame sequence → `(T, H, W, 3)` uint8 BGR
2. Resample to target pixel spacing (0.36 mm/px)
3. Remove gray UI overlays (pixels `[57, 57, 57]`)
4. Extract fan-shaped ROI via contour detection
5. Normalize pixel values to [0, 255]
6. Pad to square
7. Resize to 336×336 (LANCZOS4)
8. Resample to 24 FPS (nearest-neighbor)

Output: `(T', 336, 336, 3)` uint8 RGB stored as `.npy` + `.json` metadata per sample in `.tar` shards.

### Setup (one-time): Convert JSON annotations to SQLite

```bash
python preprocessing/json_to_sqlite.py \
  --json /path/to/combined_updated.json \
  --db /path/to/annotations.db
```

### Run the preprocessing pipeline

```bash
python preprocessing/create_webdataset.py \
  --csv <intersected_dicoms_csv> \
  --metadata-csv <dicom_metadata_csv> \
  --annotations-db /path/to/annotations.db \
  --output-dir <output_shards_dir> \
  --num-workers 6
```

### Inspect output shards

```bash
python preprocessing/load_shard_samples.py \
  --shard path/to/shard-000000.tar --n-samples 10

python preprocessing/visualize_preprocessed_dicoms.py \
  --tar path/to/shard-000000.tar --out output_dir --n 10 --fps 24
```

## Pretraining

EchoJEPAv2 uses the [V-JEPA](https://github.com/facebookresearch/jepa) codebase as its backbone. The EchoJEPA submodule contains the modified version with WebDataset support and echocardiography-specific adaptations.

### Standard pretraining (336px, 16 frames)

```bash
cd EchoJEPA
python app/main.py \
  --fname ../training/pretrain_icardio_336px_16f.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

### Text-conditioned pretraining

```bash
cd EchoJEPA
python app/main.py \
  --fname ../training/pretrain_icardio_336px_16f_text.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Key hyperparameters (see `training/pretrain_icardio_336px_16f.yaml`):

| Parameter | Value |
|---|---|
| Architecture | ViT-L/16 |
| Input | 336×336 px, 16 frames/clip |
| Tubelet size | 2 |
| Predictor depth | 12 |
| Training epochs | 200 |
| Batch size | 24 per GPU |
| LR warmup | 40 epochs |

## Downstream Evaluation

All evaluations use a **frozen attentive probe**: a 4-block, 16-head transformer attached to the frozen target encoder, trained on task-specific labels.

### LVEF regression (EchoNet-Dynamic, CAMUS)

```bash
cd EchoJEPA
python eval/eval_video_classification.py \
  --fname ../training/eval_echonet_lvef_e200_336px.yaml
```

### View classification (iCardio)

```bash
cd EchoJEPA
python eval/eval_video_classification.py \
  --fname ../training/eval_view_cls.yaml
```

### Retrieval (study and view)

```bash
python training/eval_retrieval.py \
  --checkpoint /path/to/target_encoder.pt \
  --data-dir /path/to/shards
```

## Results

### LVEF Estimation (MAE ↓)

| Dataset | EchoJEPAv2 |
|---|---|
| EchoNet-Dynamic | 3.97% |
| EchoNet-Pediatric (zero-shot) | 6.91% |
| EchoNet-Pediatric (fine-tuned) | 5.78% |
| CAMUS (zero-shot) | 10.68% |

### iCardio Downstream Tasks

| Task | Metric | EchoJEPAv2 |
|---|---|---|
| LVEF (iCardio) | MAE | 4.21% |
| RVSP | MAE | ≤5.9 mmHg |
| LVIDd | MAE | ≤0.42 cm |
| MV Regurgitation | AUC | — |

### Retrieval (Recall@k)

| k | Study | View |
|---|---|---|
| @1 | — | — |
| @5 | — | — |
| @10 | — | — |
| @20 | — | — |

*(See `training/eval_retrieval.py` for full results.)*

### View Classification (iCardio, 12 classes)

| Model | Top-1 Accuracy |
|---|---|
| EchoJEPAv2 | 92.0% |
| EchoJEPAv2+text | 92.0% |

## Environment Setup

```bash
bash training/setup_env.sh
```

Dependencies: `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `webdataset`, `transformers` (for text variant), `wandb`.

## Citation

If you use this work, please cite the original EchoJEPA paper:

```bibtex
@article{echojepa2024,
  title   = {EchoJEPA: A Latent Predictive Foundation Model for Echocardiography},
  author  = {Munim, Alif and Fallahpour, Adibvafa and Szasz, Teodora and Attarpour, Ahmadreza and Jiang, River and others},
  journal = {arXiv preprint arXiv:2602.02603},
  year    = {2024}
}
```
