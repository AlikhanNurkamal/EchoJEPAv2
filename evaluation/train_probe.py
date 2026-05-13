#!/usr/bin/env python3
"""
Train a linear probe on pre-extracted embeddings for a single task (AS / MR / TR).

Requires embeddings to be pre-extracted by extract_embeddings.py first.
Training runs on cached tensors — no encoder needed, epochs take seconds.

Usage:
    python evaluation/train_probe.py --task AS --device cuda:0
    python evaluation/train_probe.py --task MR --device cuda:1
    python evaluation/train_probe.py --task TR --device cuda:2

Output:
    evaluation/results/{task}/probe_best.pt
    evaluation/results/{task}/predictions_val.csv
    evaluation/results/{task}/predictions_test.csv
    evaluation/results/{task}/metrics.csv
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_repo = Path(__file__).resolve().parent.parent / "EchoJEPA"
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.models.linear_pooler import LinearClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_metrics_with_ci, format_metrics_table, save_metrics_report, save_predictions

TASK_CSV = {"AS": "AS.csv", "MR": "MR.csv", "TR": "TR.csv"}


def load_embeddings(emb_dir, split):
    path = emb_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. Run extract_embeddings.py --task <task> first."
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  {split}: {data['embeddings'].shape[0]} samples loaded from {path}")
    return data


def make_loader(data, batch_size, shuffle):
    embs   = data["embeddings"]   # (N, D)
    labels = data["labels"]       # (N,)
    ds = TensorDataset(embs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


@torch.no_grad()
def evaluate(probe, loader, uuids, device):
    probe.eval()
    all_logits, all_labels = [], []
    for embs, labels in loader:
        embs = embs.unsqueeze(1).to(device)  # (B, 1, D) — probe expects (B, N, D)
        logits = probe(embs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits  = torch.cat(all_logits).numpy()
    y_true  = torch.cat(all_labels).numpy()
    y_pred  = logits.argmax(axis=1)
    acc     = (y_pred == y_true).mean()
    return acc, y_true, y_pred, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["AS", "MR", "TR"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_dir = Path(__file__).resolve().parent
    emb_dir  = eval_dir / "embeddings" / args.task
    out_dir  = eval_dir / "results" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted embeddings
    print(f"Loading embeddings for {args.task} ...")
    train_data = load_embeddings(emb_dir, "train")
    valid_data = load_embeddings(emb_dir, "valid")
    test_data  = load_embeddings(emb_dir, "test")

    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_data, args.batch_size, shuffle=False)
    test_loader  = make_loader(test_data,  args.batch_size, shuffle=False)

    # Label info from CSV
    df = pd.read_csv(eval_dir.parent / "labels" / TASK_CSV[args.task])
    classes   = sorted(df["stratification"].dropna().unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    int_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(classes)
    print(f"\nTask: {args.task}  |  Classes ({num_classes}): {classes}")

    embed_dim = train_data["embeddings"].shape[1]
    probe = LinearClassifier(embed_dim=embed_dim, num_classes=num_classes, use_layernorm=True).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    probe_ckpt = out_dir / "probe_best.pt"
    best_val_acc = 0.0

    if args.eval_only:
        print(f"\nEval-only: loading {probe_ckpt} ...")
        ckpt = torch.load(probe_ckpt, map_location=args.device, weights_only=False)
        probe.load_state_dict(ckpt["probe"])
        best_val_acc = ckpt.get("val_acc", 0.0)
    else:
        print(f"\nTraining for {args.epochs} epochs ...")
        for epoch in range(1, args.epochs + 1):
            probe.train()
            total_loss, n_batches = 0.0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
            for embs, labels in pbar:
                embs   = embs.unsqueeze(1).to(args.device)   # (B, 1, D)
                labels = labels.to(args.device)
                logits = probe(embs)
                loss   = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_acc, _, _, _ = evaluate(probe, valid_loader, valid_data["uuids"], args.device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"probe": probe.state_dict(), "epoch": epoch, "val_acc": val_acc}, probe_ckpt)

            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={total_loss/n_batches:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    # Final evaluation on best checkpoint
    print(f"\nLoading best probe (val_acc={best_val_acc:.4f}) ...")
    ckpt = torch.load(probe_ckpt, map_location=args.device, weights_only=False)
    probe.load_state_dict(ckpt["probe"])

    for split_name, loader, data in [("val", valid_loader, valid_data), ("test", test_loader, test_data)]:
        print(f"\nEvaluating on {split_name} ...")
        acc, y_true, y_pred, logits = evaluate(probe, loader, data["uuids"], args.device)

        preds_df = pd.DataFrame({
            "dicom_uuid":   data["uuids"],
            "y_true":       y_true,
            "y_pred":       y_pred,
            "y_true_label": [int_to_label[i] for i in y_true],
            "y_pred_label": [int_to_label[i] for i in y_pred],
            "y_prob":       [json.dumps(logits[i].tolist()) for i in range(len(y_true))],
        })
        save_predictions(preds_df, str(out_dir / f"predictions_{split_name}.csv"))

        metrics = compute_metrics_with_ci(
            y_true=y_true, y_pred=y_pred, y_prob=logits,
            classification_mode="max", n_bootstrap=1000, confidence_level=0.95,
        )
        print(format_metrics_table(metrics))
        save_metrics_report(
            metrics, output_path=str(out_dir / "metrics.csv"),
            task_name=args.task, mode=f"linear_probe_{split_name}",
            model_name="icardio_vitl16_336px_e200", label_mapping=label_map,
        )

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
