"""
Metrics Computation with Bootstrapping

Provides functions for computing classification metrics with
confidence intervals via bootstrapping.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary"
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC)
        average: Averaging method for multi-class
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            if average == "binary":
                metrics["auc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            metrics["auc"] = 0.0
    
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        metric_fn: Function that takes (y_true, y_prob) and returns metric value
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
    
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    # Point estimate
    point_estimate = metric_fn(y_true, y_prob)
    
    # Bootstrap samples
    bootstrap_values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        try:
            value = metric_fn(y_true[indices], y_prob[indices])
            bootstrap_values.append(value)
        except:
            continue
    
    if len(bootstrap_values) == 0:
        return point_estimate, point_estimate, point_estimate
    
    # Compute confidence interval
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_values, alpha * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha) * 100)
    
    return point_estimate, lower, upper


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    classification_mode: str = "max",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics with bootstrap confidence intervals.
    classification_mode: "calibration" skips Brier; "max" computes Brier.
    
    Returns:
        Dictionary with metric names as keys, each containing
        {value, lower, upper}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    classification_mode = (classification_mode or "max").lower()
    compute_brier = classification_mode != "calibration"
    n_bins = 15
    is_multiclass = y_prob.ndim == 2 and y_prob.shape[1] > 1
    if not is_multiclass and y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)

    results = {}
    if is_multiclass:
        logits = y_prob
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        n_classes = probs.shape[1]
        if y_pred.size == 0:
            y_pred = probs.argmax(axis=1)

        def acc_fn(yt, yp):
            return accuracy_score(yt, yp)
        val, lower, upper = bootstrap_ci(y_true, y_pred, acc_fn, n_bootstrap, confidence_level)
        results["accuracy"] = {"value": val, "lower": lower, "upper": upper}

        def prec_fn(yt, yp):
            return precision_score(yt, yp, average="macro", zero_division=0)
        val, lower, upper = bootstrap_ci(y_true, y_pred, prec_fn, n_bootstrap, confidence_level)
        results["precision"] = {"value": val, "lower": lower, "upper": upper}

        def rec_fn(yt, yp):
            return recall_score(yt, yp, average="macro", zero_division=0)
        val, lower, upper = bootstrap_ci(y_true, y_pred, rec_fn, n_bootstrap, confidence_level)
        results["recall"] = {"value": val, "lower": lower, "upper": upper}

        def f1_fn(yt, yp):
            return f1_score(yt, yp, average="macro", zero_division=0)
        val, lower, upper = bootstrap_ci(y_true, y_pred, f1_fn, n_bootstrap, confidence_level)
        results["f1"] = {"value": val, "lower": lower, "upper": upper}

        try:
            def auc_fn(yt, yp):
                return roc_auc_score(yt, yp, multi_class="ovr", average="macro")
            val, lower, upper = bootstrap_ci(y_true, probs, auc_fn, n_bootstrap, confidence_level)
            results["auc"] = {"value": val, "lower": lower, "upper": upper}
        except:
            results["auc"] = {"value": 0.0, "lower": 0.0, "upper": 0.0}

        try:
            def auprc_fn(yt, yp):
                scores = []
                for cls in range(n_classes):
                    y_true_cls = (yt == cls).astype(int)
                    precision, recall, _ = precision_recall_curve(y_true_cls, yp[:, cls])
                    order = np.argsort(recall)
                    scores.append(float(np.trapz(precision[order], recall[order])))
                return float(np.mean(scores)) if scores else 0.0
            val, lower, upper = bootstrap_ci(y_true, probs, auprc_fn, n_bootstrap, confidence_level)
            results["auprc"] = {"value": val, "lower": lower, "upper": upper}
        except:
            results["auprc"] = {"value": 0.0, "lower": 0.0, "upper": 0.0}

        try:
            def ap_fn(yt, yp):
                return average_precision_score(yt, yp, average="macro")
            val, lower, upper = bootstrap_ci(y_true, probs, ap_fn, n_bootstrap, confidence_level)
            results["average_precision"] = {"value": val, "lower": lower, "upper": upper}
        except:
            results["average_precision"] = {"value": 0.0, "lower": 0.0, "upper": 0.0}

        def spec_fn(yt, yp):
            cm = confusion_matrix(yt, yp, labels=list(range(n_classes)))
            specs = []
            total = cm.sum()
            for cls in range(n_classes):
                tp = cm[cls, cls]
                fp = cm[:, cls].sum() - tp
                fn = cm[cls, :].sum() - tp
                tn = total - (tp + fp + fn)
                denom = tn + fp
                specs.append(tn / denom if denom > 0 else 0.0)
            return float(np.mean(specs)) if specs else 0.0
        val, lower, upper = bootstrap_ci(y_true, y_pred, spec_fn, n_bootstrap, confidence_level)
        results["specificity"] = {"value": val, "lower": lower, "upper": upper}

        def ece_fn(yt, yp):
            conf = yp.max(axis=1)
            pred = yp.argmax(axis=1)
            correct = (pred == yt).astype(float)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            n = len(yt)
            for i in range(n_bins):
                lo = bins[i]
                hi = bins[i + 1]
                mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
                if not np.any(mask):
                    continue
                bin_acc = float(np.mean(correct[mask]))
                bin_conf = float(np.mean(conf[mask]))
                ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
            return float(ece)
        val, lower, upper = bootstrap_ci(y_true, probs, ece_fn, n_bootstrap, confidence_level)
        results["ece"] = {"value": val, "lower": lower, "upper": upper}

        if compute_brier:
            def brier_fn(yt, yp):
                one_hot = np.eye(n_classes)[yt.astype(int)]
                return float(np.mean(np.sum((yp - one_hot) ** 2, axis=1)))
            val, lower, upper = bootstrap_ci(y_true, probs, brier_fn, n_bootstrap, confidence_level)
            results["brier_score"] = {"value": val, "lower": lower, "upper": upper}

        n = int(y_true.shape[0])
        if n > 0:
            correct = int((y_true == y_pred).sum())
            null_p = 1.0 / float(n_classes)
            var = null_p * (1.0 - null_p) / n
            if var > 0:
                z = (correct / n - null_p) / math.sqrt(var)
                results["p_value"] = math.erfc(abs(z) / math.sqrt(2.0))
            else:
                results["p_value"] = float("nan")
        else:
            results["p_value"] = float("nan")
    else:
        # Accuracy
        def acc_fn(yt, yp): 
            return accuracy_score(yt, (yp >= threshold).astype(int))
        val, lower, upper = bootstrap_ci(y_true, y_prob, acc_fn, n_bootstrap, confidence_level)
        results["accuracy"] = {"value": val, "lower": lower, "upper": upper}

        # Precision
        def prec_fn(yt, yp):
            return precision_score(yt, (yp >= threshold).astype(int), zero_division=0)
        val, lower, upper = bootstrap_ci(y_true, y_prob, prec_fn, n_bootstrap, confidence_level)
        results["precision"] = {"value": val, "lower": lower, "upper": upper}

        # Recall
        def rec_fn(yt, yp):
            return recall_score(yt, (yp >= threshold).astype(int), zero_division=0)
        val, lower, upper = bootstrap_ci(y_true, y_prob, rec_fn, n_bootstrap, confidence_level)
        results["recall"] = {"value": val, "lower": lower, "upper": upper}

        # F1
        def f1_fn(yt, yp):
            return f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
        val, lower, upper = bootstrap_ci(y_true, y_prob, f1_fn, n_bootstrap, confidence_level)
        results["f1"] = {"value": val, "lower": lower, "upper": upper}

        # AUC
        try:
            def auc_fn(yt, yp):
                return roc_auc_score(yt, yp)
            val, lower, upper = bootstrap_ci(y_true, y_prob, auc_fn, n_bootstrap, confidence_level)
            results["auc"] = {"value": val, "lower": lower, "upper": upper}
        except:
            results["auc"] = {"value": 0.0, "lower": 0.0, "upper": 0.0}

        # AUPRC
        try:
            def auprc_fn(yt, yp):
                precision, recall, _ = precision_recall_curve(yt, yp)
                order = np.argsort(recall)
                return float(np.trapz(precision[order], recall[order]))
            val, lower, upper = bootstrap_ci(y_true, y_prob, auprc_fn, n_bootstrap, confidence_level)
            results["auprc"] = {"value": val, "lower": lower, "upper": upper}
        except:
            results["auprc"] = {"value": 0.0, "lower": 0.0, "upper": 0.0}

        # Average Precision
        try:
            def ap_fn(yt, yp):
                return average_precision_score(yt, yp)
            val, lower, upper = bootstrap_ci(y_true, y_prob, ap_fn, n_bootstrap, confidence_level)
            results["average_precision"] = {"value": val, "lower": lower, "upper": upper}
        except:
            results["average_precision"] = {"value": 0.0, "lower": 0.0, "upper": 0.0}

        # Specificity
        def spec_fn(yt, yp):
            preds = (yp >= threshold).astype(int)
            tn, fp, _, _ = confusion_matrix(yt, preds, labels=[0, 1]).ravel()
            denom = tn + fp
            return float(tn / denom) if denom > 0 else 0.0
        val, lower, upper = bootstrap_ci(y_true, y_prob, spec_fn, n_bootstrap, confidence_level)
        results["specificity"] = {"value": val, "lower": lower, "upper": upper}

        def ece_fn(yt, yp):
            probs = np.clip(yp, 0.0, 1.0)
            preds = (probs >= threshold).astype(int)
            conf = np.where(preds == 1, probs, 1.0 - probs)
            correct = (preds == yt).astype(float)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            n = len(yt)
            for i in range(n_bins):
                lo = bins[i]
                hi = bins[i + 1]
                mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
                if not np.any(mask):
                    continue
                bin_acc = float(np.mean(correct[mask]))
                bin_conf = float(np.mean(conf[mask]))
                ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
            return float(ece)
        val, lower, upper = bootstrap_ci(y_true, y_prob, ece_fn, n_bootstrap, confidence_level)
        results["ece"] = {"value": val, "lower": lower, "upper": upper}

        if compute_brier:
            def brier_fn(yt, yp):
                return float(np.mean((yp - yt) ** 2))
            val, lower, upper = bootstrap_ci(y_true, y_prob, brier_fn, n_bootstrap, confidence_level)
            results["brier_score"] = {"value": val, "lower": lower, "upper": upper}

        # p-value (two-sided z-test vs 0.5 accuracy baseline)
        n = int(y_true.shape[0])
        if n > 0:
            correct = int((y_true == y_pred).sum())
            null_p = 0.5
            var = null_p * (1.0 - null_p) / n
            if var > 0:
                z = (correct / n - null_p) / math.sqrt(var)
                results["p_value"] = math.erfc(abs(z) / math.sqrt(2.0))
            else:
                results["p_value"] = float("nan")
        else:
            results["p_value"] = float("nan")
    
    return results


def _safe_metric_value(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def compute_regression_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """Compute regression metrics with bootstrap confidence intervals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results = {}

    def mae_fn(yt, yp):
        return _safe_metric_value(mean_absolute_error(yt, yp))
    val, lower, upper = bootstrap_ci(y_true, y_pred, mae_fn, n_bootstrap, confidence_level)
    results["mae"] = {"value": val, "lower": lower, "upper": upper}

    def rmse_fn(yt, yp):
        return _safe_metric_value(math.sqrt(mean_squared_error(yt, yp)))
    val, lower, upper = bootstrap_ci(y_true, y_pred, rmse_fn, n_bootstrap, confidence_level)
    results["rmse"] = {"value": val, "lower": lower, "upper": upper}

    def r2_fn(yt, yp):
        return _safe_metric_value(r2_score(yt, yp))
    val, lower, upper = bootstrap_ci(y_true, y_pred, r2_fn, n_bootstrap, confidence_level)
    results["r2"] = {"value": val, "lower": lower, "upper": upper}

    def pearson_fn(yt, yp):
        if yt.size < 2:
            return 0.0
        corr = np.corrcoef(yt, yp)[0, 1]
        return _safe_metric_value(corr)
    val, lower, upper = bootstrap_ci(y_true, y_pred, pearson_fn, n_bootstrap, confidence_level)
    results["pearson_r"] = {"value": val, "lower": lower, "upper": upper}

    def spearman_fn(yt, yp):
        if yt.size < 2:
            return 0.0
        yt_rank = pd.Series(yt).rank(method="average").to_numpy()
        yp_rank = pd.Series(yp).rank(method="average").to_numpy()
        corr = np.corrcoef(yt_rank, yp_rank)[0, 1]
        return _safe_metric_value(corr)
    val, lower, upper = bootstrap_ci(y_true, y_pred, spearman_fn, n_bootstrap, confidence_level)
    results["spearman_r"] = {"value": val, "lower": lower, "upper": upper}

    return results


def calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal threshold to maximize a given metric.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        metric: Metric to optimize ("f1", "accuracy", "precision", "recall")
    
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_value = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif metric == "precision":
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value


def ensure_unique_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: str
):
    """Save predictions to CSV, avoiding overwrite with _1, _2, ... suffixes."""
    output_path = ensure_unique_path(output_path)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    return output_path


def save_metrics_report(
    metrics: Dict,
    output_path: str,
    task_name: str = "",
    mode: str = "",
    model_name: str = "",
    threshold: Optional[float] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    zero_shot_mode: Optional[str] = None,
    prompts_used: Optional[Dict[str, List[str]]] = None,
):
    """Save metrics report to CSV."""
    row = {
        "model_name": model_name,
        "task": task_name,
        "mode": mode,
    }
    if zero_shot_mode is not None:
        row["zero_shot_mode"] = zero_shot_mode
    if label_mapping is not None:
        try:
            normalized = {str(k): int(v) for k, v in label_mapping.items()}
        except Exception:
            normalized = {str(k): str(v) for k, v in label_mapping.items()}
        row["label_mapping"] = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    if prompts_used is not None:
        row["prompts_used"] = json.dumps(prompts_used, sort_keys=True, separators=(",", ":"))
    if threshold is not None:
        row["threshold"] = float(threshold)

    for name, values in metrics.items():
        if isinstance(values, dict):
            row[name] = values.get("value")
            row[f"{name}_ci_lower"] = values.get("lower")
            row[f"{name}_ci_upper"] = values.get("upper")
        else:
            row[name] = values

    columns = list(row.keys())
    df = pd.DataFrame([row], columns=columns)
    output_path = Path(output_path)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        for col in df.columns:
            if col not in existing.columns:
                existing[col] = np.nan
        for col in existing.columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[existing.columns]
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    print(f"Saved metrics report to: {output_path}")


def format_metrics_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """Format metrics with CIs as a table string."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{'Metric':<15} {'Value':>10} {'95% CI':>20}")
    lines.append("-" * 60)
    
    for name, values in metrics.items():
        if isinstance(values, dict):
            val = values["value"]
            lower = values["lower"]
            upper = values["upper"]
            ci_str = f"[{lower:.3f}, {upper:.3f}]"
            lines.append(f"{name:<15} {val:>10.4f} {ci_str:>20}")
        else:
            lines.append(f"{name:<15} {values:>10.4f}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    preds_path = "results/aortic_stenosis/aortic_stenosis_linear_probe_preds.csv"
    output_path = "results/aortic_stenosis/aortic_stenosis_linear_probe_metrics.csv"
    threshold = 0.5
    n_bootstrap = 1000
    confidence_level = 0.95
    task_name = "aortic_stenosis"
    mode = "linear_probe"
    model_name = ""
    classification_mode = "max"

    df = pd.read_csv(preds_path)
    if "label" in df.columns:
        y_true = df["label"].to_numpy()
    elif "y_true" in df.columns:
        y_true = df["y_true"].to_numpy()
    else:
        raise ValueError("Predictions CSV must contain a 'label' column.")

    if "score" in df.columns:
        y_prob = df["score"].to_numpy()
    elif "prob" in df.columns:
        y_prob = df["prob"].to_numpy()
    elif "probability" in df.columns:
        y_prob = df["probability"].to_numpy()
    elif "y_prob" in df.columns:
        y_prob = df["y_prob"].to_numpy()
    else:
        raise ValueError("Predictions CSV must contain a probability column like 'score'.")

    if "prediction" in df.columns:
        y_pred = df["prediction"].to_numpy()
    elif "pred" in df.columns:
        y_pred = df["pred"].to_numpy()
    elif "y_pred" in df.columns:
        y_pred = df["y_pred"].to_numpy()
    else:
        y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics_with_ci(
        y_true,
        y_pred,
        y_prob,
        threshold=threshold,
        classification_mode=classification_mode,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )
    print(format_metrics_table(metrics))

    if output_path:
        save_metrics_report(
            metrics,
            output_path,
            task_name=task_name,
            mode=mode,
            model_name=model_name,
            threshold=threshold,
        )