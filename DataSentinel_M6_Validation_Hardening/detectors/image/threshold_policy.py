"""Validation-only threshold calibration policies for DataSentinel M6."""
from __future__ import annotations
import numpy as np


def choose_threshold_max_recall_under_fpr(
    y_true: np.ndarray,
    risk_probability: np.ndarray,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    """Return (threshold, F1) maximizing recall subject to a validation FPR cap.

    Tie-break order: precision, F1, then higher threshold. This is an explicit operating
    policy: avoid very large quarantine sets when that would remove legitimate data.

    Important: the FPR cap applies to the validation partition only. The held-out test FPR
    can be higher and must be reported. If ranking is weak, the selected threshold may
    yield very low recall; threshold tuning cannot repair poor ROC-AUC.
    """
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(risk_probability, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or len(y) == 0:
        raise ValueError("y_true and risk_probability must be non-empty aligned 1-D arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 (clean) and 1 (poisoned)")
    if not np.isfinite(scores).all():
        raise ValueError("risk_probability contains NaN or infinite values")
    max_fpr = float(np.clip(max_fpr, 0.0, 1.0))
    candidates = np.unique(np.concatenate([scores, [np.nextafter(scores.max(), np.inf)]]))
    negatives = int((y == 0).sum())
    positives = int((y == 1).sum())
    best_key = None
    conservative = (float(np.nextafter(scores.max(), np.inf)), 0.0)

    for threshold in candidates:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = positives - tp
        tn = negatives - fp
        fpr = fp / max(negatives, 1)
        if fpr > max_fpr + 1e-12:
            continue
        recall = tp / max(positives, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        key = (recall, precision, f1, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            conservative = (float(threshold), float(f1))
    return conservative


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute thresholded metrics plus ranking metrics without hiding class imbalance."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "quarantine_rate": float(pred.mean()),
        "quarantined_count": int(pred.sum()),
        "positive_prevalence": float(y.mean()),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["pr_auc"] = float(average_precision_score(y, scores))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out
