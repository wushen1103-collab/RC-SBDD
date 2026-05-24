"""Risk-scoring metrics used by the lightweight smoke tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(y_true), 1)
    error = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(error * len(y_true) / total)


def _safe_metric(fn, y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(fn(y_true, prob))
    except ValueError:
        return float("nan")


def summarize_risk_metrics(y_true, prob) -> dict[str, float]:
    """Return discrimination and calibration metrics for binary failure risk.

    ``y_true=1`` denotes failure/corruption/high-risk class in the smoke tests.
    """

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    return {
        "auroc": _safe_metric(roc_auc_score, y, p),
        "auprc": _safe_metric(average_precision_score, y, p),
        "brier": _safe_metric(brier_score_loss, y, p),
        "ece": _ece(y, p),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "mean_prob": float(p.mean()) if len(p) else float("nan"),
        "n": int(len(y)),
    }
