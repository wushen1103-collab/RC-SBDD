import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def expected_calibration_error(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true).astype(float)
    prob = np.asarray(prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if not np.any(mask):
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def risk_coverage_curve(y_failure, risk_score):
    y_failure = np.asarray(y_failure).astype(int)
    risk_score = np.asarray(risk_score).astype(float)
    order = np.argsort(risk_score)
    failures = y_failure[order]
    coverages, failure_rates = [], []
    for k in range(1, len(failures) + 1):
        coverages.append(k / len(failures))
        failure_rates.append(float(failures[:k].mean()))
    return np.asarray(coverages), np.asarray(failure_rates)


def summarize_risk_metrics(y_failure, risk_prob):
    y_failure = np.asarray(y_failure).astype(int)
    risk_prob = np.asarray(risk_prob).astype(float)
    cov, fr = risk_coverage_curve(y_failure, risk_prob)
    return {
        'failure_auroc': float(roc_auc_score(y_failure, risk_prob)) if len(set(y_failure)) == 2 else None,
        'brier': float(brier_score_loss(y_failure, risk_prob)),
        'ece': expected_calibration_error(y_failure, risk_prob),
        'risk_coverage_auc_failure_rate': float(np.trapz(fr, cov)),
    }
