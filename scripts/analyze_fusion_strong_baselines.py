import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


warnings.filterwarnings("ignore")

FEATURES = [
    "risk_prob",
    "qed",
    "lp_min",
    "center_dist",
    "contacts_lt_4_0_per_lig",
    "clash_lt_1_5_per_lig",
    "pock_n",
    "mol_wt",
    "heavy_atoms",
    "intramol_pass",
]
BLOCKS = {
    "risk": ["risk_prob"],
    "geometry": ["lp_min", "center_dist", "contacts_lt_4_0_per_lig", "clash_lt_1_5_per_lig", "pock_n"],
    "validity": ["intramol_pass"],
    "chemistry": ["qed", "mol_wt", "heavy_atoms"],
}
SCENARIOS = ["full", "missing_risk", "missing_geometry", "missing_validity", "missing_chemistry"]
METHODS = ["late_fusion", "logistic_stacking", "bayesian_fusion", "hgb_early", "xgboost_early", "lightgbm_early"]


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def load_pool(path, source):
    df = pd.read_csv(path, low_memory=False)
    if "kind" in df.columns:
        df = df[df["kind"] == "generated"].copy()
    df["source"] = source
    df["target_key"] = df["key"].astype(str)
    df["intramol_pass"] = df["intramol_pass"].fillna(False).astype(bool)
    df["dock_pose_pass"] = df["dock_pose_pass"].fillna(False).astype(bool)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df["label_success"] = df["dock_pose_pass"].astype(int)
    return df


def split_targets(df, seed, test_frac=0.30):
    rng = np.random.default_rng(seed)
    targets = np.array(sorted(df["target_key"].unique()))
    rng.shuffle(targets)
    n_test = max(1, int(round(len(targets) * test_frac)))
    test = set(targets[:n_test])
    return df[~df.target_key.isin(test)].copy(), df[df.target_key.isin(test)].copy()


def medians(train):
    return train[FEATURES].astype(float).median(numeric_only=True)


def apply_missing(df, train, scenario):
    out = df.copy()
    med = medians(train)
    if scenario == "full":
        return out
    block = scenario.replace("missing_", "")
    for col in BLOCKS[block]:
        out[col] = med[col]
    return out


def feature_matrix(df):
    return df[FEATURES].astype(float).to_numpy()


def base_score_matrix(df):
    risk_success = 1.0 - df["risk_prob"].astype(float).clip(0, 1)
    qed = df["qed"].astype(float).clip(0, 1)
    geom = (
        (df["lp_min"].astype(float).clip(0, 5) / 5.0)
        + (1.0 - (df["center_dist"].astype(float).clip(0, 10) / 10.0))
        + (df["contacts_lt_4_0_per_lig"].astype(float).clip(0, 6) / 6.0)
        + (1.0 - df["clash_lt_1_5_per_lig"].astype(float).clip(0, 2) / 2.0)
    ) / 4.0
    valid = df["intramol_pass"].astype(float)
    size = 1.0 - ((df["heavy_atoms"].astype(float) - 25).abs().clip(0, 40) / 40.0)
    return np.vstack([risk_success, qed, geom, valid, size]).T


def fit_method(train, method, seed):
    y = train["label_success"].to_numpy(int)
    if method == "late_fusion":
        return None
    if method == "logistic_stacking":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)).fit(base_score_matrix(train), y)
    if method == "bayesian_fusion":
        return make_pipeline(StandardScaler(), GaussianNB()).fit(feature_matrix(train), y)
    if method == "hgb_early":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=0.01, random_state=seed).fit(feature_matrix(train), y)
    if method == "xgboost_early":
        if XGBClassifier is None:
            return None
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=8,
        ).fit(feature_matrix(train), y)
    if method == "lightgbm_early":
        if LGBMClassifier is None:
            return None
        return LGBMClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=8,
            verbosity=-1,
        ).fit(feature_matrix(train), y)
    raise ValueError(method)


def predict_method(model, df, method):
    if method == "late_fusion":
        scores = base_score_matrix(df)
        weights = np.array([0.35, 0.15, 0.25, 0.20, 0.05])
        return np.clip(scores @ weights, 0, 1)
    if model is None:
        return np.full(len(df), np.nan)
    x = base_score_matrix(df) if method == "logistic_stacking" else feature_matrix(df)
    prob = model.predict_proba(x)[:, 1]
    if method == "bayesian_fusion":
        entropy = -(prob * np.log(np.clip(prob, 1e-6, 1)) + (1 - prob) * np.log(np.clip(1 - prob, 1e-6, 1)))
        prob = np.clip(prob - 0.10 * entropy / np.log(2), 0, 1)
    return prob


def ece_score(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metric_row(direction, method, scenario, pred_df):
    y = pred_df["label_success"].to_numpy(int)
    p = pred_df["fusion_prob"].to_numpy(float)
    valid = np.isfinite(p)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        auroc = auprc = np.nan
    else:
        auroc = float(roc_auc_score(y[valid], p[valid]))
        auprc = float(average_precision_score(y[valid], p[valid]))
    return {
        "direction": direction,
        "method": method,
        "scenario": scenario,
        "n": int(valid.sum()),
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(brier_score_loss(y[valid], p[valid])) if valid.sum() else np.nan,
        "ece": ece_score(y[valid], p[valid]) if valid.sum() else np.nan,
    }


def select_fusion(df, k=4):
    rows = []
    for _, group in df.groupby("target_key", sort=True):
        rows.append(group.sort_values(["fusion_prob", "qed", "risk_prob"], ascending=[False, False, True]).head(k))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def selection_row(direction, method, scenario, pred_df):
    sel = select_fusion(pred_df)
    return {
        "direction": direction,
        "method": method,
        "scenario": scenario,
        "selected": int(len(sel)),
        "targets": int(pred_df["target_key"].nunique()),
        "dock_fast": float(sel["dock_pose_pass"].mean()) if len(sel) else np.nan,
        "risk_gt_0_5": float((sel["risk_prob"] > 0.5).mean()) if len(sel) else np.nan,
        "qed": float(sel["qed"].mean()) if len(sel) else np.nan,
        "mol_fast": float(sel["intramol_pass"].mean()) if len(sel) else np.nan,
        "fusion_prob": float(sel["fusion_prob"].mean()) if len(sel) else np.nan,
    }


def run_direction(train, test, direction, seed):
    metric_rows, selection_rows = [], []
    for method in METHODS:
        model = fit_method(train, method, seed)
        for scenario in SCENARIOS:
            pert = apply_missing(test, train, scenario)
            pert["fusion_prob"] = predict_method(model, pert, method)
            metric_rows.append(metric_row(direction, method, scenario, pert))
            selection_rows.append(selection_row(direction, method, scenario, pert))
    return metric_rows, selection_rows


def main():
    diff = load_pool("results/dockfast_full_pool_fullatom_cond.csv", "DiffSBDD")
    pocket = load_pool("results/dockfast_full_pool_pocket2mol_n16_ext.csv", "Pocket2Mol")
    metrics, selections = [], []
    for i in range(3):
        seed = 20260516 + i
        tr, te = split_targets(diff, seed)
        m, s = run_direction(tr, te, "DiffSBDD_within", seed)
        metrics.extend([dict(x, seed=i) for x in m])
        selections.extend([dict(x, seed=i) for x in s])
        tr, te = split_targets(pocket, seed)
        m, s = run_direction(tr, te, "Pocket2Mol_within", seed)
        metrics.extend([dict(x, seed=i) for x in m])
        selections.extend([dict(x, seed=i) for x in s])
    for train, test, direction in [(diff, pocket, "DiffSBDD_to_Pocket2Mol"), (pocket, diff, "Pocket2Mol_to_DiffSBDD")]:
        m, s = run_direction(train, test, direction, 20260516)
        metrics.extend([dict(x, seed=0) for x in m])
        selections.extend([dict(x, seed=0) for x in s])

    met = pd.DataFrame(metrics)
    sel = pd.DataFrame(selections)
    agg_m = met.groupby(["direction", "method", "scenario"], as_index=False).agg(
        seeds=("seed", "nunique"),
        n=("n", "mean"),
        auroc=("auroc", "mean"),
        auprc=("auprc", "mean"),
        brier=("brier", "mean"),
        ece=("ece", "mean"),
    )
    agg_s = sel.groupby(["direction", "method", "scenario"], as_index=False).agg(
        seeds=("seed", "nunique"),
        selected=("selected", "mean"),
        dock_fast=("dock_fast", "mean"),
        risk_gt_0_5=("risk_gt_0_5", "mean"),
        qed=("qed", "mean"),
        mol_fast=("mol_fast", "mean"),
        fusion_prob=("fusion_prob", "mean"),
    )
    met.to_csv("results/fusion_strong_baselines_metrics.csv", index=False)
    sel.to_csv("results/fusion_strong_baselines_selection.csv", index=False)
    agg_m.to_csv("results/fusion_strong_baselines_metrics_agg.csv", index=False)
    agg_s.to_csv("results/fusion_strong_baselines_selection_agg.csv", index=False)

    view = agg_m[agg_m["scenario"].isin(["full", "missing_risk", "missing_geometry", "missing_validity"])].copy()
    lines = [
        "# Fusion Strong Baselines and Missing-Modality Calibration",
        "",
        "## Protocol",
        "",
        "- Baselines: late fusion, logistic stacking, uncertainty-aware Bayesian fusion, HGB early fusion, XGBoost early fusion, and LightGBM early fusion.",
        "- Missing-modality calibration replaces one oracle block with training medians at test time.",
        "- Metrics are true dock_fast-success AUROC/AUPRC, Brier, and ECE.",
        "",
        "## Calibration Metrics",
        "",
        "| Direction | Method | Scenario | Seeds | AUROC | AUPRC | Brier | ECE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in view.sort_values(["direction", "scenario", "method"]).itertuples(index=False):
        lines.append(f"| {row.direction} | {row.method} | {row.scenario} | {row.seeds} | {f4(row.auroc)} | {f4(row.auprc)} | {f4(row.brier)} | {f4(row.ece)} |")
    best = agg_m[agg_m.scenario == "full"].sort_values(["direction", "auroc"], ascending=[True, False]).groupby("direction").head(1)
    lines.extend(["", "## Best Full-Modality Models", "", "| Direction | Method | AUROC | AUPRC | ECE |", "|---|---|---:|---:|---:|"])
    for row in best.itertuples(index=False):
        lines.append(f"| {row.direction} | {row.method} | {f4(row.auroc)} | {f4(row.auprc)} | {f4(row.ece)} |")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. The stronger fusion baselines test whether the reliability gains are specific to one HGB fusion model.",
            "2. Missing-modality calibration exposes which oracle blocks are necessary for deployment.",
            "3. The safest claim is multi-oracle fusion robustness, not superiority of one classifier family.",
        ]
    )
    Path("experiments/FUSION_STRONG_BASELINES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
