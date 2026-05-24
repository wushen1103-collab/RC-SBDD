from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def load_dataset(path, name, dedupe_cols=None):
    df = pd.read_csv(path, low_memory=False)
    if "kind" in df.columns:
        df = df[df["kind"] == "generated"].copy()
    if dedupe_cols:
        cols = [c for c in dedupe_cols if c in df.columns]
        df = df.drop_duplicates(cols)
    df = df[df["risk_prob"].notna() & df["dock_pose_pass"].notna()].copy()
    df["dataset"] = name
    df["risk_prob"] = df["risk_prob"].astype(float).clip(0, 1)
    df["dock_pose_pass"] = df["dock_pose_pass"].fillna(False).astype(bool)
    df["dock_fast_failure"] = 1 - df["dock_pose_pass"].astype(int)
    return df


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            score += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(score)


def target_col(df):
    return "key" if "key" in df.columns else "data_id"


def oof_calibrated_predictions(df, method, folds=5):
    p = df["risk_prob"].to_numpy(float)
    y = df["dock_fast_failure"].to_numpy(int)
    if method == "raw":
        return p
    pred = np.full(len(df), np.nan, dtype=float)
    groups = np.array(sorted(df[target_col(df)].astype(str).unique()))
    fold_map = {g: i % folds for i, g in enumerate(groups)}
    group_fold = df[target_col(df)].astype(str).map(fold_map).to_numpy(int)
    for fold in range(folds):
        train = group_fold != fold
        test = group_fold == fold
        if test.sum() == 0:
            continue
        if len(np.unique(y[train])) < 2:
            pred[test] = y[train].mean()
            continue
        if method == "platt_oof":
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(p[train].reshape(-1, 1), y[train])
            pred[test] = model.predict_proba(p[test].reshape(-1, 1))[:, 1]
        elif method == "isotonic_oof":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p[train], y[train])
            pred[test] = model.predict(p[test])
        else:
            raise ValueError(method)
    return np.clip(pred, 0, 1)


def reliability_rows(df, calibrator, pred, bins=10):
    rows = []
    y = df["dock_fast_failure"].to_numpy(float)
    p = np.asarray(pred, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not m.any():
            rows.append(
                {
                    "dataset": df["dataset"].iloc[0],
                    "calibrator": calibrator,
                    "bin": i,
                    "bin_low": lo,
                    "bin_high": hi,
                    "n": 0,
                    "mean_predicted_failure": np.nan,
                    "observed_failure_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        rows.append(
            {
                "dataset": df["dataset"].iloc[0],
                "calibrator": calibrator,
                "bin": i,
                "bin_low": lo,
                "bin_high": hi,
                "n": int(m.sum()),
                "mean_predicted_failure": float(p[m].mean()),
                "observed_failure_rate": float(y[m].mean()),
                "abs_gap": float(abs(y[m].mean() - p[m].mean())),
            }
        )
    return rows


def summary_row(df, calibrator, pred):
    y = df["dock_fast_failure"].to_numpy(int)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(np.unique(y)) < 2:
        auroc = auprc = np.nan
    else:
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
    return {
        "dataset": df["dataset"].iloc[0],
        "calibrator": calibrator,
        "n": int(len(df)),
        "targets": int(df[target_col(df)].nunique()),
        "failure_rate": float(y.mean()),
        "mean_predicted_failure": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "auroc_failure": auroc,
        "auprc_failure": auprc,
    }


def main():
    datasets = [
        load_dataset("results/dockfast_full_pool_fullatom_cond.csv", "DiffSBDD_full_pool", ["key", "source_file", "mol_index"]),
        load_dataset("results/dockfast_full_pool_pocket2mol_n16_ext.csv", "Pocket2Mol_full_pool", ["data_id", "source_file", "mol_index"]),
        load_dataset(
            "results/prospective20_pocket2mol_n128_dockfast_selection.csv",
            "Prospective20_selected",
            ["policy", "data_id", "source_file", "mol_index"],
        ),
        load_dataset(
            "results/syncguide_t1000_n16_dockfast_selection.csv",
            "SYNCGuide_selected",
            ["policy", "data_id", "source_file", "mol_index"],
        ),
    ]
    summary_rows = []
    reliability_all = []
    for df in datasets:
        for calibrator in ["raw", "platt_oof", "isotonic_oof"]:
            pred = oof_calibrated_predictions(df, calibrator)
            summary_rows.append(summary_row(df, calibrator, pred))
            reliability_all.extend(reliability_rows(df, calibrator, pred))
    summary = pd.DataFrame(summary_rows)
    reliability = pd.DataFrame(reliability_all)
    summary.to_csv("results/risk_to_dockfast_calibration_summary.csv", index=False)
    reliability.to_csv("results/risk_to_dockfast_reliability_curve.csv", index=False)

    lines = [
        "# Risk-to-dockfast Calibration",
        "",
        "## Protocol",
        "",
        "`risk_prob` is evaluated as a probability of real dock_fast failure (`1 - dock_pose_pass`) on observed PoseBusters dock_fast labels.",
        "This is separate from synthetic corruption calibration and directly tests whether risk tracks downstream fast docking/pose failures.",
        "",
        "## Summary",
        "",
        "| Dataset | Calibrator | N | Targets | Failure rate | Mean predicted failure | Brier | ECE | AUROC failure | AUPRC failure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.calibrator} | {row.n} | {row.targets} | {f4(row.failure_rate)} | {f4(row.mean_predicted_failure)} | "
            f"{f4(row.brier)} | {f4(row.ece)} | {f4(row.auroc_failure)} | {f4(row.auprc_failure)} |"
        )
    worst = reliability[reliability["calibrator"] == "raw"].dropna(subset=["abs_gap"]).sort_values("abs_gap", ascending=False).head(8)
    lines.extend(["", "## Largest Raw-Risk Reliability Gaps", "", "| Dataset | Bin | N | Predicted failure | Observed failure | Gap |", "|---|---:|---:|---:|---:|---:|"])
    for row in worst.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.bin} | {row.n} | {f4(row.mean_predicted_failure)} | "
            f"{f4(row.observed_failure_rate)} | {f4(row.abs_gap)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The table reports calibration against real dock_fast outcomes and can be cited beside the synthetic corruption calibration to show deployment-facing reliability.",
        ]
    )
    Path("experiments/RISK_TO_DOCKFAST_CALIBRATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
