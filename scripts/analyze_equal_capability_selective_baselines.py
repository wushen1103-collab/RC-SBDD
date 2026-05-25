from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_fusion_strong_baselines import (
    METHODS,
    SCENARIOS,
    apply_missing,
    fit_method,
    load_pool,
    predict_method,
    split_targets,
)


SEEDS = [20260525, 20260526, 20260527, 20260528, 20260529]
ALPHA = 0.10
METHOD_NAMES = ["RC-risk"] + METHODS


def candidate_score(df: pd.DataFrame, method: str, model) -> np.ndarray:
    if method == "RC-risk":
        return 1.0 - df["risk_prob"].astype(float).clip(0, 1).to_numpy()
    return predict_method(model, df, method)


def pb_prefilter(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["intramol_pass"].fillna(False).astype(bool)].copy()
    return filtered if len(filtered) else df.copy()


def top_one(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["decision_score", "qed", "risk_prob"], ascending=[False, False, True])
        .groupby("target_key", sort=True)
        .head(1)
        .copy()
    )


def calibrate_threshold(calibration: pd.DataFrame) -> float:
    ranked = top_one(pb_prefilter(calibration))
    candidates = []
    for threshold in sorted(ranked["decision_score"].dropna().unique().tolist()):
        accepted = ranked[ranked["decision_score"] >= threshold]
        if len(accepted) < 5:
            continue
        failure = float(1.0 - accepted["dock_pose_pass"].astype(float).mean())
        candidates.append((failure <= ALPHA, len(accepted), -failure, -threshold, threshold))
    if not candidates:
        return 1.0
    feasible = [row for row in candidates if row[0]]
    chosen = max(feasible if feasible else candidates, key=lambda row: row[1:4])
    return float(chosen[-1])


def evaluate(test: pd.DataFrame, threshold: float) -> dict[str, float]:
    ranked = top_one(pb_prefilter(test))
    accepted = ranked[ranked["decision_score"] >= threshold].copy()
    targets = int(ranked["target_key"].nunique())
    covered = int(accepted["target_key"].nunique())
    return {
        "targets": targets,
        "covered_targets": covered,
        "coverage": float(covered / targets) if targets else np.nan,
        "dock_fast": float(accepted["dock_pose_pass"].mean()) if covered else np.nan,
        "selective_failure": float(1.0 - accepted["dock_pose_pass"].mean()) if covered else np.nan,
        "risk_gt_0_5": float((accepted["risk_prob"] > 0.5).mean()) if covered else np.nan,
        "qed": float(accepted["qed"].mean()) if covered else np.nan,
    }


def run_direction(source: pd.DataFrame, target: pd.DataFrame, direction: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        fit, calibration = split_targets(source, seed, test_frac=0.30)
        for method in METHOD_NAMES:
            model = None if method == "RC-risk" else fit_method(fit, method, seed)
            if method != "RC-risk" and model is None:
                continue
            for scenario in SCENARIOS:
                cal_view = apply_missing(calibration, fit, scenario)
                test_view = apply_missing(target, fit, scenario)
                cal_view["decision_score"] = candidate_score(cal_view, method, model)
                test_view["decision_score"] = candidate_score(test_view, method, model)
                threshold = calibrate_threshold(cal_view)
                out = evaluate(test_view, threshold)
                out.update({"direction": direction, "seed": seed, "method": method, "scenario": scenario, "threshold": threshold})
                rows.append(out)
    return rows


def main() -> None:
    diff = load_pool("results/dockfast_full_pool_fullatom_cond.csv", "DiffSBDD")
    pocket = load_pool("results/dockfast_full_pool_pocket2mol_n16_ext.csv", "Pocket2Mol")
    raw = pd.DataFrame(run_direction(diff, pocket, "DiffSBDD_to_Pocket2Mol") + run_direction(pocket, diff, "Pocket2Mol_to_DiffSBDD"))
    agg = raw.groupby(["direction", "method", "scenario"], as_index=False).agg(
        seeds=("seed", "nunique"),
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        dock_fast_mean=("dock_fast", "mean"),
        dock_fast_std=("dock_fast", "std"),
        selective_failure_mean=("selective_failure", "mean"),
        risk_gt_0_5_mean=("risk_gt_0_5", "mean"),
        qed_mean=("qed", "mean"),
    )
    Path("results").mkdir(exist_ok=True)
    raw.to_csv("results/equal_capability_selective_baselines_raw.csv", index=False)
    agg.to_csv("results/equal_capability_selective_baselines_summary.csv", index=False)
    print(agg[agg["scenario"] == "full"].to_string(index=False))


if __name__ == "__main__":
    main()
