import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * float(x):.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{float(x):.4f}"


def normalize_bound(row):
    method = str(row["method"])
    if method == "tc_crc_stratified":
        vals = [row.get("calib_bound_easy", np.nan), row.get("calib_bound_hard", np.nan)]
        vals = [float(v) for v in vals if pd.notna(v)]
        return max(vals) if vals else np.nan
    return float(row.get("calib_bound", np.nan)) if pd.notna(row.get("calib_bound", np.nan)) else np.nan


def build(raw):
    raw = raw.copy()
    raw = raw[raw["method"].isin(["qed_topk", "pb_qed_topk", "tc_crc_global", "tc_crc_stratified"])].copy()
    raw["calibration_bound"] = raw.apply(normalize_bound, axis=1)
    raw["alpha_value"] = raw["alpha"].fillna(np.nan)
    raw["test_loss"] = raw["loss"].astype(float)
    raw["test_pass"] = 1.0 - raw["test_loss"]
    raw["bound_satisfies_alpha"] = np.where(
        raw["alpha_value"].notna(),
        raw["calibration_bound"] <= raw["alpha_value"],
        np.nan,
    )
    raw["target_heldout_violation"] = np.where(
        raw["alpha_value"].notna(),
        raw["test_loss"] > raw["alpha_value"],
        np.nan,
    )
    raw["excess_loss"] = np.where(
        raw["alpha_value"].notna(),
        raw["test_loss"] - raw["alpha_value"],
        np.nan,
    )
    cols = [
        "generator",
        "seed",
        "method",
        "alpha_value",
        "feasible",
        "calibration_bound",
        "bound_satisfies_alpha",
        "targets",
        "coverage",
        "reach_k",
        "test_loss",
        "test_pass",
        "target_heldout_violation",
        "excess_loss",
        "dock_pose_pass",
        "risk_mean",
        "risk_gt_0_5",
        "qed_mean",
        "intramol_pass",
    ]
    return raw[cols].copy()


def summarize(df):
    rows = []
    for (generator, method), group in df.groupby(["generator", "method"], sort=True):
        row = {
            "generator": generator,
            "method": method,
            "seeds": int(group["seed"].nunique()),
            "alpha": float(group["alpha_value"].dropna().iloc[0]) if group["alpha_value"].notna().any() else np.nan,
            "feasible_rate": float(group["feasible"].mean()) if "feasible" in group else np.nan,
            "bound_satisfy_rate": float(group["bound_satisfies_alpha"].dropna().mean()) if group["bound_satisfies_alpha"].notna().any() else np.nan,
            "heldout_violation_rate": float(group["target_heldout_violation"].dropna().mean()) if group["target_heldout_violation"].notna().any() else np.nan,
            "calibration_bound_mean": float(group["calibration_bound"].mean()) if group["calibration_bound"].notna().any() else np.nan,
            "test_loss_mean": float(group["test_loss"].mean()),
            "excess_loss_mean": float(group["excess_loss"].mean()) if group["excess_loss"].notna().any() else np.nan,
            "coverage_mean": float(group["coverage"].mean()),
            "reach_k_mean": float(group["reach_k"].mean()),
            "dock_pose_pass_mean": float(group["dock_pose_pass"].mean()),
            "risk_mean": float(group["risk_mean"].mean()),
            "risk_gt_0_5_mean": float(group["risk_gt_0_5"].mean()),
            "qed_mean": float(group["qed_mean"].mean()),
            "intramol_pass_mean": float(group["intramol_pass"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_md(summary, out_md):
    order = {"qed_topk": 0, "pb_qed_topk": 1, "tc_crc_global": 2, "tc_crc_stratified": 3}
    summary = summary.copy()
    summary["order"] = summary["method"].map(order)
    lines = [
        "# Target-Heldout Selective Guarantee",
        "",
        "## Protocol",
        "",
        "- Calibration/test units are targets, not molecules.",
        "- `calibration_bound` is the finite-sample target-level CRC bound selected on calibration targets.",
        "- `heldout_violation_rate` is the fraction of random target splits where test target loss is greater than alpha.",
        "- Baselines are reported without a calibration guarantee and therefore have `NA` bound/violation fields.",
        "",
        "| Generator | Method | Seeds | Alpha | Feasible | Bound<=alpha | Heldout violation | Cal bound | Test loss | Excess loss | Coverage | Reach K | dock_fast | Mean risk | Risk >0.5 | QED | mol_fast |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["generator", "order"]).itertuples(index=False):
        lines.append(
            f"| {row.generator} | {row.method} | {row.seeds} | {f4(row.alpha)} | "
            f"{pct(row.feasible_rate)} | {pct(row.bound_satisfy_rate)} | {pct(row.heldout_violation_rate)} | "
            f"{f4(row.calibration_bound_mean)} | {f4(row.test_loss_mean)} | {f4(row.excess_loss_mean)} | "
            f"{pct(row.coverage_mean)} | {pct(row.reach_k_mean)} | {pct(row.dock_pose_pass_mean)} | "
            f"{f4(row.risk_mean)} | {pct(row.risk_gt_0_5_mean)} | {f4(row.qed_mean)} | {pct(row.intramol_pass_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This supports the paper's formal claim only for target-exchangeable calibration/test splits within a generator/domain. Cross-generator and BindingMOAD transfer remain empirical external-validity checks.",
        ]
    )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/target_conditional_crc.csv")
    ap.add_argument("--out-csv", default="results/target_heldout_selective_guarantee.csv")
    ap.add_argument("--out-summary", default="results/target_heldout_selective_guarantee_summary.csv")
    ap.add_argument("--out-md", default="experiments/TARGET_HELDOUT_SELECTIVE_GUARANTEE.md")
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    table = build(raw)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    summary = summarize(table)
    summary.to_csv(args.out_summary, index=False)
    write_md(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
