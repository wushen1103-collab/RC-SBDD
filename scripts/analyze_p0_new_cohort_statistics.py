from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


COHORTS = [
    ("molcraft_crossdock_t100", "results/molcraft_crossdock_t100_n16_dockfast_selection.csv"),
    ("bindingmoad_task1_strict35_sensitivity", "results/bindingmoad_task1_strict35_pocketflow_n16_dockfast_selection.csv"),
]
METRICS = {
    "dock_fast": ("dock_pose_pass", "higher"),
    "protein_pass": ("protein_pass", "higher"),
    "mol_fast": ("intramol_pass", "higher"),
    "risk_prob": ("risk_prob", "lower"),
    "risk_gt_0_5": ("risk_prob", "lower"),
    "qed": ("qed", "higher"),
}
BASELINE = "pb_qed"
METHOD = "pb_rc_select"
RNG = np.random.default_rng(20260525)


def as_numeric(group, metric):
    column, _ = METRICS[metric]
    if metric == "risk_gt_0_5":
        return (pd.to_numeric(group[column], errors="coerce") > 0.5).astype(float)
    if column in {"dock_pose_pass", "protein_pass", "intramol_pass"}:
        return group[column].fillna(False).astype(bool).astype(float)
    return pd.to_numeric(group[column], errors="coerce")


def per_target(frame, metric):
    rows = []
    for (target, policy), group in frame[frame["policy"].isin([BASELINE, METHOD])].groupby(["key", "policy"]):
        rows.append({"target": target, "policy": policy, "value": float(as_numeric(group, metric).mean())})
    wide = pd.DataFrame(rows).pivot(index="target", columns="policy", values="value")
    return wide[[BASELINE, METHOD]].dropna()


def bootstrap_ci(delta, n_boot=10000):
    indices = RNG.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def cliffs_delta(method, baseline):
    return float((np.sign(method[:, None] - baseline[None, :])).mean())


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    q = np.empty(len(p))
    running = 1.0
    for reverse_index, index in enumerate(order[::-1], 1):
        rank = len(p) - reverse_index + 1
        running = min(running, p[index] * len(p) / rank)
        q[index] = running
    return np.clip(q, 0.0, 1.0)


def main():
    rows = []
    for block, path in COHORTS:
        if not Path(path).exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        for metric, (_, direction) in METRICS.items():
            paired = per_target(frame, metric)
            baseline = paired[BASELINE].to_numpy(float)
            method = paired[METHOD].to_numpy(float)
            delta = method - baseline
            low, high = bootstrap_ci(delta)
            p_value = 1.0 if np.allclose(delta, 0.0) else float(wilcoxon(method, baseline).pvalue)
            rows.append(
                {
                    "block": block,
                    "metric": metric,
                    "baseline": BASELINE,
                    "method": METHOD,
                    "targets": len(paired),
                    "baseline_mean": baseline.mean(),
                    "method_mean": method.mean(),
                    "delta_method_minus_baseline": delta.mean(),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_p": p_value,
                    "cliffs_delta": cliffs_delta(method, baseline),
                    "desired_direction": direction,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_p0_family"] = bh_fdr(result["wilcoxon_p"])
    result.to_csv("results/p0_new_cohort_target_level_statistics.csv", index=False)
    lines = [
        "# P0 New-Cohort Target-Level Statistics",
        "",
        "- Comparisons use PB-RC versus PB-QED on the same targets.",
        "- MolCRAFT-T100 is an expanded direct-output positive cohort.",
        "- BindingMOAD strict35 is a declared ligand-matching threshold sensitivity cohort (minimum similarity 0.03), not a replacement for the strict33 primary row.",
        "- Wilcoxon tests are corrected within this newly declared P0 addendum family with Benjamini-Hochberg FDR.",
        "",
        "| Block | Metric | Targets | PB-QED | PB-RC | Delta | 95% CI | p | FDR-q | Cliff |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.block} | {row.metric} | {row.targets} | {row.baseline_mean:.4f} | {row.method_mean:.4f} | "
            f"{row.delta_method_minus_baseline:.4f} | [{row.bootstrap_ci_low:.4f}, {row.bootstrap_ci_high:.4f}] | "
            f"{row.wilcoxon_p:.4g} | {row.fdr_q_p0_family:.4g} | {row.cliffs_delta:.4f} |"
        )
    Path("experiments/P0_NEW_COHORT_TARGET_LEVEL_STATISTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
