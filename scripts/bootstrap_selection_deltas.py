import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COMPARISONS = [
    ("fullatom_cond", "rc_select", "qed", "RC-Select vs QED"),
    ("fullatom_cond", "qed_minus_risk", "qed", "QED-Risk vs QED"),
    ("fullatom_cond", "pb_rc_select", "pb_qed", "PB+RC vs PB+QED"),
    ("fullatom_cond", "pb_qed_minus_risk", "pb_qed", "PB+QED-Risk vs PB+QED"),
    ("fullatom_joint", "rc_select", "qed", "RC-Select vs QED"),
    ("fullatom_joint", "pb_rc_select", "pb_qed", "PB+RC vs PB+QED"),
]

METRICS = ["dock_pose_pass", "protein_pass", "risk_prob", "qed"]


def load_rows():
    base = pd.read_csv("results/posebusters_dockfast_selection.csv")
    pb = pd.read_csv("results/posebusters_dockfast_pb_selection.csv")
    return pd.concat([base, pb], ignore_index=True)


def per_target(df):
    rows = []
    for (set_name, policy, key), group in df.groupby(["set", "policy", "key"], sort=True):
        rows.append(
            {
                "set": set_name,
                "policy": policy,
                "key": key,
                "dock_pose_pass": float(group["dock_pose_pass"].fillna(False).astype(bool).mean()),
                "protein_pass": float(group["protein_pass"].fillna(False).astype(bool).mean()),
                "risk_prob": float(group["risk_prob"].mean()),
                "qed": float(group["qed"].mean()),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_delta(a, b, rng, n_boot):
    delta = a - b
    observed = float(delta.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(delta)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(delta[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_pos = float((boots > 0).mean())
    return observed, float(lo), float(hi), p_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-csv", default="results/selection_bootstrap_deltas.csv")
    ap.add_argument("--out-md", default="experiments/SELECTION_BOOTSTRAP_DELTAS.md")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    target = per_target(load_rows())
    rows = []
    for set_name, policy, baseline, label in COMPARISONS:
        left = target[(target["set"] == set_name) & (target["policy"] == policy)]
        right = target[(target["set"] == set_name) & (target["policy"] == baseline)]
        merged = left.merge(right, on="key", suffixes=("_policy", "_baseline"))
        for metric in METRICS:
            obs, lo, hi, p_pos = bootstrap_delta(
                merged[f"{metric}_policy"].to_numpy(dtype=float),
                merged[f"{metric}_baseline"].to_numpy(dtype=float),
                rng,
                args.n_boot,
            )
            rows.append(
                {
                    "set": set_name,
                    "comparison": label,
                    "policy": policy,
                    "baseline": baseline,
                    "metric": metric,
                    "targets": int(len(merged)),
                    "delta_mean": obs,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "bootstrap_p_delta_gt_0": p_pos,
                }
            )
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    lines = [
        "# Selection Bootstrap Deltas",
        "",
        f"Bootstrap resamples targets with replacement; n_boot={args.n_boot}; seed={args.seed}.",
        "",
        "| Set | Comparison | Metric | Targets | Delta mean | 95% CI | P(delta>0) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = row["metric"]
        if metric in {"dock_pose_pass", "protein_pass"}:
            delta = 100 * row["delta_mean"]
            lo = 100 * row["ci95_low"]
            hi = 100 * row["ci95_high"]
            fmt = f"{delta:+.1f} pp"
            ci = f"[{lo:+.1f}, {hi:+.1f}]"
        else:
            fmt = f"{row['delta_mean']:+.4f}"
            ci = f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
        lines.append(
            f"| {row['set']} | {row['comparison']} | {metric} | {row['targets']} | "
            f"{fmt} | {ci} | {row['bootstrap_p_delta_gt_0']:.3f} |"
        )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
