"""Reproduce target-paired selection statistics from frozen numerical pools."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from reproduce import select


ROOT = Path(__file__).resolve().parent
REPLICATES = 10000
SEED = 20261105


def choose(group, policy, k):
    if policy in ("PB-QED", "PB-RC"):
        return select(group, policy, k)
    pool = group[group.intramol_pass].copy()
    if policy == "PB-weighted":
        return pool.assign(score=pool.qed - pool.risk_prob).sort_values(
            ["score", "qed"], ascending=[False, False]).head(k)
    if policy == "Single-distance":
        return pool.sort_values(["lp_min", "qed"], ascending=[False, False]).head(k)
    if policy == "Full filter":
        return pool[pool.dock_pose_pass].sort_values(
            ["qed", "risk_prob"], ascending=[False, True]).head(k)
    raise ValueError(policy)


def paired(method, reference, generator, metric, comparator):
    pair = pd.concat([method[metric].rename("a"), reference[metric].rename("b")], axis=1).dropna()
    difference = (pair.a - pair.b).to_numpy()
    rng = np.random.default_rng(SEED)
    draws = difference[rng.integers(0, len(difference),
                                   size=(REPLICATES, len(difference)))].mean(axis=1)
    nonzero = difference[np.abs(difference) > 1e-12]
    ranks = rankdata(np.abs(nonzero))
    effect = ((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
              if len(nonzero) else 0.0)
    return {"setting": generator, "metric": metric, "method": "PB-RC",
            "comparator": comparator, "targets": len(pair),
            "baseline_mean": float(pair.b.mean()), "method_mean": float(pair.a.mean()),
            "delta": float(difference.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "wilcoxon_p": float(wilcoxon(difference).pvalue) if len(nonzero) else 1.0,
            "paired_rank_biserial": float(effect)}


def fdr(frame):
    p = frame.wilcoxon_p.to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return frame.assign(fdr_q=np.minimum(adjusted, 1.0))


def main():
    primary, controls = [], []
    for generator, k in [("DiffSBDD", 10), ("Pocket2Mol", 4)]:
        pool = pd.read_csv(ROOT / "data" / (generator + ".csv"))
        aggregates = {}
        for policy in ("PB-QED", "PB-RC", "PB-weighted", "Single-distance", "Full filter"):
            chosen = pd.concat([choose(group, policy, k)
                                for _, group in pool.groupby("key", sort=True)])
            aggregates[policy] = chosen.groupby("key").agg(
                dock_fast=("dock_pose_pass", "mean"), qed=("qed", "mean"))
        for metric in ("dock_fast", "qed"):
            primary.append(paired(aggregates["PB-RC"], aggregates["PB-QED"],
                                  generator, metric, "PB-QED"))
        for comparator in ("PB-weighted", "Single-distance", "Full filter"):
            for metric in ("dock_fast", "qed"):
                controls.append(paired(aggregates["PB-RC"], aggregates[comparator],
                                       generator, metric, comparator))
    primary = fdr(pd.DataFrame(primary))
    controls = fdr(pd.DataFrame(controls))
    result = {"bootstrap_replicates": REPLICATES, "seed": SEED,
              "fdr_families": {"primary": 4, "controls": 12},
              "primary": primary.to_dict("records"),
              "controls": controls.to_dict("records")}
    (ROOT / "STATISTICS_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(primary.to_string(index=False))
    print("PASS: target-paired bootstrap, Wilcoxon, rank-biserial effect, and separate-family FDR.")


if __name__ == "__main__":
    main()
