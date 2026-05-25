from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


RNG = np.random.default_rng(20260525)
FILES = [
    "results/gnina_redock_main100_top4_pose_quality.csv",
    "results/gnina_redock_bindingmoad_v100_top4_pose_quality.csv",
    "results/gnina_redock_p0_expansion_pose_quality.csv",
]
METRICS = {
    "after_redock_dock_fast": "higher",
    "gnina_affinity": "lower",
    "gnina_cnnscore": "higher",
    "gnina_cnnaffinity": "higher",
    "redock_rmsd": "lower",
}


def bootstrap(delta: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    if len(delta) == 1:
        return float(delta[0]), float(delta[0])
    index = RNG.integers(0, len(delta), size=(n_boot, len(delta)))
    draws = delta[index].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bh_fdr(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    q = np.empty(len(p))
    running = 1.0
    for reversed_index, idx in enumerate(order[::-1], start=1):
        rank = len(p) - reversed_index + 1
        running = min(running, p[idx] * len(p) / rank)
        q[idx] = running
    return np.clip(q, 0, 1)


def main() -> None:
    frames = [pd.read_csv(path, low_memory=False) for path in FILES if Path(path).exists()]
    if not frames:
        raise FileNotFoundError("No confirmatory redocking pose-quality input found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for source, source_data in data.groupby("source", sort=True):
        selected = source_data[source_data["policy"].isin(["pb_qed", "pb_rc_select"])].copy()
        for metric, direction in METRICS.items():
            target_means = selected.groupby(["target_id", "policy"], as_index=False)[metric].mean()
            wide = target_means.pivot(index="target_id", columns="policy", values=metric).dropna()
            if not {"pb_qed", "pb_rc_select"}.issubset(wide.columns) or len(wide) < 2:
                continue
            baseline = wide["pb_qed"].to_numpy(float)
            method = wide["pb_rc_select"].to_numpy(float)
            delta = method - baseline
            ci_low, ci_high = bootstrap(delta)
            p = 1.0 if np.allclose(delta, 0) else float(wilcoxon(method, baseline, zero_method="wilcox").pvalue)
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "direction": direction,
                    "targets": len(wide),
                    "pb_qed": baseline.mean(),
                    "pb_rc_select": method.mean(),
                    "delta": delta.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": p,
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_q_confirmatory_family"] = bh_fdr(out["wilcoxon_p"].tolist())
    out.to_csv("results/independent_redock_confirmation_statistics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
