"""Build one deduplicated statistical registry with a global BH-FDR column."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bh_fdr(p_values: pd.Series) -> np.ndarray:
    values = p_values.fillna(1.0).astype(float).to_numpy()
    order = np.argsort(values)
    q_values = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_index, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_index + 1
        running = min(running, values[index] * len(values) / rank)
        q_values[index] = running
    return np.clip(q_values, 0.0, 1.0)


def family_for(block: str, source: str) -> str:
    if source == "sota_external":
        return "generator_external"
    if block == "main_fullatom_cond":
        return "controlled_primary"
    if block.startswith("prospective"):
        return "prospective_confirmation"
    return "ood_secondary"


def normalized_key(frame: pd.DataFrame) -> pd.Series:
    block = frame["block"].astype(str).str.replace(r"^sota_", "", regex=True)
    return pd.Series(
        list(zip(block, frame["metric"], frame["baseline"], frame["method"])),
        index=frame.index,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-stats", required=True)
    ap.add_argument("--sota-stats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target = pd.read_csv(args.target_stats)
    target["registry_source"] = "target_stats"
    sota = pd.read_csv(args.sota_stats)
    sota["registry_source"] = "sota_external"
    if "role" not in target:
        target["role"] = ""

    input_rows = len(target) + len(sota)
    external_keys = set(normalized_key(sota))
    target = target[~normalized_key(target).isin(external_keys)].copy()
    registry = pd.concat([target, sota], ignore_index=True, sort=False)
    registry["hypothesis_family"] = [
        family_for(block, source) for block, source in zip(registry["block"], registry["registry_source"])
    ]
    registry["fdr_q_global_registry"] = bh_fdr(registry["wilcoxon_p"])
    registry = registry.sort_values(
        ["hypothesis_family", "block", "metric", "baseline", "method"]
    ).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out, index=False)
    print(
        {
            "rows": len(registry),
            "duplicates_removed": input_rows - len(registry),
            "sync_rows": int(registry["block"].astype(str).str.contains("sync", case=False).sum()),
            "out": str(out),
        }
    )


if __name__ == "__main__":
    main()
