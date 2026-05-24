"""Verify that lightweight paper source-data snapshots are readable.

This script is intentionally small. It lets reviewers check the headline
numbers without downloading large molecular structures or generated SDF files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_source_data"


def _load(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    target = _load("target_level_statistical_tests.csv")
    sota = _load("final_sota_target_level_statistics.csv")
    multi = _load("multiobjective_selection_summary.csv")
    fusion = _load("fusion_strong_baselines_metrics_agg.csv")
    calibration = _load("risk_to_dockfast_calibration_summary.csv")
    selective = _load("target_heldout_selective_guarantee_summary.csv")
    runtime = _load("runtime_memory_throughput.csv")

    official = target[target["metric"].astype(str).str.contains("dock", case=False, na=False)].head(1)
    best_multi = multi.sort_values("dock_pose_pass", ascending=False).head(5)
    best_fusion = fusion.sort_values("brier").groupby(["direction", "scenario"], as_index=False).first()
    full = best_fusion[best_fusion["scenario"] == "full"][["direction", "brier", "ece"]].rename(
        columns={"brier": "full_brier", "ece": "full_ece"}
    )
    contrib = best_fusion.merge(full, on="direction")
    contrib = contrib[contrib["scenario"] != "full"].copy()
    contrib["delta_brier"] = contrib["brier"] - contrib["full_brier"]
    modality = (
        contrib.groupby("scenario", as_index=False)
        .agg(delta_brier_mean=("delta_brier", "mean"), delta_brier_max=("delta_brier", "max"))
        .sort_values("delta_brier_mean", ascending=False)
    )

    summary = {
        "source_data_dir": str(DATA),
        "n_files": len(list(DATA.glob("*.csv"))),
        "official_dockfast_row": official.to_dict(orient="records"),
        "sota_rows": int(len(sota)),
        "multiobjective_top5": best_multi[["source", "policy", "dock_pose_pass", "risk_mean", "qed_mean"]].to_dict(
            orient="records"
        ),
        "modality_delta_brier": modality.to_dict(orient="records"),
        "calibration_rows": int(len(calibration)),
        "selective_rows": int(len(selective)),
        "runtime_rows": int(len(runtime)),
    }

    out = ROOT / "logs" / "snapshot_smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

