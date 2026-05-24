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


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(float(actual) - expected) > tol:
        raise AssertionError(f"{name}: {actual} != {expected}")


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
    sync = _load("syncguide_t1000_n16_dockfast_selection_summary.csv")
    sync_vina = _load("vina_score_syncguide_t1000_n16_top1_summary.csv")
    sync_gnina = _load("gnina_score_syncguide_t1000_n16_top1_summary.csv")

    official = target[target["metric"].astype(str).str.contains("dock", case=False, na=False)].head(1)
    if official.empty:
        raise AssertionError("official dock-fast row not found")
    official_row = official.iloc[0]
    _assert_close("official_delta", official_row["delta_method_minus_baseline"], 0.169)
    _assert_close("official_baseline_mean", official_row["baseline_mean"], 0.798)
    _assert_close("official_method_mean", official_row["method_mean"], 0.967)

    pocketflow = sota[(sota["block"] == "pocketflow_crossdock100") & (sota["metric"] == "dock_fast")]
    if pocketflow.empty:
        raise AssertionError("PocketFlow dock-fast row not found")
    _assert_close("pocketflow_delta", pocketflow.iloc[0]["delta_method_minus_baseline"], 0.1125)

    bindingmoad = sota[(sota["block"] == "bindingmoad_v100_holdout") & (sota["metric"] == "dock_fast")]
    if bindingmoad.empty:
        raise AssertionError("BindingMOAD v100 dock-fast row not found")
    _assert_close("bindingmoad_delta", bindingmoad.iloc[0]["delta_method_minus_baseline"], 0.0975)

    syncguide = sota[(sota["block"] == "syncguide_t1000_n16") & (sota["metric"] == "dock_fast")]
    if syncguide.empty:
        raise AssertionError("SYNC-Guide dock-fast row not found")
    _assert_close("syncguide_delta", syncguide.iloc[0]["delta_method_minus_baseline"], 0.03)

    sync_pb = sync[sync["policy"] == "pb_rc_select"]
    if sync_pb.empty:
        raise AssertionError("SYNC-Guide PB-RC summary row not found")
    _assert_close("syncguide_pb_rc_high_risk", sync_pb.iloc[0]["risk_gt_0_5"], 0.04145077720207254)

    sync_vina_pb = sync_vina[sync_vina["policy"] == "pb_rc_select"]
    if sync_vina_pb.empty:
        raise AssertionError("SYNC-Guide Vina PB-RC row not found")
    _assert_close("syncguide_vina_pb_rc_dockfast", sync_vina_pb.iloc[0]["dock_pose_pass"], 1.0)

    sync_gnina_pb = sync_gnina[sync_gnina["policy"] == "pb_rc_select"]
    if sync_gnina_pb.empty:
        raise AssertionError("SYNC-Guide GNINA PB-RC row not found")
    _assert_close("syncguide_gnina_pb_rc_dockfast", sync_gnina_pb.iloc[0]["dock_pose_pass"], 1.0)

    diffsbdd_crc = selective[
        (selective["generator"] == "DiffSBDD") & (selective["method"] == "tc_crc_stratified")
    ]
    if diffsbdd_crc.empty:
        raise AssertionError("DiffSBDD target-heldout CRC row not found")
    _assert_close("diffsbdd_crc_violation", diffsbdd_crc.iloc[0]["heldout_violation_rate"], 0.0)

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
        "syncguide_summary_rows": int(len(sync)),
        "syncguide_vina_rows": int(len(sync_vina)),
        "syncguide_gnina_rows": int(len(sync_gnina)),
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
