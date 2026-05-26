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
    ltr = _load("learning_to_rank_selection_summary.csv")
    bm_molpilot = _load("bindingmoad_molpilot_v50_dockfast_selection_summary.csv")
    known_ligand = _load("known_ligand_similarity_enrichment_summary.csv")
    p0_sota = _load("p0_sota_generator_unified_summary.csv")
    p0_calibration = _load("generator_shift_adaptive_calibration_p0_summary.csv")
    prospective_cases = _load("p0_prospective3_case_targets.csv")
    prospective_aizynth = _load("aizynthfinder_prospective20_top1_summary.csv")
    joint_shift = _load("joint_shift_generator_protein_scaffold_statistics_p1.csv")
    contact_p1 = _load("contact_counterfactual_faithfulness_p1_aggregate_summary.csv")
    kinase_p1 = _load("kinase_selectivity_aware_expanded_summary.csv")

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

    molcraft = p0_sota[(p0_sota["generator"] == "MolCRAFT") & (p0_sota["policy"] == "PB-RC")]
    if molcraft.empty:
        raise AssertionError("MolCRAFT PB-RC P0 row not found")
    _assert_close("molcraft_pb_rc_dockfast", molcraft.iloc[0]["dock_fast"], 0.9975)
    _assert_close("molcraft_pb_rc_high_risk", molcraft.iloc[0]["risk_gt_0_5"], 0.0)

    molpilot_frame = p0_sota[
        (p0_sota["generator"] == "MolPilot-framefix") & (p0_sota["policy"] == "PB-RC")
    ]
    if molpilot_frame.empty:
        raise AssertionError("MolPilot-framefix PB-RC P0 row not found")
    _assert_close("molpilot_framefix_pb_rc_dockfast", molpilot_frame.iloc[0]["dock_fast"], 0.035)
    _assert_close("molpilot_framefix_pb_rc_high_risk", molpilot_frame.iloc[0]["risk_gt_0_5"], 0.985)

    molpilot_cal = p0_calibration[
        (p0_calibration["heldout_source"] == "MolPilot-framefix")
        & (p0_calibration["method"] == "raw_risk")
        & (p0_calibration["calib_targets"] == 0)
    ]
    if molpilot_cal.empty:
        raise AssertionError("MolPilot-framefix raw calibration row not found")
    _assert_close("molpilot_framefix_raw_ece", molpilot_cal.iloc[0]["ece_mean"], 0.017802328462855233)

    pros_rc = prospective_aizynth[prospective_aizynth["policy"] == "pb_rc_select"]
    if pros_rc.empty:
        raise AssertionError("Prospective20 PB-RC AiZynth row not found")
    _assert_close("prospective20_pb_rc_aizynth_solved", pros_rc.iloc[0]["solved_rate"], 0.35)
    if len(prospective_cases) != 6:
        raise AssertionError(f"expected 6 prospective case rows, found {len(prospective_cases)}")

    pocketflow_joint = joint_shift[
        (joint_shift["axis"] == "family_and_generated_scaffold_unseen")
        & (joint_shift["source"] == "PocketFlow")
        & (joint_shift["metric"] == "dock_pose_pass")
    ]
    if pocketflow_joint.empty:
        raise AssertionError("PocketFlow joint-shift dock-fast row not found")
    _assert_close("pocketflow_joint_shift_dockfast_delta", pocketflow_joint.iloc[0]["delta_method_minus_baseline"], 0.099537037037037)

    molcraft_joint = joint_shift[
        (joint_shift["axis"] == "family_and_generated_scaffold_unseen")
        & (joint_shift["source"] == "MolCRAFT-100")
        & (joint_shift["metric"] == "risk_gt_0_5")
    ]
    if molcraft_joint.empty:
        raise AssertionError("MolCRAFT joint-shift risk-tail row not found")
    _assert_close("molcraft_joint_shift_highrisk_delta", molcraft_joint.iloc[0]["delta_method_minus_baseline"], -0.0722222222222222)

    if len(contact_p1) < 4:
        raise AssertionError("P1 aggregate contact faithfulness summary is incomplete")
    diffsbdd_contact = contact_p1[contact_p1["source"] == "DiffSBDD-fullpool"]
    if diffsbdd_contact.empty:
        raise AssertionError("DiffSBDD aggregate contact faithfulness row not found")
    _assert_close(
        "diffsbdd_contact_mask_delta",
        diffsbdd_contact.iloc[0]["contact_masked_risk_delta"],
        -0.0811936604767127,
        tol=1e-6,
    )

    selectivity_rc = kinase_p1[kinase_p1["policy"] == "selectivity_rc"]
    if selectivity_rc.empty:
        raise AssertionError("expanded kinase selectivity RC row not found")
    _assert_close("expanded_kinase_targets", selectivity_rc.iloc[0]["n"], 12)
    _assert_close("expanded_kinase_target_best", selectivity_rc.iloc[0]["target_is_best"], 1 / 3)

    diffsbdd_crc = selective[
        (selective["generator"] == "DiffSBDD") & (selective["method"] == "tc_crc_stratified")
    ]
    if diffsbdd_crc.empty:
        raise AssertionError("DiffSBDD target-heldout CRC row not found")
    _assert_close("diffsbdd_crc_violation", diffsbdd_crc.iloc[0]["heldout_violation_rate"], 0.0)

    ltr_row = ltr[
        (ltr["setting"] == "DiffSBDD_to_Pocket2Mol")
        & (ltr["policy"] == "constrained_xgbrank")
        & (ltr["mode"] == "selective_alpha10")
    ]
    if ltr_row.empty:
        raise AssertionError("learning-to-rank constrained XGB transfer row not found")
    _assert_close("ltr_constrained_xgb_coverage", ltr_row.iloc[0]["coverage_mean"], 0.769811320754717)

    bm_molpilot_pb = bm_molpilot[bm_molpilot["policy"] == "pb_rc_select"]
    if bm_molpilot_pb.empty:
        raise AssertionError("BindingMOAD MolPilot PB-RC row not found")
    _assert_close("bindingmoad_molpilot_pb_rc_dockfast", bm_molpilot_pb.iloc[0]["dock_pose_pass"], 0.035)
    _assert_close("bindingmoad_molpilot_pb_rc_high_risk", bm_molpilot_pb.iloc[0]["risk_gt_0_5"], 1.0)

    known_row = known_ligand[
        (known_ligand["pool"] == "DiffSBDD_official") & (known_ligand["policy"] == "pb_rc_select")
    ]
    if known_row.empty:
        raise AssertionError("known-ligand DiffSBDD PB-RC row not found")
    _assert_close("known_ligand_diffsbdd_pb_rc_max_tanimoto", known_row.iloc[0]["max_tanimoto_mean"], 0.1746822592840945)

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
        "learning_to_rank_rows": int(len(ltr)),
        "bindingmoad_molpilot_rows": int(len(bm_molpilot)),
        "known_ligand_rows": int(len(known_ligand)),
        "p0_sota_rows": int(len(p0_sota)),
        "p0_calibration_rows": int(len(p0_calibration)),
        "p0_prospective_case_rows": int(len(prospective_cases)),
        "p1_joint_shift_rows": int(len(joint_shift)),
        "p1_contact_sources": int(len(contact_p1)),
        "p1_kinase_rows": int(len(kinase_p1)),
    }

    out = ROOT / "logs" / "snapshot_smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
