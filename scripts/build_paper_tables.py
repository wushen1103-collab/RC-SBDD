from pathlib import Path

import json
import pandas as pd


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.1f}%"


def f4(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


def risk_proxy_table():
    if Path("results/risk_model_ablation.csv").exists():
        df = pd.read_csv("results/risk_model_ablation.csv")
        overall = df[df["mode"] == "overall"].copy()
        order = ["heuristic", "logreg", "hgb", "rf", "mlp"]
        rows = []
        for scorer in order:
            sub = overall[overall["classifier"] == scorer]
            rows.append(
                {
                    "Risk scorer": scorer,
                    "Seeds": sub["seed"].nunique(),
                    "AUROC": f"{sub['failure_auroc'].mean():.4f} +/- {sub['failure_auroc'].std(ddof=1):.4f}",
                    "Brier": f"{sub['brier'].mean():.4f} +/- {sub['brier'].std(ddof=1):.4f}",
                    "ECE": f"{sub['ece'].mean():.4f} +/- {sub['ece'].std(ddof=1):.4f}",
                }
            )
        return pd.DataFrame(rows)

    df = pd.read_csv("results/risk_proxy_seed_summary.csv")
    rows = []
    for scorer in ["model", "heuristic"]:
        sub = df[(df["scorer"] == scorer) & (df["mode"] == "overall")]
        rows.append(
            {
                "Scorer": scorer,
                "Seeds": sub["seed"].nunique(),
                "AUROC": f"{sub['failure_auroc'].mean():.4f} +/- {sub['failure_auroc'].std(ddof=0):.4f}",
                "Brier": f"{sub['brier'].mean():.4f} +/- {sub['brier'].std(ddof=0):.4f}",
                "ECE": f"{sub['ece'].mean():.4f} +/- {sub['ece'].std(ddof=0):.4f}",
            }
        )
    return pd.DataFrame(rows)


def risk_scorer_selection_table():
    df = pd.read_csv("results/risk_scorer_selection_ablation/selection_summary.csv")
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_rc_select"]
    df = df[df["policy"].isin(keep)].copy()
    order = {p: i for i, p in enumerate(keep)}
    df["order"] = df["policy"].map(order)
    rows = []
    for row in df.sort_values(["order", "scorer"]).itertuples(index=False):
        rows.append(
            {
                "Scorer": row.scorer,
                "Policy": row.policy,
                "N": row.selected,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Above native p95": pct(row.above_native_p95),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.molfast_core_pass),
                "Overlap vs logreg": pct(row.overlap_with_logreg_same_policy),
            }
        )
    return pd.DataFrame(rows)


def failure_signal_table():
    df = pd.read_csv("results/failure_taxonomy/selection_failure_signals.csv")
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_rc_select"]
    df = df[df["policy"].isin(keep)].copy()
    order = {p: i for i, p in enumerate(keep)}
    df["order"] = df["policy"].map(order)
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.selected,
                "High risk": pct(row.risk_high),
                "Above native p95": pct(row.above_native_p95),
                "High-risk steric": pct(row.high_risk_steric_or_too_close),
                "High-risk center": pct(row.high_risk_center_shift),
                "High-risk weak contact": pct(row.high_risk_weak_contact),
                "High-risk multi-signal": pct(row.high_risk_multi_signal),
                "Mean QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def rejection_budget_table():
    df = pd.read_csv("results/rejection_budget.csv")
    keep = df[
        (df["threshold"].isin(["native_p90", "native_p95", "fixed_0_5"]))
        & (df["pool"].isin(["all", "mol_fast"]))
    ].copy()
    order_threshold = {"native_p90": 0, "native_p95": 1, "fixed_0_5": 2}
    order_pool = {"all": 0, "mol_fast": 1}
    keep["order_threshold"] = keep["threshold"].map(order_threshold)
    keep["order_pool"] = keep["pool"].map(order_pool)
    rows = []
    for row in keep.sort_values(["order_pool", "order_threshold"]).itertuples(index=False):
        rows.append(
            {
                "Pool": row.pool,
                "Threshold": row.threshold,
                "Risk cutoff": f4(row.risk_cutoff),
                "Accept rate": pct(row.accept_rate),
                "Targets >=K": pct(row.targets_ge_k),
                "Selected N": row.selected,
                "Mean risk": f4(row.selected_risk_mean),
                "Risk >0.5": pct(row.selected_risk_gt_0_5),
                "Mean QED": f4(row.selected_qed_mean),
                "mol_fast pass": pct(row.selected_molfast_pass),
            }
        )
    return pd.DataFrame(rows)


def generated_benchmark_table():
    risk = pd.read_csv("results/diffsbdd_zenodo_risk_comparison.csv")
    mol = pd.read_csv("results/posebusters_molfast_summary.csv")
    df = risk.merge(mol[["set", "all_core_pass"]], on="set", how="left")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Generated set": row.set,
                "Targets": row.matched_targets,
                "Mols": row.records,
                "Mean QED": f4(row.qed_mean),
                "Mean risk": f4(row.risk_mean),
                "Median risk": f4(row.risk_median),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "mol_fast pass": pct(row.all_core_pass),
            }
        )
    return pd.DataFrame(rows)


def official_selection_table():
    base = pd.read_csv("results/posebusters_dockfast_selection_summary.csv")
    pb = pd.read_csv("results/posebusters_dockfast_pb_selection_summary.csv")
    df = pd.concat([base, pb], ignore_index=True)
    base_raw = pd.read_csv("results/posebusters_dockfast_selection.csv")
    pb_raw = pd.read_csv("results/posebusters_dockfast_pb_selection.csv")
    raw = pd.concat([base_raw, pb_raw], ignore_index=True)
    risk_gt = raw.groupby(["set", "policy"])["risk_prob"].apply(lambda x: float((x > 0.5).mean())).to_dict()
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    df = df[(df["set"] == "fullatom_cond") & (df["policy"].isin(keep))].copy()
    order = {p: i for i, p in enumerate(keep)}
    df["order"] = df["policy"].map(order)
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "K": row.k,
                "N": row.n,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(risk_gt[(row.set, row.policy)]),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.intramol_pass),
                "Protein pass": pct(row.protein_pass),
                "Dock pose pass": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_table():
    df = pd.read_csv("results/selection_bootstrap_deltas.csv")
    keep = df[
        (df["set"] == "fullatom_cond")
        & (df["comparison"].isin(["RC-Select vs QED", "QED-Risk vs QED", "PB+RC vs PB+QED", "PB+QED-Risk vs PB+QED"]))
        & (df["metric"].isin(["dock_pose_pass", "risk_prob", "qed"]))
    ].copy()
    rows = []
    for row in keep.itertuples(index=False):
        if row.metric == "dock_pose_pass":
            delta = f"{100 * row.delta_mean:+.1f} pp"
            ci = f"[{100 * row.ci95_low:+.1f}, {100 * row.ci95_high:+.1f}]"
        else:
            delta = f"{row.delta_mean:+.4f}"
            ci = f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]"
        rows.append(
            {
                "Comparison": row.comparison,
                "Metric": row.metric,
                "Targets": row.targets,
                "Delta": delta,
                "95% CI": ci,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table():
    df = pd.read_csv("results/selection_sensitivity.csv")
    df = df[
        (df["set"] == "fullatom_cond")
        & (df["threshold"] == "p95")
        & (df["policy"].isin(["qed", "rc_select", "pb_qed", "pb_rc_select"]))
    ].copy()
    rows = []
    for row in df.sort_values(["k", "policy"]).itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "K": row.k,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.all_core_pass),
            }
        )
    return pd.DataFrame(rows)


def ca_boundary_table():
    df = pd.read_csv("results/posebusters_dockfast_ca_selection_summary.csv")
    keep = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    df = df[df["policy"].isin(keep)].copy()
    rows = []
    for row in df.sort_values(["set", "policy"]).itertuples(index=False):
        rows.append(
            {
                "Set": row.set,
                "Policy": row.policy,
                "N": row.n,
                "Mean risk": f4(row.risk_mean),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.intramol_pass),
                "Protein pass": pct(row.protein_pass),
                "Dock pose pass": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def local_t500_table():
    df = pd.read_csv("results/local_t500_dockfast_selection_summary.csv")
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    df = df[df["policy"].isin(keep)].copy()
    order = {p: i for i, p in enumerate(keep)}
    df["order"] = df["policy"].map(order)
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.molfast_core_pass),
                "Protein pass": pct(row.protein_pass),
                "Dock pose pass": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def online_rejection_table():
    df = pd.read_csv("results/online_rejection_control_fullatom_cond.csv")
    keep = [
        ("stream_first_k", False),
        ("risk_gate_native_p90", False),
        ("risk_gate_native_p95", False),
        ("risk_gate_native_p95", True),
        ("risk_gate_native_p99", False),
    ]
    rows = []
    for policy, require_molfast in keep:
        row = df[(df["policy"] == policy) & (df["require_molfast"] == require_molfast)].iloc[0]
        rows.append(
            {
                "Policy": row.policy,
                "Threshold": row.threshold,
                "mol_fast gate": bool(row.require_molfast),
                "Targets reach K": pct(row.targets_reach_k),
                "Mean seen": f4(row.mean_seen),
                "Mean accepts": f4(row.mean_accepts),
                "Reject until stop": pct(row.mean_rejection_rate_until_stop),
                "Selected N": row.selected_n,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.molfast_pass),
            }
        )
    return pd.DataFrame(rows)


def targetdiff_boundary_table():
    sel = pd.read_csv("results/targetdiff_t250_n64_selection_summary.csv")
    dock = pd.read_csv("results/targetdiff_t250_n64_dockfast_selection_summary.csv")
    rows = []
    for policy in ["all", "qed", "risk", "qed_minus_risk", "rc_select"]:
        srow = sel[sel["policy"] == policy].iloc[0]
        dsub = dock[dock["policy"] == policy]
        dock_pose = dsub["dock_pose_pass"].iloc[0] if len(dsub) else float("nan")
        protein = dsub["protein_pass"].iloc[0] if len(dsub) else float("nan")
        rows.append(
            {
                "Policy": policy,
                "N": int(srow.n),
                "Targets": int(srow.targets),
                "Mean risk": f4(srow.risk_mean),
                "Risk >0.5": pct(srow.risk_gt_0_5),
                "Mean QED": f4(srow.qed_mean),
                "mol_fast pass": pct(srow.all_core_pass),
                "Protein pass": pct(protein) if pd.notna(protein) else "NA",
                "Dock pose pass": pct(dock_pose) if pd.notna(dock_pose) else "NA",
            }
        )
    return pd.DataFrame(rows)


def targetdiff_online_table():
    df = pd.read_csv("results/online_rejection_control_targetdiff_t250_n64.csv")
    keep = [
        ("stream_first_k", False),
        ("risk_gate_native_p90", False),
        ("risk_gate_native_p95", False),
        ("risk_gate_native_p99", False),
        ("risk_gate_native_p95", True),
    ]
    rows = []
    for policy, require_molfast in keep:
        row = df[(df["policy"] == policy) & (df["require_molfast"] == require_molfast)].iloc[0]
        rows.append(
            {
                "Policy": row.policy,
                "Threshold": row.threshold,
                "mol_fast gate": bool(row.require_molfast),
                "Targets reach K": pct(row.targets_reach_k),
                "Mean accepts": f4(row.mean_accepts),
                "Selected N": row.selected_n,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.molfast_pass),
            }
        )
    return pd.DataFrame(rows)


def vina_score_table():
    df = pd.read_csv("results/vina_score_fullatom_cond_top1_summary.csv")
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    order = {p: i for i, p in enumerate(keep)}
    df = df[df["policy"].isin(keep)].copy()
    df["order"] = df["policy"].map(order)
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "Scored": row.scored,
                "Vina mean": f4(row.vina_mean),
                "Vina median": f4(row.vina_median),
                "Vina IQR": f"[{f4(row.vina_p25)}, {f4(row.vina_p75)}]",
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "dock_fast pass": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def full_pool_dockfast_table():
    df = pd.read_csv("results/dockfast_full_pool_fullatom_cond.csv")
    rows = [
        {
            "Pool": "fullatom_cond_all_candidates",
            "Targets": df["key"].nunique(),
            "Mols": len(df),
            "Mean risk": f4(df["risk_prob"].mean()),
            "Risk >0.5": pct((df["risk_prob"] > 0.5).mean()),
            "Mean QED": f4(df["qed"].mean()),
            "mol_fast pass": pct(df["intramol_pass"].mean()),
            "Protein pass": pct(df["protein_pass"].mean()),
            "Dock pose pass": pct(df["dock_pose_pass"].mean()),
        }
    ]
    return pd.DataFrame(rows)


def conformal_risk_control_table():
    df = pd.read_csv("results/conformal_risk_control_fullatom_cond_agg.csv")
    keep = [
        ("qed_topk", float("nan")),
        ("pb_qed_topk", float("nan")),
        ("crc_molfast_risk", 0.05),
        ("crc_molfast_risk", 0.10),
        ("crc_molfast_risk", 0.20),
        ("crc_risk", 0.30),
    ]
    rows = []
    for method, alpha in keep:
        if pd.isna(alpha):
            row = df[(df["method"] == method) & (df["alpha"].isna())].iloc[0]
            alpha_text = "NA"
        else:
            row = df[(df["method"] == method) & (df["alpha"] == alpha)].iloc[0]
            alpha_text = f"{alpha:.2f}"
        rows.append(
            {
                "Method": method,
                "Alpha": alpha_text,
                "Seeds": int(row.seeds),
                "Feasible": pct(row.feasible_rate),
                "Coverage": pct(row.coverage_mean),
                "Reach K": pct(row.targets_reach_k_mean),
                "Test loss": f4(row.loss_mean),
                "dock_fast pass": pct(row.dock_pose_pass_mean),
                "Mean risk": f4(row.risk_mean_mean),
                "Risk >0.5": pct(row.risk_gt_0_5_mean),
                "Mean QED": f4(row.qed_mean_mean),
                "Tau": f4(row.tau_mean),
            }
        )
    return pd.DataFrame(rows)


def adaptive_generation_control_table():
    df = pd.read_csv("results/adaptive_generation_control_agg.csv")
    keep = [
        ("first_k", float("nan"), 100),
        ("qed_after_budget", float("nan"), 100),
        ("adaptive_crc_molfast", 0.05, 40),
        ("adaptive_crc_molfast", 0.05, 100),
        ("adaptive_crc_molfast", 0.10, 40),
        ("adaptive_crc_molfast", 0.20, 40),
    ]
    rows = []
    for policy, alpha, budget in keep:
        if pd.isna(alpha):
            row = df[(df["policy"] == policy) & (df["alpha"].isna()) & (df["max_budget"] == budget)].iloc[0]
            alpha_text = "NA"
        else:
            row = df[(df["policy"] == policy) & (df["alpha"] == alpha) & (df["max_budget"] == budget)].iloc[0]
            alpha_text = f"{alpha:.2f}"
        rows.append(
            {
                "Policy": policy,
                "Alpha": alpha_text,
                "Budget": int(row.max_budget),
                "Reach K": pct(row.targets_reach_k_mean),
                "Mean seen": f4(row.mean_seen_mean),
                "Mean accepts": f4(row.mean_accepts_mean),
                "dock_fast pass": pct(row.dock_pose_pass_mean),
                "Mean risk": f4(row.risk_mean_mean),
                "Risk >0.5": pct(row.risk_gt_0_5_mean),
                "Mean QED": f4(row.qed_mean_mean),
                "Tau": f4(row.tau_mean),
            }
        )
    return pd.DataFrame(rows)


def vina_redock_table():
    df = pd.read_csv("results/vina_redock_fullatom_cond_top1_summary.csv")
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    order = {p: i for i, p in enumerate(keep)}
    df = df[df["policy"].isin(keep)].copy()
    df["order"] = df["policy"].map(order)
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "Scored": row.scored,
                "Vina mean": f4(row.vina_mean),
                "Vina median": f4(row.vina_median),
                "Vina IQR": f"[{f4(row.vina_p25)}, {f4(row.vina_p75)}]",
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "dock_fast pass": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def rank_auc(labels, scores):
    y = pd.Series(labels).astype(int)
    s = pd.Series(scores).astype(float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bindingmoad_transfer_table():
    df = pd.read_csv("results/bindingmoad_risk_transfer.csv")
    auc = rank_auc(df["label_failure"], df["risk_prob"])
    rows = []
    for mode, group in df.groupby("mode", sort=True):
        rows.append(
            {
                "Mode": mode,
                "N": len(group),
                "Mean risk": f4(group["risk_prob"].mean()),
                "Median risk": f4(group["risk_prob"].median()),
                "Risk >0.5": pct((group["risk_prob"] > 0.5).mean()),
                "Native-vs-corruption AUROC": f4(auc) if mode == "native" else "",
            }
        )
    return pd.DataFrame(rows)


def pocket2mol_crossgen_table():
    sel = pd.read_csv("results/pocket2mol_crossgen_n16_ext_selection_summary.csv")
    dock = pd.read_csv("results/pocket2mol_crossgen_n16_ext_dockfast_selection_summary.csv")
    collect = pd.read_csv("results/pocket2mol_crossgen_n16_ext_collect_summary.csv")
    success = {
        "Policy": "generation_pool",
        "N": int(sel[sel["policy"] == "all"]["n"].iloc[0]),
        "Targets": int(sel[sel["policy"] == "all"]["targets"].iloc[0]),
        "Attempted targets": int(len(collect)),
        "Mean risk": f4(sel[sel["policy"] == "all"]["risk_mean"].iloc[0]),
        "Risk >0.5": pct(sel[sel["policy"] == "all"]["risk_gt_0_5"].iloc[0]),
        "Mean QED": f4(sel[sel["policy"] == "all"]["qed_mean"].iloc[0]),
        "mol_fast pass": pct(sel[sel["policy"] == "all"]["all_core_pass"].iloc[0]),
        "Protein pass": "NA",
        "Dock pose pass": "NA",
    }
    keep = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    order = {p: i for i, p in enumerate(keep)}
    dock = dock[dock["policy"].isin(keep)].copy()
    dock["order"] = dock["policy"].map(order)
    rows = [success]
    for row in dock.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Attempted targets": int(len(collect)),
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.molfast_core_pass),
                "Protein pass": pct(row.protein_pass),
                "Dock pose pass": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def synthesis_oracle_table():
    df = pd.read_csv("results/synthesis_oracle_selection_summary.csv")
    keep_exp = ["DiffSBDD_official", "Pocket2Mol_transfer"]
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    main_diffsbdd = (df["experiment"] == "DiffSBDD_official") & (df["set"] == "fullatom_cond")
    pocket2mol = df["experiment"] == "Pocket2Mol_transfer"
    df = df[(main_diffsbdd | pocket2mol) & df["policy"].isin(keep_policy)].copy()
    order_exp = {name: i for i, name in enumerate(keep_exp)}
    order_policy = {name: i for i, name in enumerate(keep_policy)}
    df["order_exp"] = df["experiment"].map(order_exp)
    df["order_policy"] = df["policy"].map(order_policy)
    rows = []
    for row in df.sort_values(["order_exp", "order_policy"]).itertuples(index=False):
        rows.append(
            {
                "Generator": row.experiment,
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "SA mean": f4(row.sa_mean),
                "SA <=6": pct(row.sa_pass_le_6),
                "RAscore mean": f4(row.rascore_mean),
                "RAscore >=0.5": pct(row.rascore_pass_ge_0_5),
                "PAINS-free": pct(row.pains_free),
                "Lipinski pass": pct(row.lipinski_pass_le_1),
                "Veber pass": pct(row.veber_pass),
                "Mean QED": f4(row.qed_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
            }
        )
    return pd.DataFrame(rows)


def functional_interaction_table():
    df = pd.read_csv("results/functional_interaction_selection_summary.csv")
    keep_exp = ["DiffSBDD_official", "Pocket2Mol_transfer"]
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    main_diffsbdd = (df["experiment"] == "DiffSBDD_official") & (df["set"] == "fullatom_cond")
    pocket2mol = df["experiment"] == "Pocket2Mol_transfer"
    df = df[(main_diffsbdd | pocket2mol) & df["policy"].isin(keep_policy)].copy()
    order_exp = {name: i for i, name in enumerate(keep_exp)}
    order_policy = {name: i for i, name in enumerate(keep_policy)}
    df["order_exp"] = df["experiment"].map(order_exp)
    df["order_policy"] = df["policy"].map(order_policy)
    rows = []
    for row in df.sort_values(["order_exp", "order_policy"]).itertuples(index=False):
        rows.append(
            {
                "Generator": row.experiment,
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "H-bonds": f4(row.hbond_mean),
                "Hydrophobic": f4(row.hydrophobic_mean),
                "Functional contacts": f4(row.functional_contacts_mean),
                "Native profile recovery": pct(row.native_profile_recovery),
                "Native contact ratio": f4(row.native_contact_ratio),
                "dock_fast pass": pct(row.dock_pose_pass),
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def cross_generator_crc_table():
    df = pd.read_csv("results/cross_generator_crc.csv")
    baseline = df["method"].isin(["qed_topk", "pb_qed_topk"])
    crc = (df["method"] == "crc_molfast_risk") & (~df["weighted"]) & (df["alpha"].isin([0.05, 0.10, 0.20]))
    df = df[baseline | crc].copy()
    method_order = {"qed_topk": 0, "pb_qed_topk": 1, "crc_molfast_risk": 2}
    df["method_order"] = df["method"].map(method_order)
    df["alpha_order"] = df["alpha"].fillna(-1)
    rows = []
    for row in df.sort_values(["calibration_generator", "test_generator", "method_order", "alpha_order"]).itertuples(index=False):
        rows.append(
            {
                "Calibration": row.calibration_generator,
                "Test": row.test_generator,
                "Method": row.method,
                "Alpha": "NA" if pd.isna(row.alpha) else f"{row.alpha:.2f}",
                "Feasible": bool(row.feasible),
                "Selected": row.selected_n,
                "Targets": row.targets,
                "Reach K": pct(row.reach_k),
                "Coverage": pct(row.coverage),
                "Loss": f4(row.loss),
                "dock_fast pass": pct(row.dock_pose_pass),
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def prospective_case_table():
    sel = pd.read_csv("results/prospective_pocket2mol_n128_selection_summary.csv")
    dock = pd.read_csv("results/prospective_pocket2mol_n128_dockfast_selection_summary.csv")
    vina = pd.read_csv("results/vina_redock_prospective_pocket2mol_n128_summary.csv")
    keep = ["qed", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    order = {p: i for i, p in enumerate(keep)}
    pool = sel[sel["policy"] == "all"].iloc[0]
    rows = [
        {
            "Policy": "generation_pool",
            "N": int(pool.n),
            "Targets": int(pool.targets),
            "Mean risk": f4(pool.risk_mean),
            "Risk >0.5": pct(pool.risk_gt_0_5),
            "Mean QED": f4(pool.qed_mean),
            "mol_fast pass": pct(pool.all_core_pass),
            "Protein pass": "NA",
            "Dock pose pass": "NA",
            "Vina median": "NA",
            "Vina failures": "NA",
        }
    ]
    df = dock[dock["policy"].isin(keep)].merge(vina, on="policy", how="left", suffixes=("", "_vina"))
    df["order"] = df["policy"].map(order)
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast pass": pct(row.molfast_core_pass),
                "Protein pass": pct(row.protein_pass),
                "Dock pose pass": pct(row.dock_pose_pass),
                "Vina median": f4(row.vina_median) if pd.notna(row.vina_median) else "NA",
                "Vina failures": int(row.failures) if pd.notna(row.failures) else "NA",
            }
        )
    return pd.DataFrame(rows)


def target_conditional_crc_table():
    df = pd.read_csv("results/target_conditional_crc_agg.csv")
    keep = ["qed_topk", "pb_qed_topk", "tc_crc_global", "tc_crc_stratified"]
    order = {p: i for i, p in enumerate(keep)}
    df = df[df["method"].isin(keep)].copy()
    df["order"] = df["method"].map(order)
    rows = []
    for row in df.sort_values(["generator", "order"]).itertuples(index=False):
        rows.append(
            {
                "Generator": row.generator,
                "Method": row.method,
                "Seeds": row.seeds,
                "Feasible": pct(row.feasible_rate),
                "Reach K": pct(row.reach_k_mean),
                "Loss": f4(row.loss_mean),
                "dock_fast pass": pct(row.dock_pose_pass_mean),
                "Mean risk": f4(row.risk_mean_mean),
                "Risk >0.5": pct(row.risk_gt_0_5_mean),
                "Mean QED": f4(row.qed_mean_mean),
                "mol_fast pass": pct(row.intramol_pass_mean),
            }
        )
    return pd.DataFrame(rows)


def route_synthesis_proxy_table():
    df = pd.read_csv("results/route_synthesis_proxy_summary.csv")
    keep_exp = ["DiffSBDD_official", "Pocket2Mol_transfer"]
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    main_diffsbdd = (df["experiment"] == "DiffSBDD_official") & (df["set"] == "fullatom_cond")
    pocket2mol = df["experiment"] == "Pocket2Mol_transfer"
    df = df[(main_diffsbdd | pocket2mol) & df["policy"].isin(keep_policy)].copy()
    order_e = {p: i for i, p in enumerate(keep_exp)}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df["order_e"] = df["experiment"].map(order_e)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_e", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Generator": row.experiment,
                "Policy": row.policy,
                "N": row.n,
                "Route success": pct(row.route_proxy_success),
                "Steps median": f4(row.estimated_steps_median),
                "Complexity": f4(row.route_complexity_mean),
                "Stock-like": pct(row.stock_like_fraction_mean),
                "BRICS frags": f4(row.brics_fragments_mean),
                "SA": f4(row.sa_mean),
                "RAscore": f4(row.rascore_mean),
                "Mean QED": f4(row.qed_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
            }
        )
    return pd.DataFrame(rows)


def high_fidelity_oracle_table():
    df = pd.read_csv("results/high_fidelity_oracle_triad.csv")
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1, "Pocket2Mol_prospective": 2}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df = df[df["policy"].isin(keep_policy)].copy()
    df["order_s"] = df["source"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_s", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "Policy": row.policy,
                "N": row.n,
                "dock_fast": pct(row.dock_pose_pass),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "Redock median": f4(row.redock_median),
                "Delta vs family-QED": f4(row.redock_delta_vs_family_qed),
                "Route success": pct(row.route_proxy_success),
                "Native recovery": pct(row.native_profile_recovery),
                "Consensus": pct(row.consensus_score),
            }
        )
    return pd.DataFrame(rows)


def pocket_robustness_table():
    df = pd.concat(
        [
            pd.read_csv("results/pocket_robustness_diffsbdd_summary.csv"),
            pd.read_csv("results/pocket_robustness_pocket2mol_summary.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    keep = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_p = {p: i for i, p in enumerate(keep)}
    df = df[df["policy"].isin(keep)].copy()
    df["order_s"] = df["source_label"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_s", "sigma", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source_label,
                "Policy": row.policy,
                "Sigma": row.sigma,
                "N": row.n,
                "Targets": row.targets,
                "Replicates": row.replicates,
                "Mean risk": f4(row.risk_mean),
                "Mean QED": f4(row.qed_mean),
                "Protein pass": pct(row.protein_pass),
                "Dock pose pass": pct(row.dock_pose_pass),
                "Min-dist protein": pct(row.minimum_distance_to_protein),
                "Vol overlap protein": pct(row.volume_overlap_with_protein),
            }
        )
    return pd.DataFrame(rows)


def multi_oracle_fusion_table():
    df = pd.read_csv("results/multi_oracle_fusion_selection_agg.csv")
    keep_direction = [
        "DiffSBDD_within",
        "Pocket2Mol_within",
        "DiffSBDD_to_Pocket2Mol",
        "Pocket2Mol_to_DiffSBDD",
    ]
    keep_policy = ["qed", "pb_qed", "rc_select", "fusion", "pb_fusion", "fusion_rc"]
    df = df[df["direction"].isin(keep_direction) & df["policy"].isin(keep_policy)].copy()
    order_d = {name: i for i, name in enumerate(keep_direction)}
    order_p = {name: i for i, name in enumerate(keep_policy)}
    df["order_d"] = df["direction"].map(order_d)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_d", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Direction": row.direction,
                "Policy": row.policy,
                "Seeds": row.seeds,
                "Reach K": pct(row.reach_k_mean),
                "Selected": f4(row.selected_n_mean),
                "dock_fast": pct(row.dock_pose_pass_mean),
                "Risk >0.5": pct(row.risk_gt_0_5_mean),
                "Mean QED": f4(row.qed_mean_mean),
                "mol_fast": pct(row.intramol_pass_mean),
                "Fusion prob": f4(row.fusion_prob_mean_mean),
            }
        )
    return pd.DataFrame(rows)


def risk_explanation_faithfulness_table():
    df = pd.read_csv("results/risk_explanation_feature_ablation_summary.csv")
    keep_source = ["DiffSBDD", "Pocket2Mol", "DiffSBDD_to_Pocket2Mol", "Pocket2Mol_to_DiffSBDD"]
    keep_group = ["risk_only", "geometry_only", "chemistry_only", "validity_only", "all_without_risk", "all"]
    df = df[df["source"].isin(keep_source) & df["feature_group"].isin(keep_group)].copy()
    order_s = {name: i for i, name in enumerate(keep_source)}
    order_g = {name: i for i, name in enumerate(keep_group)}
    df["order_s"] = df["source"].map(order_s)
    df["order_g"] = df["feature_group"].map(order_g)
    rows = []
    for row in df.sort_values(["order_s", "order_g"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "Feature group": row.feature_group,
                "Seeds": row.seeds,
                "Failure AUROC": f"{row.auroc_mean:.4f} +/- {row.auroc_std:.4f}",
            }
        )
    return pd.DataFrame(rows)


def external_oracle_audit_table():
    audit_path = Path("logs/external_oracle_resource_audit.json")
    if not audit_path.exists():
        return pd.DataFrame([{"Resource": "external_oracle_audit", "Status": "missing", "Notes": "audit json not found"}])
    meta = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = []
    for name in ["aizynthfinder", "rdchiral", "rxnmapper", "gnina", "vina", "rdkit", "gdown"]:
        ok = bool(meta.get("modules", {}).get(name))
        rows.append({"Resource": f"module:{name}", "Status": "available" if ok else "missing", "Notes": "importlib probe"})
    for name in ["gnina", "aizynthcli", "aizynthapp", "gdown"]:
        path = meta.get("binaries", {}).get(name)
        rows.append({"Resource": f"binary:{name}", "Status": "available" if path else "missing", "Notes": path or "not on PATH"})
    gnina = meta.get("gnina_release", {})
    if gnina.get("ok"):
        assets = "; ".join(f"{a.get('name')} ({a.get('size_gb')} GB)" for a in gnina.get("assets", []))
        rows.append({"Resource": "GNINA release", "Status": "found", "Notes": f"{gnina.get('tag')}: {assets}"})
    else:
        rows.append({"Resource": "GNINA release", "Status": "probe failed", "Notes": gnina.get("error", "")})
    generators = meta.get("generators", {})
    for name in ["DiffSBDD", "Pocket2Mol", "TargetDiff", "DecompDiff", "CBGBench"]:
        item = generators.get(name, {})
        rows.append({"Resource": name, "Status": "present" if item.get("exists") else "missing", "Notes": item.get("path", "")})
    for name in ["CBGBench_generated_sdf_count", "DecompDiff_checkpoint_count"]:
        count = int(generators.get(name, 0))
        status = "present" if count > 0 else "missing"
        rows.append({"Resource": name, "Status": status, "Notes": f"count={count}"})
    for name, item in meta.get("local_oracles", {}).items():
        rows.append(
            {
                "Resource": name,
                "Status": "present" if item.get("exists") else "missing",
                "Notes": f"{item.get('path', '')}; size={item.get('size', 0)}",
            }
        )
    return pd.DataFrame(rows)


def gnina_selection_table():
    df = pd.read_csv("results/gnina_selection_summary.csv")
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df = df[df["policy"].isin(keep_policy)].copy()
    df["order_s"] = df["source"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_s", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "Policy": row.policy,
                "Scored": row.scored,
                "CNNscore": f4(row.cnnscore_mean),
                "CNNaffinity": f4(row.cnnaffinity_mean),
                "Affinity": f4(row.affinity_mean),
                "dock_fast": pct(row.dock_pose_pass),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def aizynthfinder_route_table():
    df = pd.read_csv("results/aizynthfinder_selection_summary.csv")
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df = df[df["policy"].isin(keep_policy)].copy()
    df["order_s"] = df["source"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_s", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Solved": pct(row.solved_rate),
                "Top score": f4(row.top_score_mean),
                "Steps median": f4(row.route_steps_median),
                "In-stock/total": f"{f4(row.precursors_in_stock_mean)}/{f4(row.precursors_total_mean)}",
                "Search time": f4(row.search_time_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "dock_fast": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def multiobjective_baseline_table():
    df = pd.read_csv("results/multiobjective_selection_summary.csv")
    keep_policy = [
        "qed",
        "qed_minus_risk",
        "pb_qed",
        "pb_qed_minus_risk",
        "pb_rc_select",
        "pb_weighted_qed_risk_1_00",
        "pb_pareto_qed_risk",
        "diverse_rc",
    ]
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df = df[df["policy"].isin(keep_policy)].copy()
    df["order_s"] = df["source"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_s", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "Policy": row.policy,
                "N": row.n,
                "Reach K": pct(row.reach_k),
                "dock_fast": pct(row.dock_pose_pass),
                "Protein pass": pct(row.protein_pass),
                "mol_fast": pct(row.intramol_pass),
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def protein_scaffold_ood_table():
    df = pd.read_csv("results/protein_scaffold_ood_selection_summary.csv")
    keep_axis = ["protein_unseen_train", "native_scaffold_unseen_train", "generated_scaffold_unseen_train"]
    keep_policy = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    df = df[df["axis"].isin(keep_axis) & df["policy"].isin(keep_policy)].copy()
    df = df[df["ood_value"].astype(bool)].copy()
    order_a = {p: i for i, p in enumerate(keep_axis)}
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df["order_a"] = df["axis"].map(order_a)
    df["order_s"] = df["source"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for row in df.sort_values(["order_a", "order_s", "order_p"]).itertuples(index=False):
        rows.append(
            {
                "OOD axis": row.axis,
                "Source": row.source,
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "dock_fast": pct(row.dock_pose_pass),
                "Protein pass": pct(row.protein_pass),
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def targetdiff_t50_boundary_table():
    df = pd.read_csv("results/targetdiff_t50_boundary_summary.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Generator": row.generator,
                "Targets": row.targets_with_native,
                "Nonempty SDFs": row.nonempty_sdf_files,
                "Generated mols": row.generated_molecules,
                "Mean risk": f4(row.generated_risk_mean),
                "Risk >0.5": pct(row.generated_risk_gt_0_5),
                "Mean QED": f4(row.generated_qed_mean),
                "mol_fast mols": row.molfast_molecules,
                "mol_fast pass": pct(row.molfast_all_core_pass),
                "Native risk": f4(row.native_risk_mean),
            }
        )
    return pd.DataFrame(rows)


def syncguide_third_generator_table():
    df = pd.read_csv("results/syncguide_t1000_n16_third_generator_summary.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Generator": row.generator,
                "Targets": row.targets,
                "SDF files": row.sdf_files,
                "Generated mols": row.generated_molecules,
                "Mean risk": f4(row.generated_risk_mean),
                "Risk >0.5": pct(row.generated_risk_gt_0_5),
                "Mean QED": f4(row.generated_qed_mean),
                "mol_fast": pct(row.molfast_all_core_pass),
                "RC dock_fast": pct(row.rc_dock_pose_pass),
                "PB+QED dock_fast": pct(row.pb_qed_dock_pose_pass),
                "PB+RC dock_fast": pct(row.pb_rc_dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def weight_transfer_control_table():
    df = pd.read_csv("results/weight_transfer_control.csv")
    keep_policy = [
        "rc_select",
        "weighted_qed_risk_0",
        "weighted_qed_risk_1",
        "weighted_qed_risk_2",
        "weighted_qed_risk_4",
        "weighted_qed_risk_8",
    ]
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1, "SYNC-Guide": 2}
    order_p = {p: i for i, p in enumerate(keep_policy)}
    df = df[(df["filter"] == "raw") & df["policy"].isin(keep_policy)].copy()
    df["order_s"] = df["source"].map(order_s)
    df["order_p"] = df["policy"].map(order_p)
    rows = []
    for _, row in df.sort_values(["order_s", "order_p"]).iterrows():
        rows.append(
            {
                "Source": row["source"],
                "Policy": row["policy"],
                "Lambda": "" if pd.isna(row["lambda"]) else row["lambda"],
                "N": row["n"],
                "Targets": row["targets"],
                "Threshold viol.": pct(row["threshold_violation"]),
                "Risk >0.5": pct(row["risk_gt_0_5"]),
                "Mean risk": f4(row["risk_mean"]),
                "Mean QED": f4(row["qed_mean"]),
                "mol_fast": pct(row["mol_fast"]),
            }
        )
    return pd.DataFrame(rows)


def fusion_oracle_robustness_table():
    df = pd.read_csv("results/fusion_oracle_robustness_agg.csv")
    keep_scenarios = [
        "full",
        "missing_risk",
        "missing_geometry",
        "missing_validity",
        "noisy_risk_0_10",
        "noisy_geometry_0_25sd",
        "noisy_all_oracles",
    ]
    df = df[(df["policy"] == "fusion_rc") & df["scenario"].isin(keep_scenarios)].copy()
    order_s = {name: i for i, name in enumerate(keep_scenarios)}
    df["order_s"] = df["scenario"].map(order_s)
    rows = []
    for row in df.sort_values(["direction", "order_s"]).itertuples(index=False):
        rows.append(
            {
                "Direction": row.direction,
                "Scenario": row.scenario,
                "Seeds": row.seeds,
                "Reach K": pct(row.reach_k_mean),
                "dock_fast": pct(row.dock_pose_pass_mean),
                "Risk >0.5": pct(row.risk_gt_0_5_mean),
                "Mean QED": f4(row.qed_mean_mean),
                "mol_fast": pct(row.intramol_pass_mean),
            }
        )
    return pd.DataFrame(rows)


def prospective20_case_table():
    df = pd.read_csv("results/prospective20_pocket2mol_n128_case_summary.csv")
    keep = ["raw_pool", "qed", "rc_select", "pb_qed", "pb_rc_select"]
    order = {name: i for i, name in enumerate(keep)}
    df = df[df["scope"].isin(keep)].copy()
    df["order"] = df["scope"].map(order)
    rows = []
    for row in df.sort_values("order").itertuples(index=False):
        rows.append(
            {
                "Scope": row.scope,
                "Targets": row.targets,
                "Molecules": row.molecules,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast": pct(row.mol_fast_pass),
                "dock_fast": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def runtime_memory_table():
    df = pd.read_csv("results/runtime_memory_throughput.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Stage": row.stage,
                "Items": row.items,
                "Molecules": row.molecules,
                "Wall sec": f4(row.wall_sec),
                "Median target sec": f4(row.median_target_sec),
                "Throughput mol/s": f4(row.throughput_mol_per_sec),
                "Max RSS MB": f4(row.max_rss_mb),
                "GPU MB": f4(row.gpu_mem_mb),
            }
        )
    return pd.DataFrame(rows)


def contact_counterfactual_table():
    df = pd.read_csv("results/contact_counterfactual_faithfulness_summary.csv")
    rows = []
    for row in df.sort_values("source").itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "N": row.n,
                "Orig risk": f4(row.orig_risk),
                "Residue delta": f4(row.residue_deleted_risk_delta),
                "Residue drop": pct(row.residue_deleted_risk_decrease),
                "Atom delta": f4(row.atom_deleted_risk_delta),
                "Atom drop": pct(row.atom_deleted_risk_decrease),
                "Mask delta": f4(row.contact_masked_risk_delta),
                "Mask drop": pct(row.contact_masked_risk_decrease),
            }
        )
    return pd.DataFrame(rows)


def short_md_stability_table():
    df = pd.read_csv("results/short_md_stability_top3.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Data": row.data_id,
                "Mol": row.mol_index,
                "Risk": f4(row.risk_prob),
                "QED": f4(row.qed),
                "Min E": f4(row.minimized_interaction_kj_mol),
                "MD mean E": f4(row.md_mean_interaction_kj_mol),
                "Final RMSD A": f4(row.final_ligand_rmsd_ang),
                "Contact retention": f4(row.final_contact_retention),
                "Stable": row.stable_proxy,
            }
        )
    return pd.DataFrame(rows)


def kinase_selectivity_table():
    df = pd.read_csv("results/kinase_selectivity_case_summary.csv")
    rows = []
    for row in df.sort_values("ligand_target").itertuples(index=False):
        rows.append(
            {
                "Ligand target": row.ligand_target,
                "Target score": f4(row.target_score),
                "Best off-target": f4(row.best_offtarget_score),
                "Margin": f4(row.selectivity_margin),
                "Target rank": row.target_rank,
                "Target best": row.target_is_best,
                "Risk": f4(row.ligand_risk),
                "QED": f4(row.ligand_qed),
            }
        )
    return pd.DataFrame(rows)


def rcsbdd_bench_table():
    df = pd.read_csv("benchmarks/RC-SBDD-Bench/manifest.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Benchmark": "RC-SBDD-Bench",
                "Targets": row.targets,
                "Raw candidates": row.raw_candidates,
                "Selected dock_fast rows": row.selected_dockfast_rows,
                "Candidate SDFs": row.candidate_sdf_files,
                "Pocket files": row.pocket_files,
            }
        )
    return pd.DataFrame(rows)


def calibrated_weight_transfer_table():
    df = pd.read_csv("results/calibrated_weight_transfer.csv")
    df = df[df["calibration_source"] != df["test_source"]].copy()
    keep = ["transferred_weighted", "source_rc", "oracle_weighted"]
    df = df[df["eval_label"].isin(keep)].copy()
    rows = []
    for row in df.sort_values(["calibration_source", "test_source", "eval_label"]).itertuples(index=False):
        rows.append(
            {
                "Calibration": row.calibration_source,
                "Test": row.test_source,
                "Eval": row.eval_label,
                "Lambda": f4(row.chosen_lambda),
                "N": row.n,
                "Violation": pct(row.threshold_violation),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "QED": f4(row.qed_mean),
                "mol_fast": pct(row.mol_fast),
                "dock_fast": pct(row.dock_fast),
            }
        )
    return pd.DataFrame(rows)


def selectivity_aware_kinase_table():
    df = pd.read_csv("results/kinase_selectivity_aware_summary.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Target best": pct(row.target_is_best),
                "Mean margin": f4(row.mean_margin),
                "Median rank": f4(row.median_rank),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "QED": f4(row.mean_qed),
                "dock_fast": pct(row.dock_fast),
            }
        )
    return pd.DataFrame(rows)


def bindingmoad_test_pose_table():
    df = pd.read_csv("results/bindingmoad_test_pose_selection_summary.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean risk": f4(row.mean_risk),
                "RMSD": f4(row.mean_pose_rmsd),
                "RMSD<=1A": pct(row.rmsd_le_1),
                "RMSD<=2A": pct(row.rmsd_le_2),
                "Native top4": pct(row.native_in_top4),
                "QED": f4(row.mean_qed),
            }
        )
    return pd.DataFrame(rows)


def short_md_top10_policy_table():
    df = pd.read_csv("results/short_md_top10_policy_comparison_summary.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Stable": pct(row.stable_proxy),
                "Final RMSD A": f4(row.mean_final_rmsd),
                "Contact retention": f4(row.mean_contact_retention),
                "Min E": f4(row.mean_min_energy),
                "MD mean E": f4(row.mean_md_energy),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "QED": f4(row.mean_qed),
                "dock_fast": pct(row.dock_fast),
            }
        )
    return pd.DataFrame(rows)


def public_sota_output_audit_table():
    df = pd.read_csv("results/public_sota_output_audit.csv")
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "Source": row.name,
                "Status": row.status,
                "Assets": row.asset_probe if isinstance(row.asset_probe, str) and row.asset_probe else row.file_count,
                "Error": row.error if isinstance(row.error, str) else "",
            }
        )
    return pd.DataFrame(rows)


def rcsbdd_bench100_table():
    manifest = pd.read_csv("benchmarks/RC-SBDD-Bench100/manifest.csv").iloc[0]
    gen = pd.read_csv("benchmarks/RC-SBDD-Bench100/generator_manifest.csv")
    rows = [
        {
            "Scope": "Total",
            "Targets": manifest.targets,
            "Candidates": manifest.candidates,
            "dock_fast labels": manifest.dockfast_label_rows,
            "Risk labels": manifest.candidates,
            "SDF files": manifest.candidate_sdf_files,
        }
    ]
    for row in gen.sort_values("generator").itertuples(index=False):
        rows.append(
            {
                "Scope": row.generator,
                "Targets": row.targets,
                "Candidates": row.candidates,
                "dock_fast labels": row.dockfast_label_rows,
                "Risk labels": row.risk_label_rows,
                "SDF files": row.candidate_sdf_files,
            }
        )
    return pd.DataFrame(rows)


def fusion_strong_baselines_table():
    df = pd.read_csv("results/fusion_strong_baselines_metrics_agg.csv")
    grouped = (
        df.groupby(["method", "scenario"], as_index=False)
        .agg(
            directions=("direction", "nunique"),
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
        )
        .sort_values(["scenario", "auroc"], ascending=[True, False])
    )
    rows = []
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "Method": row.method,
                "Scenario": row.scenario,
                "Directions": row.directions,
                "AUROC": f4(row.auroc),
                "AUPRC": f4(row.auprc),
                "Brier": f4(row.brier),
                "ECE": f4(row.ece),
            }
        )
    return pd.DataFrame(rows)


def target_level_statistics_table():
    df = pd.read_csv("results/target_level_statistical_tests.csv")
    df = df[df["metric"].isin(["dock_fast", "risk_gt_0_5", "qed"])].copy()
    rows = []
    for row in df.sort_values(["block", "metric"]).itertuples(index=False):
        rows.append(
            {
                "Block": row.block,
                "Metric": row.metric,
                "Targets": row.targets,
                "Baseline": f4(row.baseline_mean),
                "Method": f4(row.method_mean),
                "Delta": f4(row.delta_method_minus_baseline),
                "95% CI": f"[{f4(row.bootstrap_ci_low)}, {f4(row.bootstrap_ci_high)}]",
                "p": f4(row.wilcoxon_p),
                "FDR q": f4(row.fdr_q),
                "Cliff": f4(row.cliffs_delta),
                "Improves": row.improves,
            }
        )
    return pd.DataFrame(rows)


def risk_to_dockfast_calibration_table():
    df = pd.read_csv("results/risk_to_dockfast_calibration_summary.csv")
    rows = []
    for row in df.sort_values(["dataset", "calibrator"]).itertuples(index=False):
        rows.append(
            {
                "Dataset": row.dataset,
                "Calibrator": row.calibrator,
                "N": row.n,
                "Targets": row.targets,
                "Failure": pct(row.failure_rate),
                "Pred fail": f4(row.mean_predicted_failure),
                "Brier": f4(row.brier),
                "ECE": f4(row.ece),
                "AUROC fail": f4(row.auroc_failure),
                "AUPRC fail": f4(row.auprc_failure),
            }
        )
    return pd.DataFrame(rows)


def optional_missing_table(name):
    return pd.DataFrame([{"Status": f"Missing required result for {name}"}])


def pocketflow_crossdock_table():
    path = Path("results/pocketflow_crossdock_n16_dockfast_selection_summary.csv")
    run_path = Path("results/pocketflow_crossdock_n16_run_summary.csv")
    if not path.exists():
        return optional_missing_table("PocketFlow CrossDock")
    df = pd.read_csv(path)
    run = pd.read_csv(run_path) if run_path.exists() else pd.DataFrame()
    rows = []
    for row in df.sort_values("policy").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast": pct(row.molfast_core_pass),
                "Protein pass": pct(row.protein_pass),
                "dock_fast": pct(row.dock_pose_pass),
                "Generated targets": int((run.returncode == 0).sum()) if len(run) and "returncode" in run.columns else "NA",
            }
        )
    return pd.DataFrame(rows)


def gnina_redock_t25_table():
    path = Path("results/gnina_redock_t25_summary.csv")
    if not path.exists():
        return optional_missing_table("GNINA redock T25")
    df = pd.read_csv(path)
    rows = []
    for row in df.sort_values(["source", "policy"]).itertuples(index=False):
        rows.append(
            {
                "Source": row.source,
                "Policy": row.policy,
                "Attempted": row.attempted,
                "Redocked": row.redocked,
                "Success": pct(row.success_rate),
                "CNNscore": f4(row.cnnscore_mean),
                "CNNaffinity": f4(row.cnnaffinity_mean),
                "Affinity": f4(row.affinity_mean),
                "dock_fast": pct(row.dock_fast_pass),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "QED": f4(row.qed_mean),
            }
        )
    return pd.DataFrame(rows)


def calibration_stress_table():
    coverage = Path("results/selective_coverage_risk_curves.csv")
    shift = Path("results/generator_shift_calibration.csv")
    size = Path("results/calibration_set_size_sensitivity.csv")
    if not coverage.exists() or not shift.exists() or not size.exists():
        return optional_missing_table("Calibration stress tests")
    curves = pd.read_csv(coverage)
    rows = []
    for row in curves[curves["tau"].isin([0.25, 0.50, 0.75, 1.00])].sort_values(["generator", "tau"]).itertuples(index=False):
        rows.append(
            {
                "Block": "coverage",
                "Generator": row.generator,
                "Setting": f"tau={row.tau:.2f}",
                "Coverage": pct(row.molecule_coverage),
                "Target coverage": pct(row.target_coverage),
                "dock_fast": pct(row.dock_pose_pass),
                "Failure": pct(row.dock_failure),
                "Mean risk": f4(row.mean_risk),
                "QED": f4(row.mean_qed),
            }
        )
    sh = pd.read_csv(shift)
    for row in sh.sort_values(["calibration_source", "evaluation_generator", "method"]).itertuples(index=False):
        rows.append(
            {
                "Block": "shift",
                "Generator": f"{row.calibration_source}->{row.evaluation_generator}",
                "Setting": row.method,
                "Coverage": "NA",
                "Target coverage": "NA",
                "dock_fast": "NA",
                "Failure": pct(row.base_failure),
                "Mean risk": f"ECE {f4(row.ece)} / Brier {f4(row.brier)}",
                "QED": f"AUROC {f4(row.auroc)}",
            }
        )
    return pd.DataFrame(rows)


def bindingmoad_pocketflow_table():
    path = Path("results/bindingmoad_pocketflow_n16_dockfast_selection_summary.csv")
    if not path.exists():
        return optional_missing_table("BindingMOAD PocketFlow")
    df = pd.read_csv(path)
    rows = []
    for row in df.sort_values("policy").itertuples(index=False):
        rows.append(
            {
                "Policy": row.policy,
                "N": row.n,
                "Targets": row.targets,
                "Mean risk": f4(row.risk_mean),
                "Risk >0.5": pct(row.risk_gt_0_5),
                "Mean QED": f4(row.qed_mean),
                "mol_fast": pct(row.molfast_core_pass),
                "Protein pass": pct(row.protein_pass),
                "dock_fast": pct(row.dock_pose_pass),
            }
        )
    return pd.DataFrame(rows)


def rcsbdd_bench_v1_table():
    path = Path("benchmarks/RC-SBDD-Bench-v1/manifest.csv")
    gen_path = Path("benchmarks/RC-SBDD-Bench-v1/generator_manifest.csv")
    if not path.exists() or not gen_path.exists():
        return optional_missing_table("RC-SBDD-Bench v1")
    manifest = pd.read_csv(path).iloc[0]
    gen = pd.read_csv(gen_path)
    rows = [
        {
            "Scope": "Total",
            "Targets": manifest.targets,
            "Rows": manifest.candidate_rows,
            "dock_fast labels": manifest.dockfast_label_rows,
            "SDF files": manifest.candidate_sdf_files,
        }
    ]
    for row in gen.sort_values("generator").itertuples(index=False):
        rows.append(
            {
                "Scope": row.generator,
                "Targets": row.targets,
                "Rows": row.rows,
                "dock_fast labels": row.dockfast_label_rows,
                "SDF files": row.candidate_sdf_files,
            }
        )
    return pd.DataFrame(rows)


def to_md(df):
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(str(c) for c in cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def main():
    sections = [
        ("Table 1. Risk Estimator Ablation", risk_proxy_table()),
        ("Table 2. Risk Scorer Selection Stability", risk_scorer_selection_table()),
        ("Table 3. Failure Signal Taxonomy under Selection", failure_signal_table()),
        ("Table 4. Rejection Budget under Native-Calibrated Risk Thresholds", rejection_budget_table()),
        ("Table 5. Official DiffSBDD Generated-Set Risk Audit", generated_benchmark_table()),
        ("Table 6. Main Selection Results on Official Full-Atom Conditional Set", official_selection_table()),
        ("Table 7. Target-Level Bootstrap Deltas", bootstrap_table()),
        ("Table 8. K Sensitivity under Native p95 Threshold", sensitivity_table()),
        ("Table 9. CA-Conditioned Boundary Case", ca_boundary_table()),
        ("Table 10. Local t500 Generation Closed-Loop Check", local_t500_table()),
        ("Table 11. Online Rejection Control on Official Full-Atom Conditional Set", online_rejection_table()),
        ("Table 12. TargetDiff Cross-Generator Boundary Check", targetdiff_boundary_table()),
        ("Table 13. TargetDiff Online Rejection Boundary Check", targetdiff_online_table()),
        ("Table 14. Fixed-Pose Vina Score Sanity Check", vina_score_table()),
        ("Table 15. Full-Pool dock_fast Label Audit", full_pool_dockfast_table()),
        ("Table 16. Conformal Risk Control on Official Full-Atom Conditional Set", conformal_risk_control_table()),
        ("Table 17. Adaptive Generation Control on Official Full-Atom Conditional Stream", adaptive_generation_control_table()),
        ("Table 18. AutoDock Vina Redocking Sanity Check", vina_redock_table()),
        ("Table 19. BindingMOAD External Risk Transfer", bindingmoad_transfer_table()),
        ("Table 20. Pocket2Mol Cross-Generator Transfer Check", pocket2mol_crossgen_table()),
        ("Table 21. Synthesis and Drugability Oracle Check", synthesis_oracle_table()),
        ("Table 22. Functional Interaction Oracle Check", functional_interaction_table()),
        ("Table 23. Leave-Generator-Out CRC Transfer", cross_generator_crc_table()),
        ("Table 24. Three-Target Prospective Case Study", prospective_case_table()),
        ("Table 25. Target-Conditional CRC", target_conditional_crc_table()),
        ("Table 26. Route-Level Synthesis Proxy", route_synthesis_proxy_table()),
        ("Table 27. High-Fidelity Oracle Triad", high_fidelity_oracle_table()),
        ("Table 28. Pocket Coordinate Robustness", pocket_robustness_table()),
        ("Table 29. Multi-Oracle Reliability Fusion", multi_oracle_fusion_table()),
        ("Table 30. Risk Explanation Faithfulness", risk_explanation_faithfulness_table()),
        ("Table 31. External Oracle and Third-Generator Audit", external_oracle_audit_table()),
        ("Table 32. GNINA CNN Scoring", gnina_selection_table()),
        ("Table 33. AiZynthFinder Route Search", aizynthfinder_route_table()),
        ("Table 34. Multi-Objective Selection Baselines", multiobjective_baseline_table()),
        ("Table 35. Protein and Scaffold OOD Selection", protein_scaffold_ood_table()),
        ("Table 36. TargetDiff 50-Target Boundary", targetdiff_t50_boundary_table()),
        ("Table 37. SYNC-Guide Third Positive Generator", syncguide_third_generator_table()),
        ("Table 38. Weight Transfer and Risk-Control Audit", weight_transfer_control_table()),
        ("Table 39. Fusion Missing/Noisy Oracle Robustness", fusion_oracle_robustness_table()),
        ("Table 40. Prospective20 Pocket2Mol n128 Case Study", prospective20_case_table()),
        ("Table 41. Runtime Memory Throughput", runtime_memory_table()),
        ("Table 42. Contact Counterfactual Faithfulness", contact_counterfactual_table()),
        ("Table 43. Short MD Stability Top3", short_md_stability_table()),
        ("Table 44. Kinase-Family Selectivity Case", kinase_selectivity_table()),
        ("Table 45. RC-SBDD-Bench Manifest", rcsbdd_bench_table()),
        ("Table 46. Calibration-Selected Weighted Baseline Transfer", calibrated_weight_transfer_table()),
        ("Table 47. Selectivity-Aware Kinase Case", selectivity_aware_kinase_table()),
        ("Table 48. BindingMOAD Test Pose-Selection Holdout", bindingmoad_test_pose_table()),
        ("Table 49. Short MD Top10 Policy Comparison", short_md_top10_policy_table()),
        ("Table 50. Public SOTA Output Audit", public_sota_output_audit_table()),
        ("Table 51. RC-SBDD-Bench100 Manifest", rcsbdd_bench100_table()),
        ("Table 52. Strong Fusion Baselines and Missing-Modality Calibration", fusion_strong_baselines_table()),
        ("Table 53. Target-Level Statistical Tests", target_level_statistics_table()),
        ("Table 54. Risk-to-dockfast Calibration", risk_to_dockfast_calibration_table()),
        ("Table 55. PocketFlow CrossDock Generated Outputs", pocketflow_crossdock_table()),
        ("Table 56. GNINA Local Redocking T25", gnina_redock_t25_table()),
        ("Table 57. Calibration Stress Tests", calibration_stress_table()),
        ("Table 58. BindingMOAD PocketFlow De Novo", bindingmoad_pocketflow_table()),
        ("Table 59. RC-SBDD-Bench v1 Release", rcsbdd_bench_v1_table()),
    ]
    lines = [
        "# Paper Tables Only",
        "",
        "This file intentionally contains tables only. Figures are deferred until the experiment design is frozen.",
        "",
    ]
    for title, df in sections:
        lines.extend([f"## {title}", "", to_md(df), ""])
    out = Path("experiments/PAPER_TABLES_ONLY.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
