import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


POLICY_ORDER = [
    "qed",
    "risk",
    "qed_minus_risk",
    "pb_qed",
    "pb_qed_minus_risk",
    "rc_select",
    "pb_rc_select",
    "weighted_qed_risk_0_25",
    "weighted_qed_risk_0_50",
    "weighted_qed_risk_1_00",
    "weighted_qed_risk_2_00",
    "pb_weighted_qed_risk_1_00",
    "pareto_qed_risk",
    "pb_pareto_qed_risk",
    "diverse_qed",
    "diverse_rc",
]


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def load_pool(source):
    if source == "DiffSBDD_official":
        df = pd.read_csv("results/dockfast_full_pool_fullatom_cond.csv")
        df = df[df["kind"] == "generated"].copy()
        df["source"] = source
        df["target_id"] = df["key"]
        k = 10
    elif source == "Pocket2Mol_transfer":
        df = pd.read_csv("results/dockfast_full_pool_pocket2mol_n16_ext.csv")
        df = df[df["kind"] == "generated"].copy()
        df["source"] = source
        df["target_id"] = df["key"]
        k = 4
    else:
        raise ValueError(source)
    df["intramol_pass"] = df["intramol_pass"].fillna(False).astype(bool)
    df["dock_pose_pass"] = df["dock_pose_pass"].fillna(False).astype(bool)
    return df, k


def fp_from_sdf(path):
    try:
        mol = next((mol for mol in Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True) if mol is not None), None)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except Exception:
        return None


def greedy_diverse(group, k, safe=False):
    pool = group.copy()
    if safe:
        native = pd.read_csv("results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
        tau = float(native[native["kind"] == "native"]["risk_prob"].quantile(0.95))
        pool = pool[pool["risk_prob"] <= tau].copy()
        if pool.empty:
            pool = group.copy()
    pool = pool.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(max(k * 8, k)).copy()
    pool["fp"] = [fp_from_sdf(path) for path in pool["mol_pred"]]
    selected = []
    remaining = list(pool.index)
    while remaining and len(selected) < k:
        best_idx = None
        best_score = -1e9
        for idx in remaining:
            row = pool.loc[idx]
            novelty = 1.0
            if selected and row["fp"] is not None:
                sims = [
                    DataStructs.TanimotoSimilarity(row["fp"], pool.loc[j, "fp"])
                    for j in selected
                    if pool.loc[j, "fp"] is not None
                ]
                if sims:
                    novelty = 1.0 - max(sims)
            score = float(row["qed"]) - 0.5 * float(row["risk_prob"]) + 0.25 * novelty
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return pool.loc[selected].drop(columns=["fp"], errors="ignore")


def pareto_select(group, k, require_pb=False):
    pool = group[group["intramol_pass"]].copy() if require_pb else group.copy()
    if pool.empty:
        pool = group.copy()
    values = pool[["qed", "risk_prob"]].to_numpy(float)
    dominated = np.zeros(len(pool), dtype=bool)
    for i, (q_i, r_i) in enumerate(values):
        better_or_equal = (values[:, 0] >= q_i) & (values[:, 1] <= r_i)
        strictly_better = (values[:, 0] > q_i) | (values[:, 1] < r_i)
        dominated[i] = bool(np.any(better_or_equal & strictly_better))
    front = pool.loc[~dominated].copy()
    front["selection_score"] = front["qed"] - front["risk_prob"]
    front = front.sort_values(["selection_score", "qed"], ascending=[False, False])
    if len(front) >= k:
        return front.head(k)
    rest = pool.drop(front.index, errors="ignore").copy()
    rest["selection_score"] = rest["qed"] - rest["risk_prob"]
    rest = rest.sort_values(["selection_score", "qed"], ascending=[False, False])
    return pd.concat([front, rest.head(k - len(front))], axis=0)


def select_policy(group, policy, k, tau):
    if policy == "qed":
        return group.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(k)
    if policy == "risk":
        return group.sort_values(["risk_prob", "qed"], ascending=[True, False]).head(k)
    if policy == "qed_minus_risk":
        scored = group.assign(selection_score=group["qed"] - group["risk_prob"])
        return scored.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k)
    if policy == "pb_qed":
        pool = group[group["intramol_pass"]].copy()
        return pool.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(k)
    if policy == "pb_qed_minus_risk":
        pool = group[group["intramol_pass"]].copy()
        scored = pool.assign(selection_score=pool["qed"] - pool["risk_prob"])
        return scored.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k)
    if policy == "rc_select":
        safe = group[group["risk_prob"] <= tau].sort_values(["qed", "risk_prob"], ascending=[False, True])
        if len(safe) >= k:
            return safe.head(k)
        rest = group.drop(safe.index, errors="ignore").assign(selection_score=group["qed"] - group["risk_prob"])
        return pd.concat([safe, rest.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k - len(safe))])
    if policy == "pb_rc_select":
        pool = group[group["intramol_pass"]].copy()
        safe = pool[pool["risk_prob"] <= tau].sort_values(["qed", "risk_prob"], ascending=[False, True])
        if len(safe) >= k:
            return safe.head(k)
        rest = pool.drop(safe.index, errors="ignore").assign(selection_score=pool["qed"] - pool["risk_prob"])
        return pd.concat([safe, rest.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k - len(safe))])
    if policy.startswith("weighted_qed_risk_"):
        lam = float(policy.rsplit("_", 1)[-1]) / 100 if policy.endswith(("25", "50")) else float(policy.rsplit("_", 1)[-1])
        if policy.endswith("0_25"):
            lam = 0.25
        elif policy.endswith("0_50"):
            lam = 0.50
        elif policy.endswith("1_00"):
            lam = 1.00
        elif policy.endswith("2_00"):
            lam = 2.00
        scored = group.assign(selection_score=group["qed"] - lam * group["risk_prob"])
        return scored.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k)
    if policy == "pb_weighted_qed_risk_1_00":
        pool = group[group["intramol_pass"]].copy()
        scored = pool.assign(selection_score=pool["qed"] - pool["risk_prob"])
        return scored.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k)
    if policy == "pareto_qed_risk":
        return pareto_select(group, k, require_pb=False)
    if policy == "pb_pareto_qed_risk":
        return pareto_select(group, k, require_pb=True)
    if policy == "diverse_qed":
        return greedy_diverse(group, k, safe=False)
    if policy == "diverse_rc":
        return greedy_diverse(group, k, safe=True)
    raise ValueError(policy)


def summarize(selected):
    rows = []
    for (source, policy), group in selected.groupby(["source", "policy"], sort=True):
        expected_k = 10 if source == "DiffSBDD_official" else 4
        rows.append(
            {
                "source": source,
                "policy": policy,
                "n": int(len(group)),
                "targets": int(group["target_id"].nunique()),
                "reach_k": float((group.groupby("target_id").size() >= expected_k).mean()),
                "dock_pose_pass": float(group["dock_pose_pass"].mean()),
                "protein_pass": float(group["protein_pass"].fillna(False).astype(bool).mean())
                if "protein_pass" in group
                else np.nan,
                "intramol_pass": float(group["intramol_pass"].mean()),
                "risk_mean": float(group["risk_prob"].mean()),
                "risk_gt_0_5": float((group["risk_prob"] > 0.5).mean()),
                "qed_mean": float(group["qed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary, out_md):
    order_s = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_p = {p: i for i, p in enumerate(POLICY_ORDER)}
    summary = summary.copy()
    summary["order_s"] = summary["source"].map(order_s)
    summary["order_p"] = summary["policy"].map(order_p)
    lines = [
        "# Multi-Objective Selection Baselines",
        "",
        "## Protocol",
        "",
        "- Candidate pools: full dock_fast-labelled DiffSBDD official pool and Pocket2Mol transfer pool.",
        "- Baselines include weighted-sum QED-risk, mol_fast-gated weighted sum, Pareto QED-risk, and diversity-aware selection.",
        "- The purpose is to test whether RC is replaceable by ordinary scalarization or Pareto ranking.",
        "",
        "## Summary",
        "",
        "| Source | Policy | N | Targets | Reach K | dock_fast | Protein pass | mol_fast | Mean risk | Risk >0.5 | Mean QED |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["order_s", "order_p"]).itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.policy} | {row.n} | {row.targets} | {pct(row.reach_k)} | "
            f"{pct(row.dock_pose_pass)} | {pct(row.protein_pass)} | {pct(row.intramol_pass)} | "
            f"{f4(row.risk_mean)} | {pct(row.risk_gt_0_5)} | {f4(row.qed_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. If weighted or Pareto baselines improve dock_fast but retain high risk, RC remains necessary as a calibrated rejection/control layer.",
            "2. If PB+weighted closes the gap, the manuscript should frame RC as a simpler calibrated alternative rather than claiming scalarization cannot work.",
            "3. Diversity-aware baselines test whether the gain is merely avoiding similar bad molecules.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["DiffSBDD_official", "Pocket2Mol_transfer"])
    ap.add_argument("--out-csv", default="results/multiobjective_selection.csv")
    ap.add_argument("--out-summary", default="results/multiobjective_selection_summary.csv")
    ap.add_argument("--out-md", default="experiments/MULTIOBJECTIVE_SELECTION_BASELINES.md")
    args = ap.parse_args()

    native = pd.read_csv("results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
    tau = float(native[native["kind"] == "native"]["risk_prob"].quantile(0.95))
    all_selected = []
    for source in args.sources:
        pool, k = load_pool(source)
        for policy in POLICY_ORDER:
            pieces = []
            for _, group in pool.groupby("target_id", sort=True):
                selected = select_policy(group, policy, k, tau).copy()
                selected["policy"] = policy
                pieces.append(selected)
            if pieces:
                all_selected.append(pd.concat(pieces, ignore_index=True, sort=False))
    selected = pd.concat(all_selected, ignore_index=True, sort=False)
    selected.to_csv(args.out_csv, index=False)
    summary = summarize(selected)
    summary.to_csv(args.out_summary, index=False)
    write_report(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
