import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from vina import Vina
from openbabel import pybel


RDLogger.DisableLog("rdApp.*")

KINASE_IDS = {
    2: "GRK4",
    9: "IPMK",
    16: "M3K14",
    25: "PAK4",
    26: "PHKG1",
    76: "ABL2",
}


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def convert_with_openbabel(in_path, out_path, in_format, add_h=True, rigid_receptor=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mol = next(pybel.readfile(in_format, str(in_path)), None)
    if mol is None:
        raise ValueError(f"OpenBabel could not read {in_path}")
    if add_h:
        mol.addh()
    mol.write("pdbqt", str(out_path), overwrite=True)
    if rigid_receptor:
        lines = [
            line
            for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.startswith(("ATOM", "HETATM", "TER", "END"))
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise ValueError(f"OpenBabel wrote empty PDBQT for {in_path}")
    return out_path


def ligand_center_and_box(sdf_path, padding=10.0):
    mol = next((m for m in Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False) if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not read native ligand coordinates: {sdf_path}")
    conf = mol.GetConformer()
    coords = np.asarray([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    center = coords.mean(axis=0)
    box = np.maximum(coords.max(axis=0) - coords.min(axis=0) + padding, 20.0)
    return center.tolist(), box.tolist()


def test_key(row):
    return f"{Path(row.pocket_path).stem}_{Path(row.ligand_path).stem}"


def prepare_targets(index_path, raw_root):
    index = pd.read_csv(index_path)
    rows = []
    for data_id, name in KINASE_IDS.items():
        row = index.iloc[data_id]
        pocket = Path(raw_root) / row.pocket_path
        native = Path(raw_root) / row.ligand_path
        center, box = ligand_center_and_box(native)
        rows.append(
            {
                "data_id": data_id,
                "target": name,
                "key": test_key(row),
                "pocket": str(pocket),
                "native_ligand": str(native),
                "center": center,
                "box": box,
            }
        )
    return pd.DataFrame(rows)


def prepare_ligands(selection_csv, targets, top_per_target):
    sel = pd.read_csv(selection_csv)
    sel = sel[(sel["set"] == "fullatom_cond") & (sel["policy"] == "pb_rc_select")].copy()
    key_to_target = targets.set_index("key")["target"].to_dict()
    key_to_data = targets.set_index("key")["data_id"].to_dict()
    sel = sel[sel["key"].isin(key_to_target)].copy()
    sel["target"] = sel["key"].map(key_to_target)
    sel["data_id"] = sel["key"].map(key_to_data)
    sel = sel.sort_values(["target", "dock_pose_pass", "risk_prob", "qed"], ascending=[True, False, True, False])
    return sel.groupby("target", sort=True).head(top_per_target).copy()


def dock_pair(payload):
    ligand, receptor, work_dir, exhaustiveness, cpu = payload
    pair_id = f"{ligand['target']}_m{int(ligand['mol_index']):03d}__{receptor['target']}"
    try:
        work = Path(work_dir) / "pairs" / pair_id
        rec_pdbqt = work / "receptor.pdbqt"
        lig_pdbqt = work / "ligand.pdbqt"
        convert_with_openbabel(receptor["pocket"], rec_pdbqt, "pdb", add_h=True, rigid_receptor=True)
        convert_with_openbabel(ligand["mol_pred"], lig_pdbqt, "sdf", add_h=True)
        v = Vina(sf_name="vina", cpu=cpu, verbosity=0)
        v.set_receptor(str(rec_pdbqt))
        v.compute_vina_maps(center=receptor["center"], box_size=receptor["box"])
        v.set_ligand_from_file(str(lig_pdbqt))
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)
        score = float(v.energies(n_poses=1)[0][0])
        err = ""
        success = True
    except Exception as exc:
        score = np.nan
        err = str(exc)
        success = False
    return {
        "ligand_target": ligand["target"],
        "ligand_data_id": int(ligand["data_id"]),
        "ligand_key": ligand["key"],
        "ligand_mol_index": int(ligand["mol_index"]),
        "receptor_target": receptor["target"],
        "receptor_data_id": int(receptor["data_id"]),
        "is_native_target": ligand["target"] == receptor["target"],
        "vina_score": score,
        "success": success,
        "error": err,
        "ligand_risk": float(ligand["risk_prob"]),
        "ligand_qed": float(ligand["qed"]),
        "dock_pose_pass": bool(ligand.get("dock_pose_pass", False)),
        "mol_pred": ligand["mol_pred"],
        "mol_cond": ligand["mol_cond"],
    }


def candidate_summary(raw):
    rows = []
    for (target, mol_index), group in raw[raw.success].groupby(["ligand_target", "ligand_mol_index"], sort=True):
        target_row = group[group.is_native_target]
        off = group[~group.is_native_target]
        if target_row.empty or off.empty:
            continue
        target_score = float(target_row.vina_score.iloc[0])
        best_off = float(off.vina_score.min())
        rank = int(group.sort_values("vina_score").reset_index(drop=True).query("is_native_target").index[0] + 1)
        rows.append(
            {
                "ligand_target": target,
                "ligand_mol_index": int(mol_index),
                "target_score": target_score,
                "best_offtarget_score": best_off,
                "selectivity_margin": best_off - target_score,
                "target_rank": rank,
                "target_is_best": rank == 1,
                "ligand_risk": float(target_row.ligand_risk.iloc[0]),
                "ligand_qed": float(target_row.ligand_qed.iloc[0]),
                "dock_pose_pass": bool(target_row.dock_pose_pass.iloc[0]),
                "mol_pred": target_row.mol_pred.iloc[0],
                "mol_cond": target_row.mol_cond.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def select_policy(group, policy, tau):
    if policy == "plain_pb_rc_qed":
        return group.sort_values(["ligand_qed", "ligand_risk"], ascending=[False, True]).head(1)
    if policy == "plain_pb_rc_lowrisk":
        return group.sort_values(["ligand_risk", "ligand_qed"], ascending=[True, False]).head(1)
    if policy == "selectivity_oracle":
        return group.sort_values(["selectivity_margin", "ligand_qed"], ascending=[False, False]).head(1)
    if policy == "selectivity_rc":
        safe = group[group.ligand_risk <= tau].copy()
        pool = safe if len(safe) else group
        return pool.sort_values(["selectivity_margin", "ligand_qed"], ascending=[False, False]).head(1)
    if policy == "selectivity_utility":
        scored = group.assign(selection_score=group.selectivity_margin + 0.2 * group.ligand_qed - 0.5 * group.ligand_risk)
        return scored.sort_values(["selection_score", "ligand_qed"], ascending=[False, False]).head(1)
    raise ValueError(policy)


def summarize_policies(candidates, tau):
    policies = [
        "plain_pb_rc_qed",
        "plain_pb_rc_lowrisk",
        "selectivity_oracle",
        "selectivity_rc",
        "selectivity_utility",
    ]
    rows = []
    selected_rows = []
    for policy in policies:
        selected = []
        for _, group in candidates.groupby("ligand_target", sort=True):
            selected.append(select_policy(group, policy, tau))
        selected = pd.concat(selected, axis=0)
        selected["policy"] = policy
        selected_rows.append(selected)
        rows.append(
            {
                "policy": policy,
                "n": int(len(selected)),
                "target_is_best": float(selected.target_is_best.mean()),
                "mean_margin": float(selected.selectivity_margin.mean()),
                "median_rank": float(selected.target_rank.median()),
                "mean_risk": float(selected.ligand_risk.mean()),
                "risk_gt_0_5": float((selected.ligand_risk > 0.5).mean()),
                "mean_qed": float(selected.ligand_qed.mean()),
                "dock_fast": float(selected.dock_pose_pass.mean()),
            }
        )
    return pd.DataFrame(rows), pd.concat(selected_rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/if3-crossdocked2020/index_test.csv")
    ap.add_argument("--raw-root", default="data/raw/if3-crossdocked2020/crossdocked_pocket10")
    ap.add_argument("--selection-csv", default="results/posebusters_dockfast_pb_selection.csv")
    ap.add_argument("--top-per-target", type=int, default=8)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--work-dir", default="results/kinase_selectivity_aware_work")
    args = ap.parse_args()

    tau = float(
        pd.read_csv("results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
        .query("kind == 'native'")
        .risk_prob.quantile(0.95)
    )
    targets = prepare_targets(args.index, args.raw_root)
    ligands = prepare_ligands(args.selection_csv, targets, args.top_per_target)
    payloads = [
        (ligand, receptor, args.work_dir, args.exhaustiveness, args.cpu)
        for ligand in ligands.to_dict(orient="records")
        for receptor in targets.to_dict(orient="records")
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(dock_pair, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    raw = pd.DataFrame(rows)
    raw.to_csv("results/kinase_selectivity_aware_crossdock.csv", index=False)
    candidates = candidate_summary(raw)
    candidates.to_csv("results/kinase_selectivity_aware_candidates.csv", index=False)
    summary, selected = summarize_policies(candidates, tau)
    summary.to_csv("results/kinase_selectivity_aware_summary.csv", index=False)
    selected.to_csv("results/kinase_selectivity_aware_selected.csv", index=False)

    lines = [
        "# Selectivity-Aware Kinase Case",
        "",
        "## Protocol",
        "",
        f"- Targets: {len(KINASE_IDS)} kinase-like CrossDocked pockets.",
        f"- Candidates: top {args.top_per_target} PB+RC DiffSBDD molecules per kinase target.",
        f"- Cross-docking pairs attempted: {len(raw)}; successful: {int(raw.success.sum())}.",
        "- Selectivity margin = best off-target Vina score - target Vina score. Positive is target-preferred.",
        "- This is a computational selectivity oracle; it is not pharmacological profiling.",
        "",
        "## Policy Summary",
        "",
        "| Policy | N | Target best | Mean margin | Median rank | Risk >0.5 | Mean QED | dock_fast |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.policy} | {row.n} | {pct(row.target_is_best)} | {f4(row.mean_margin)} | "
            f"{f4(row.median_rank)} | {pct(row.risk_gt_0_5)} | {f4(row.mean_qed)} | {pct(row.dock_fast)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Plain RC is a pocket-reliability selector, not a selectivity optimizer.",
            "2. Adding an explicit off-target oracle gives a direct way to select target-preferred molecules within a kinase family.",
            "3. The safest manuscript framing is: RC controls geometric reliability; selectivity-aware RC is an optional objective layer.",
        ]
    )
    Path("experiments/SELECTIVITY_AWARE_KINASE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
