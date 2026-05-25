from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


POOLS = [
    ("DiffSBDD_official", "results/multiobjective_selection.csv", None),
    ("PocketFlow_CrossDock", "results/pocketflow_crossdock_n16_dockfast_selection.csv", None),
    ("BindingMOAD_PocketFlow_v100", "results/bindingmoad_pocketflow_n16_v100_dockfast_selection.csv", "data/processed/bindingmoad_pocketflow_v100/manifest.csv"),
    ("BindingMOAD_MolPilot_v50", "results/bindingmoad_molpilot_v50_dockfast_selection.csv", "data/processed/bindingmoad_pocketflow_v100/manifest.csv"),
]

POLICIES = ["qed", "pb_qed", "pb_rc_select", "rc_select", "qed_minus_risk", "pb_qed_minus_risk"]


def read_first_mol(path: str | Path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    return next((mol for mol in suppl if mol is not None), None)


def fp(mol):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def tanimoto(a, b) -> float:
    if a is None or b is None:
        return np.nan
    return float(DataStructs.TanimotoSimilarity(a, b))


def crossdock_native_from_pocket(pocket_path: str) -> str:
    p = Path(str(pocket_path))
    stem = p.stem
    if stem.endswith("_pocket10"):
        return str(p.with_name(stem.replace("_pocket10", "") + ".sdf"))
    return str(p.with_suffix(".sdf"))


def native_map_from_manifest(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    m = pd.read_csv(path)
    return dict(zip(m["key"].astype(str), m["native_ligand_path"].astype(str)))


def load_pool(name: str, selection_csv: str, manifest_csv: str | None) -> pd.DataFrame:
    df = pd.read_csv(selection_csv)
    df = df[df["policy"].astype(str).isin(POLICIES)].copy()
    df["pool"] = name
    mapping = native_map_from_manifest(manifest_csv)
    if mapping:
        df["native_ligand_path"] = df["key"].astype(str).map(mapping)
    else:
        df["native_ligand_path"] = df["mol_cond"].astype(str).map(crossdock_native_from_pocket)
    return df


def compute_pool(df: pd.DataFrame) -> pd.DataFrame:
    native_fp_cache = {}
    mol_fp_cache = {}
    rows = []
    for row in df.itertuples(index=False):
        native_path = str(row.native_ligand_path)
        mol_path = str(row.mol_pred)
        if native_path not in native_fp_cache:
            native_fp_cache[native_path] = fp(read_first_mol(native_path))
        if mol_path not in mol_fp_cache:
            mol_fp_cache[mol_path] = fp(read_first_mol(mol_path))
        rows.append(
            {
                "pool": row.pool,
                "policy": row.policy,
                "key": row.key,
                "mol_index": getattr(row, "mol_index", np.nan),
                "mol_pred": mol_path,
                "native_ligand_path": native_path,
                "known_ligand_tanimoto": tanimoto(mol_fp_cache[mol_path], native_fp_cache[native_path]),
                "dock_pose_pass": bool(getattr(row, "dock_pose_pass", False)),
                "risk_prob": float(getattr(row, "risk_prob", np.nan)),
                "qed": float(getattr(row, "qed", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    per_target = (
        rows.groupby(["pool", "policy", "key"], as_index=False)
        .agg(
            max_known_ligand_tanimoto=("known_ligand_tanimoto", "max"),
            mean_known_ligand_tanimoto=("known_ligand_tanimoto", "mean"),
            any_dock_fast=("dock_pose_pass", "max"),
            mean_risk=("risk_prob", "mean"),
            mean_qed=("qed", "mean"),
        )
    )
    return per_target.groupby(["pool", "policy"], as_index=False).agg(
        targets=("key", "nunique"),
        max_tanimoto_mean=("max_known_ligand_tanimoto", "mean"),
        max_tanimoto_median=("max_known_ligand_tanimoto", "median"),
        mean_tanimoto=("mean_known_ligand_tanimoto", "mean"),
        target_any_dock_fast=("any_dock_fast", "mean"),
        risk_mean=("mean_risk", "mean"),
        qed_mean=("mean_qed", "mean"),
    )


def write_report(summary: pd.DataFrame, out_md: str) -> None:
    lines = [
        "# Known-Ligand Similarity Enrichment Audit",
        "",
        "## Protocol",
        "",
        "- Each selected molecule is compared to the matched co-crystal/native ligand using ECFP4 Tanimoto similarity.",
        "- The analysis is retrospective and computational; it is a known-ligand proximity proxy, not wet activity validation.",
        "- Metrics are summarized at target level by the maximum selected-molecule similarity per target.",
        "",
        "## Summary",
        "",
        "| Pool | Policy | Targets | Max-Tanimoto mean | Max-Tanimoto median | Mean Tanimoto | Any dock-fast | Risk | QED |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {p: i for i, p in enumerate(POLICIES)}
    summary = summary.assign(_order=summary["policy"].map(order).fillna(99))
    for row in summary.sort_values(["pool", "_order"]).itertuples(index=False):
        lines.append(
            f"| {row.pool} | {row.policy} | {row.targets} | {row.max_tanimoto_mean:.3f} | "
            f"{row.max_tanimoto_median:.3f} | {row.mean_tanimoto:.3f} | {row.target_any_dock_fast:.3f} | "
            f"{row.risk_mean:.3f} | {row.qed_mean:.3f} |"
        )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frames = []
    for name, selection_csv, manifest_csv in POOLS:
        if not Path(selection_csv).exists():
            continue
        frames.append(compute_pool(load_pool(name, selection_csv, manifest_csv)))
    rows = pd.concat(frames, ignore_index=True)
    summary = summarize(rows)
    rows.to_csv("results/known_ligand_similarity_enrichment_rows.csv", index=False)
    summary.to_csv("results/known_ligand_similarity_enrichment_summary.csv", index=False)
    write_report(summary, "experiments/KNOWN_LIGAND_SIMILARITY_ENRICHMENT.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
