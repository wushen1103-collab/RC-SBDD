import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SETS = {
    "fullatom_cond": "crossdocked_fullatom_cond",
    "fullatom_joint": "crossdocked_fullatom_joint",
    "ca_cond": "crossdocked_ca_cond",
    "ca_joint": "crossdocked_ca_joint",
}

INTRAMOL_COLUMNS = [
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
]

PROTEIN_COLUMNS = [
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "volume_overlap_with_protein",
]


def test_key(row):
    return f"{Path(row.pocket_path).stem}_{Path(row.ligand_path).stem}"


def top_k(group, k, policy, native_p95):
    if policy == "qed":
        return group.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(k)
    if policy == "risk":
        return group.sort_values(["risk_prob", "qed"], ascending=[True, False]).head(k)
    if policy == "qed_minus_risk":
        scored = group.assign(selection_score=group["qed"] - group["risk_prob"])
        return scored.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k)
    if policy == "rc_select":
        safe = group[group["risk_prob"] <= native_p95].copy()
        safe = safe.sort_values(["qed", "risk_prob"], ascending=[False, True])
        if len(safe) >= k:
            return safe.head(k)
        fill = group.drop(safe.index).assign(selection_score=group["qed"] - group["risk_prob"])
        fill = fill.sort_values(["selection_score", "qed"], ascending=[False, False])
        return pd.concat([safe, fill.head(k - len(safe))], axis=0)
    raise ValueError(f"Unknown policy: {policy}")


def bool_mean(df, col):
    if col not in df.columns:
        return np.nan
    return float(df[col].fillna(False).astype(bool).mean())


def bool_all(df, cols):
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[present].fillna(False).astype(bool).all(axis=1)


def extract_one_mol(source_file, mol_index, out_path):
    supplier = Chem.SDMolSupplier(str(source_file), sanitize=False, removeHs=False)
    mol = None
    for idx, candidate in enumerate(supplier):
        if idx == int(mol_index):
            mol = candidate
            break
    if mol is None:
        raise ValueError(f"Could not read molecule index {mol_index} from {source_file}")
    mol.SetProp("_Name", f"{Path(source_file).stem}__mol{int(mol_index):03d}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return out_path


def build_selection(args, label, native_p95, key_to_pocket):
    stem = SETS[label]
    risk_path = Path(f"results/diffsbdd_zenodo_{stem}_risk_scores.csv")
    risk = pd.read_csv(risk_path)
    gen = risk[risk.kind == "generated"].copy()
    if any(policy.startswith("pb_") for policy in args.policies):
        pb_path = Path(args.molfast_dir) / f"{label}.csv"
        pb = pd.read_csv(pb_path)
        pb = pb.rename(columns={"file": "source_file", "position": "mol_index"})
        pb["molfast_core_pass"] = pb[INTRAMOL_COLUMNS].fillna(False).astype(bool).all(axis=1)
        gen = gen.merge(pb[["source_file", "mol_index", "molfast_core_pass"]], on=["source_file", "mol_index"], how="left")
        gen["molfast_core_pass"] = gen["molfast_core_pass"].fillna(False).astype(bool)
    if args.target_limit:
        keep_keys = sorted(gen["key"].unique())[: args.target_limit]
        gen = gen[gen["key"].isin(keep_keys)].copy()

    rows = []
    for policy in args.policies:
        base_policy = policy[3:] if policy.startswith("pb_") else policy
        policy_gen = gen[gen["molfast_core_pass"]].copy() if policy.startswith("pb_") else gen
        if policy_gen.empty:
            continue
        selected = pd.concat(
            [top_k(group, args.k, base_policy, native_p95) for _, group in policy_gen.groupby("key", sort=True)],
            axis=0,
        )
        for rank, row in selected.groupby("key", sort=True).cumcount().items():
            selected.at[rank, "policy_rank"] = int(row)
        for row in selected.itertuples(index=False):
            if row.key not in key_to_pocket:
                raise KeyError(f"No pocket path for key {row.key}")
            rel = Path(row.source_file)
            out_name = f"{row.key}__m{int(row.mol_index):03d}.sdf"
            out_path = Path(args.work_dir) / "extracted" / label / policy / out_name
            extract_one_mol(rel, int(row.mol_index), out_path)
            rows.append(
                {
                    "set": label,
                    "policy": policy,
                    "key": row.key,
                    "mol_index": int(row.mol_index),
                    "source_file": str(row.source_file),
                    "mol_pred": str(out_path),
                    "mol_cond": str(Path(args.raw_root) / key_to_pocket[row.key]),
                    "risk_prob": float(row.risk_prob),
                    "qed": float(row.qed),
                    "lp_min": float(row.lp_min),
                    "center_dist": float(row.center_dist),
                    "molfast_core_pass": bool(getattr(row, "molfast_core_pass", False)),
                }
            )
    return pd.DataFrame(rows)


def summarize(df):
    row = {
        "n": int(len(df)),
        "targets": int(df["key"].nunique()),
        "risk_mean": float(df["risk_prob"].mean()),
        "qed_mean": float(df["qed"].mean()),
    }
    for col in INTRAMOL_COLUMNS + PROTEIN_COLUMNS:
        row[col] = bool_mean(df, col)
    row["intramol_pass"] = float(df["intramol_pass"].astype(bool).mean()) if "intramol_pass" in df else np.nan
    row["protein_pass"] = float(df["protein_pass"].astype(bool).mean()) if "protein_pass" in df else np.nan
    row["dock_pose_pass"] = float(df["dock_pose_pass"].astype(bool).mean()) if "dock_pose_pass" in df else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["fullatom_cond"])
    ap.add_argument("--policies", nargs="+", default=["qed", "rc_select"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target-limit", type=int, default=25, help="0 means all targets.")
    ap.add_argument("--index", default="data/processed/if3-crossdocked2020/index_test.csv")
    ap.add_argument("--raw-root", default="data/raw/if3-crossdocked2020/crossdocked_pocket10")
    ap.add_argument("--molfast-dir", default="results/posebusters_molfast")
    ap.add_argument("--work-dir", default="results/posebusters_dockfast_selection")
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--out-csv", default="results/posebusters_dockfast_selection.csv")
    ap.add_argument("--out-summary-csv", default="results/posebusters_dockfast_selection_summary.csv")
    ap.add_argument("--out-md", default="logs/posebusters_dockfast_selection_summary.md")
    args = ap.parse_args()

    index = pd.read_csv(args.index)
    key_to_pocket = {test_key(row): row.pocket_path for row in index.itertuples(index=False)}
    native_scores = pd.read_csv("results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
    native_p95 = float(native_scores[native_scores.kind == "native"]["risk_prob"].quantile(0.95))

    selected_tables = [build_selection(args, label, native_p95, key_to_pocket) for label in args.sets]
    selected = pd.concat(selected_tables, axis=0).reset_index(drop=True)
    table = selected[["mol_pred", "mol_cond"]].copy()

    buster = PoseBusters(config="dock_fast", max_workers=args.max_workers, chunk_size=args.chunk_size)
    pb = buster.bust_table(table, full_report=False).reset_index()
    pb = pb.rename(columns={"file": "mol_pred", "position": "posebusters_position"})

    merged = selected.merge(pb, on="mol_pred", how="left")
    merged["intramol_pass"] = bool_all(merged, INTRAMOL_COLUMNS)
    merged["protein_pass"] = bool_all(merged, PROTEIN_COLUMNS)
    merged["dock_pose_pass"] = bool_all(merged, INTRAMOL_COLUMNS + PROTEIN_COLUMNS)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    rows = []
    for (label, policy), group in merged.groupby(["set", "policy"], sort=True):
        row = {"set": label, "policy": policy, "k": args.k, "target_limit": args.target_limit}
        row.update(summarize(group))
        rows.append(row)
    summary = pd.DataFrame(rows)
    Path(args.out_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary_csv, index=False)

    lines = [
        "# PoseBusters dock_fast Selection Summary",
        "",
        f"Selection uses K={args.k}; target limit={args.target_limit or 'all'}; native risk p95={native_p95:.4f}.",
        "",
        "| Set | Policy | N | Targets | Mean risk | Mean QED | Intramol pass | Protein pass | Dock pose pass | Protein max-dist | Min-dist protein | Vol overlap protein |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['set']} | {row['policy']} | {row['n']} | {row['targets']} | "
            f"{row['risk_mean']:.4f} | {row['qed_mean']:.4f} | "
            f"{100*row['intramol_pass']:.1f}% | {100*row['protein_pass']:.1f}% | {100*row['dock_pose_pass']:.1f}% | "
            f"{100*row.get('protein-ligand_maximum_distance', np.nan):.1f}% | "
            f"{100*row.get('minimum_distance_to_protein', np.nan):.1f}% | "
            f"{100*row.get('volume_overlap_with_protein', np.nan):.1f}% |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: `dock_fast` checks local protein-pocket geometry for extracted generated poses. It is an external plausibility check, not a replacement for the calibrated interaction-risk proxy.",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
