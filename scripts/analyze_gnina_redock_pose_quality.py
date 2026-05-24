import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign


RDLogger.DisableLog("rdApp.*")

CORE_COLUMNS = [
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


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * float(x):.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{float(x):.4f}"


def read_first_mol(path):
    if not path or not Path(path).exists() or Path(path).stat().st_size == 0:
        return None
    return next((m for m in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False) if m is not None), None)


def pose_rmsd(before, after):
    mol_a = read_first_mol(before)
    mol_b = read_first_mol(after)
    if mol_a is None or mol_b is None:
        return np.nan
    try:
        return float(rdMolAlign.GetBestRMS(Chem.RemoveHs(mol_a), Chem.RemoveHs(mol_b)))
    except Exception:
        return np.nan


def bool_all(df, cols):
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].fillna(False).astype(bool).all(axis=1)


def load_inputs(paths):
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df["input_scores_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No GNINA redock score CSVs found.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["gnina_redock_success"].fillna(False).astype(bool)].copy()
    df = df[df["gnina_redocked_sdf"].astype(str).map(lambda x: Path(x).exists())].copy()
    return df.reset_index(drop=True)


def summarize(df):
    rows = []
    for (source, policy), group in df.groupby(["source", "policy"], sort=True):
        rows.append(
            {
                "source": source,
                "policy": policy,
                "n": int(len(group)),
                "targets": int(group["target_id"].nunique()) if "target_id" in group else int(group["key"].nunique()),
                "before_dock_fast": float(group["dock_pose_pass_bool"].fillna(False).astype(bool).mean()) if "dock_pose_pass_bool" in group else np.nan,
                "after_molfast": float(group["after_redock_molfast_pass"].mean()),
                "after_protein_pass": float(group["after_redock_protein_pass"].mean()),
                "after_dock_fast": float(group["after_redock_dock_fast"].mean()),
                "redock_rmsd_mean": float(group["redock_rmsd"].mean()),
                "redock_rmsd_median": float(group["redock_rmsd"].median()),
                "cnnscore_mean": float(group["gnina_cnnscore"].mean()),
                "cnnaffinity_mean": float(group["gnina_cnnaffinity"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_md(summary, out_md):
    lines = [
        "# GNINA Redock Pose Quality",
        "",
        "## Protocol",
        "",
        "- Inputs are existing GNINA local-redocking outputs.",
        "- `redock_rmsd` is RDKit best RMSD between selected input pose and GNINA redocked pose.",
        "- `after_dock_fast` reruns PoseBusters `dock_fast` on the redocked pose against the same pocket.",
        "",
        "| Source | Policy | N | Targets | Before dock_fast | After mol_fast | After protein pass | After dock_fast | RMSD mean | RMSD median | CNNscore | CNNaffinity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["source", "policy"]).itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.policy} | {row.n} | {row.targets} | "
            f"{pct(row.before_dock_fast)} | {pct(row.after_molfast)} | {pct(row.after_protein_pass)} | {pct(row.after_dock_fast)} | "
            f"{f4(row.redock_rmsd_mean)} | {f4(row.redock_rmsd_median)} | {f4(row.cnnscore_mean)} | {f4(row.cnnaffinity_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This table separates neural docking plausibility from geometric survival after local optimization. A method that keeps high after-redock dock_fast while lowering risk is stronger than one that only improves pre-redock selection.",
        ]
    )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/gnina_redock_t25_scores.csv",
            "results/gnina_redock_pocketflow_t25_top4_scores.csv",
        ],
    )
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--out-csv", default="results/gnina_redock_pose_quality.csv")
    ap.add_argument("--out-summary", default="results/gnina_redock_pose_quality_summary.csv")
    ap.add_argument("--out-md", default="experiments/GNINA_REDOCK_POSE_QUALITY.md")
    args = ap.parse_args()

    df = load_inputs(args.inputs)
    df["redock_rmsd"] = [pose_rmsd(a, b) for a, b in zip(df["mol_pred"], df["gnina_redocked_sdf"])]
    buster = PoseBusters(config="dock_fast", max_workers=args.max_workers, chunk_size=args.chunk_size)
    pb = buster.bust_table(
        df[["gnina_redocked_sdf", "mol_cond"]].rename(columns={"gnina_redocked_sdf": "mol_pred"}),
        full_report=False,
    ).reset_index()
    pb = pb.rename(columns={"file": "gnina_redocked_sdf", "position": "redocked_position"})
    merged = df.merge(pb, on="gnina_redocked_sdf", how="left", suffixes=("", "_after"))
    merged["after_redock_molfast_pass"] = bool_all(merged, CORE_COLUMNS)
    merged["after_redock_protein_pass"] = bool_all(merged, PROTEIN_COLUMNS)
    merged["after_redock_dock_fast"] = bool_all(merged, CORE_COLUMNS + PROTEIN_COLUMNS)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    summary = summarize(merged)
    summary.to_csv(args.out_summary, index=False)
    write_md(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
