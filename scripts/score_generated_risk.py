import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

from rcsbdd.features.interaction import feature_names, featurize_interaction

RDLogger.DisableLog("rdApp.*")

ATOM_MAP = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "B": 4,
    "BR": 5,
    "CL": 6,
    "P": 7,
    "I": 8,
    "F": 9,
}


def normalize_stem(stem):
    return stem.replace("-", "_")


def test_key(row):
    return f"{Path(row.pocket_path).stem}_{Path(row.ligand_path).stem}"


def read_pocket_positions(path, center=None, radius=None):
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            element = line[76:78].strip().upper()
            atom_name = line[12:16].strip().upper()
            if not element:
                element = "".join(ch for ch in atom_name if ch.isalpha())[:2].upper()
            if element.startswith("H") or atom_name.startswith("H"):
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    if not coords:
        raise ValueError(f"No pocket atoms parsed from {path}")
    arr = np.asarray(coords, dtype=np.float32)
    if center is not None and radius is not None:
        center = np.asarray(center, dtype=np.float32).reshape(1, 3)
        keep = np.linalg.norm(arr - center, axis=1) < float(radius)
        arr = arr[keep]
        if arr.shape[0] == 0:
            raise ValueError(f"No pocket atoms remained after center filter for {path}")
    return arr


def atom_type(atom):
    symbol = atom.GetSymbol().upper()
    return ATOM_MAP.get(symbol, 10)


def mol_heavy_positions(mol):
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    if not coords:
        raise ValueError("molecule has no heavy atoms")
    return np.asarray(coords, dtype=np.float32)


def mol_to_item(mol, pock_pos):
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = mol.GetConformer()
    lig_pos = []
    lig_atom_type = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        lig_pos.append([pos.x, pos.y, pos.z])
        lig_atom_type.append(atom_type(atom))
    if not lig_pos:
        raise ValueError("molecule has no heavy atoms")
    return {
        "lig_pos": np.asarray(lig_pos, dtype=np.float32),
        "lig_atom_type": np.asarray(lig_atom_type, dtype=np.int64),
        "pock_pos": pock_pos,
    }


def sanitize_copy(mol):
    mol_copy = Chem.Mol(mol)
    Chem.SanitizeMol(mol_copy)
    return mol_copy


def descriptor_row(mol):
    return {
        "qed": float(QED.qed(mol)),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def score_mol(model, mol, pock_pos, names):
    item = mol_to_item(mol, pock_pos)
    x = featurize_interaction(item, include_pocket_feat=False)
    if x.shape[0] != len(names):
        raise ValueError(f"feature length mismatch: got {x.shape[0]}, expected {len(names)}")
    prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    named = {name: float(value) for name, value in zip(names, x)}
    return prob, named


def describe(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "frac_gt_0_5": float(np.mean(arr > 0.5)),
        "frac_gt_0_8": float(np.mean(arr > 0.8)),
    }


def load_model(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("include_pocket_feat", True):
        raise ValueError("Model includes LMDB-only pocket features; train with --geometry-only before scoring raw structures.")
    names = payload.get("feature_names") or feature_names(include_pocket_feat=False)
    expected = feature_names(include_pocket_feat=False)
    if list(names) != list(expected):
        raise ValueError("Model feature names do not match current geometry-only interaction features.")
    return payload["model"], list(names), payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/risk_proxy_hard_geom_logreg/model.pkl")
    ap.add_argument("--sdf-dir", default="data/generated/diffsbdd_zenodo/crossdocked_fullatom_cond")
    ap.add_argument("--index", default="data/processed/if3-crossdocked2020/index_test.csv")
    ap.add_argument("--raw-root", default="data/raw/if3-crossdocked2020/crossdocked_pocket10")
    ap.add_argument("--out-csv", default="results/diffsbdd_zenodo_risk_scores.csv")
    ap.add_argument("--out-json", default="logs/diffsbdd_zenodo_risk_summary.json")
    ap.add_argument("--out-md", default="logs/diffsbdd_zenodo_risk_summary.md")
    ap.add_argument("--max-generated-per-target", type=int, default=0, help="0 scores every generated molecule.")
    ap.add_argument("--pocket-center-cutoff", type=float, default=10.0, help="Filter pocket atoms by distance to the native/reference ligand center; matches the IF3 LMDB schema.")
    args = ap.parse_args()

    model, names, payload = load_model(args.model)
    index = pd.read_csv(args.index)
    row_by_key = {test_key(row): row for row in index.itertuples(index=False)}
    raw_root = Path(args.raw_root)
    context_by_key = {}

    rows = []
    failures = []

    for key, row in row_by_key.items():
        pocket_path = raw_root / row.pocket_path
        ligand_path = raw_root / row.ligand_path
        try:
            supplier = Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=True)
            mol = next((m for m in supplier if m is not None), None)
            if mol is None:
                raise ValueError("native ligand not readable")
            mol_san = sanitize_copy(mol)
            native_center = mol_heavy_positions(mol_san).mean(axis=0)
            pock_pos = read_pocket_positions(pocket_path, center=native_center, radius=args.pocket_center_cutoff)
            context_by_key[key] = {"pock_pos": pock_pos, "native_center": native_center}
            prob, feat = score_mol(model, mol_san, pock_pos, names)
            desc = descriptor_row(mol_san)
            rows.append({
                "kind": "native",
                "key": key,
                "mol_index": 0,
                "source_file": str(ligand_path),
                "risk_prob": prob,
                "lp_min": feat["lp_min"],
                "center_dist": feat["center_dist"],
                "contacts_lt_4_0_per_lig": feat["contacts_lt_4_0_per_lig"],
                "clash_lt_1_5_per_lig": feat["clash_lt_1_5_per_lig"],
                "pock_n": feat["pock_n"],
                **desc,
            })
        except Exception as exc:
            failures.append({"kind": "native", "key": key, "error": str(exc)})

    sdf_dir = Path(args.sdf_dir)
    for sdf_path in sorted(sdf_dir.glob("*.sdf")):
        if sdf_path.name.startswith("._"):
            continue
        key = normalize_stem(sdf_path.stem.removesuffix("_gen"))
        row = row_by_key.get(key)
        if row is None:
            failures.append({"kind": "generated", "file": str(sdf_path), "error": "no matching test row"})
            continue
        ctx = context_by_key.get(key)
        if ctx is None:
            failures.append({"kind": "generated", "key": key, "file": str(sdf_path), "error": "native context unavailable"})
            continue
        pock_pos = ctx["pock_pos"]
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=True)
        kept = 0
        for mol_index, mol in enumerate(supplier):
            if args.max_generated_per_target and kept >= args.max_generated_per_target:
                break
            if mol is None:
                failures.append({"kind": "generated", "key": key, "file": str(sdf_path), "mol_index": mol_index, "error": "not readable"})
                continue
            try:
                mol_san = sanitize_copy(mol)
                prob, feat = score_mol(model, mol_san, pock_pos, names)
                desc = descriptor_row(mol_san)
            except Exception as exc:
                failures.append({"kind": "generated", "key": key, "file": str(sdf_path), "mol_index": mol_index, "error": str(exc)})
                continue
            rows.append({
                "kind": "generated",
                "key": key,
                "mol_index": mol_index,
                "source_file": str(sdf_path),
                "risk_prob": prob,
                "lp_min": feat["lp_min"],
                "center_dist": feat["center_dist"],
                "contacts_lt_4_0_per_lig": feat["contacts_lt_4_0_per_lig"],
                "clash_lt_1_5_per_lig": feat["clash_lt_1_5_per_lig"],
                "pock_n": feat["pock_n"],
                **desc,
            })
            kept += 1

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    native = df[df.kind == "native"] if not df.empty else pd.DataFrame()
    generated = df[df.kind == "generated"] if not df.empty else pd.DataFrame()
    per_key_generated = generated.groupby("key")["risk_prob"].mean() if not generated.empty else pd.Series(dtype=float)
    native_by_key = native.set_index("key")["risk_prob"] if not native.empty else pd.Series(dtype=float)
    paired = pd.concat([native_by_key.rename("native"), per_key_generated.rename("generated_mean")], axis=1).dropna()
    paired_delta = (paired["generated_mean"] - paired["native"]).tolist() if not paired.empty else []

    summary = {
        "model": str(args.model),
        "model_modes": payload.get("modes"),
        "feature_dim": len(names),
        "test_targets": int(len(index)),
        "rows": int(len(df)),
        "failures": failures[:50],
        "failure_count": int(len(failures)),
        "native_risk": describe(native["risk_prob"].tolist() if not native.empty else []),
        "generated_risk": describe(generated["risk_prob"].tolist() if not generated.empty else []),
        "generated_target_mean_risk": describe(per_key_generated.tolist()),
        "paired_generated_mean_minus_native": describe(paired_delta),
        "qed_generated": describe(generated["qed"].tolist() if not generated.empty else []),
        "qed_native": describe(native["qed"].tolist() if not native.empty else []),
        "lp_min_generated": describe(generated["lp_min"].tolist() if not generated.empty else []),
        "lp_min_native": describe(native["lp_min"].tolist() if not native.empty else []),
        "pock_n_native": describe(native["pock_n"].tolist() if not native.empty else []),
        "targets_with_generated": int(generated["key"].nunique()) if not generated.empty else 0,
        "generated_molecules": int(len(generated)),
        "native_molecules": int(len(native)),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# DiffSBDD Zenodo Risk Summary",
        "",
        f"- model: `{args.model}`",
        f"- generated molecules scored: {summary['generated_molecules']}",
        f"- native ligands scored: {summary['native_molecules']}",
        f"- failures: {summary['failure_count']}",
        "",
        "| group | n | mean risk | median | p90 | p95 | frac > 0.5 | frac > 0.8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in [("native", summary["native_risk"]), ("generated", summary["generated_risk"]), ("generated target mean", summary["generated_target_mean_risk"]), ("paired delta", summary["paired_generated_mean_minus_native"])]:
        md.append(
            f"| {label} | {stats.get('n', 0)} | {stats.get('mean', float('nan')):.4f} | {stats.get('median', float('nan')):.4f} | "
            f"{stats.get('p90', float('nan')):.4f} | {stats.get('p95', float('nan')):.4f} | {stats.get('frac_gt_0_5', float('nan')):.4f} | {stats.get('frac_gt_0_8', float('nan')):.4f} |"
        )
    md.append("")
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
