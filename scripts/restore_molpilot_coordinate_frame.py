import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def pdb_heavy_centroid(path):
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip().upper()
            atom_name = line[12:16].strip().upper()
            if element.startswith("H") or atom_name.startswith("H"):
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No pocket atoms parsed from {path}")
    return np.asarray(coords, dtype=float).mean(axis=0)


def mol_centroid(mol):
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return np.asarray(coords, dtype=float).mean(axis=0)


def translate_mol(mol, offset):
    restored = Chem.Mol(mol)
    conf = restored.GetConformer()
    for atom_idx in range(restored.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, (pos.x - offset[0], pos.y - offset[1], pos.z - offset[2]))
    return restored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sdf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    source_dir = Path(args.sdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in manifest.itertuples(index=False):
        source_path = source_dir / f"{rec.key}_gen.sdf"
        if not source_path.exists():
            rows.append({"data_id": int(rec.data_id), "key": rec.key, "records": 0, "status": "missing"})
            continue
        offset = pdb_heavy_centroid(rec.pocket_path)
        supplier = Chem.SDMolSupplier(str(source_path), sanitize=False, removeHs=False)
        output_path = out_dir / source_path.name
        writer = Chem.SDWriter(str(output_path))
        count = 0
        before = []
        after = []
        for mol in supplier:
            if mol is None:
                continue
            restored = translate_mol(mol, offset)
            writer.write(restored)
            before.append(float(np.linalg.norm(mol_centroid(mol) - offset)))
            after.append(float(np.linalg.norm(mol_centroid(restored) - offset)))
            count += 1
        writer.close()
        rows.append(
            {
                "data_id": int(rec.data_id),
                "key": rec.key,
                "records": count,
                "offset_x": offset[0],
                "offset_y": offset[1],
                "offset_z": offset[2],
                "center_dist_before_mean": np.mean(before) if before else np.nan,
                "center_dist_after_mean": np.mean(after) if after else np.nan,
                "status": "ok" if count else "empty",
            }
        )
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out[["records", "center_dist_before_mean", "center_dist_after_mean"]].describe().to_string())


if __name__ == "__main__":
    main()
