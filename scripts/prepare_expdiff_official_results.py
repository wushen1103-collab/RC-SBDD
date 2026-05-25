from __future__ import annotations

import argparse
import re
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
SDF_RE = re.compile(r"^sdf_results/sdf_(\d+)/res_(\d+)\.sdf$")


def read_first_mol(payload: bytes) -> Chem.Mol | None:
    supplier = Chem.ForwardSDMolSupplier(BytesIO(payload), sanitize=False, removeHs=False)
    return next((mol for mol in supplier if mol is not None), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/molcraft_crossdock_t100/manifest.csv")
    ap.add_argument("--sdf-zip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest).sort_values("data_id").reset_index(drop=True)
    by_target: dict[int, list[tuple[int, str]]] = {}
    with zipfile.ZipFile(args.sdf_zip) as zf:
        for name in zf.namelist():
            match = SDF_RE.match(name)
            if not match:
                continue
            data_id = int(match.group(1))
            res_id = int(match.group(2))
            by_target.setdefault(data_id, []).append((res_id, name))

        rows = []
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in manifest.itertuples(index=False):
            data_id = int(rec.data_id)
            key = str(rec.key)
            target_files = sorted(by_target.get(data_id, []))
            out_sdf = out_dir / f"{key}_gen.sdf"
            writer = Chem.SDWriter(str(out_sdf))
            written = 0
            for res_id, name in target_files:
                mol = read_first_mol(zf.read(name))
                if mol is None:
                    rows.append(
                        {
                            "data_id": data_id,
                            "key": key,
                            "expdiff_res_id": res_id,
                            "source_zip_member": name,
                            "written_mol_index": "",
                            "status": "unreadable",
                        }
                    )
                    continue
                mol.SetProp("_Name", f"expdiff_data{data_id:03d}_res{res_id:03d}")
                mol.SetIntProp("expdiff_data_id", data_id)
                mol.SetIntProp("expdiff_res_id", res_id)
                writer.write(mol)
                rows.append(
                    {
                        "data_id": data_id,
                        "key": key,
                        "expdiff_res_id": res_id,
                        "source_zip_member": name,
                        "written_mol_index": written,
                        "status": "written",
                    }
                )
                written += 1
            writer.close()
            if written == 0 and out_sdf.exists():
                out_sdf.unlink()

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_manifest, index=False)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(out_csv, index=False)
    summary = {
        "targets_in_manifest": int(len(manifest)),
        "targets_with_generated_mols": int(meta[meta["status"].eq("written")]["data_id"].nunique()) if len(meta) else 0,
        "written_molecules": int(meta["status"].eq("written").sum()) if len(meta) else 0,
        "unreadable_molecules": int(meta["status"].eq("unreadable").sum()) if len(meta) else 0,
    }
    print(summary)


if __name__ == "__main__":
    main()
