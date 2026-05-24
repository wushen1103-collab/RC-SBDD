import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score_generated_risk import descriptor_row, load_model, read_pocket_positions, sanitize_copy, score_mol


RDLogger.DisableLog("rdApp.*")


def score_sdf(model, names, key, data_id, pocket_path, sdf_path, kind):
    pock_pos = read_pocket_positions(pocket_path)
    rows, failures = [], []
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=True)
    for mol_index, mol in enumerate(supplier):
        if mol is None:
            failures.append({"key": key, "data_id": data_id, "kind": kind, "source_file": str(sdf_path), "mol_index": mol_index, "error": "unreadable mol"})
            continue
        try:
            mol_san = sanitize_copy(mol)
            prob, feat = score_mol(model, mol_san, pock_pos, names)
            rows.append(
                {
                    "kind": kind,
                    "data_id": int(data_id),
                    "key": key,
                    "mol_index": int(mol_index),
                    "source_file": str(sdf_path),
                    "mol_cond": str(pocket_path),
                    "risk_prob": float(prob),
                    "lp_min": feat["lp_min"],
                    "center_dist": feat["center_dist"],
                    "contacts_lt_4_0_per_lig": feat["contacts_lt_4_0_per_lig"],
                    "clash_lt_1_5_per_lig": feat["clash_lt_1_5_per_lig"],
                    "pock_n": feat["pock_n"],
                    **descriptor_row(mol_san),
                }
            )
        except Exception as exc:
            failures.append({"key": key, "data_id": data_id, "kind": kind, "source_file": str(sdf_path), "mol_index": mol_index, "error": str(exc)})
    return rows, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sdf-dir", required=True)
    ap.add_argument("--model", default="results/risk_proxy_hard_geom_logreg/model.pkl")
    ap.add_argument("--include-native", action="store_true")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    model, names, _ = load_model(args.model)
    manifest = pd.read_csv(args.manifest)
    rows, failures = [], []
    sdf_dir = Path(args.sdf_dir)
    for rec in manifest.itertuples(index=False):
        data_id = int(rec.data_id)
        key = str(rec.key)
        pocket = Path(rec.pocket_path)
        gen_sdf = sdf_dir / f"{key}_gen.sdf"
        if args.include_native and hasattr(rec, "native_ligand_path") and Path(rec.native_ligand_path).exists():
            r, f = score_sdf(model, names, key, data_id, pocket, Path(rec.native_ligand_path), "native")
            rows.extend(r)
            failures.extend(f)
        if gen_sdf.exists():
            r, f = score_sdf(model, names, key, data_id, pocket, gen_sdf, "generated")
            rows.extend(r)
            failures.extend(f)
        else:
            failures.append({"key": key, "data_id": data_id, "kind": "generated", "source_file": str(gen_sdf), "error": "generated SDF missing"})
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = {
        "targets": int(out["data_id"].nunique()) if len(out) else 0,
        "rows": int(len(out)),
        "generated_rows": int((out["kind"] == "generated").sum()) if len(out) else 0,
        "native_rows": int((out["kind"] == "native").sum()) if len(out) else 0,
        "failures": len(failures),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"summary": summary, "failures": failures}, indent=2), encoding="utf-8")
    lines = [
        "# Manifest SDF Risk Scoring",
        "",
        f"- Targets scored: {summary['targets']}.",
        f"- Rows: {summary['rows']}; generated rows: {summary['generated_rows']}; native rows: {summary['native_rows']}.",
        f"- Failures: {summary['failures']}.",
    ]
    if len(out):
        gen = out[out["kind"] == "generated"]
        lines.extend(
            [
                f"- Generated mean risk: {gen.risk_prob.mean():.4f}.",
                f"- Generated risk >0.5: {100 * (gen.risk_prob > 0.5).mean():.1f}%.",
                f"- Generated mean QED: {gen.qed.mean():.4f}.",
            ]
        )
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
