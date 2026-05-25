import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def mol_to_smiles(path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    mol = next((mol for mol in supplier if mol is not None), None)
    if mol is None:
        raise ValueError(f"Cannot read molecule from {path}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-csv", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--policies", nargs="*", default=[])
    ap.add_argument("--first-per-target", action="store_true")
    ap.add_argument("--max-unique-smiles", type=int, default=0)
    ap.add_argument("--out-map", required=True)
    ap.add_argument("--out-smiles", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.selection_csv)
    if args.policies:
        df = df[df["policy"].isin(args.policies)].copy()
    if args.first_per_target:
        df = df.sort_values(
            ["policy", "key", "qed", "risk_prob"],
            ascending=[True, True, False, True],
        )
        df = df.groupby(["policy", "key"], sort=True).head(1).copy()
    rows, failures = [], []
    for idx, row in enumerate(df.itertuples(index=False)):
        try:
            rows.append(
                {
                    "selection_row": idx,
                    "source": args.source,
                    "policy": row.policy,
                    "target_id": row.key,
                    "key": row.key,
                    "data_id": getattr(row, "data_id", ""),
                    "mol_index": getattr(row, "mol_index", ""),
                    "mol_pred": row.mol_pred,
                    "mol_cond": row.mol_cond,
                    "smiles": mol_to_smiles(row.mol_pred),
                    "risk_prob": row.risk_prob,
                    "qed": row.qed,
                    "dock_pose_pass": getattr(row, "dock_pose_pass", pd.NA),
                }
            )
        except Exception as exc:
            failures.append({"selection_row": idx, "mol_pred": getattr(row, "mol_pred", ""), "error": str(exc)})

    mapped = pd.DataFrame(rows)
    unique = mapped.drop_duplicates("smiles").sort_values("smiles").reset_index(drop=True)
    if args.max_unique_smiles > 0:
        keep = set(unique.head(args.max_unique_smiles)["smiles"])
        mapped = mapped[mapped["smiles"].isin(keep)].copy()
        unique = unique[unique["smiles"].isin(keep)].copy()
    smiles_to_id = {smiles: f"mol_{i:06d}" for i, smiles in enumerate(unique["smiles"])}
    mapped["aizynth_id"] = mapped["smiles"].map(smiles_to_id)
    Path(args.out_map).parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(args.out_map, index=False)
    with open(args.out_smiles, "w", encoding="utf-8") as handle:
        for smiles, mol_id in smiles_to_id.items():
            handle.write(f"{smiles} {mol_id}\n")
    if failures:
        pd.DataFrame(failures).to_csv(Path(args.out_map).with_suffix(".failures.csv"), index=False)
    print({"rows": len(mapped), "unique_smiles": len(unique), "failures": len(failures)})


if __name__ == "__main__":
    main()
