import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


MAIN_POLICIES = ["qed", "rc_select", "pb_qed", "pb_rc_select"]


def mol_to_smiles(path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    mol = next((mol for mol in supplier if mol is not None), None)
    if mol is None:
        raise ValueError(f"Cannot read molecule from {path}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_diffsbdd(top_per_target):
    df = pd.concat(
        [
            pd.read_csv("results/posebusters_dockfast_selection.csv"),
            pd.read_csv("results/posebusters_dockfast_pb_selection.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    df = df[(df["set"] == "fullatom_cond") & (df["policy"].isin(MAIN_POLICIES))].copy()
    df["source"] = "DiffSBDD_official"
    df["target_id"] = df["key"]
    return select_top(df, top_per_target)


def load_pocket2mol(top_per_target):
    df = pd.read_csv("results/pocket2mol_crossgen_n16_ext_dockfast_selection.csv")
    df = df[df["policy"].isin(MAIN_POLICIES)].copy()
    df["source"] = "Pocket2Mol_transfer"
    df["target_id"] = df["key"]
    return select_top(df, top_per_target)


def load_syncguide(top_per_target):
    df = pd.read_csv("results/syncguide_t1000_n16_dockfast_selection.csv")
    df = df[df["policy"].isin(MAIN_POLICIES)].copy()
    df["source"] = "SYNC-Guide"
    df["target_id"] = df["key"]
    return select_top(df, top_per_target)


def load_pocketflow(top_per_target):
    df = pd.read_csv("results/pocketflow_crossdock_n16_dockfast_selection.csv")
    df = df[df["policy"].isin(MAIN_POLICIES)].copy()
    df["source"] = "PocketFlow"
    df["target_id"] = df["key"]
    return select_top(df, top_per_target)


def load_molpilot(top_per_target):
    df = pd.read_csv("results/molpilot_crossdock_t50_n16_dockfast_selection.csv")
    df = df[df["policy"].isin(MAIN_POLICIES)].copy()
    df["source"] = "MolPilot"
    df["target_id"] = df["key"]
    return select_top(df, top_per_target)


def select_top(df, top_per_target):
    df = df.sort_values(["source", "policy", "target_id", "qed", "risk_prob"], ascending=[True, True, True, False, True])
    return df.groupby(["source", "policy", "target_id"], sort=True).head(top_per_target).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-target", type=int, default=1)
    ap.add_argument("--sources", nargs="+", default=["diffsbdd", "pocket2mol"])
    ap.add_argument("--max-unique-smiles", type=int, default=0)
    ap.add_argument("--out-map", default="results/aizynthfinder_selection_map.csv")
    ap.add_argument("--out-smiles", default="results/aizynthfinder_selection_smiles.txt")
    args = ap.parse_args()

    parts = []
    if "diffsbdd" in args.sources:
        parts.append(load_diffsbdd(args.top_per_target))
    if "pocket2mol" in args.sources:
        parts.append(load_pocket2mol(args.top_per_target))
    if "syncguide" in args.sources or "sync" in args.sources:
        parts.append(load_syncguide(args.top_per_target))
    if "pocketflow" in args.sources:
        parts.append(load_pocketflow(args.top_per_target))
    if "molpilot" in args.sources:
        parts.append(load_molpilot(args.top_per_target))
    if not parts:
        raise ValueError("No source selected")
    df = pd.concat(parts, ignore_index=True, sort=False)

    rows = []
    failures = []
    for idx, row in enumerate(df.itertuples(index=False)):
        try:
            smiles = mol_to_smiles(row.mol_pred)
            rows.append(
                {
                    "selection_row": idx,
                    "source": row.source,
                    "policy": row.policy,
                    "target_id": row.target_id,
                    "key": getattr(row, "key", ""),
                    "data_id": getattr(row, "data_id", ""),
                    "mol_index": getattr(row, "mol_index", ""),
                    "mol_pred": row.mol_pred,
                    "mol_cond": row.mol_cond,
                    "smiles": smiles,
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

    out_map = Path(args.out_map)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out_map, index=False)

    out_smiles = Path(args.out_smiles)
    out_smiles.parent.mkdir(parents=True, exist_ok=True)
    with open(out_smiles, "w", encoding="utf-8") as handle:
        for smiles, mol_id in smiles_to_id.items():
            handle.write(f"{smiles} {mol_id}\n")

    print(
        {
            "selection_rows": len(df),
            "mapped_rows": len(mapped),
            "unique_smiles": len(unique),
            "failures": len(failures),
            "out_map": str(out_map),
            "out_smiles": str(out_smiles),
        }
    )
    if failures:
        pd.DataFrame(failures).to_csv(out_map.with_suffix(".failures.csv"), index=False)


if __name__ == "__main__":
    main()
