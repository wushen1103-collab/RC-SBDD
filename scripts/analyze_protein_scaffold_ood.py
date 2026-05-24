import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


TRAILING_NUMBERS = re.compile(r"(_\d+)+$")


def protein_tag(path):
    folder = Path(path).parts[0]
    return TRAILING_NUMBERS.sub("", folder)


def protein_family(path):
    tag = protein_tag(path)
    return tag.split("_")[0]


def scaffold_from_sdf(path):
    try:
        mol = next((mol for mol in Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True) if mol is not None), None)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or Chem.MolToSmiles(mol)
    except Exception:
        return ""


def scaffold_task(item):
    idx, path = item
    return idx, scaffold_from_sdf(path)


def build_reference(index_train, raw_root, cache_json, max_workers):
    cache_path = Path(cache_json)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    train = pd.read_csv(index_train)
    train_proteins = sorted({protein_tag(path) for path in train["pocket_path"]})
    train_families = sorted({protein_family(path) for path in train["pocket_path"]})
    ligand_paths = [str(Path(raw_root) / path) for path in train["ligand_path"]]
    scaffolds = set()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(scaffold_task, item) for item in enumerate(ligand_paths)]
        for i, future in enumerate(as_completed(futures), start=1):
            _, scaffold = future.result()
            if scaffold:
                scaffolds.add(scaffold)
            if i % 10000 == 0:
                print(f"train_scaffolds_done={i}", flush=True)
    ref = {
        "train_proteins": train_proteins,
        "train_families": train_families,
        "train_scaffolds": sorted(scaffolds),
        "train_rows": int(len(train)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(ref), encoding="utf-8")
    return ref


def test_key(row):
    return f"{Path(row.pocket_path).stem}_{Path(row.ligand_path).stem}"


def test_metadata(index_test, raw_root, train_ref):
    test = pd.read_csv(index_test)
    rows = []
    train_proteins = set(train_ref["train_proteins"])
    train_families = set(train_ref["train_families"])
    train_scaffolds = set(train_ref["train_scaffolds"])
    for row in test.itertuples(index=False):
        native_path = Path(raw_root) / row.ligand_path
        native_scaffold = scaffold_from_sdf(native_path)
        rows.append(
            {
                "key": test_key(row),
                "pocket_path_rel": row.pocket_path,
                "ligand_path_rel": row.ligand_path,
                "protein_tag": protein_tag(row.pocket_path),
                "protein_family": protein_family(row.pocket_path),
                "protein_unseen_train": protein_tag(row.pocket_path) not in train_proteins,
                "family_unseen_train": protein_family(row.pocket_path) not in train_families,
                "native_scaffold": native_scaffold,
                "native_scaffold_unseen_train": native_scaffold not in train_scaffolds if native_scaffold else True,
            }
        )
    return pd.DataFrame(rows)


def load_selection_tables():
    diff = pd.concat(
        [
            pd.read_csv("results/posebusters_dockfast_selection.csv"),
            pd.read_csv("results/posebusters_dockfast_pb_selection.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    diff = diff[(diff["set"] == "fullatom_cond") & (diff["policy"].isin(["qed", "rc_select", "pb_qed", "pb_rc_select"]))].copy()
    diff["source"] = "DiffSBDD_official"
    p2m = pd.read_csv("results/pocket2mol_crossgen_n16_ext_dockfast_selection.csv")
    p2m = p2m[p2m["policy"].isin(["qed", "rc_select", "pb_qed", "pb_rc_select"])].copy()
    p2m["source"] = "Pocket2Mol_transfer"
    return pd.concat([diff, p2m], ignore_index=True, sort=False)


def annotate_generated_scaffold(df, max_workers):
    paths = list(df["mol_pred"])
    scaffolds = [""] * len(paths)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(scaffold_task, item) for item in enumerate(paths)]
        for future in as_completed(futures):
            idx, scaffold = future.result()
            scaffolds[idx] = scaffold
    df = df.copy()
    df["generated_scaffold"] = scaffolds
    return df


def summarize(df):
    axes = [
        "protein_unseen_train",
        "family_unseen_train",
        "native_scaffold_unseen_train",
        "generated_scaffold_unseen_train",
        "generated_scaffold_novel_vs_native",
    ]
    rows = []
    for axis in axes:
        for (source, policy, value), group in df.groupby(["source", "policy", axis], sort=True, dropna=False):
            rows.append(
                {
                    "axis": axis,
                    "ood_value": bool(value),
                    "source": source,
                    "policy": policy,
                    "n": int(len(group)),
                    "targets": int(group["key"].nunique()),
                    "dock_pose_pass": float(group["dock_pose_pass"].fillna(False).astype(bool).mean()),
                    "protein_pass": float(group["protein_pass"].fillna(False).astype(bool).mean()),
                    "risk_mean": float(group["risk_prob"].mean()),
                    "risk_gt_0_5": float((group["risk_prob"] > 0.5).mean()),
                    "qed_mean": float(group["qed"].mean()),
                }
            )
    return pd.DataFrame(rows)


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def write_report(summary, out_md):
    show_axes = ["protein_unseen_train", "native_scaffold_unseen_train", "generated_scaffold_unseen_train"]
    show_policies = ["qed", "rc_select", "pb_qed", "pb_rc_select"]
    main = summary[(summary["axis"].isin(show_axes)) & (summary["policy"].isin(show_policies))].copy()
    order_axis = {name: i for i, name in enumerate(show_axes)}
    order_source = {"DiffSBDD_official": 0, "Pocket2Mol_transfer": 1}
    order_policy = {name: i for i, name in enumerate(show_policies)}
    main["order_axis"] = main["axis"].map(order_axis)
    main["order_source"] = main["source"].map(order_source)
    main["order_policy"] = main["policy"].map(order_policy)
    lines = [
        "# Protein and Scaffold OOD Selection Audit",
        "",
        "## Protocol",
        "",
        "- Protein OOD: test protein/family parsed from CrossDocked directory names and compared with the training index.",
        "- Scaffold OOD: Bemis-Murcko scaffolds of train ligands, native test ligands, and generated molecules.",
        "- This is an OOD audit of selected molecules under identical target pools; it tests whether RC gains survive protein/scaffold novelty strata.",
        "",
        "## Summary",
        "",
        "| Axis | OOD | Source | Policy | N | Targets | dock_fast | Protein pass | Mean risk | Risk >0.5 | Mean QED |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main.sort_values(["order_axis", "ood_value", "order_source", "order_policy"]).itertuples(index=False):
        lines.append(
            f"| {row.axis} | {row.ood_value} | {row.source} | {row.policy} | {row.n} | {row.targets} | "
            f"{pct(row.dock_pose_pass)} | {pct(row.protein_pass)} | {f4(row.risk_mean)} | {pct(row.risk_gt_0_5)} | {f4(row.qed_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Protein/scaffold OOD strata should be used to bound generality claims beyond the official split.",
            "2. If an OOD stratum is tiny, report it as an audit rather than a powered benchmark.",
            "3. The key reviewer question is whether RC still reduces high-risk molecules and preserves dock_fast in unseen-protein or unseen-scaffold strata.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-train", default="data/processed/if3-crossdocked2020/index_train.csv")
    ap.add_argument("--index-test", default="data/processed/if3-crossdocked2020/index_test.csv")
    ap.add_argument("--raw-root", default="data/raw/if3-crossdocked2020/crossdocked_pocket10")
    ap.add_argument("--cache-json", default="results/ood_train_reference.json")
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--out-csv", default="results/protein_scaffold_ood_selection.csv")
    ap.add_argument("--out-summary", default="results/protein_scaffold_ood_selection_summary.csv")
    ap.add_argument("--out-md", default="experiments/PROTEIN_SCAFFOLD_OOD_SELECTION.md")
    args = ap.parse_args()

    ref = build_reference(args.index_train, args.raw_root, args.cache_json, args.max_workers)
    meta = test_metadata(args.index_test, args.raw_root, ref)
    selected = load_selection_tables()
    selected = annotate_generated_scaffold(selected, args.max_workers)
    train_scaffolds = set(ref["train_scaffolds"])
    selected = selected.merge(meta, on="key", how="left")
    selected["generated_scaffold_unseen_train"] = [
        scaffold not in train_scaffolds if scaffold else True for scaffold in selected["generated_scaffold"]
    ]
    selected["generated_scaffold_novel_vs_native"] = selected["generated_scaffold"] != selected["native_scaffold"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_csv, index=False)
    summary = summarize(selected)
    summary.to_csv(args.out_summary, index=False)
    write_report(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
