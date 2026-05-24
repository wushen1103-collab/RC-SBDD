import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


RAW_ROOT = Path("data/raw/if3-crossdocked2020/crossdocked_pocket10")
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


def short_hash(text, n=10):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:n]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_key(row):
    return f"{Path(row.pocket_path).stem}_{Path(row.ligand_path).stem}"


def as_bool(series):
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_copy(src, dst):
    src = Path(src)
    if not src.exists() or src.stat().st_size == 0:
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return str(dst)


def reset(out):
    if out.exists():
        shutil.rmtree(out)
    for sub in ["pockets", "native_ligands", "candidates", "labels", "splits", "evaluation"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def build_targets(out, limit):
    idx = pd.read_csv("data/processed/if3-crossdocked2020/index_test.csv").head(limit).copy()
    rows = []
    for i, row in idx.iterrows():
        key = test_key(row)
        pocket = RAW_ROOT / row.pocket_path
        native = RAW_ROOT / row.ligand_path
        family_proxy = Path(row.pocket_path).parts[0] if Path(row.pocket_path).parts else ""
        rows.append(
            {
                "data_id": int(i),
                "target_key": key,
                "protein_family_proxy": family_proxy,
                "source_pocket_path": str(pocket),
                "source_native_ligand_path": str(native),
                "pocket_path": safe_copy(pocket, out / "pockets" / f"data{i:03d}_{Path(row.pocket_path).name}"),
                "native_ligand_path": safe_copy(native, out / "native_ligands" / f"data{i:03d}_{Path(row.ligand_path).name}"),
            }
        )
    targets = pd.DataFrame(rows)
    targets.to_csv(out / "targets.csv", index=False)
    split = targets[["data_id", "target_key", "protein_family_proxy"]].copy()
    split["split"] = "official_test100"
    split["family_ood_fold"] = split.groupby("protein_family_proxy").ngroup() % 5
    split.to_csv(out / "splits" / "official_target_split.csv", index=False)
    return targets


def copy_candidate_files(out, labels, generator):
    mapping = {}
    gen_dir = out / "candidates" / generator
    for src in sorted(labels["source_file"].dropna().astype(str).unique()):
        p = Path(src)
        if not p.exists() or p.stat().st_size == 0:
            continue
        dst = gen_dir / f"{short_hash(src)}_{p.name}"
        mapping[src] = safe_copy(p, dst)
    return mapping


def normalize(df, targets, generator, source_scope, file_map):
    tmap = targets.set_index("target_key")["data_id"].to_dict()
    if "target_key" not in df.columns:
        df["target_key"] = df["key"].astype(str)
    if "data_id" not in df.columns:
        df["data_id"] = df["target_key"].map(tmap)
    if "mol_fast_pass" not in df.columns:
        if "intramol_pass" in df.columns:
            df["mol_fast_pass"] = as_bool(df["intramol_pass"])
        elif set(CORE_COLUMNS).issubset(df.columns):
            df["mol_fast_pass"] = df[CORE_COLUMNS].apply(as_bool).all(axis=1)
        else:
            df["mol_fast_pass"] = pd.NA
    for col in ["risk_prob", "qed", "protein_pass", "dock_pose_pass", "mol_index", "source_file", "mol_pred", "mol_cond"]:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[
        [
            "data_id",
            "target_key",
            "mol_index",
            "source_file",
            "mol_pred",
            "mol_cond",
            "risk_prob",
            "qed",
            "mol_fast_pass",
            "protein_pass",
            "dock_pose_pass",
        ]
    ].copy()
    out["generator"] = generator
    out["label_scope"] = source_scope
    out["bench_candidate_file"] = out["source_file"].map(file_map).fillna("")
    return out[
        [
            "generator",
            "data_id",
            "target_key",
            "mol_index",
            "source_file",
            "bench_candidate_file",
            "mol_pred",
            "mol_cond",
            "risk_prob",
            "qed",
            "mol_fast_pass",
            "protein_pass",
            "dock_pose_pass",
            "label_scope",
        ]
    ]


def load_diffsbdd():
    p = Path("results/dockfast_full_pool_fullatom_cond.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    return df[df["kind"].astype(str) == "generated"].copy()


def load_pocket2mol():
    p = Path("results/dockfast_full_pool_pocket2mol_n16_ext.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    return df[df["kind"].astype(str) == "generated"].copy()


def load_syncguide():
    risk_p = Path("results/syncguide_t1000_n16_risk_scores.csv")
    mol_p = Path("results/syncguide_t1000_n16_molfast.csv")
    if not risk_p.exists():
        return pd.DataFrame()
    df = pd.read_csv(risk_p)
    df = df[df["kind"].astype(str) == "generated"].copy()
    if mol_p.exists():
        mol = pd.read_csv(mol_p).rename(columns={"file": "source_file"})
        mol["mol_index"] = mol["position"].astype(int)
        mol["mol_fast_pass"] = mol[CORE_COLUMNS].apply(as_bool).all(axis=1)
        df = df.merge(mol[["source_file", "mol_index", "mol_fast_pass"]], on=["source_file", "mol_index"], how="left")
    dock_p = Path("results/syncguide_t1000_n16_dockfast_selection.csv")
    if dock_p.exists():
        dock = pd.read_csv(dock_p, low_memory=False).drop_duplicates(["data_id", "source_file", "mol_index"])
        df = df.merge(dock[["data_id", "source_file", "mol_index", "protein_pass", "dock_pose_pass"]], on=["data_id", "source_file", "mol_index"], how="left")
    return df


def load_pocketflow():
    risk_p = Path("results/pocketflow_crossdock_n16_risk_scores.csv")
    if not risk_p.exists():
        return pd.DataFrame()
    df = pd.read_csv(risk_p)
    df = df[df["kind"].astype(str) == "generated"].copy()
    if "data_id" not in df.columns:
        idx = pd.read_csv("data/processed/if3-crossdocked2020/index_test.csv")
        key_to_data = {test_key(row): i for i, row in enumerate(idx.itertuples(index=False))}
        df["data_id"] = df["key"].map(key_to_data)
    mol_p = Path("results/pocketflow_crossdock_n16_molfast.csv")
    if mol_p.exists():
        mol = pd.read_csv(mol_p).rename(columns={"file": "source_file"})
        mol["mol_index"] = mol["position"].astype(int)
        mol["mol_fast_pass"] = mol[CORE_COLUMNS].apply(as_bool).all(axis=1)
        df = df.merge(mol[["source_file", "mol_index", "mol_fast_pass"]], on=["source_file", "mol_index"], how="left")
    dock_p = Path("results/pocketflow_crossdock_n16_dockfast_selection.csv")
    if dock_p.exists():
        dock = pd.read_csv(dock_p, low_memory=False).drop_duplicates(["data_id", "source_file", "mol_index"])
        df = df.merge(dock[["data_id", "source_file", "mol_index", "protein_pass", "dock_pose_pass"]], on=["data_id", "source_file", "mol_index"], how="left")
    return df


def write_score_submission(out):
    script = r'''import argparse
from pathlib import Path
import pandas as pd

def as_bool(s):
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(Path(__file__).resolve().parents[1] / "labels" / "oracle_labels.csv"))
    ap.add_argument("--submission", required=True, help="CSV with generator,target_key,mol_index or bench_candidate_file")
    ap.add_argument("--out", default="submission_scores.csv")
    args = ap.parse_args()
    labels = pd.read_csv(args.labels, low_memory=False)
    sub = pd.read_csv(args.submission)
    keys = ["generator", "target_key", "mol_index"] if "mol_index" in sub.columns else ["bench_candidate_file"]
    merged = sub.merge(labels, on=keys, how="left", suffixes=("_submitted", ""))
    for col in ["mol_fast_pass", "protein_pass", "dock_pose_pass"]:
        if col in merged.columns:
            merged[col + "_bool"] = as_bool(merged[col])
    summary = {
        "submitted": len(sub),
        "matched": int(merged["risk_prob"].notna().sum()),
        "targets": int(merged["target_key"].nunique()) if "target_key" in merged.columns else 0,
        "mean_risk": float(merged["risk_prob"].mean()),
        "mean_qed": float(merged["qed"].mean()),
        "mol_fast_pass": float(merged.get("mol_fast_pass_bool", pd.Series(dtype=bool)).mean()) if "mol_fast_pass_bool" in merged else None,
        "dock_pose_pass": float(merged.get("dock_pose_pass_bool", pd.Series(dtype=bool)).mean()) if "dock_pose_pass_bool" in merged else None,
    }
    pd.DataFrame([summary]).to_csv(args.out, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))

if __name__ == "__main__":
    main()
'''
    path = out / "evaluation" / "score_submission.py"
    path.write_text(script, encoding="utf-8")


def write_docs(out, manifest, gen_summary):
    (out / "DATASET_CARD.md").write_text(
        "\n".join(
            [
                "# RC-SBDD-Bench v1 Dataset Card",
                "",
                "## Purpose",
                "",
                "RC-SBDD-Bench v1 is a compact 100-target reliability benchmark for structure-based de novo molecular generation and candidate selection.",
                "",
                "## Contents",
                "",
                "- CrossDocked2020 pocket and native ligand files for the official 100-target split.",
                "- Candidate SDF outputs from DiffSBDD, Pocket2Mol, SYNC-Guide, and PocketFlow when available.",
                "- Candidate-level oracle labels: risk score, QED, mol_fast, protein/dock_fast labels where available.",
                "- Official target split and a protein-family proxy fold assignment.",
                "",
                "## Intended Use",
                "",
                "Use the benchmark to evaluate selection protocols over fixed generator outputs. Do not train on labels from the official test split.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "METRICS.md").write_text(
        "\n".join(
            [
                "# Metrics",
                "",
                "- `risk_prob`: calibrated geometry failure probability from the RC model.",
                "- `mol_fast_pass`: PoseBusters intramolecular validity checks.",
                "- `protein_pass`: PoseBusters protein-ligand checks.",
                "- `dock_pose_pass`: complete PoseBusters dock_fast pass label.",
                "- `qed`: RDKit quantitative estimate of drug-likeness.",
                "",
                "Primary evaluation should report target-level paired comparisons, confidence intervals, and FDR-corrected significance tests.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"method": "example_qed", "generator": "DiffSBDD", "targets": 100, "dock_pose_pass": "", "mean_risk": "", "mean_qed": "", "notes": ""},
            {"method": "example_rc_select", "generator": "DiffSBDD", "targets": 100, "dock_pose_pass": "", "mean_risk": "", "mean_qed": "", "notes": ""},
        ]
    ).to_csv(out / "LEADERBOARD_TEMPLATE.csv", index=False)
    (out / "README.md").write_text(
        "\n".join(
            [
                "# RC-SBDD-Bench v1",
                "",
                f"- Targets: {manifest['targets']}",
                f"- Generators: {manifest['generators']}",
                f"- Candidate label rows: {manifest['candidate_rows']}",
                f"- Candidate SDF files: {manifest['candidate_sdf_files']}",
                "",
                "See `DATASET_CARD.md`, `METRICS.md`, `labels/oracle_labels.csv`, `splits/official_target_split.csv`, and `evaluation/score_submission.py`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RC-SBDD-Bench v1 Release",
        "",
        "| Targets | Generators | Candidate rows | Candidate SDFs | dock_fast labels |",
        "|---:|---:|---:|---:|---:|",
        f"| {manifest['targets']} | {manifest['generators']} | {manifest['candidate_rows']} | {manifest['candidate_sdf_files']} | {manifest['dockfast_label_rows']} |",
        "",
        "## Generator Coverage",
        "",
        "| Generator | Targets | Rows | SDF files | dock_fast labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in gen_summary.itertuples(index=False):
        lines.append(f"| {row.generator} | {row.targets} | {row.rows} | {row.candidate_sdf_files} | {row.dockfast_label_rows} |")
    Path("experiments/RC_SBDD_BENCH_V1_RELEASE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hashes(out):
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/RC-SBDD-Bench-v1")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    reset(out)
    targets = build_targets(out, args.limit)
    loaders = [
        ("DiffSBDD", "full_pool_dockfast", load_diffsbdd),
        ("Pocket2Mol", "full_pool_dockfast", load_pocket2mol),
        ("SYNC-Guide", "risk_molfast_selected_dockfast", load_syncguide),
        ("PocketFlow", "risk_molfast_selected_dockfast", load_pocketflow),
    ]
    label_parts = []
    file_counts = {}
    for generator, scope, loader in loaders:
        df = loader()
        if df.empty:
            continue
        if "key" in df.columns:
            df = df[df["key"].astype(str).isin(set(targets["target_key"]))].copy()
        fmap = copy_candidate_files(out, df, generator)
        file_counts[generator] = len(fmap)
        label_parts.append(normalize(df, targets, generator, scope, fmap))
    labels = pd.concat(label_parts, ignore_index=True, sort=False) if label_parts else pd.DataFrame()
    labels.to_csv(out / "labels" / "oracle_labels.csv", index=False)
    labels[labels["dock_pose_pass"].notna()].to_csv(out / "labels" / "dockfast_available_labels.csv", index=False)
    gen_summary = (
        labels.groupby("generator", sort=True)
        .agg(targets=("target_key", "nunique"), rows=("target_key", "size"), dockfast_label_rows=("dock_pose_pass", lambda x: int(x.notna().sum())))
        .reset_index()
    )
    gen_summary["candidate_sdf_files"] = gen_summary["generator"].map(file_counts).fillna(0).astype(int)
    gen_summary.to_csv(out / "generator_manifest.csv", index=False)
    manifest = {
        "benchmark": "RC-SBDD-Bench-v1",
        "targets": int(targets["target_key"].nunique()),
        "generators": int(labels["generator"].nunique()) if len(labels) else 0,
        "candidate_rows": int(len(labels)),
        "dockfast_label_rows": int(labels["dock_pose_pass"].notna().sum()) if len(labels) else 0,
        "candidate_sdf_files": int(sum(file_counts.values())),
        "pocket_files": len(list((out / "pockets").glob("*.pdb"))),
        "native_ligand_files": len(list((out / "native_ligands").glob("*.sdf"))),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([manifest]).to_csv(out / "manifest.csv", index=False)
    write_score_submission(out)
    write_docs(out, manifest, gen_summary)
    write_hashes(out)
    print(Path("experiments/RC_SBDD_BENCH_V1_RELEASE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
