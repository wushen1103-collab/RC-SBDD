import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


TRAILING_NUMBERS = re.compile(r"(_\d+)+$")

SOURCES = [
    ("PocketFlow", "results/pocketflow_crossdock_n16_dockfast_selection.csv"),
    ("MolCRAFT-100", "results/molcraft_crossdock_t100_n16_dockfast_selection.csv"),
    ("ExpDiff-100", "results/expdiff_official_t100_nall_dockfast_selection.csv"),
    ("MolPilot-framefix", "results/molpilot_crossdock_t50_n16_framefix_dockfast_selection.csv"),
]

METRICS = {
    "dock_pose_pass": "higher",
    "risk_prob": "lower",
    "risk_gt_0_5": "lower",
    "qed": "higher",
}


def protein_tag(path):
    folder = Path(path).parts[0]
    return TRAILING_NUMBERS.sub("", folder)


def protein_family(path):
    return protein_tag(path).split("_")[0]


def test_key(row):
    return f"{Path(row.pocket_path).stem}_{Path(row.ligand_path).stem}"


def scaffold_from_sdf(path):
    try:
        mol = next((m for m in Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True) if m is not None), None)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or Chem.MolToSmiles(mol)
    except Exception:
        return ""


def scaffold_task(item):
    idx, path = item
    return idx, scaffold_from_sdf(path)


def bh_fdr(p_values):
    vals = np.asarray([1.0 if pd.isna(v) else float(v) for v in p_values], dtype=float)
    n = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]
    adj = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adj[order[i]] = min(running, 1.0)
    return adj


def wilcoxon_p(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x, y in zip(a, b):
        gt += int(x > y)
        lt += int(x < y)
    return float((gt - lt) / len(a))


def paired_bootstrap_ci(diff, n_boot=2000, seed=20260526):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


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
        for future in as_completed(futures):
            _, scaffold = future.result()
            if scaffold:
                scaffolds.add(scaffold)
    ref = {
        "train_proteins": train_proteins,
        "train_families": train_families,
        "train_scaffolds": sorted(scaffolds),
        "train_rows": int(len(train)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(ref), encoding="utf-8")
    return ref


def test_metadata(index_test, raw_root, ref):
    test = pd.read_csv(index_test)
    train_proteins = set(ref["train_proteins"])
    train_families = set(ref["train_families"])
    train_scaffolds = set(ref["train_scaffolds"])
    rows = []
    for row in test.itertuples(index=False):
        native_path = Path(raw_root) / row.ligand_path
        native_scaffold = scaffold_from_sdf(native_path)
        rows.append(
            {
                "key": test_key(row),
                "pocket_path_rel": row.pocket_path,
                "protein_tag": protein_tag(row.pocket_path),
                "protein_family": protein_family(row.pocket_path),
                "protein_unseen_train": protein_tag(row.pocket_path) not in train_proteins,
                "family_unseen_train": protein_family(row.pocket_path) not in train_families,
                "native_scaffold": native_scaffold,
                "native_scaffold_unseen_train": native_scaffold not in train_scaffolds if native_scaffold else True,
            }
        )
    return pd.DataFrame(rows)


def annotate_generated_scaffold(df, max_workers):
    paths = list(df["mol_pred"])
    scaffolds = [""] * len(paths)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(scaffold_task, item) for item in enumerate(paths)]
        for future in as_completed(futures):
            idx, scaffold = future.result()
            scaffolds[idx] = scaffold
    out = df.copy()
    out["generated_scaffold"] = scaffolds
    return out


def load_external_selection():
    frames = []
    for source, path in SOURCES:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df = df[df["policy"].isin(["pb_qed", "pb_rc_select"])].copy()
        df["source"] = source
        frames.append(df)
    if not frames:
        raise RuntimeError("no external generator selection CSVs found")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_joint_axes(df, train_scaffolds):
    out = df.copy()
    out["generated_scaffold_unseen_train"] = [
        scaffold not in train_scaffolds if scaffold else True for scaffold in out["generated_scaffold"]
    ]
    out["family_and_generated_scaffold_unseen"] = out["family_unseen_train"] & out["generated_scaffold_unseen_train"]
    out["family_and_native_scaffold_unseen"] = out["family_unseen_train"] & out["native_scaffold_unseen_train"]
    out["protein_family_or_scaffold_unseen"] = out["family_unseen_train"] | out["generated_scaffold_unseen_train"]
    out["risk_gt_0_5"] = out["risk_prob"] > 0.5
    return out


def summarize_rows(df):
    axes = [
        "family_unseen_train",
        "native_scaffold_unseen_train",
        "generated_scaffold_unseen_train",
        "family_and_generated_scaffold_unseen",
        "family_and_native_scaffold_unseen",
        "protein_family_or_scaffold_unseen",
    ]
    rows = []
    for axis in axes:
        subset = df[df[axis].fillna(False)].copy()
        for (source, policy), group in subset.groupby(["source", "policy"], sort=True):
            rows.append(
                {
                    "axis": axis,
                    "source": source,
                    "policy": policy,
                    "rows": int(len(group)),
                    "targets": int(group["key"].nunique()),
                    "dock_pose_pass": float(group["dock_pose_pass"].fillna(False).astype(bool).mean()),
                    "risk_prob": float(group["risk_prob"].mean()),
                    "risk_gt_0_5": float((group["risk_prob"] > 0.5).mean()),
                    "qed": float(group["qed"].mean()),
                }
            )
    return pd.DataFrame(rows)


def target_level_stats(df):
    axes = [
        "family_unseen_train",
        "native_scaffold_unseen_train",
        "generated_scaffold_unseen_train",
        "family_and_generated_scaffold_unseen",
        "family_and_native_scaffold_unseen",
        "protein_family_or_scaffold_unseen",
    ]
    rows = []
    for axis in axes:
        subset = df[df[axis].fillna(False)].copy()
        if subset.empty:
            continue
        for source, source_df in subset.groupby("source", sort=True):
            for metric, direction in METRICS.items():
                pivot = (
                    source_df.groupby(["key", "policy"], sort=True)[metric]
                    .mean()
                    .unstack()
                    .dropna(subset=["pb_qed", "pb_rc_select"])
                )
                if pivot.empty:
                    continue
                diff = pivot["pb_rc_select"].to_numpy(dtype=float) - pivot["pb_qed"].to_numpy(dtype=float)
                ci_low, ci_high = paired_bootstrap_ci(diff)
                rows.append(
                    {
                        "axis": axis,
                        "source": source,
                        "metric": metric,
                        "targets": int(len(pivot)),
                        "baseline_mean": float(pivot["pb_qed"].mean()),
                        "method_mean": float(pivot["pb_rc_select"].mean()),
                        "delta_method_minus_baseline": float(diff.mean()),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "wilcoxon_p": wilcoxon_p(diff),
                        "cliffs_delta": cliffs_delta(pivot["pb_rc_select"], pivot["pb_qed"]),
                        "desired_direction": direction,
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_q_joint_shift_family"] = bh_fdr(out["wilcoxon_p"].tolist())
        out["improves"] = [
            (d > 0 and direction == "higher") or (d < 0 and direction == "lower")
            for d, direction in zip(out["delta_method_minus_baseline"], out["desired_direction"])
        ]
    return out


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def write_report(summary, stats, out_md):
    show_axes = [
        "family_and_generated_scaffold_unseen",
        "family_and_native_scaffold_unseen",
        "protein_family_or_scaffold_unseen",
    ]
    lines = [
        "# P1 Joint Generator and Protein/Scaffold Shift",
        "",
        "## Protocol",
        "",
        "- Joint shift is defined as a recent/unseen generator output evaluated inside protein-family or scaffold novelty strata relative to the CrossDocked training index.",
        "- The table keeps PB-QED and PB-RC on identical selected rows per target and reports target-level paired statistics for PB-RC minus PB-QED.",
        "- This is stronger than separate OOD and generator-shift reporting because both shifts are active at the same time.",
        "",
        "## Summary",
        "",
        "| Axis | Source | Policy | Rows | Targets | dock_fast | Risk | Risk >0.5 | QED |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    main = summary[summary["axis"].isin(show_axes)].copy()
    for row in main.sort_values(["axis", "source", "policy"]).itertuples(index=False):
        lines.append(
            f"| {row.axis} | {row.source} | {row.policy} | {row.rows} | {row.targets} | "
            f"{pct(row.dock_pose_pass)} | {f4(row.risk_prob)} | {pct(row.risk_gt_0_5)} | {f4(row.qed)} |"
        )
    lines.extend(
        [
            "",
            "## Target-level paired tests",
            "",
            "| Axis | Source | Metric | Targets | PB-QED | PB-RC | Delta | 95% CI | FDR q |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    main_stats = stats[(stats["axis"].isin(show_axes)) & (stats["metric"].isin(["dock_pose_pass", "risk_gt_0_5", "risk_prob"]))].copy()
    for row in main_stats.sort_values(["axis", "source", "metric"]).itertuples(index=False):
        lines.append(
            f"| {row.axis} | {row.source} | {row.metric} | {row.targets} | {f4(row.baseline_mean)} | "
            f"{f4(row.method_mean)} | {f4(row.delta_method_minus_baseline)} | "
            f"[{f4(row.bootstrap_ci_low)}, {f4(row.bootstrap_ci_high)}] | {f4(row.fdr_q_joint_shift_family)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. The joint-shift table should be used as a boundary-aware transfer audit rather than a new primary benchmark.",
            "2. Strong generators with near-ceiling dock-fast mainly test residual risk-tail removal; stress generators test whether RC refuses unusable pools.",
            "3. The manuscript should claim reliable selection under declared joint shift only where target-level paired effects remain favorable.",
        ]
    )
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-train", default="data/processed/if3-crossdocked2020/index_train.csv")
    ap.add_argument("--index-test", default="data/processed/if3-crossdocked2020/index_test.csv")
    ap.add_argument("--raw-root", default="data/raw/if3-crossdocked2020/crossdocked_pocket10")
    ap.add_argument("--cache-json", default="results/ood_train_reference.json")
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--out-csv", default="results/joint_shift_generator_protein_scaffold_selection_p1.csv")
    ap.add_argument("--out-summary", default="results/joint_shift_generator_protein_scaffold_summary_p1.csv")
    ap.add_argument("--out-stats", default="results/joint_shift_generator_protein_scaffold_statistics_p1.csv")
    ap.add_argument("--out-md", default="experiments/JOINT_SHIFT_GENERATOR_PROTEIN_SCAFFOLD_P1.md")
    args = ap.parse_args()

    ref = build_reference(args.index_train, args.raw_root, args.cache_json, args.max_workers)
    meta = test_metadata(args.index_test, args.raw_root, ref)
    selected = load_external_selection()
    selected = annotate_generated_scaffold(selected, args.max_workers)
    selected = selected.merge(meta, on="key", how="left")
    selected = add_joint_axes(selected, set(ref["train_scaffolds"]))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_csv, index=False)
    summary = summarize_rows(selected)
    stats = target_level_stats(selected)
    summary.to_csv(args.out_summary, index=False)
    stats.to_csv(args.out_stats, index=False)
    write_report(summary, stats, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
