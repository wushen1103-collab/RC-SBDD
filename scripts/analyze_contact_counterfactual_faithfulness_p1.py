import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_contact_counterfactual_faithfulness import (
    f4,
    load_model,
    pct,
    predict,
    process_row,
    read_ligand,
    read_pocket_atoms,
    summarize,
)


SOURCES = [
    ("DiffSBDD-fullpool", "results/dockfast_full_pool_fullatom_cond.csv", None),
    ("PocketFlow", "results/pocketflow_crossdock_n16_dockfast_selection.csv", None),
    ("MolCRAFT-100", "results/molcraft_crossdock_t100_n16_dockfast_selection.csv", None),
    ("ExpDiff-100", "results/expdiff_official_t100_nall_dockfast_selection.csv", None),
]


def normalize_pool(path, source):
    df = pd.read_csv(path, low_memory=False)
    if "kind" in df.columns:
        df = df[df["kind"] == "generated"].copy()
    if "policy" in df.columns:
        # Keep unique molecules, regardless of which selector first exposed them.
        df = df.sort_values(["key", "risk_prob", "qed"], ascending=[True, False, False])
        df = df.drop_duplicates(["key", "mol_index"], keep="first")
    df = df[df["mol_pred"].notna() & df["mol_cond"].notna()].copy()
    df["source"] = source
    return df


def select_one_per_target(df, max_targets):
    selected = (
        df.sort_values(["key", "risk_prob", "qed"], ascending=[True, False, False])
        .groupby("key", sort=True)
        .head(1)
        .copy()
    )
    selected = selected.sort_values(["risk_prob", "qed"], ascending=[False, False]).head(max_targets)
    return selected


def build_lowrisk_median(paths, model, names):
    lowrisk_features = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        if "kind" in df.columns:
            df = df[df["kind"] == "generated"].copy()
        if "policy" in df.columns:
            df = df.drop_duplicates(["key", "mol_index"], keep="first")
        df = df[df["risk_prob"] <= 0.20].head(80)
        for row in df.itertuples(index=False):
            try:
                lig_pos, lig_type, _ = read_ligand(row.mol_pred)
                atoms = read_pocket_atoms(row.mol_cond)
                pock_pos = np.vstack([a["xyz"] for a in atoms]).astype(np.float32)
                _, x = predict(model, lig_pos, lig_type, pock_pos, names)
                lowrisk_features.append(x)
            except Exception:
                continue
    if not lowrisk_features:
        raise RuntimeError("could not build low-risk feature medians")
    return np.median(np.vstack(lowrisk_features), axis=0)


def write_report(summary, out_md):
    lines = [
        "# P1 100-Target Contact Counterfactual Faithfulness",
        "",
        "## Protocol",
        "",
        "- For each source, select at most one highest-risk generated molecule per target, up to 100 targets.",
        "- Recompute RC-SBDD risk after deleting contact residues, deleting contact ligand atoms, and masking distance/contact features to low-risk medians.",
        "- This is an aggregate explanation-faithfulness audit of the risk scorer, not a chemically valid edit or a biochemical mechanism claim.",
        "",
        "## Summary",
        "",
        "| Source | N | Orig risk | Residue delta | Residue drop | Atom delta | Atom drop | Mask delta | Mask drop | Close residues | Close atoms | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.n} | {f4(row.orig_risk)} | "
            f"{f4(row.residue_deleted_risk_delta)} | {pct(row.residue_deleted_risk_decrease)} | "
            f"{f4(row.atom_deleted_risk_delta)} | {pct(row.atom_deleted_risk_decrease)} | "
            f"{f4(row.contact_masked_risk_delta)} | {pct(row.contact_masked_risk_decrease)} | "
            f"{f4(row.close_residues_mean)} | {f4(row.close_lig_atoms_mean)} | {row.failures} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Negative deltas after contact-feature masking support faithfulness of the learned contact-risk channel at aggregate level.",
            "2. Residue and atom deletion are stricter perturbations and should be interpreted as counterfactual sensitivity, not molecular design edits.",
            "3. The main text should cite this table briefly and keep detailed per-source rows in the supplementary material.",
        ]
    )
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/risk_proxy_hard_geom_logreg/model.pkl")
    ap.add_argument("--max-targets-per-source", type=int, default=100)
    ap.add_argument("--contact-cutoff", type=float, default=2.0)
    ap.add_argument("--out-csv", default="results/contact_counterfactual_faithfulness_p1_aggregate.csv")
    ap.add_argument("--out-summary-csv", default="results/contact_counterfactual_faithfulness_p1_aggregate_summary.csv")
    ap.add_argument("--out-md", default="experiments/CONTACT_COUNTERFACTUAL_FAITHFULNESS_P1_AGGREGATE.md")
    args = ap.parse_args()

    model, names = load_model(args.model)
    existing_paths = [path for _, path, _ in SOURCES if Path(path).exists()]
    lowrisk_median = build_lowrisk_median(existing_paths, model, names)

    rows = []
    failures = []
    for source, path, _ in SOURCES:
        if not Path(path).exists():
            continue
        pool = normalize_pool(path, source)
        selected = select_one_per_target(pool, args.max_targets_per_source)
        for row in selected.itertuples(index=False):
            try:
                rows.append(process_row(row, model, names, args.contact_cutoff, lowrisk_median))
            except Exception as exc:
                failures.append({"source": source, "key": getattr(row, "key", ""), "error": str(exc)})

    out = pd.DataFrame(rows)
    summaries = []
    for source, group in out.groupby("source", sort=True):
        source_failures = sum(1 for item in failures if item["source"] == source)
        summaries.append({"source": source, **summarize(group), "failures": source_failures})
    summary = pd.DataFrame(summaries)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary.to_csv(args.out_summary_csv, index=False)
    write_report(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))
    if failures:
        print(f"failures={len(failures)} first={failures[:3]}")


if __name__ == "__main__":
    main()
