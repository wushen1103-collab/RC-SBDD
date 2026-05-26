import argparse
from pathlib import Path

import pandas as pd

import analyze_selectivity_aware_kinase as base


EXPANDED_KINASE_IDS = {
    2: "GRK4",
    9: "IPMK",
    11: "KS6A3",
    16: "M3K14",
    25: "PAK4",
    26: "PHKG1",
    41: "TBK1",
    76: "ABL2",
    80: "AKT1",
    87: "CDK6-4AUA",
    88: "CDK6-2F2C",
    98: "DYRK2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/if3-crossdocked2020/index_test.csv")
    ap.add_argument("--raw-root", default="data/raw/if3-crossdocked2020/crossdocked_pocket10")
    ap.add_argument("--selection-csv", default="results/posebusters_dockfast_pb_selection.csv")
    ap.add_argument("--top-per-target", type=int, default=8)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--cpu", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=24)
    ap.add_argument("--work-dir", default="results/kinase_selectivity_aware_expanded_work")
    ap.add_argument("--out-prefix", default="results/kinase_selectivity_aware_expanded")
    ap.add_argument("--out-md", default="experiments/SELECTIVITY_AWARE_KINASE_EXPANDED.md")
    args = ap.parse_args()

    base.KINASE_IDS = EXPANDED_KINASE_IDS
    tau = float(
        pd.read_csv("results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
        .query("kind == 'native'")
        .risk_prob.quantile(0.95)
    )
    targets = base.prepare_targets(args.index, args.raw_root)
    ligands = base.prepare_ligands(args.selection_csv, targets, args.top_per_target)
    payloads = [
        (ligand, receptor, args.work_dir, args.exhaustiveness, args.cpu)
        for ligand in ligands.to_dict(orient="records")
        for receptor in targets.to_dict(orient="records")
    ]
    rows = []
    with base.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(base.dock_pair, payload) for payload in payloads]
        for future in base.as_completed(futures):
            rows.append(future.result())
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{args.out_prefix}_crossdock.csv", index=False)
    candidates = base.candidate_summary(raw)
    candidates.to_csv(f"{args.out_prefix}_candidates.csv", index=False)
    summary, selected = base.summarize_policies(candidates, tau)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    selected.to_csv(f"{args.out_prefix}_selected.csv", index=False)

    lines = [
        "# Expanded Selectivity-Aware Kinase Pilot",
        "",
        "## Protocol",
        "",
        f"- Targets: {len(EXPANDED_KINASE_IDS)} kinase or kinase-like CrossDocked pockets.",
        f"- Candidates: top {args.top_per_target} PB-RC DiffSBDD molecules per target.",
        f"- Cross-docking pairs attempted: {len(raw)}; successful: {int(raw.success.sum())}.",
        "- Selectivity margin = best off-target Vina score - target Vina score. Positive is target-preferred.",
        "- This is a computational selectivity pilot, not pharmacological selectivity profiling.",
        "",
        "## Policy Summary",
        "",
        "| Policy | N | Target best | Mean margin | Median rank | Risk >0.5 | Mean QED | dock_fast |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.policy} | {row.n} | {base.pct(row.target_is_best)} | {base.f4(row.mean_margin)} | "
            f"{base.f4(row.median_rank)} | {base.pct(row.risk_gt_0_5)} | {base.f4(row.mean_qed)} | {base.pct(row.dock_fast)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Expanded cross-docking makes the pilot less anecdotal, but it remains a computational selectivity audit rather than a core RC-SBDD claim.",
            "2. Plain PB-RC optimizes pocket reliability, whereas selectivity-aware variants require an explicit off-target oracle.",
            "3. The main manuscript should state that selectivity can be layered on top of RC, not that RC alone solves kinase selectivity.",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
