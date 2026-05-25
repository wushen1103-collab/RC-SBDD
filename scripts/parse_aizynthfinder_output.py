import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def load_aizynth_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        return pd.DataFrame(data["data"])
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported AiZynthFinder output format: {path}")


def summarize(df):
    rows = []
    for keys, group in df.groupby(["source", "policy"], sort=True):
        row = dict(zip(["source", "policy"], keys))
        row.update(
            {
                "n": int(len(group)),
                "targets": int(group["target_id"].nunique()),
                "unique_smiles": int(group["smiles"].nunique()),
                "solved_rate": float(group["is_solved"].fillna(False).astype(bool).mean()),
                "top_score_mean": float(group["top_score"].mean()),
                "route_steps_mean": float(group["number_of_steps"].replace(0, np.nan).mean()),
                "route_steps_median": float(group["number_of_steps"].replace(0, np.nan).median()),
                "precursors_in_stock_mean": float(group["number_of_precursors_in_stock"].mean()),
                "precursors_total_mean": float(group["number_of_precursors"].mean()),
                "search_time_mean": float(group["search_time"].mean()),
                "risk_mean": float(group["risk_prob"].mean()),
                "risk_gt_0_5": float((group["risk_prob"] > 0.5).mean()),
                "qed_mean": float(group["qed"].mean()),
                "dock_pose_pass": float(group["dock_pose_pass"].fillna(False).astype(bool).mean())
                if "dock_pose_pass" in group
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary, out_md):
    order_source = {
        "DiffSBDD_official": 0,
        "Pocket2Mol_transfer": 1,
        "PocketFlow": 2,
        "MolCRAFT-CrossDock-T50": 3,
        "MolPilot-CrossDock-T50-FrameRestored": 4,
        "Prospective20-Pocket2Mol": 5,
        "SGEDiff": 6,
        "MolPilot": 7,
    }
    order_policy = {"qed": 0, "rc_select": 1, "pb_qed": 2, "pb_rc_select": 3}
    summary = summary.copy()
    summary["order_source"] = summary["source"].map(order_source).fillna(99)
    summary["order_policy"] = summary["policy"].map(order_policy).fillna(99)
    lines = [
        "# AiZynthFinder Retrosynthesis Route Search",
        "",
        "## Protocol",
        "",
        "- Planner: AiZynthFinder 4.4.1.",
        "- Public data: USPTO expansion policy, ringbreaker policy, USPTO filter, and ZINC stock downloaded with `download_public_data`.",
        "- Input: selected top molecules from DiffSBDD official and Pocket2Mol transfer policies.",
        "- This is a true tree-search retrosynthesis oracle, replacing BRICS/RECAP route proxy as the strongest synthesis evidence.",
        "",
        "## Summary",
        "",
        "| Source | Policy | N | Targets | Unique smiles | Solved | Top score | Steps median | In-stock precursors | Search time | Mean QED | Risk >0.5 | dock_fast |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["order_source", "order_policy"]).itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.policy} | {row.n} | {row.targets} | {row.unique_smiles} | "
            f"{pct(row.solved_rate)} | {f4(row.top_score_mean)} | {f4(row.route_steps_median)} | "
            f"{f4(row.precursors_in_stock_mean)}/{f4(row.precursors_total_mean)} | {f4(row.search_time_mean)} | "
            f"{f4(row.qed_mean)} | {pct(row.risk_gt_0_5)} | {pct(row.dock_pose_pass)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. This table should be used as the main synthesis-route evidence; the BRICS/RECAP proxy becomes a lightweight supporting diagnostic.",
            "2. Low solved rates should be reported honestly because generated SBDD molecules can be geometrically plausible yet hard under a strict stock-constrained planner.",
            "3. The fair comparison is policy-level preservation or improvement of solved rate and route score under the same AiZynthFinder stock and policies.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-csv", default="results/aizynthfinder_selection_map.csv")
    ap.add_argument("--aizynth-json", default="results/aizynthfinder_selection_routes.json")
    ap.add_argument("--out-csv", default="results/aizynthfinder_selection_routes.csv")
    ap.add_argument("--out-summary", default="results/aizynthfinder_selection_summary.csv")
    ap.add_argument("--out-md", default="experiments/AIZYNTHFINDER_ROUTE_SEARCH.md")
    args = ap.parse_args()

    mapping = pd.read_csv(args.map_csv)
    routes = load_aizynth_json(args.aizynth_json)
    unique = mapping[["aizynth_id", "smiles"]].drop_duplicates("aizynth_id").reset_index(drop=True)
    route_by_smiles = routes.rename(columns={"target": "smiles"})
    merged_unique = unique.merge(route_by_smiles, on="smiles", how="left")
    merged = mapping.merge(merged_unique.drop(columns=["smiles"]), on="aizynth_id", how="left")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    summary = summarize(merged)
    summary.to_csv(args.out_summary, index=False)
    write_report(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
