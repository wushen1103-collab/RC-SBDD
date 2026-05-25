from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scripts.evaluate_gnina_redock_selection import as_bool, load_selection_file, redock_one, summarize, write_report


def load_bindingmoad_selection(path: str, policies: list[str], top_k: int, limit_targets: int) -> pd.DataFrame:
    df = load_selection_file(path, "BindingMOAD_holdout100", policies)
    df = df.dropna(subset=["mol_pred", "mol_cond"]).copy()
    df = df[
        df["mol_pred"].astype(str).map(lambda x: Path(x).exists())
        & df["mol_cond"].astype(str).map(lambda x: Path(x).exists())
    ].copy()
    df["dock_pose_pass_bool"] = as_bool(df["dock_pose_pass"])
    df = df.sort_values(
        ["source", "policy", "target_id", "qed", "risk_prob"],
        ascending=[True, True, True, False, True],
    )
    df = df.groupby(["source", "policy", "target_id"], sort=True).head(top_k)
    df["selection_rank"] = df.groupby(["source", "policy", "target_id"], sort=True).cumcount() + 1
    if limit_targets > 0:
        kept = []
        for _, group in df.groupby(["source", "policy"], sort=True):
            kept.append(group.head(limit_targets * top_k))
        df = pd.concat(kept, ignore_index=True)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/bindingmoad_pocketflow_n16_v100_dockfast_selection.csv")
    ap.add_argument("--gnina-bin", default="tools/gnina")
    ap.add_argument("--policies", nargs="+", default=["pb_qed", "pb_rc_select"])
    ap.add_argument("--top-k-per-target", type=int, default=4)
    ap.add_argument("--limit-targets", type=int, default=100)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--work-dir", default="results/gnina_redock_bindingmoad_v100_top4")
    ap.add_argument("--out-csv", default="results/gnina_redock_bindingmoad_v100_top4_scores.csv")
    ap.add_argument("--out-summary", default="results/gnina_redock_bindingmoad_v100_top4_summary.csv")
    ap.add_argument("--out-md", default="experiments/GNINA_REDOCK_BINDINGMOAD_V100_TOP4.md")
    args = ap.parse_args()

    df = load_bindingmoad_selection(args.input, args.policies, args.top_k_per_target, args.limit_targets)
    records = [row._asdict() for row in df.itertuples(index=False)]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = [
            pool.submit(
                redock_one,
                record,
                args.gnina_bin,
                args.work_dir,
                args.exhaustiveness,
                args.timeout,
                [],
            )
            for record in records
        ]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 25 == 0:
                print(f"bindingmoad_gnina_redock_done={i}/{len(records)}", flush=True)
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    result_summary = summarize(out)
    result_summary.to_csv(args.out_summary, index=False)
    write_report(result_summary, args)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
