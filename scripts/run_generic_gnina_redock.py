from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scripts.evaluate_gnina_redock_selection import as_bool, redock_one, summarize, write_report


def load_selection(path: str, source: str, policies: list[str], top_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["policy"].isin(policies)].copy()
    df["source"] = source
    df["target_id"] = df["key"].astype(str) if "key" in df.columns else df["data_id"].astype(str)
    df = df.dropna(subset=["mol_pred", "mol_cond"]).copy()
    df = df[
        df["mol_pred"].astype(str).map(lambda value: Path(value).exists())
        & df["mol_cond"].astype(str).map(lambda value: Path(value).exists())
    ].copy()
    df["dock_pose_pass_bool"] = as_bool(df["dock_pose_pass"])
    df = df.sort_values(
        ["source", "policy", "target_id", "qed", "risk_prob"],
        ascending=[True, True, True, False, True],
    )
    df = df.groupby(["source", "policy", "target_id"], sort=True).head(top_k)
    df["selection_rank"] = df.groupby(["source", "policy", "target_id"], sort=True).cumcount() + 1
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gnina-bin", default="tools/gnina")
    ap.add_argument("--policies", nargs="+", default=["pb_qed", "pb_rc_select"])
    ap.add_argument("--top-k-per-target", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    data = load_selection(args.input, args.source, args.policies, args.top_k_per_target)
    records = [row._asdict() for row in data.itertuples(index=False)]
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
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 25 == 0:
                print(f"gnina_redock_done={index}/{len(records)}", flush=True)
    result = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)
    summary = summarize(result)
    summary.to_csv(args.out_summary, index=False)
    write_report(summary, args)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
