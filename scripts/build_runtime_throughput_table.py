import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TIME_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})::")


def f2(x):
    return "NA" if pd.isna(x) else f"{x:.2f}"


def parse_elapsed(text):
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    if not match:
        return np.nan
    parts = match.group(1).split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(parts[0])
    except ValueError:
        return np.nan


def parse_time_v(path):
    p = Path(path)
    if not p.exists():
        return {"wall_sec": np.nan, "max_rss_mb": np.nan}
    text = p.read_text(encoding="utf-8", errors="ignore")
    wall = parse_elapsed(text)
    rss = np.nan
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match:
        rss = int(match.group(1)) / 1024.0
    return {"wall_sec": wall, "max_rss_mb": rss}


def parse_log_times(path):
    p = Path(path)
    if not p.exists():
        return None
    times = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TIME_RE.search(line)
        if match:
            base, ms = match.groups()
            times.append(datetime.strptime(f"{base}.{ms}", "%Y-%m-%d %H:%M:%S.%f"))
    if len(times) < 2:
        return None
    return min(times), max(times), (max(times) - min(times)).total_seconds()


def gpu_mem_snapshot(path):
    p = Path(path)
    if not p.exists():
        return np.nan
    vals = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            try:
                vals.append(float(parts[2]))
            except ValueError:
                pass
    return max(vals) if vals else np.nan


def generation_row(summary_path, gpu_snapshot):
    df_all = pd.read_csv(summary_path)
    df_all = df_all[(df_all["returncode"] == 0) & (df_all["written_sdf"] > 0)].copy()
    if "source_batch" in df_all.columns:
        df = df_all[df_all["source_batch"] == "prospective20_current"].copy()
    else:
        df = df_all.copy()
    per_target = []
    batch_bounds = {}
    for log_path in df["log_path"]:
        parsed = parse_log_times(log_path)
        if parsed is None:
            continue
        start, end, elapsed = parsed
        batch = str(Path(log_path).parent)
        if batch not in batch_bounds:
            batch_bounds[batch] = [start, end]
        else:
            batch_bounds[batch][0] = min(batch_bounds[batch][0], start)
            batch_bounds[batch][1] = max(batch_bounds[batch][1], end)
        per_target.append(elapsed)
    wall = (
        sum((end - start).total_seconds() for start, end in batch_bounds.values())
        if batch_bounds
        else np.nan
    )
    molecules = int(df["written_sdf"].sum())
    return {
        "stage": "Pocket2Mol n128 generation/current targets",
        "items": int(len(df)),
        "molecules": molecules,
        "wall_sec": wall,
        "median_target_sec": float(np.median(per_target)) if per_target else np.nan,
        "throughput_mol_per_sec": molecules / wall if wall and wall > 0 else np.nan,
        "max_rss_mb": np.nan,
        "gpu_mem_mb": gpu_mem_snapshot(gpu_snapshot),
        "source": summary_path,
    }


def time_row(stage, time_path, count_path=None, count_kind="rows"):
    meta = parse_time_v(time_path)
    count = np.nan
    if count_path and Path(count_path).exists():
        df = pd.read_csv(count_path)
        if count_kind == "generated" and "kind" in df.columns:
            count = int((df["kind"] == "generated").sum())
        else:
            count = int(len(df))
    wall = meta["wall_sec"]
    return {
        "stage": stage,
        "items": count,
        "molecules": count,
        "wall_sec": wall,
        "median_target_sec": np.nan,
        "throughput_mol_per_sec": count / wall if pd.notna(count) and wall and wall > 0 else np.nan,
        "max_rss_mb": meta["max_rss_mb"],
        "gpu_mem_mb": np.nan,
        "source": time_path,
    }


def write_report(df, out_md):
    lines = [
        "# Runtime / Memory / Throughput",
        "",
        "## Protocol",
        "",
        "- CPU stages use `/usr/bin/time -v` maximum resident set size and wall-clock time.",
        "- Pocket2Mol generation wall time is reconstructed from per-target log timestamps; GPU memory is an observed active-process snapshot during the extra prospective run, not a profiler peak.",
        "- Throughput is computed as processed/generated molecules per wall-clock second when a molecule count is meaningful.",
        "",
        "## Table",
        "",
        "| Stage | Items | Molecules | Wall sec | Median target sec | Throughput mol/s | Max RSS MB | GPU memory MB | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.items} | {row.molecules} | {f2(row.wall_sec)} | {f2(row.median_target_sec)} | "
            f"{f2(row.throughput_mol_per_sec)} | {f2(row.max_rss_mb)} | {f2(row.gpu_mem_mb)} | `{row.source}` |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. This table separates GPU generation cost from CPU oracle/selection cost.",
            "2. The GPU memory number should be reported as an observed runtime snapshot unless a dedicated profiler run is added.",
            "3. Selection and analysis stages are lightweight relative to molecule generation and external oracles.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prospective-summary", default="results/prospective20_pocket2mol_n128_success20_run_summary.csv")
    ap.add_argument("--gpu-snapshot", default="logs/runtime/gpu_memory_snapshot_during_prospective20_extra.csv")
    ap.add_argument("--out-csv", default="results/runtime_memory_throughput.csv")
    ap.add_argument("--out-md", default="experiments/RUNTIME_MEMORY_THROUGHPUT.md")
    args = ap.parse_args()

    rows = [
        generation_row(args.prospective_summary, args.gpu_snapshot),
        time_row("Fusion missing/noisy robustness", "logs/runtime/fusion_oracle_robustness_time.txt", "results/fusion_oracle_robustness.csv"),
        time_row("Prospective20 risk scoring", "logs/runtime/prospective20_risk_time.txt", "results/prospective20_pocket2mol_n128_risk_scores.csv", "generated"),
        time_row("Prospective20 mol_fast", "logs/runtime/prospective20_molfast_time.txt", "results/prospective20_pocket2mol_n128_molfast.csv"),
        time_row("Prospective20 dock_fast selection", "logs/runtime/prospective20_dockfast_time.txt", "results/prospective20_pocket2mol_n128_dockfast_selection.csv"),
        time_row("Contact counterfactual faithfulness", "logs/runtime/contact_counterfactual_time.txt", "results/contact_counterfactual_faithfulness.csv"),
    ]
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    write_report(out, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
