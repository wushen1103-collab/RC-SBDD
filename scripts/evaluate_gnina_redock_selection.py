import argparse
import hashlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


CNN_SCORE_RE = re.compile(r"CNNscore\s*[:=]\s*([-+0-9.eE]+)")
CNN_AFF_RE = re.compile(r"CNNaffinity\s*[:=]\s*([-+0-9.eE]+)")
AFF_RE = re.compile(r"Affinity\s*[:=]\s*([-+0-9.eE]+)")


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def short_hash(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


def as_bool(series):
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def parse_output(text):
    cnn_score = CNN_SCORE_RE.search(text)
    cnn_aff = CNN_AFF_RE.search(text)
    aff = AFF_RE.search(text)
    return {
        "gnina_affinity": float(aff.group(1)) if aff else np.nan,
        "gnina_cnnscore": float(cnn_score.group(1)) if cnn_score else np.nan,
        "gnina_cnnaffinity": float(cnn_aff.group(1)) if cnn_aff else np.nan,
    }


def parse_output_sdf(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return {}
    try:
        mol = next((m for m in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False) if m is not None), None)
        if mol is None:
            return {}
        mapping = {
            "minimizedAffinity": "gnina_affinity",
            "CNNscore": "gnina_cnnscore",
            "CNNaffinity": "gnina_cnnaffinity",
            "CNN_VS": "gnina_cnn_vs",
            "CNNaffinity_variance": "gnina_cnnaffinity_variance",
        }
        out = {}
        for prop, col in mapping.items():
            if mol.HasProp(prop):
                out[col] = float(mol.GetProp(prop))
        return out
    except Exception:
        return {}


def load_selection_file(path, source, policies):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    if "policy" not in df.columns:
        return pd.DataFrame()
    df = df[df["policy"].isin(policies)].copy()
    if df.empty:
        return df
    df["source"] = source
    if "target_id" not in df.columns:
        df["target_id"] = df["key"].astype(str) if "key" in df.columns else df.get("data_id", df.index).astype(str)
    for col in ["mol_pred", "mol_cond", "risk_prob", "qed", "dock_pose_pass"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_core_selection(policies, limit_targets, sources, top_k_per_target):
    files = [
        ("results/posebusters_dockfast_selection.csv", "DiffSBDD_official"),
        ("results/posebusters_dockfast_pb_selection.csv", "DiffSBDD_official"),
        ("results/pocket2mol_crossgen_n16_ext_dockfast_selection.csv", "Pocket2Mol_transfer"),
        ("results/syncguide_t1000_n16_dockfast_selection.csv", "SYNC-Guide"),
        ("results/pocketflow_crossdock_n16_dockfast_selection.csv", "PocketFlow"),
    ]
    parts = [load_selection_file(path, source, policies) for path, source in files]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise FileNotFoundError("No selection CSVs found for GNINA redocking.")
    df = pd.concat(parts, ignore_index=True, sort=False)
    df = df.dropna(subset=["mol_pred", "mol_cond"]).copy()
    df = df[df["mol_pred"].astype(str).map(lambda x: Path(x).exists()) & df["mol_cond"].astype(str).map(lambda x: Path(x).exists())]
    if sources:
        allowed = set(sources)
        df = df[df["source"].isin(allowed)].copy()
        if df.empty:
            raise ValueError(f"No rows left after source filter: {sorted(allowed)}")
    df["dock_pose_pass_bool"] = as_bool(df["dock_pose_pass"]) if "dock_pose_pass" in df.columns else False
    df = df.sort_values(["source", "policy", "target_id", "qed", "risk_prob"], ascending=[True, True, True, False, True])
    df = df.groupby(["source", "policy", "target_id"], sort=True).head(top_k_per_target).copy()
    if limit_targets > 0:
        kept = []
        for (source, policy), group in df.groupby(["source", "policy"], sort=True):
            kept.append(group.head(limit_targets * top_k_per_target))
        df = pd.concat(kept, ignore_index=True)
    return df.reset_index(drop=True)


def redock_one(record, gnina_bin, work_dir, exhaustiveness, timeout, extra_args):
    rid = short_hash(f"{record['source']}|{record['policy']}|{record['target_id']}|{record['mol_pred']}")
    out_sdf = Path(work_dir) / record["source"] / record["policy"] / f"{rid}.sdf"
    log_path = out_sdf.with_suffix(".log")
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    if out_sdf.exists() and out_sdf.stat().st_size > 0:
        parsed = parse_output_sdf(out_sdf)
        record.update(parsed)
        record.update(
            {
                "gnina_redock_success": True,
                "gnina_returncode": 0,
                "gnina_redocked_sdf": str(out_sdf),
                "gnina_log": str(log_path),
                "gnina_error": "",
                "skipped_existing": True,
            }
        )
        return record
    cmd = [
        gnina_bin,
        "-r",
        str(record["mol_cond"]),
        "-l",
        str(record["mol_pred"]),
        "--autobox_ligand",
        str(record["mol_pred"]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "1",
        "-o",
        str(out_sdf),
    ] + list(extra_args)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text("command=" + " ".join(cmd) + "\n\n" + text, encoding="utf-8", errors="ignore")
        parsed = parse_output(text)
        success = proc.returncode == 0 and out_sdf.exists() and out_sdf.stat().st_size > 0
        if success:
            parsed.update(parse_output_sdf(out_sdf))
        record.update(parsed)
        record.update(
            {
                "gnina_redock_success": success,
                "gnina_returncode": proc.returncode,
                "gnina_redocked_sdf": str(out_sdf) if out_sdf.exists() else "",
                "gnina_log": str(log_path),
                "gnina_error": "" if success else text[-1000:],
                "skipped_existing": False,
            }
        )
    except Exception as exc:
        record.update(
            {
                "gnina_affinity": np.nan,
                "gnina_cnnscore": np.nan,
                "gnina_cnnaffinity": np.nan,
                "gnina_redock_success": False,
                "gnina_returncode": -1,
                "gnina_redocked_sdf": "",
                "gnina_log": str(log_path),
                "gnina_error": f"{type(exc).__name__}: {exc}",
                "skipped_existing": False,
            }
        )
    return record


def summarize(df):
    rows = []
    for (source, policy), group in df.groupby(["source", "policy"], sort=True):
        ok = group[group["gnina_redock_success"].astype(bool)].copy()
        rows.append(
            {
                "source": source,
                "policy": policy,
                "attempted": int(len(group)),
                "redocked": int(len(ok)),
                "success_rate": float(len(ok) / len(group)) if len(group) else np.nan,
                "cnnscore_mean": float(ok["gnina_cnnscore"].mean()) if len(ok) else np.nan,
                "cnnscore_median": float(ok["gnina_cnnscore"].median()) if len(ok) else np.nan,
                "cnnaffinity_mean": float(ok["gnina_cnnaffinity"].mean()) if len(ok) else np.nan,
                "affinity_mean": float(ok["gnina_affinity"].mean()) if len(ok) else np.nan,
                "dock_fast_pass": float(ok["dock_pose_pass_bool"].mean()) if len(ok) else np.nan,
                "risk_gt_0_5": float((ok["risk_prob"].astype(float) > 0.5).mean()) if len(ok) else np.nan,
                "qed_mean": float(ok["qed"].astype(float).mean()) if len(ok) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary, args):
    lines = [
        "# GNINA Local Redocking",
        "",
        "## Protocol",
        "",
        f"- GNINA command: receptor + selected ligand with `--autobox_ligand`, `--exhaustiveness {args.exhaustiveness}`, `--num_modes 1`.",
        f"- Unit: top-{args.top_k_per_target} molecule(s) per target, generator, and policy.",
        "- This is stronger than `--score_only` because the ligand is locally optimized before neural docking scores are parsed.",
        "",
        "## Summary",
        "",
        "| Source | Policy | Attempted | Redocked | Success | CNNscore mean | CNNaffinity mean | Affinity mean | dock_fast | Risk >0.5 | QED |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["source", "policy"]).itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.policy} | {row.attempted} | {row.redocked} | {pct(row.success_rate)} | "
            f"{f4(row.cnnscore_mean)} | {f4(row.cnnaffinity_mean)} | {f4(row.affinity_mean)} | "
            f"{pct(row.dock_fast_pass)} | {pct(row.risk_gt_0_5)} | {f4(row.qed_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "1. GNINA redocking is used as an orthogonal high-fidelity sanity check, not as the optimized target of RC.",
            "2. A strong result is that RC improves geometry reliability while retaining competitive neural docking plausibility.",
        ]
    )
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnina-bin", default="tools/gnina")
    ap.add_argument("--policies", nargs="+", default=["qed", "pb_qed", "rc_select", "pb_rc_select"])
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--limit-targets", type=int, default=100)
    ap.add_argument("--top-k-per-target", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--exhaustiveness", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--extra-args", nargs="*", default=[])
    ap.add_argument("--work-dir", default="results/gnina_redock_selection")
    ap.add_argument("--out-csv", default="results/gnina_redock_selection_scores.csv")
    ap.add_argument("--out-summary", default="results/gnina_redock_selection_summary.csv")
    ap.add_argument("--out-md", default="experiments/GNINA_REDOCK_SELECTION.md")
    args = ap.parse_args()

    gnina_bin = str(Path(args.gnina_bin))
    if not Path(gnina_bin).exists():
        raise FileNotFoundError(gnina_bin)
    df = load_core_selection(args.policies, args.limit_targets, args.sources, args.top_k_per_target)
    records = [row._asdict() for row in df.itertuples(index=False)]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = [pool.submit(redock_one, rec, gnina_bin, args.work_dir, args.exhaustiveness, args.timeout, args.extra_args) for rec in records]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 25 == 0:
                print(f"gnina_redock_done={i}/{len(records)}", flush=True)
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = summarize(out)
    summary.to_csv(args.out_summary, index=False)
    write_report(summary, args)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
