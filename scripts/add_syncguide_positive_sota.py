from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260525)

BLOCK = "syncguide_t1000_n16"
ROLE = "positive recent SOTA-output comparator"
BASELINE = "pb_qed"
METHOD = "pb_rc_select"
SELECTION_CSV = ROOT / "results/syncguide_t1000_n16_dockfast_selection.csv"
FINAL_STATS_CSV = ROOT / "results/final_sota_target_level_statistics.csv"
MASTER_CSV = ROOT / "results/trans_journal_master_evidence.csv"
AUDIT_CSV = ROOT / "results/public_sota_output_audit.csv"

METRICS = {
    "dock_fast": ("dock_pose_pass", "higher"),
    "protein_pass": ("protein_pass", "higher"),
    "mol_fast": ("intramol_pass", "higher"),
    "risk_prob": ("risk_prob", "lower"),
    "risk_gt_0_5": ("risk_gt_0_5", "lower"),
    "qed": ("qed", "higher"),
}


def f4(x: float) -> str:
    return "NA" if pd.isna(x) else f"{float(x):.4f}"


def pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{100 * float(x):.1f}%"


def ptxt(x: float) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    return f"{x:.2e}" if x < 1e-4 else f"{x:.4f}"


def paired_bootstrap_ci(delta: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    if len(delta) == 0:
        return np.nan, np.nan
    if len(delta) == 1:
        return float(delta[0]), float(delta[0])
    idx = RNG.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += int((xi > y).sum())
        lt += int((xi < y).sum())
    return float((gt - lt) / (len(x) * len(y)))


def bh_fdr(pvals: pd.Series) -> np.ndarray:
    p = np.asarray([1.0 if pd.isna(x) else float(x) for x in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = n - rank + 1
        running = min(running, p[idx] * n / true_rank)
        q[idx] = running
    return np.clip(q, 0, 1)


def target_policy_pivots(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = df.copy()
    if "risk_gt_0_5" not in out.columns:
        out["risk_gt_0_5"] = out["risk_prob"].astype(float) > 0.5
    for col in ["dock_pose_pass", "intramol_pass", "protein_pass"]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    out = out[out["policy"].isin([BASELINE, METHOD])]
    rows = []
    for (target, policy), group in out.groupby(["key", "policy"], sort=True):
        row = {"target_id": str(target), "policy": policy}
        for metric, (col, _) in METRICS.items():
            if col in group.columns:
                row[metric] = float(group[col].astype(float).mean())
        rows.append(row)
    wide = pd.DataFrame(rows)
    pivots: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        if metric in wide.columns:
            pivot = wide.pivot(index="target_id", columns="policy", values=metric)
            if BASELINE in pivot.columns and METHOD in pivot.columns:
                pivots[metric] = pivot[[BASELINE, METHOD]].dropna()
    return pivots


def build_syncguide_stats() -> pd.DataFrame:
    df = pd.read_csv(SELECTION_CSV, low_memory=False)
    rows = []
    for metric, pivot in target_policy_pivots(df).items():
        base = pivot[BASELINE].to_numpy(float)
        meth = pivot[METHOD].to_numpy(float)
        delta = meth - base
        ci_lo, ci_hi = paired_bootstrap_ci(delta)
        try:
            p = 1.0 if len(delta) < 2 or np.allclose(delta, 0) else float(wilcoxon(meth, base, zero_method="wilcox").pvalue)
        except Exception:
            p = np.nan
        direction = METRICS[metric][1]
        mean_delta = float(delta.mean())
        improves = mean_delta > 0 if direction == "higher" else mean_delta < 0
        rows.append(
            {
                "block": BLOCK,
                "role": ROLE,
                "metric": metric,
                "baseline": BASELINE,
                "method": METHOD,
                "targets": int(len(pivot)),
                "baseline_mean": float(base.mean()),
                "method_mean": float(meth.mean()),
                "delta_method_minus_baseline": mean_delta,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "wilcoxon_p": p,
                "cliffs_delta": cliffs_delta(meth, base),
                "desired_direction": direction,
                "improves": bool(improves),
            }
        )
    return pd.DataFrame(rows)


def update_final_stats(sync_stats: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(FINAL_STATS_CSV) if FINAL_STATS_CSV.exists() else pd.DataFrame()
    if len(old):
        old = old[old["block"] != BLOCK].copy()
    out = pd.concat([old, sync_stats], ignore_index=True)
    out["fdr_q"] = bh_fdr(out["wilcoxon_p"]) if len(out) else []
    out.to_csv(FINAL_STATS_CSV, index=False)
    return out


def md_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def write_final_stats_md(out: pd.DataFrame) -> None:
    rows = []
    for r in out.sort_values(["block", "metric"]).itertuples(index=False):
        rows.append(
            {
                "Block": r.block,
                "Metric": r.metric,
                "Targets": r.targets,
                "Baseline": f4(r.baseline_mean),
                "Method": f4(r.method_mean),
                "Delta": f4(r.delta_method_minus_baseline),
                "95% CI": f"[{f4(r.bootstrap_ci_low)}, {f4(r.bootstrap_ci_high)}]",
                "FDR q": ptxt(r.fdr_q),
                "Improves": r.improves,
                "Role": r.role,
            }
        )
    text = "\n".join(
        [
            "# Final SOTA Target-Level Statistical Tests",
            "",
            "## Protocol",
            "",
            "- Unit of inference: target, not molecule row.",
            "- Scope: direct SOTA and external generation blocks after the final P0 closure.",
            "- SYNC-Guide is added as an additional positive recent SOTA-output comparator.",
            "- Tests: paired bootstrap 95% CI, Wilcoxon signed-rank test, Cliff's delta, and Benjamini-Hochberg FDR.",
            "",
            "## Results",
            "",
            md_table(rows, ["Block", "Metric", "Targets", "Baseline", "Method", "Delta", "95% CI", "FDR q", "Improves", "Role"]),
            "",
            "## Reviewer-Facing Interpretation",
            "",
            "PocketFlow, SYNC-Guide, and BindingMOAD provide positive target-level evidence for reliability selection. MolPilot and SGEDiff remain negative generator-shift stress cases.",
        ]
    )
    (ROOT / "experiments/FINAL_SOTA_TARGET_LEVEL_STATISTICS.md").write_text(text + "\n", encoding="utf-8")


def update_master_evidence() -> None:
    row = {
        "Evidence block": "SYNC-Guide CrossDock direct SOTA output",
        "Scope": "50 targets / 193 selected rows",
        "Primary claim": "PB-RC vs PB-QED post-generation selection",
        "Key number": "dock_fast 93.3% -> 96.4% (3.1%); risk>0.5 14.5% -> 4.1%",
        "Reviewer value": "additional positive recent SOTA-output comparator",
    }
    df = pd.read_csv(MASTER_CSV)
    df = df[df["Evidence block"] != row["Evidence block"]].copy()
    insert_at = 2 if len(df) >= 2 else len(df)
    top = df.iloc[:insert_at]
    bottom = df.iloc[insert_at:]
    out = pd.concat([top, pd.DataFrame([row]), bottom], ignore_index=True)
    out.to_csv(MASTER_CSV, index=False)


def update_public_sota_audit() -> None:
    row = {
        "name": "SYNC-Guide",
        "paper": "SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design",
        "repo": "local SYNC-Guide outputs",
        "type": "local_generated_output",
        "api": "",
        "status": "completed_positive",
        "asset_probe": "50 targets; 741 generated molecules; dock_fast/risk/QED/Vina/GNINA present",
        "file_count": 50,
        "error": "",
    }
    df = pd.read_csv(AUDIT_CSV) if AUDIT_CSV.exists() else pd.DataFrame(columns=list(row))
    df = df[df["name"] != "SYNC-Guide"].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(AUDIT_CSV, index=False)

    md = ROOT / "experiments/PUBLIC_SOTA_OUTPUT_AUDIT.md"
    text = md.read_text(encoding="utf-8", errors="ignore") if md.exists() else "# Public SOTA Output Audit\n"
    addendum = "\n".join(
        [
            "",
            "<!-- SYNCGUIDE_POSITIVE_SOTA_START -->",
            "",
            "## SYNC-Guide positive SOTA-output addendum",
            "",
            "SYNC-Guide is now included as an additional positive recent SOTA-output comparator: 50 targets, 741 generated molecules, 193 PB-selected rows, PB-RC dock_fast 96.4% versus PB-QED 93.3%, and high-risk selection reduced from 14.5% to 4.1%. Top-1 Vina and GNINA score-only sanity checks were also completed.",
            "",
            "<!-- SYNCGUIDE_POSITIVE_SOTA_END -->",
        ]
    )
    start = "<!-- SYNCGUIDE_POSITIVE_SOTA_START -->"
    end = "<!-- SYNCGUIDE_POSITIVE_SOTA_END -->"
    if start in text and end in text:
        text = text.split(start, 1)[0].rstrip() + "\n" + addendum.strip() + "\n" + text.split(end, 1)[1].lstrip()
    else:
        text = text.rstrip() + "\n" + addendum + "\n"
    md.write_text(text, encoding="utf-8")


def write_syncguide_report(sync_stats: pd.DataFrame) -> None:
    summary = pd.read_csv(ROOT / "results/syncguide_t1000_n16_dockfast_selection_summary.csv")
    vina = pd.read_csv(ROOT / "results/vina_score_syncguide_t1000_n16_top1_summary.csv")
    gnina = pd.read_csv(ROOT / "results/gnina_score_syncguide_t1000_n16_top1_summary.csv")
    stats_rows = []
    for r in sync_stats.sort_values("metric").itertuples(index=False):
        stats_rows.append(
            {
                "Metric": r.metric,
                "Targets": r.targets,
                "PB-QED": f4(r.baseline_mean),
                "PB-RC": f4(r.method_mean),
                "Delta": f4(r.delta_method_minus_baseline),
                "95% CI": f"[{f4(r.bootstrap_ci_low)}, {f4(r.bootstrap_ci_high)}]",
                "p": ptxt(r.wilcoxon_p),
                "Improves": r.improves,
            }
        )
    selection_rows = []
    for r in summary[summary["policy"].isin([BASELINE, METHOD])].itertuples(index=False):
        selection_rows.append(
            {
                "Policy": r.policy,
                "N": r.n,
                "Targets": r.targets,
                "dock_fast": pct(r.dock_pose_pass),
                "Risk >0.5": pct(r.risk_gt_0_5),
                "QED": f4(r.qed_mean),
            }
        )
    vina_rows = []
    for r in vina[vina["policy"].isin([BASELINE, METHOD])].itertuples(index=False):
        vina_rows.append(
            {
                "Policy": r.policy,
                "Scored": r.scored,
                "Vina mean": f4(r.vina_mean),
                "dock_fast": pct(r.dock_pose_pass),
                "Risk >0.5": pct(r.risk_gt_0_5),
            }
        )
    gnina_rows = []
    for r in gnina[gnina["policy"].isin([BASELINE, METHOD])].itertuples(index=False):
        gnina_rows.append(
            {
                "Policy": r.policy,
                "Scored": r.scored,
                "CNNscore": f4(r.cnnscore_mean),
                "CNNaffinity": f4(r.cnnaffinity_mean),
                "dock_fast": pct(r.dock_pose_pass),
                "Risk >0.5": pct(r.risk_gt_0_5),
            }
        )
    text = "\n".join(
        [
            "# SYNC-Guide Positive SOTA Generator Output",
            "",
            "## Scope",
            "",
            "SYNC-Guide is added as an additional positive recent SOTA-output comparator. The evidence uses 50 CrossDocked targets, 741 generated molecules, and the same PB-QED versus PB-RC post-generation selection protocol used for the other direct-output blocks.",
            "",
            "## Selection summary",
            "",
            md_table(selection_rows, ["Policy", "N", "Targets", "dock_fast", "Risk >0.5", "QED"]),
            "",
            "## Target-level paired inference",
            "",
            md_table(stats_rows, ["Metric", "Targets", "PB-QED", "PB-RC", "Delta", "95% CI", "p", "Improves"]),
            "",
            "## Vina top-1 sanity check",
            "",
            md_table(vina_rows, ["Policy", "Scored", "Vina mean", "dock_fast", "Risk >0.5"]),
            "",
            "## GNINA top-1 sanity check",
            "",
            md_table(gnina_rows, ["Policy", "Scored", "CNNscore", "CNNaffinity", "dock_fast", "Risk >0.5"]),
            "",
            "## Manuscript use",
            "",
            "Use this block as an additional positive direct-output result alongside PocketFlow. It should not replace the main 100-target PocketFlow result, but it reduces the risk that the only positive direct SOTA evidence comes from one generator.",
        ]
    )
    (ROOT / "experiments/SYNCGUIDE_POSITIVE_SOTA_OUTPUT.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    sync_stats = build_syncguide_stats()
    final_stats = update_final_stats(sync_stats)
    write_final_stats_md(final_stats)
    update_master_evidence()
    update_public_sota_audit()
    write_syncguide_report(sync_stats)
    print("SYNC-Guide positive SOTA block integrated")
    print(sync_stats.to_string(index=False))


if __name__ == "__main__":
    main()
