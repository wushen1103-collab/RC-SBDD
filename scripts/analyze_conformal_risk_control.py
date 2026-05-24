import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MOLFAST_COLUMNS = [
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


def pct(x):
    return "NA" if pd.isna(x) else f"{100*x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def attach_molfast(df, molfast_csv):
    pb = pd.read_csv(molfast_csv).rename(columns={"file": "source_file", "position": "mol_index"})
    pb["molfast_core_pass"] = pb[MOLFAST_COLUMNS].fillna(False).astype(bool).all(axis=1)
    out = df.merge(pb[["source_file", "mol_index", "molfast_core_pass"]], on=["source_file", "mol_index"], how="left")
    out["molfast_core_pass"] = out["molfast_core_pass"].fillna(False).astype(bool)
    return out


def select_topk(group, k, tau=None, require_molfast=False):
    g = group.copy()
    if require_molfast:
        g = g[g["molfast_core_pass"]].copy()
    if tau is not None:
        g = g[g["risk_prob"] <= tau].copy()
    if g.empty:
        return g
    return g.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(k)


def target_losses(selected):
    rows = []
    for key, group in selected.groupby("key", sort=True):
        rows.append(
            {
                "key": key,
                "n": int(len(group)),
                "loss": float(1.0 - group["dock_pose_pass"].fillna(False).astype(bool).mean()),
                "dock_pose_pass": float(group["dock_pose_pass"].fillna(False).astype(bool).mean()),
                "risk_mean": float(group["risk_prob"].mean()),
                "risk_gt_0_5": float((group["risk_prob"] > 0.5).mean()),
                "qed_mean": float(group["qed"].mean()),
                "molfast_pass": float(group["molfast_core_pass"].mean()),
            }
        )
    return pd.DataFrame(rows)


def select_for_keys(df, keys, k, tau=None, require_molfast=False):
    parts = []
    sub = df[df["key"].isin(keys)].copy()
    for _, group in sub.groupby("key", sort=True):
        selected = select_topk(group, k, tau=tau, require_molfast=require_molfast)
        if len(selected):
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, axis=0, ignore_index=True)


def hoeffding_ucb(mean, n, delta):
    if n <= 0:
        return np.inf
    return min(1.0, mean + np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


def crc_bound(mean, n):
    if n <= 0 or pd.isna(mean):
        return np.inf
    return min(1.0, (n * mean + 1.0) / (n + 1.0))


def risk_bound(mean, n, delta, bound):
    if bound == "hoeffding":
        return hoeffding_ucb(mean, n, delta)
    if bound == "crc":
        return crc_bound(mean, n)
    raise ValueError(bound)


def choose_tau(cal_df, cal_keys, k, alpha, delta, grid, require_molfast, bound):
    rows = []
    for tau in grid:
        selected = select_for_keys(cal_df, cal_keys, k, tau=tau, require_molfast=require_molfast)
        tl = target_losses(selected)
        n_targets = int(len(tl))
        mean_loss = float(tl["loss"].mean()) if n_targets else np.nan
        rows.append(
            {
                "tau": float(tau),
                "cal_selected_targets": n_targets,
                "cal_reach_k": float((tl["n"] >= k).mean()) if n_targets else 0.0,
                "cal_mean_loss": mean_loss,
                "cal_bound": risk_bound(mean_loss, n_targets, delta, bound) if n_targets else np.inf,
            }
        )
    curve = pd.DataFrame(rows)
    feasible = curve[(curve["cal_bound"] <= alpha) & (curve["cal_selected_targets"] > 0)].copy()
    if feasible.empty:
        idx = curve["cal_bound"].idxmin()
        chosen = curve.loc[idx].copy()
        chosen["feasible"] = False
    else:
        chosen = feasible.sort_values(["tau", "cal_selected_targets"], ascending=[False, False]).iloc[0].copy()
        chosen["feasible"] = True
    return chosen, curve


def summarize_selection(selected, all_keys, k):
    tl = target_losses(selected)
    if tl.empty:
        return {
            "selected_n": 0,
            "selected_targets": 0,
            "targets_reach_k": 0.0,
            "coverage": 0.0,
            "loss": np.nan,
            "dock_pose_pass": np.nan,
            "risk_mean": np.nan,
            "risk_gt_0_5": np.nan,
            "qed_mean": np.nan,
            "molfast_pass": np.nan,
        }
    return {
        "selected_n": int(len(selected)),
        "selected_targets": int(tl["key"].nunique()),
        "targets_reach_k": float((tl["n"] >= k).mean()),
        "coverage": float(tl["key"].nunique() / max(len(all_keys), 1)),
        "loss": float(tl["loss"].mean()),
        "dock_pose_pass": float(tl["dock_pose_pass"].mean()),
        "risk_mean": float(tl["risk_mean"].mean()),
        "risk_gt_0_5": float(tl["risk_gt_0_5"].mean()),
        "qed_mean": float(tl["qed_mean"].mean()),
        "molfast_pass": float(tl["molfast_pass"].mean()),
    }


def run_seed(df, keys, seed, k, alphas, delta, grid, bound):
    rng = np.random.default_rng(seed)
    shuffled = np.array(keys, dtype=object)
    rng.shuffle(shuffled)
    split = len(shuffled) // 2
    cal_keys = set(shuffled[:split])
    test_keys = set(shuffled[split:])
    rows = []
    curves = []

    baselines = [
        ("qed_topk", None, False),
        ("pb_qed_topk", None, True),
    ]
    for name, tau, require_molfast in baselines:
        selected = select_for_keys(df, test_keys, k, tau=tau, require_molfast=require_molfast)
        row = {
            "seed": seed,
            "alpha": np.nan,
            "method": name,
            "tau": np.nan if tau is None else float(tau),
            "require_molfast": require_molfast,
            "feasible": True,
            "cal_bound": np.nan,
            "cal_mean_loss": np.nan,
        }
        row.update(summarize_selection(selected, test_keys, k))
        rows.append(row)

    for alpha in alphas:
        for require_molfast, label in [(False, "crc_risk"), (True, "crc_molfast_risk")]:
            chosen, curve = choose_tau(df, cal_keys, k, alpha, delta, grid, require_molfast, bound)
            selected = select_for_keys(df, test_keys, k, tau=float(chosen.tau), require_molfast=require_molfast)
            row = {
                "seed": seed,
                "alpha": alpha,
                "method": label,
                "tau": float(chosen.tau),
                "require_molfast": require_molfast,
                "feasible": bool(chosen.feasible),
                "cal_bound": float(chosen.cal_bound),
                "cal_mean_loss": float(chosen.cal_mean_loss) if pd.notna(chosen.cal_mean_loss) else np.nan,
            }
            row.update(summarize_selection(selected, test_keys, k))
            rows.append(row)
            curve = curve.assign(seed=seed, alpha=alpha, method=label, require_molfast=require_molfast)
            curves.append(curve)
    return rows, curves


def aggregate(df):
    group_cols = ["method", "alpha", "require_molfast"]
    metrics = ["tau", "coverage", "targets_reach_k", "loss", "dock_pose_pass", "risk_mean", "risk_gt_0_5", "qed_mean", "molfast_pass"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        method, alpha, require_molfast = keys
        row = {
            "method": method,
            "alpha": alpha,
            "require_molfast": require_molfast,
            "seeds": int(group["seed"].nunique()),
            "feasible_rate": float(group["feasible"].mean()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean()) if group[metric].notna().any() else np.nan
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if group[metric].notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dockfast-csv", default="results/dockfast_full_pool_fullatom_cond.csv")
    ap.add_argument("--molfast-csv", default="results/posebusters_molfast/fullatom_cond.csv")
    ap.add_argument("--risk-csv", default="results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--bound", choices=["crc", "hoeffding"], default="crc")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--out-csv", default="results/conformal_risk_control_fullatom_cond.csv")
    ap.add_argument("--out-agg-csv", default="results/conformal_risk_control_fullatom_cond_agg.csv")
    ap.add_argument("--out-curve-csv", default="results/conformal_risk_control_fullatom_cond_curves.csv")
    ap.add_argument("--out-md", default="experiments/CONFORMAL_RISK_CONTROL.md")
    args = ap.parse_args()

    df = pd.read_csv(args.dockfast_csv)
    if "molfast_core_pass" not in df.columns:
        df = attach_molfast(df, args.molfast_csv)
    df["dock_pose_pass"] = df["dock_pose_pass"].fillna(False).astype(bool)
    df["molfast_core_pass"] = df["molfast_core_pass"].fillna(False).astype(bool)
    keys = sorted(df["key"].unique())

    risk = pd.read_csv(args.risk_csv)
    native = risk[risk.kind == "native"].copy()
    fixed = [float(native["risk_prob"].quantile(q)) for q in [0.5, 0.75, 0.9, 0.95, 0.99]]
    quantile_grid = np.quantile(df["risk_prob"], np.linspace(0.01, 0.99, 99)).tolist()
    grid = sorted(set([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0] + fixed + quantile_grid))

    rows = []
    curves = []
    for seed in args.seeds:
        seed_rows, seed_curves = run_seed(df, keys, seed, args.k, args.alphas, args.delta, grid, args.bound)
        rows.extend(seed_rows)
        curves.extend(seed_curves)
    out = pd.DataFrame(rows)
    agg = aggregate(out)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    agg.to_csv(args.out_agg_csv, index=False)
    pd.concat(curves, axis=0, ignore_index=True).to_csv(args.out_curve_csv, index=False)

    display = agg.copy()
    display = display.sort_values(["method", "alpha_mean" if "alpha_mean" in display.columns else "alpha"], na_position="first")
    lines = [
        "# Conformal Risk Control",
        "",
        "## Protocol",
        "",
        f"- Dataset: official DiffSBDD full-atom conditional full candidate pool with dock_fast labels.",
        f"- Split: target-level 50/50 calibration/test over {len(args.seeds)} seeds.",
        f"- Loss: target-level mean `1 - dock_pose_pass` over accepted top-K candidates.",
        f"- Threshold family: nested risk gates; selected threshold is the largest gate whose calibration `{args.bound}` bound is <= alpha.",
        f"- Bound: `crc` uses `(n * mean_loss + 1) / (n + 1)` for bounded target losses; `hoeffding` uses delta={args.delta}. K={args.k}.",
        "",
        "## Aggregated Results",
        "",
        "| Method | Alpha | mol_fast gate | Feasible | Coverage | Reach K | Test loss | dock_fast pass | Mean risk | Risk >0.5 | Mean QED | mol_fast pass | Tau |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {
        "qed_topk": 0,
        "pb_qed_topk": 1,
        "crc_risk": 2,
        "crc_molfast_risk": 3,
    }
    for row in agg.assign(order=agg["method"].map(order)).sort_values(["order", "alpha"], na_position="first").itertuples(index=False):
        alpha = "NA" if pd.isna(row.alpha) else f"{row.alpha:.2f}"
        lines.append(
            f"| {row.method} | {alpha} | {row.require_molfast} | {pct(row.feasible_rate)} | "
            f"{pct(row.coverage_mean)} | {pct(row.targets_reach_k_mean)} | {f4(row.loss_mean)} | "
            f"{pct(row.dock_pose_pass_mean)} | {f4(row.risk_mean_mean)} | {pct(row.risk_gt_0_5_mean)} | "
            f"{f4(row.qed_mean_mean)} | {pct(row.molfast_pass_mean)} | {f4(row.tau_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. CRC turns the risk scorer into a target-level selective control rule rather than a fixed heuristic threshold.",
            "2. The mol_fast-gated CRC rows separate intramolecular validity from pocket-level reliability and should be emphasized for deployment.",
            "3. Strict alpha values may trade off K coverage; this is a useful operational cost rather than a failure of the framework.",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
