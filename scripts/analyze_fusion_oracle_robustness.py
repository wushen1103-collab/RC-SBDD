import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


FEATURES = [
    "risk_prob",
    "qed",
    "lp_min",
    "center_dist",
    "contacts_lt_4_0_per_lig",
    "clash_lt_1_5_per_lig",
    "pock_n",
    "mol_wt",
    "heavy_atoms",
    "intramol_pass",
]

BLOCKS = {
    "risk": ["risk_prob"],
    "chemistry": ["qed", "mol_wt", "heavy_atoms"],
    "geometry": ["lp_min", "center_dist", "contacts_lt_4_0_per_lig", "clash_lt_1_5_per_lig", "pock_n"],
    "validity": ["intramol_pass"],
}

SCENARIOS = [
    "full",
    "missing_risk",
    "missing_geometry",
    "missing_validity",
    "noisy_risk_0_10",
    "noisy_geometry_0_25sd",
    "noisy_all_oracles",
]

POLICIES = ["qed", "pb_qed", "rc_select", "fusion", "pb_fusion", "fusion_rc"]


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def load_pool(path, source):
    df = pd.read_csv(path, low_memory=False)
    if "kind" in df.columns:
        df = df[df["kind"] == "generated"].copy()
    df["source"] = source
    df["target_key"] = df["key"].astype(str)
    df["intramol_pass"] = df["intramol_pass"].fillna(False).astype(bool)
    df["dock_pose_pass"] = df["dock_pose_pass"].fillna(False).astype(bool)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df["label_success"] = df["dock_pose_pass"].astype(int)
    return df


def native_threshold(path):
    df = pd.read_csv(path)
    return float(df[df["kind"] == "native"]["risk_prob"].quantile(0.95))


def split_targets(df, seed, test_frac):
    rng = np.random.default_rng(seed)
    targets = np.array(sorted(df["target_key"].unique()))
    rng.shuffle(targets)
    n_test = max(1, int(round(len(targets) * test_frac)))
    test_keys = set(targets[:n_test])
    return df[~df["target_key"].isin(test_keys)].copy(), df[df["target_key"].isin(test_keys)].copy()


def fit_model(train, seed):
    x = train[FEATURES].astype(float).to_numpy()
    y = train["label_success"].to_numpy(int)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, l2_regularization=0.01, random_state=seed)
    model.fit(x, y)
    return model


def medians_and_scales(train):
    med = train[FEATURES].astype(float).median(numeric_only=True)
    scale = train[FEATURES].astype(float).std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    return med, scale


def perturb(test, train, scenario, seed):
    out = test.copy()
    med, scale = medians_and_scales(train)
    rng = np.random.default_rng(seed)
    if scenario == "full":
        return out
    if scenario == "missing_risk":
        for col in BLOCKS["risk"]:
            out[col] = med[col]
        return out
    if scenario == "missing_geometry":
        for col in BLOCKS["geometry"]:
            out[col] = med[col]
        return out
    if scenario == "missing_validity":
        for col in BLOCKS["validity"]:
            out[col] = med[col]
        return out
    if scenario == "noisy_risk_0_10":
        out["risk_prob"] = np.clip(out["risk_prob"].astype(float) + rng.normal(0.0, 0.10, len(out)), 0.0, 1.0)
        return out
    if scenario == "noisy_geometry_0_25sd":
        for col in BLOCKS["geometry"]:
            out[col] = out[col].astype(float) + rng.normal(0.0, 0.25 * float(scale[col]), len(out))
        return out
    if scenario == "noisy_all_oracles":
        out["risk_prob"] = np.clip(out["risk_prob"].astype(float) + rng.normal(0.0, 0.10, len(out)), 0.0, 1.0)
        out["qed"] = np.clip(out["qed"].astype(float) + rng.normal(0.0, 0.05, len(out)), 0.0, 1.0)
        for col in BLOCKS["geometry"]:
            out[col] = out[col].astype(float) + rng.normal(0.0, 0.25 * float(scale[col]), len(out))
        mask = rng.random(len(out)) < 0.10
        vals = out["intramol_pass"].astype(bool).to_numpy()
        vals[mask] = ~vals[mask]
        out["intramol_pass"] = vals
        return out
    raise ValueError(scenario)


def predict_under_scenario(model, train, test, scenario, seed):
    pert = perturb(test, train, scenario, seed)
    x = pert[FEATURES].astype(float).to_numpy()
    pert["fusion_success_prob"] = model.predict_proba(x)[:, 1]
    return pert


def model_metrics(df):
    y = df["label_success"].to_numpy(int)
    pred = df["fusion_success_prob"].to_numpy(float)
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {"auroc": float(roc_auc_score(y, pred)), "auprc": float(average_precision_score(y, pred))}


def top_k(group, policy, k, tau):
    pool = group.copy()
    if policy.startswith("pb_"):
        pool = pool[pool["intramol_pass"].astype(bool)].copy()
    if policy in {"rc_select", "fusion_rc"}:
        pool = pool[pool["risk_prob"] <= tau].copy()
    if policy == "fusion_rc":
        pool = pool[pool["intramol_pass"].astype(bool)].copy()
    if pool.empty:
        return pool
    if policy in {"fusion", "pb_fusion", "fusion_rc"}:
        return pool.sort_values(["fusion_success_prob", "qed", "risk_prob"], ascending=[False, False, True]).head(k)
    return pool.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(k)


def evaluate_policies(df, k, tau):
    rows = []
    for policy in POLICIES:
        selected = []
        target_rows = []
        for key, group in df.groupby("target_key", sort=True):
            sel = top_k(group, policy, k, tau)
            if not sel.empty:
                selected.append(sel)
            target_rows.append({"target_key": key, "selected": len(sel), "reach_k": len(sel) >= k})
        target = pd.DataFrame(target_rows)
        if selected:
            out = pd.concat(selected, ignore_index=True)
            rows.append(
                {
                    "policy": policy,
                    "selected_n": int(len(out)),
                    "targets": int(df["target_key"].nunique()),
                    "reach_k": float(target["reach_k"].mean()),
                    "coverage": float((target["selected"] > 0).mean()),
                    "dock_pose_pass": float(out["dock_pose_pass"].mean()),
                    "risk_mean": float(out["risk_prob"].mean()),
                    "risk_gt_0_5": float((out["risk_prob"] > 0.5).mean()),
                    "qed_mean": float(out["qed"].mean()),
                    "intramol_pass": float(out["intramol_pass"].mean()),
                    "fusion_prob_mean": float(out["fusion_success_prob"].mean()),
                }
            )
        else:
            rows.append(
                {
                    "policy": policy,
                    "selected_n": 0,
                    "targets": int(df["target_key"].nunique()),
                    "reach_k": float(target["reach_k"].mean()),
                    "coverage": 0.0,
                    "dock_pose_pass": np.nan,
                    "risk_mean": np.nan,
                    "risk_gt_0_5": np.nan,
                    "qed_mean": np.nan,
                    "intramol_pass": np.nan,
                    "fusion_prob_mean": np.nan,
                }
            )
    return rows


def run_eval(train, test, direction, seed_idx, seed, k, tau):
    model = fit_model(train, seed)
    rows = []
    metrics = []
    for scenario in SCENARIOS:
        pred = predict_under_scenario(model, train, test, scenario, seed + 1000 * (SCENARIOS.index(scenario) + 1))
        mm = model_metrics(pred)
        metrics.append({"direction": direction, "seed": seed_idx, "scenario": scenario, **mm})
        for row in evaluate_policies(pred, k, tau):
            rows.append({"direction": direction, "seed": seed_idx, "scenario": scenario, **row})
    return rows, metrics


def aggregate(raw):
    metric_cols = [
        "selected_n",
        "targets",
        "reach_k",
        "coverage",
        "dock_pose_pass",
        "risk_mean",
        "risk_gt_0_5",
        "qed_mean",
        "intramol_pass",
        "fusion_prob_mean",
    ]
    rows = []
    for keys, group in raw.groupby(["direction", "scenario", "policy"], sort=True):
        row = dict(zip(["direction", "scenario", "policy"], keys))
        row["seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(metrics):
    rows = []
    for keys, group in metrics.groupby(["direction", "scenario"], sort=True):
        row = dict(zip(["direction", "scenario"], keys))
        row["seeds"] = int(group["seed"].nunique())
        for col in ["auroc", "auprc"]:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(agg, metrics, out_md):
    keep_policy = ["fusion", "pb_fusion", "fusion_rc", "rc_select", "pb_qed"]
    scenario_order = {s: i for i, s in enumerate(SCENARIOS)}
    policy_order = {p: i for i, p in enumerate(keep_policy)}
    view = agg[agg["policy"].isin(keep_policy)].copy()
    view["scenario_order"] = view["scenario"].map(scenario_order)
    view["policy_order"] = view["policy"].map(policy_order)
    metrics = metrics.copy()
    metrics["scenario_order"] = metrics["scenario"].map(scenario_order)
    lines = [
        "# Fusion Missing/Noisy Oracle Robustness",
        "",
        "## Protocol",
        "",
        "- Train the HGB fusion model with all oracle features, then corrupt only the test-time oracle inputs.",
        "- Missing risk/geometry/validity replaces that oracle block by the training-set median.",
        "- Noisy risk adds sigma=0.10 probability noise; noisy geometry adds 0.25 training-SD noise; noisy-all additionally perturbs QED and flips 10% validity labels.",
        "- Evaluation uses target-level splits and cross-generator transfer, with the same K and native risk threshold as the main fusion experiment.",
        "",
        "## Model Robustness",
        "",
        "| Direction | Scenario | Seeds | AUROC | AUPRC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["direction", "scenario_order"]).itertuples(index=False):
        lines.append(f"| {row.direction} | {row.scenario} | {row.seeds} | {f4(row.auroc_mean)} | {f4(row.auprc_mean)} |")
    lines.extend(
        [
            "",
            "## Selection Robustness",
            "",
            "| Direction | Scenario | Policy | Seeds | Reach K | dock_fast | Risk >0.5 | Mean QED | mol_fast |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in view.sort_values(["direction", "scenario_order", "policy_order"]).itertuples(index=False):
        lines.append(
            f"| {row.direction} | {row.scenario} | {row.policy} | {row.seeds} | {pct(row.reach_k_mean)} | "
            f"{pct(row.dock_pose_pass_mean)} | {pct(row.risk_gt_0_5_mean)} | {f4(row.qed_mean_mean)} | {pct(row.intramol_pass_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. This tests whether fusion depends on a single fragile oracle at deployment time.",
            "2. The manuscript should claim robustness only when fusion+RC remains close to the full-oracle result under missing/noisy blocks.",
            "3. If noisy-all degrades, it is a realistic limitation rather than a failure of the RC layer, because RC still supplies a thresholdable safety fallback.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffsbdd", default="results/dockfast_full_pool_fullatom_cond.csv")
    ap.add_argument("--pocket2mol", default="results/dockfast_full_pool_pocket2mol_n16_ext.csv")
    ap.add_argument("--official-risk", default="results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--out-csv", default="results/fusion_oracle_robustness.csv")
    ap.add_argument("--out-agg-csv", default="results/fusion_oracle_robustness_agg.csv")
    ap.add_argument("--out-model-csv", default="results/fusion_oracle_robustness_model_metrics.csv")
    ap.add_argument("--out-md", default="experiments/FUSION_ORACLE_ROBUSTNESS.md")
    args = ap.parse_args()

    diff = load_pool(args.diffsbdd, "DiffSBDD")
    pocket = load_pool(args.pocket2mol, "Pocket2Mol")
    tau = native_threshold(args.official_risk)
    rows = []
    metric_rows = []
    for i in range(args.seeds):
        seed = 20260516 + i
        for source, df in [("DiffSBDD", diff), ("Pocket2Mol", pocket)]:
            train, test = split_targets(df, seed, args.test_frac)
            part_rows, part_metrics = run_eval(train, test, f"{source}_within", i, seed, args.k, tau)
            rows.extend(part_rows)
            metric_rows.extend(part_metrics)
    for train, test, direction in [
        (diff, pocket, "DiffSBDD_to_Pocket2Mol"),
        (pocket, diff, "Pocket2Mol_to_DiffSBDD"),
    ]:
        part_rows, part_metrics = run_eval(train, test, direction, 0, 20260516, args.k, tau)
        rows.extend(part_rows)
        metric_rows.extend(part_metrics)

    raw = pd.DataFrame(rows)
    metrics = pd.DataFrame(metric_rows)
    agg = aggregate(raw)
    metric_agg = aggregate_metrics(metrics)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    agg.to_csv(args.out_agg_csv, index=False)
    metric_agg.to_csv(args.out_model_csv, index=False)
    write_report(agg, metric_agg, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
