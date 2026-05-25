from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from xgboost import XGBRanker


SEEDS = [20260525, 20260526, 20260527, 20260528, 20260529]
ALPHA = 0.10

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


def load_pool(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "kind" in df.columns:
        df = df[df["kind"].astype(str).str.lower().eq("generated")].copy()
    df["source"] = source
    df["target_key"] = df["key"].astype(str)
    for col in ["intramol_pass", "protein_pass", "dock_pose_pass"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df.reset_index(drop=True)


def target_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    targets = np.array(sorted(df["target_key"].unique()))
    rng.shuffle(targets)
    n = len(targets)
    n_train = max(1, int(0.60 * n))
    n_cal = max(1, int(0.20 * n))
    train_t = set(targets[:n_train])
    cal_t = set(targets[n_train : n_train + n_cal])
    test_t = set(targets[n_train + n_cal :])
    if not test_t:
        test_t = set(targets[-max(1, n // 5) :])
        train_t = set(targets) - cal_t - test_t
    return (
        df[df["target_key"].isin(train_t)].copy(),
        df[df["target_key"].isin(cal_t)].copy(),
        df[df["target_key"].isin(test_t)].copy(),
    )


def prepare_xy(train: pd.DataFrame, view: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, list[int] | None, pd.Series]:
    med = train[FEATURES].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x = view[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    for col in FEATURES:
        x[col] = x[col].fillna(med.get(col, 0.0)).astype(float)
    y = None
    groups = None
    if "dock_pose_pass" in view.columns:
        y = view["dock_pose_pass"].astype(int).to_numpy()
        groups = view.groupby("target_key", sort=False).size().astype(int).tolist()
    return x.to_numpy(float), y, groups, med


def sorted_for_ranker(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["target_key", "mol_index"], kind="mergesort").reset_index(drop=True)


def fit_lgbm(train: pd.DataFrame, seed: int) -> tuple[LGBMRanker, pd.Series]:
    train = sorted_for_ranker(train)
    x, y, groups, med = prepare_xy(train, train)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=160,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=seed,
        verbose=-1,
    )
    model.fit(x, y, group=groups)
    return model, med


def fit_xgb(train: pd.DataFrame, seed: int) -> tuple[XGBRanker, pd.Series]:
    train = sorted_for_ranker(train)
    x, y, groups, med = prepare_xy(train, train)
    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.90,
        colsample_bytree=0.90,
        random_state=seed,
        tree_method="hist",
    )
    model.fit(x, y, group=groups, verbose=False)
    return model, med


def predict_model(model, med: pd.Series, df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    for col in FEATURES:
        x[col] = x[col].fillna(med.get(col, 0.0)).astype(float)
    return np.asarray(model.predict(x.to_numpy(float)), dtype=float)


def add_scores(df: pd.DataFrame, models: dict, tau: float) -> pd.DataFrame:
    out = df.copy()
    out["score_rc"] = 1.0 - out["risk_prob"].astype(float).clip(0, 1)
    for name, payload in models.items():
        model, med = payload
        pred = predict_model(model, med, out)
        out[f"score_{name}"] = pred
    out["tau"] = tau
    return out


def select_one(group: pd.DataFrame, policy: str) -> pd.DataFrame:
    pool = group.copy()
    if policy.startswith("pb_") or policy.startswith("constrained_"):
        pb = pool[pool["intramol_pass"].fillna(False).astype(bool)].copy()
        if len(pb):
            pool = pb
    if policy.startswith("constrained_"):
        safe = pool[pool["risk_prob"].astype(float) <= float(pool["tau"].iloc[0])].copy()
        if len(safe):
            pool = safe
    if policy in {"pb_rc", "rc"}:
        score_col = "score_rc"
        order = [score_col, "qed"]
    elif "lambdamart" in policy:
        score_col = "score_lambdamart"
        order = [score_col, "qed", "score_rc"]
    elif "xgbrank" in policy:
        score_col = "score_xgbrank"
        order = [score_col, "qed", "score_rc"]
    else:
        raise ValueError(policy)
    return pool.sort_values(order, ascending=[False] * len(order)).head(1)


def top_one(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    return pd.concat([select_one(group, policy) for _, group in df.groupby("target_key", sort=True)], axis=0)


def calibrate_threshold(calibration: pd.DataFrame, policy: str) -> float:
    ranked = top_one(calibration, policy)
    if policy in {"pb_rc", "rc"}:
        score_col = "score_rc"
    elif "lambdamart" in policy:
        score_col = "score_lambdamart"
    elif "xgbrank" in policy:
        score_col = "score_xgbrank"
    else:
        raise ValueError(policy)
    candidates = []
    for threshold in sorted(ranked[score_col].dropna().unique()):
        accepted = ranked[ranked[score_col] >= threshold]
        if len(accepted) < 3:
            continue
        failure = 1.0 - float(accepted["dock_pose_pass"].mean())
        candidates.append((failure <= ALPHA, len(accepted), -failure, -float(threshold), float(threshold)))
    if not candidates:
        return np.inf
    feasible = [row for row in candidates if row[0]]
    return max(feasible if feasible else candidates, key=lambda x: x[1:4])[-1]


def summarize_selection(sel: pd.DataFrame, total_targets: int) -> dict[str, float]:
    covered = int(sel["target_key"].nunique())
    return {
        "targets": total_targets,
        "covered_targets": covered,
        "coverage": covered / total_targets if total_targets else np.nan,
        "dock_fast": float(sel["dock_pose_pass"].mean()) if covered else np.nan,
        "failure": float(1.0 - sel["dock_pose_pass"].mean()) if covered else np.nan,
        "risk_gt_0_5": float((sel["risk_prob"] > 0.5).mean()) if covered else np.nan,
        "qed": float(sel["qed"].mean()) if covered else np.nan,
    }


def evaluate_fixed(test: pd.DataFrame, policy: str) -> dict[str, float]:
    ranked = top_one(test, policy)
    out = summarize_selection(ranked, int(test["target_key"].nunique()))
    out["mode"] = "fixed_top1"
    out["threshold"] = np.nan
    return out


def evaluate_selective(cal: pd.DataFrame, test: pd.DataFrame, policy: str) -> dict[str, float]:
    threshold = calibrate_threshold(cal, policy)
    ranked = top_one(test, policy)
    if policy in {"pb_rc", "rc"}:
        score_col = "score_rc"
    elif "lambdamart" in policy:
        score_col = "score_lambdamart"
    else:
        score_col = "score_xgbrank"
    accepted = ranked[ranked[score_col] >= threshold].copy()
    out = summarize_selection(accepted, int(test["target_key"].nunique()))
    out["mode"] = "selective_alpha10"
    out["threshold"] = threshold
    return out


def run_within(df: pd.DataFrame, source: str, tau: float) -> list[dict]:
    rows = []
    for seed in SEEDS:
        train, cal, test = target_split(df, seed)
        models = {"lambdamart": fit_lgbm(train, seed), "xgbrank": fit_xgb(train, seed)}
        cal_s = add_scores(cal, models, tau)
        test_s = add_scores(test, models, tau)
        for policy in ["pb_rc", "pb_lambdamart", "constrained_lambdamart", "pb_xgbrank", "constrained_xgbrank"]:
            for out in [evaluate_fixed(test_s, policy), evaluate_selective(cal_s, test_s, policy)]:
                out.update({"setting": f"{source}_target_heldout", "seed": seed, "policy": policy})
                rows.append(out)
    return rows


def run_transfer(source_df: pd.DataFrame, target_df: pd.DataFrame, direction: str, tau: float) -> list[dict]:
    rows = []
    for seed in SEEDS:
        train, cal, _ = target_split(source_df, seed)
        models = {"lambdamart": fit_lgbm(train, seed), "xgbrank": fit_xgb(train, seed)}
        cal_s = add_scores(cal, models, tau)
        target_s = add_scores(target_df, models, tau)
        for policy in ["pb_rc", "pb_lambdamart", "constrained_lambdamart", "pb_xgbrank", "constrained_xgbrank"]:
            for out in [evaluate_fixed(target_s, policy), evaluate_selective(cal_s, target_s, policy)]:
                out.update({"setting": direction, "seed": seed, "policy": policy})
                rows.append(out)
    return rows


def write_report(summary: pd.DataFrame, out_md: str) -> None:
    view = summary[(summary["mode"] == "selective_alpha10") & (summary["setting"].str.contains("to|target_heldout", regex=True))].copy()
    lines = [
        "# Learning-to-Rank Selection Baselines",
        "",
        "## Protocol",
        "",
        "- Grouped ranking baselines are trained at target level using LightGBM LambdaMART and XGBoost Ranker.",
        "- Features match the available selection evidence and exclude the dock-fast label.",
        "- The rankers receive the same intramolecular PoseBusters prefilter and a risk-constrained variant.",
        "- Selective rows calibrate a target-level failure budget of alpha=0.10 on held-out calibration targets.",
        "",
        "## Selective summary",
        "",
        "| Setting | Policy | Coverage | dock_fast | Failure | Risk >0.5 | QED |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in view.sort_values(["setting", "policy"]).itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.policy} | {row.coverage_mean:.3f} | {row.dock_fast_mean:.3f} | "
            f"{row.failure_mean:.3f} | {row.risk_gt_0_5_mean:.3f} | {row.qed_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Learning-to-rank baselines test whether RC-SBDD is merely an ordinary ranker. "
            "If a ranker wins a fixed-budget table but loses calibrated selective behavior, the manuscript should claim calibrated control rather than universal rank dominance.",
        ]
    )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    diff = load_pool("results/dockfast_full_pool_fullatom_cond.csv", "DiffSBDD")
    pocket = load_pool("results/dockfast_full_pool_pocket2mol_n16_ext.csv", "Pocket2Mol")
    native = pd.read_csv("results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
    tau = float(native[native["kind"] == "native"]["risk_prob"].quantile(0.95))
    rows = []
    rows += run_within(diff, "DiffSBDD", tau)
    rows += run_within(pocket, "Pocket2Mol", tau)
    rows += run_transfer(diff, pocket, "DiffSBDD_to_Pocket2Mol", tau)
    rows += run_transfer(pocket, diff, "Pocket2Mol_to_DiffSBDD", tau)
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["setting", "policy", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        targets_mean=("targets", "mean"),
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        dock_fast_mean=("dock_fast", "mean"),
        dock_fast_std=("dock_fast", "std"),
        failure_mean=("failure", "mean"),
        risk_gt_0_5_mean=("risk_gt_0_5", "mean"),
        qed_mean=("qed", "mean"),
        threshold_mean=("threshold", "mean"),
    )
    Path("results").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    raw.to_csv("results/learning_to_rank_selection_raw.csv", index=False)
    agg.to_csv("results/learning_to_rank_selection_summary.csv", index=False)
    write_report(agg, "experiments/LEARNING_TO_RANK_SELECTION_BASELINES.md")
    print(agg[agg["mode"] == "selective_alpha10"].to_string(index=False))


if __name__ == "__main__":
    main()
