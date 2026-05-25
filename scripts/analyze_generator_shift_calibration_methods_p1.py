from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SOURCE_CANDIDATES = [
    ("DiffSBDD", ["results/posebusters_dockfast_selection.csv"]),
    ("DiffSBDD-PB", ["results/posebusters_dockfast_pb_selection.csv"]),
    ("Pocket2Mol", ["results/pocket2mol_crossgen_n16_ext_dockfast_selection.csv"]),
    ("SYNC-Guide", ["results/syncguide_t1000_n16_dockfast_selection.csv"]),
    ("PocketFlow", ["results/pocketflow_crossdock_n16_dockfast_selection.csv"]),
    (
        "MolCRAFT",
        [
            "results/molcraft_crossdock_t100_n16_dockfast_selection.csv",
            "results/molcraft_crossdock_t50_n16_dockfast_selection.csv",
        ],
    ),
    ("MolPilot-framefix", ["results/molpilot_crossdock_t50_n16_framefix_dockfast_selection.csv"]),
]
POLICY = "pb_rc_select"
CALIB_SIZES = [0, 5, 10, 20]
N_SEEDS = 20


def as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def ece_score(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1.0 else y_prob < hi)
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(y_prob[mask]) - np.mean(y_true[mask]))
    return float(ece)


def metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    result = {
        "test_n": int(len(y_true)),
        "failure_rate": float(np.mean(y_true)),
        "mean_pred_failure": float(np.mean(y_prob)),
        "ece": ece_score(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auroc": np.nan,
        "auprc": np.nan,
    }
    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
        result["auprc"] = float(average_precision_score(y_true, y_prob))
    return result


def first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return path
    return None


def load_deployment_rows():
    frames = []
    provenance = []
    for source, candidates in SOURCE_CANDIDATES:
        path = first_existing(candidates)
        if path is None:
            continue
        data = pd.read_csv(path, low_memory=False)
        if not {"policy", "risk_prob", "dock_pose_pass"}.issubset(data.columns):
            continue
        data = data[data["policy"].astype(str) == POLICY].copy()
        if data.empty:
            continue
        data["target_id"] = data["key"].astype(str) if "key" in data.columns else data["data_id"].astype(str)
        data["risk_prob"] = pd.to_numeric(data["risk_prob"], errors="coerce")
        data = data.dropna(subset=["risk_prob"]).copy()
        data["failure"] = (~as_bool(data["dock_pose_pass"])).astype(int)
        # Top-1 is the decision unit; the remaining top-k molecules are not independent targets.
        data = data.sort_values(["target_id", "risk_prob"]).groupby("target_id", as_index=False).head(1)
        frames.append(data[["target_id", "risk_prob", "failure"]].assign(source=source))
        provenance.append({"source": source, "path": path, "targets": int(data["target_id"].nunique())})
    if not frames:
        raise FileNotFoundError("No compatible PB-RC selection tables found.")
    rows = pd.concat(frames, ignore_index=True)
    pd.DataFrame(provenance).to_csv("results/generator_shift_calibration_methods_p1_sources.csv", index=False)
    return rows


def fit_platt(train):
    if train["failure"].nunique() < 2:
        return None
    model = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    model.fit(train[["risk_prob"]], train["failure"])
    return lambda frame: model.predict_proba(frame[["risk_prob"]])[:, 1]


def fit_beta(train):
    if train["failure"].nunique() < 2:
        return None
    x = np.clip(train["risk_prob"].to_numpy(float), 1e-6, 1.0 - 1e-6)
    features = np.c_[np.log(x), -np.log1p(-x)]
    model = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    model.fit(features, train["failure"])

    def predict(frame):
        score = np.clip(frame["risk_prob"].to_numpy(float), 1e-6, 1.0 - 1e-6)
        return model.predict_proba(np.c_[np.log(score), -np.log1p(-score)])[:, 1]

    return predict


def fit_isotonic(train):
    if train["risk_prob"].nunique() < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(train["risk_prob"].to_numpy(float), train["failure"].to_numpy(int))
    return lambda frame: model.predict(frame["risk_prob"].to_numpy(float))


CALIBRATORS = {"platt": fit_platt, "beta": fit_beta, "isotonic": fit_isotonic}


def evaluate_heldout(rows, source, calib_targets, seed):
    train = rows[rows["source"] != source].copy()
    held = rows[rows["source"] == source].copy()
    rng = np.random.default_rng(seed)
    target_ids = np.asarray(sorted(held["target_id"].unique()))
    chosen = set()
    if calib_targets > 0 and len(target_ids) > calib_targets:
        chosen = set(rng.choice(target_ids, size=calib_targets, replace=False))
    calib = held[held["target_id"].isin(chosen)].copy()
    test = held[~held["target_id"].isin(chosen)].copy() if chosen else held.copy()
    if test.empty:
        return []
    result = []

    def add(method, pred, train_n):
        result.append(
            {
                "heldout_source": source,
                "method": method,
                "calib_targets": calib_targets,
                "seed": seed,
                "train_n": int(train_n),
                **metrics(test["failure"], pred),
            }
        )

    add("raw_risk", test["risk_prob"].to_numpy(float), 0)
    logo_predictors = {}
    for name, fitter in CALIBRATORS.items():
        predictor = fitter(train)
        if predictor is not None:
            logo_predictors[name] = predictor
            add(f"logo_{name}", predictor(test), len(train))
    if calib_targets:
        adapted_train = pd.concat([train, calib], ignore_index=True)
        for name, fitter in CALIBRATORS.items():
            predictor = fitter(adapted_train)
            if predictor is not None:
                add(f"adapted_{name}", predictor(test), len(adapted_train))
                if name == "beta" and name in logo_predictors:
                    weight = min(1.0, calib_targets / 20.0)
                    pred = (1.0 - weight) * logo_predictors[name](test) + weight * predictor(test)
                    add("shrinkage_beta", pred, len(adapted_train))
    return result


def bh_fdr(values):
    p = np.asarray(values, dtype=float)
    out = np.full(len(p), np.nan)
    mask = np.isfinite(p)
    order = np.argsort(p[mask])
    pv = p[mask][order]
    adjusted = np.minimum.accumulate((pv * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    temp = np.empty(len(pv))
    temp[order] = adjusted
    out[np.where(mask)[0]] = temp
    return out


def comparison_tests(raw):
    reference = raw[raw["method"] == "raw_risk"]
    rows = []
    for size in [5, 10, 20]:
        ref = reference[reference["calib_targets"] == size][["heldout_source", "seed", "ece", "brier"]]
        for method in sorted(raw.loc[raw["calib_targets"] == size, "method"].unique()):
            if method == "raw_risk":
                continue
            method_rows = raw[(raw["calib_targets"] == size) & (raw["method"] == method)][
                ["heldout_source", "seed", "ece", "brier"]
            ]
            joined = ref.merge(method_rows, on=["heldout_source", "seed"], suffixes=("_raw", "_method"))
            for metric in ["ece", "brier"]:
                delta = joined[f"{metric}_method"] - joined[f"{metric}_raw"]
                try:
                    pvalue = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
                except ValueError:
                    pvalue = np.nan
                rows.append(
                    {
                        "calib_targets": size,
                        "method": method,
                        "metric": metric,
                        "pairs": int(len(delta)),
                        "delta_vs_raw_mean": float(delta.mean()),
                        "improved_fraction": float((delta < 0).mean()),
                        "wilcoxon_p": pvalue,
                    }
                )
    tests = pd.DataFrame(rows)
    tests["bh_fdr"] = bh_fdr(tests["wilcoxon_p"].to_numpy(float))
    return tests


def write_report(summary, tests, sources):
    focus = summary[summary["calib_targets"].isin([0, 20])].copy()
    lines = [
        "# P1 Generator-Shift Calibration Method Comparison",
        "",
        "## Protocol",
        "",
        f"- Deployment unit: one `{POLICY}` top-1 decision per target; top-k duplicates are not treated as independent rows.",
        "- Shift protocol: hold out each generator in turn; adapt using 0/5/10/20 held-out targets.",
        "- Compared calibrators: raw risk, leave-generator-out (LOGO) Platt, beta and isotonic calibration, target-adapted counterparts, and a shrinkage beta variant.",
        "- Primary probability endpoints: target-level Brier score and ECE; method-versus-raw Wilcoxon tests are corrected together with Benjamini--Hochberg FDR.",
        "",
        "## Source Coverage",
        "",
        "| Generator | Selection file | Targets |",
        "|---|---|---:|",
    ]
    for row in sources.itertuples(index=False):
        lines.append(f"| {row.source} | `{row.path}` | {row.targets} |")
    lines.extend(
        [
            "",
            "## Mean Results Across Held-Out Generators",
            "",
            "| Calib targets | Method | Test targets | ECE | Brier | AUROC | AUPRC |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    group = (
        focus.groupby(["calib_targets", "method"], as_index=False)
        .agg(test_targets=("test_n_mean", "sum"), ece=("ece_mean", "mean"), brier=("brier_mean", "mean"), auroc=("auroc_mean", "mean"), auprc=("auprc_mean", "mean"))
        .sort_values(["calib_targets", "brier"])
    )
    for row in group.itertuples(index=False):
        lines.append(
            f"| {row.calib_targets} | {row.method} | {int(row.test_targets)} | {row.ece:.4f} | {row.brier:.4f} | {row.auroc:.4f} | {row.auprc:.4f} |"
        )
    lines.extend(
        [
            "",
            "## FDR-Controlled Comparison Against Raw Risk",
            "",
            "| Calib targets | Method | Metric | Pairs | Delta (method - raw) | Improved | Wilcoxon p | BH-FDR |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tests.sort_values(["calib_targets", "metric", "bh_fdr"]).itertuples(index=False):
        lines.append(
            f"| {row.calib_targets} | {row.method} | {row.metric} | {row.pairs} | {row.delta_vs_raw_mean:.4f} | "
            f"{100 * row.improved_fraction:.1f}% | {row.wilcoxon_p:.4g} | {row.bh_fdr:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment promotes generator-shift calibration from a diagnostic curve to a pre-specified method comparison. It tests whether a deployment-time target calibration budget yields statistically defensible probability improvements under generator shift; methods that fail to beat raw risk after FDR correction are reported as boundaries rather than positive claims.",
        ]
    )
    Path("experiments/GENERATOR_SHIFT_CALIBRATION_METHOD_COMPARISON_P1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = load_deployment_rows()
    rows.to_csv("results/generator_shift_calibration_methods_p1_input_top1.csv", index=False)
    results = []
    for source in sorted(rows["source"].unique()):
        for size in CALIB_SIZES:
            seeds = [0] if size == 0 else list(range(N_SEEDS))
            for seed in seeds:
                results.extend(evaluate_heldout(rows, source, size, seed))
    raw = pd.DataFrame(results)
    raw.to_csv("results/generator_shift_calibration_methods_p1.csv", index=False)
    summary = (
        raw.groupby(["heldout_source", "method", "calib_targets"], as_index=False)
        .agg(
            test_n_mean=("test_n", "mean"),
            test_n_std=("test_n", "std"),
            failure_rate_mean=("failure_rate", "mean"),
            failure_rate_std=("failure_rate", "std"),
            mean_pred_failure_mean=("mean_pred_failure", "mean"),
            mean_pred_failure_std=("mean_pred_failure", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
        )
    )
    summary.to_csv("results/generator_shift_calibration_methods_p1_summary.csv", index=False)
    tests = comparison_tests(raw)
    tests.to_csv("results/generator_shift_calibration_methods_p1_fdr.csv", index=False)
    sources = pd.read_csv("results/generator_shift_calibration_methods_p1_sources.csv")
    write_report(summary, tests, sources)
    print(Path("experiments/GENERATOR_SHIFT_CALIBRATION_METHOD_COMPARISON_P1.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
