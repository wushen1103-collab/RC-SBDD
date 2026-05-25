from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SOURCES = [
    ("DiffSBDD", "results/posebusters_dockfast_selection.csv"),
    ("DiffSBDD-PB", "results/posebusters_dockfast_pb_selection.csv"),
    ("Pocket2Mol", "results/pocket2mol_crossgen_n16_ext_dockfast_selection.csv"),
    ("SYNC-Guide", "results/syncguide_t1000_n16_dockfast_selection.csv"),
    ("PocketFlow", "results/pocketflow_crossdock_n16_dockfast_selection.csv"),
    ("MolCRAFT", "results/molcraft_crossdock_t50_n16_dockfast_selection.csv"),
    ("MolPilot-framefix", "results/molpilot_crossdock_t50_n16_framefix_dockfast_selection.csv"),
]


def as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def ece_score(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1 else y_prob < hi)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(np.mean(y_prob[mask]) - np.mean(y_true[mask]))
    return float(ece)


def safe_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {
        "test_n": int(len(y_true)),
        "failure_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "mean_pred_failure": float(np.mean(y_prob)) if len(y_prob) else np.nan,
        "ece": ece_score(y_true, y_prob) if len(y_true) else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) >= 1 and len(y_true) else np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    return out


def load_rows():
    parts = []
    for source, path in SOURCES:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if "dock_pose_pass" not in df.columns or "risk_prob" not in df.columns:
            continue
        target = df["key"].astype(str) if "key" in df.columns else df.get("data_id", df.index).astype(str)
        out = pd.DataFrame(
            {
                "source": source,
                "policy": df["policy"].astype(str),
                "target_id": target,
                "risk_prob": pd.to_numeric(df["risk_prob"], errors="coerce"),
                "dock_pose_pass": as_bool(df["dock_pose_pass"]),
                "qed": pd.to_numeric(df.get("qed", np.nan), errors="coerce"),
            }
        )
        out = out.dropna(subset=["risk_prob"]).copy()
        out["failure"] = (~out["dock_pose_pass"]).astype(int)
        parts.append(out)
    if not parts:
        raise FileNotFoundError("No compatible selection tables were found.")
    data = pd.concat(parts, ignore_index=True)
    data["risk_prob"] = data["risk_prob"].clip(0, 1)
    return data


def fit_iso(train):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    x = train["risk_prob"].to_numpy(float)
    y = train["failure"].to_numpy(int)
    if len(np.unique(x)) < 2:
        return None
    iso.fit(x, y)
    return iso


def evaluate_source(data, heldout_source, calib_targets=0, seed=0):
    train = data[data["source"] != heldout_source].copy()
    held = data[data["source"] == heldout_source].copy()
    if held.empty:
        return []
    rng = np.random.default_rng(seed)
    target_ids = np.array(sorted(held["target_id"].unique()))
    calib_ids = set()
    if calib_targets > 0 and len(target_ids) > calib_targets:
        calib_ids = set(rng.choice(target_ids, size=calib_targets, replace=False))
    calib = held[held["target_id"].isin(calib_ids)].copy()
    test = held[~held["target_id"].isin(calib_ids)].copy() if calib_ids else held
    if test.empty:
        return []

    rows = []
    raw = safe_metrics(test["failure"], test["risk_prob"])
    rows.append(
        {
            "heldout_source": heldout_source,
            "method": "raw_risk",
            "calib_targets": calib_targets,
            "seed": seed,
            "train_n": 0,
            **raw,
        }
    )

    iso_train = fit_iso(train)
    if iso_train is not None:
        pred = iso_train.predict(test["risk_prob"].to_numpy(float))
        rows.append(
            {
                "heldout_source": heldout_source,
                "method": "leave_generator_out_isotonic",
                "calib_targets": calib_targets,
                "seed": seed,
                "train_n": int(len(train)),
                **safe_metrics(test["failure"], pred),
            }
        )

    adapted_train = pd.concat([train, calib], ignore_index=True) if len(calib) else train
    iso_adapt = fit_iso(adapted_train)
    if iso_adapt is not None:
        pred = iso_adapt.predict(test["risk_prob"].to_numpy(float))
        rows.append(
            {
                "heldout_source": heldout_source,
                "method": "target_adapted_isotonic",
                "calib_targets": calib_targets,
                "seed": seed,
                "train_n": int(len(adapted_train)),
                **safe_metrics(test["failure"], pred),
            }
        )
    return rows


def fmt4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def write_report(raw, summary):
    lines = [
        "# P0 Generator-Shift Adaptive Calibration",
        "",
        "## Protocol",
        "",
        "- Unit: selected molecules with PoseBusters dock_fast labels from all available generator outputs.",
        "- Failure label: `1 - dock_fast pass`; score: RC risk probability.",
        "- Evaluation: leave-one-generator-out calibration plus 5/10/20 held-out target recalibration.",
        "- Calibrator: isotonic regression fit without using the held-out generator unless explicitly listed as target-adapted.",
        "",
        "## Summary",
        "",
        "| Held-out generator | Method | Calib targets | Test N | Failure | Pred. failure | ECE | Brier | AUROC | AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    view = summary.copy()
    order = {"raw_risk": 0, "leave_generator_out_isotonic": 1, "target_adapted_isotonic": 2}
    view["order"] = view["method"].map(order).fillna(99)
    for row in view.sort_values(["heldout_source", "calib_targets", "order"]).itertuples(index=False):
        lines.append(
            f"| {row.heldout_source} | {row.method} | {row.calib_targets} | {int(row.test_n_mean)} | "
            f"{pct(row.failure_rate_mean)} | {pct(row.mean_pred_failure_mean)} | {fmt4(row.ece_mean)} | "
            f"{fmt4(row.brier_mean)} | {fmt4(row.auroc_mean)} | {fmt4(row.auprc_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Raw RC scores remain rank-informative under generator shift, but probability calibration can drift sharply for very easy or pathological generators.",
            "2. Leave-one-generator-out isotonic calibration tests whether a universal mapping from RC risk to dock_fast failure is sufficient.",
            "3. Target-adapted calibration quantifies how many held-out targets are needed to restore calibration under a new generator distribution.",
        ]
    )
    Path("experiments/GENERATOR_SHIFT_ADAPTIVE_CALIBRATION_P0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    data = load_rows()
    data.to_csv("results/generator_shift_adaptive_calibration_input_rows.csv", index=False)
    rows = []
    for source in sorted(data["source"].unique()):
        for calib_targets in [0, 5, 10, 20]:
            seeds = [0] if calib_targets == 0 else list(range(10))
            for seed in seeds:
                rows.extend(evaluate_source(data, source, calib_targets=calib_targets, seed=seed))
    raw = pd.DataFrame(rows)
    raw.to_csv("results/generator_shift_adaptive_calibration_p0.csv", index=False)
    agg_spec = {
        col: ["mean", "std"]
        for col in [
            "test_n",
            "failure_rate",
            "mean_pred_failure",
            "ece",
            "brier",
            "auroc",
            "auprc",
        ]
    }
    summary = raw.groupby(["heldout_source", "method", "calib_targets"], as_index=False).agg(agg_spec)
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]
    summary.to_csv("results/generator_shift_adaptive_calibration_p0_summary.csv", index=False)
    write_report(raw, summary)
    print(Path("experiments/GENERATOR_SHIFT_ADAPTIVE_CALIBRATION_P0.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
