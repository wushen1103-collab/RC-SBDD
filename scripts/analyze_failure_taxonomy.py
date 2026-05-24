import argparse
from pathlib import Path

import pandas as pd

INTRAMOL_COLUMNS = [
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
    return f"{100 * x:.1f}%"


def f4(x):
    return f"{x:.4f}"


def top_k(group, k, policy, native_p95):
    if policy == "qed":
        return group.sort_values(["qed", "risk_prob"], ascending=[False, True]).head(k)
    if policy == "qed_minus_risk":
        scored = group.assign(selection_score=group["qed"] - group["risk_prob"])
        return scored.sort_values(["selection_score", "qed"], ascending=[False, False]).head(k)
    if policy == "rc_select":
        safe = group[group["risk_prob"] <= native_p95].copy()
        safe = safe.sort_values(["qed", "risk_prob"], ascending=[False, True])
        if len(safe) >= k:
            return safe.head(k)
        fill = group.drop(safe.index).assign(selection_score=group["qed"] - group["risk_prob"])
        fill = fill.sort_values(["selection_score", "qed"], ascending=[False, False])
        return pd.concat([safe, fill.head(k - len(safe))], axis=0)
    raise ValueError(f"Unknown policy: {policy}")


def attach_molfast(gen, molfast_path):
    pb = pd.read_csv(molfast_path)
    pb = pb.rename(columns={"file": "source_file", "position": "mol_index"})
    pb["molfast_core_pass"] = pb[INTRAMOL_COLUMNS].fillna(False).astype(bool).all(axis=1)
    merged = gen.merge(pb[["source_file", "mol_index", "molfast_core_pass"]], on=["source_file", "mol_index"], how="left")
    merged["molfast_core_pass"] = merged["molfast_core_pass"].fillna(False).astype(bool)
    return merged


def build_thresholds(native):
    return {
        "native_risk_p95": float(native["risk_prob"].quantile(0.95)),
        "lp_min_p05": float(native["lp_min"].quantile(0.05)),
        "center_dist_p95": float(native["center_dist"].quantile(0.95)),
        "contacts_lt_4_0_per_lig_p05": float(native["contacts_lt_4_0_per_lig"].quantile(0.05)),
        "clash_lt_1_5_per_lig_p95": float(native["clash_lt_1_5_per_lig"].quantile(0.95)),
    }


def add_taxonomy(df, thresholds):
    out = df.copy()
    clash_floor = max(thresholds["clash_lt_1_5_per_lig_p95"], 1e-6)
    out["risk_high"] = out["risk_prob"] > 0.5
    out["above_native_p95"] = out["risk_prob"] > thresholds["native_risk_p95"]
    out["steric_or_too_close"] = (out["clash_lt_1_5_per_lig"] > clash_floor) | (out["lp_min"] < thresholds["lp_min_p05"])
    out["center_shift"] = out["center_dist"] > thresholds["center_dist_p95"]
    out["weak_contact"] = out["contacts_lt_4_0_per_lig"] < thresholds["contacts_lt_4_0_per_lig_p05"]
    out["signal_count"] = out[["steric_or_too_close", "center_shift", "weak_contact"]].astype(int).sum(axis=1)

    def primary(row):
        if not row.risk_high:
            return "not_high_risk"
        if row.signal_count > 1:
            return "multi_signal"
        if row.steric_or_too_close:
            return "steric_or_too_close"
        if row.center_shift:
            return "center_shift"
        if row.weak_contact:
            return "weak_contact"
        return "other_high_risk"

    out["primary_failure"] = out.apply(primary, axis=1)
    return out


def summarize_primary(gen):
    high = gen[gen["risk_high"]].copy()
    rows = []
    for label, group in high.groupby("primary_failure", sort=True):
        rows.append(
            {
                "primary_failure": label,
                "count": int(len(group)),
                "frac_all_generated": float(len(group) / max(len(gen), 1)),
                "frac_high_risk": float(len(group) / max(len(high), 1)),
                "risk_mean": float(group["risk_prob"].mean()),
                "qed_mean": float(group["qed"].mean()),
                "molfast_core_pass": float(group["molfast_core_pass"].mean()) if "molfast_core_pass" in group else float("nan"),
            }
        )
    order = {
        "multi_signal": 0,
        "steric_or_too_close": 1,
        "center_shift": 2,
        "weak_contact": 3,
        "other_high_risk": 4,
    }
    out = pd.DataFrame(rows)
    out["order"] = out["primary_failure"].map(order)
    return out.sort_values("order").drop(columns=["order"])


def summarize_signal_prevalence(gen):
    high = gen[gen["risk_high"]].copy()
    rows = []
    for label in ["steric_or_too_close", "center_shift", "weak_contact"]:
        mask = high[label]
        rows.append(
            {
                "signal": label,
                "count": int(mask.sum()),
                "frac_high_risk": float(mask.mean()) if len(high) else 0.0,
                "risk_mean_when_present": float(high.loc[mask, "risk_prob"].mean()) if mask.any() else float("nan"),
                "qed_mean_when_present": float(high.loc[mask, "qed"].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_selection(selected, policy):
    high = selected["risk_high"]
    return {
        "policy": policy,
        "selected": int(len(selected)),
        "targets": int(selected["key"].nunique()),
        "risk_high": float(selected["risk_high"].mean()),
        "above_native_p95": float(selected["above_native_p95"].mean()),
        "high_risk_steric_or_too_close": float((high & selected["steric_or_too_close"]).mean()),
        "high_risk_center_shift": float((high & selected["center_shift"]).mean()),
        "high_risk_weak_contact": float((high & selected["weak_contact"]).mean()),
        "high_risk_multi_signal": float((high & (selected["signal_count"] > 1)).mean()),
        "qed_mean": float(selected["qed"].mean()),
        "molfast_core_pass": float(selected["molfast_core_pass"].mean()),
    }


def to_md(df, formatters=None):
    formatters = formatters or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(formatters.get(col, str)(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-csv", default="results/diffsbdd_zenodo_crossdocked_fullatom_cond_risk_scores.csv")
    ap.add_argument("--molfast-csv", default="results/posebusters_molfast/fullatom_cond.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", default="results/failure_taxonomy")
    ap.add_argument("--out-md", default="experiments/FAILURE_TAXONOMY.md")
    args = ap.parse_args()

    df = pd.read_csv(args.risk_csv)
    native = df[df["kind"] == "native"].copy()
    gen = attach_molfast(df[df["kind"] == "generated"].copy(), args.molfast_csv)
    thresholds = build_thresholds(native)
    gen = add_taxonomy(gen, thresholds)

    primary = summarize_primary(gen)
    signals = summarize_signal_prevalence(gen)
    policies = ["qed", "qed_minus_risk", "rc_select", "pb_qed", "pb_qed_minus_risk", "pb_rc_select"]
    selection_rows = []
    for policy in policies:
        base_policy = policy[3:] if policy.startswith("pb_") else policy
        pool = gen[gen["molfast_core_pass"]].copy() if policy.startswith("pb_") else gen
        selected = pd.concat(
            [top_k(group, args.k, base_policy, thresholds["native_risk_p95"]) for _, group in pool.groupby("key", sort=True)],
            axis=0,
        )
        selection_rows.append(summarize_selection(selected, policy))
    selection = pd.DataFrame(selection_rows)
    thresholds_df = pd.DataFrame([thresholds])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds_df.to_csv(out_dir / "thresholds.csv", index=False)
    primary.to_csv(out_dir / "primary_high_risk_taxonomy.csv", index=False)
    signals.to_csv(out_dir / "signal_prevalence.csv", index=False)
    selection.to_csv(out_dir / "selection_failure_signals.csv", index=False)

    pct_cols = {
        "frac_all_generated",
        "frac_high_risk",
        "molfast_core_pass",
        "risk_high",
        "above_native_p95",
        "high_risk_steric_or_too_close",
        "high_risk_center_shift",
        "high_risk_weak_contact",
        "high_risk_multi_signal",
    }
    four_cols = {"risk_mean", "qed_mean", "risk_mean_when_present", "qed_mean_when_present"}
    formatters = {col: pct for col in pct_cols}
    formatters.update({col: f4 for col in four_cols})

    lines = [
        "# Failure Taxonomy",
        "",
        "## Protocol",
        "",
        "- Dataset: official DiffSBDD full-atom conditional generated set.",
        "- Thresholds are native-calibrated from the matched test ligands.",
        "- High risk means risk_prob > 0.5; above-native-p95 uses the native risk 95th percentile.",
        "- Failure signals are multi-label; the primary taxonomy isolates high-risk molecules into one dominant category.",
        "",
        "## Native-Calibrated Thresholds",
        "",
        to_md(thresholds_df, {col: f4 for col in thresholds_df.columns}),
        "",
        "## Primary Taxonomy Among High-Risk Generated Molecules",
        "",
        to_md(primary, formatters),
        "",
        "## Multi-Label Signal Prevalence Among High-Risk Generated Molecules",
        "",
        to_md(signals, formatters),
        "",
        "## Selection Failure Signals",
        "",
        to_md(selection, formatters),
        "",
        "## Findings",
        "",
        "1. High-risk generated molecules are not a single failure mode; multiple geometric signals often co-occur.",
        "2. QED-only selection preserves many high-risk geometric failures, while RC-Select and QED-Risk suppress them under the same candidate budget.",
        "3. The taxonomy is a mechanistic explanation of the risk axis, not a new labeling source; all thresholds come from native test ligands.",
    ]
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
