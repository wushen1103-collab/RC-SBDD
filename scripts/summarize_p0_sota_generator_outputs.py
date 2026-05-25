from pathlib import Path

import pandas as pd


GENERATORS = {
    "PocketFlow": {
        "risk": "results/pocketflow_crossdock_n16_risk_scores.csv",
        "dock": "results/pocketflow_crossdock_n16_dockfast_selection_summary.csv",
        "gnina": "results/pocketflow_crossdock_n16_gnina_score_top1_summary.csv",
        "vina": "results/vina_redock_pocketflow_crossdock_n16_top1_summary.csv",
        "aizynth": "results/aizynthfinder_pocketflow_summary.csv",
        "policies": {"pb_qed": "PB-QED", "pb_rc_select": "PB-RC"},
    },
    "MolCRAFT": {
        "risk": "results/molcraft_crossdock_t100_n16_risk_scores.csv",
        "dock": "results/molcraft_crossdock_t100_n16_dockfast_selection_summary.csv",
        "gnina": "results/gnina_redock_molcraft_t100_top4_summary.csv",
        "vina": "results/molcraft_crossdock_t100_n16_vina_redock_top1_summary.csv",
        "aizynth": "results/aizynthfinder_molcraft_t100_top1_summary.csv",
        "policies": {"pb_qed": "PB-QED", "pb_rc_select": "PB-RC"},
    },
    "MolPilot-framefix": {
        "risk": "results/molpilot_crossdock_t50_n16_framefix_risk_scores.csv",
        "dock": "results/molpilot_crossdock_t50_n16_framefix_dockfast_selection_summary.csv",
        "gnina": "results/molpilot_crossdock_t50_n16_framefix_gnina_score_top1_summary.csv",
        "vina": "results/molpilot_crossdock_t50_n16_framefix_vina_redock_top1_summary.csv",
        "aizynth": "results/aizynthfinder_molpilot_framefix_top1_summary.csv",
        "policies": {"pb_qed": "PB-QED", "pb_rc_select": "PB-RC"},
    },
}


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f3(x):
    return "NA" if pd.isna(x) else f"{x:.3f}"


def read_csv(path):
    p = Path(path)
    return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()


def get_policy(df, policy):
    if df.empty or "policy" not in df.columns:
        return pd.Series(dtype=object)
    out = df[df["policy"] == policy]
    return out.iloc[0] if len(out) else pd.Series(dtype=object)


def main():
    rows = []
    for generator, spec in GENERATORS.items():
        risk = read_csv(spec["risk"])
        gen = risk[risk.get("kind", pd.Series(dtype=str)) == "generated"].copy() if len(risk) else pd.DataFrame()
        if len(gen) and "data_id" in gen:
            pool_targets = int(gen["data_id"].nunique())
        elif len(gen) and "key" in gen:
            pool_targets = int(gen["key"].nunique())
        else:
            pool_targets = 0
        pool_mols = int(len(gen))
        pool_risk = float(gen["risk_prob"].mean()) if len(gen) else float("nan")
        dock = read_csv(spec["dock"])
        gnina = read_csv(spec["gnina"])
        vina = read_csv(spec["vina"])
        aiz = read_csv(spec["aizynth"])
        for policy, label in spec["policies"].items():
            d = get_policy(dock, policy)
            g = get_policy(gnina, policy)
            v = get_policy(vina, policy)
            a = get_policy(aiz, policy)
            rows.append(
                {
                    "generator": generator,
                    "policy": label,
                    "pool_targets": pool_targets,
                    "pool_molecules": pool_mols,
                    "pool_risk_mean": pool_risk,
                    "selected_n": int(d.get("n", 0)) if len(d) else 0,
                    "selected_targets": int(d.get("targets", 0)) if len(d) else 0,
                    "risk_gt_0_5": float(d.get("risk_gt_0_5", float("nan"))) if len(d) else float("nan"),
                    "dock_fast": float(d.get("dock_pose_pass", float("nan"))) if len(d) else float("nan"),
                    "qed": float(d.get("qed_mean", float("nan"))) if len(d) else float("nan"),
                    "gnina_cnnscore": float(g.get("cnnscore_mean", float("nan"))) if len(g) else float("nan"),
                    "gnina_cnnaffinity": float(g.get("cnnaffinity_mean", float("nan"))) if len(g) else float("nan"),
                    "vina_redock": float(v.get("vina_mean", float("nan"))) if len(v) else float("nan"),
                    "aizynth_solved": float(a.get("solved_rate", float("nan"))) if len(a) else float("nan"),
                    "aizynth_steps_median": float(a.get("route_steps_median", float("nan"))) if len(a) else float("nan"),
                    "aizynth_n": int(a.get("n", 0)) if len(a) else 0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv("results/p0_sota_generator_unified_summary.csv", index=False)
    lines = [
        "# P0 Recent SOTA Generator Output Unified Summary",
        "",
        "## Protocol",
        "",
        "- Generators: PocketFlow, MolCRAFT, and MolPilot with deterministic coordinate-frame restoration.",
        "- Common evidence: RC risk, PoseBusters dock_fast, GNINA neural scoring, Vina redocking top-1, and AiZynthFinder top-1 retrosynthesis.",
        "- For MolCRAFT-T100, the GNINA values are from complete top-4 local redocking (400 molecules per policy); older score-only summaries are not used because their missingness is policy dependent.",
        "- Selection policies: PB-QED and PB-RC with the same per-target selection budget.",
        "- MolPilot-framefix is retained as a negative external boundary because its released sample coordinate frame required restoration yet still fails protein-geometry checks.",
        "",
        "## Unified Table",
        "",
        "| Generator | Policy | Pool | Selected | Risk >0.5 | dock_fast | QED | GNINA CNNscore | CNNaff. | Vina redock | AiZynth solved | Steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in out.itertuples(index=False):
        pool = f"{row.pool_targets}/{row.pool_molecules}"
        selected = f"{row.selected_targets}/{row.selected_n}"
        lines.append(
            f"| {row.generator} | {row.policy} | {pool} | {selected} | {pct(row.risk_gt_0_5)} | "
            f"{pct(row.dock_fast)} | {f3(row.qed)} | {f3(row.gnina_cnnscore)} | {f3(row.gnina_cnnaffinity)} | "
            f"{f3(row.vina_redock)} | {pct(row.aizynth_solved)} | {f3(row.aizynth_steps_median)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. MolCRAFT-T100 provides a strong positive SOTA generator: PB-RC removes high-risk selected poses and preserves near-ceiling dock-fast success under complete GNINA redocking.",
            "2. PocketFlow shows the typical trade-off: PB-RC sharply improves dock_fast and risk tail at a modest QED cost.",
            "3. MolPilot-framefix is an intentionally retained failure boundary: synthesis can remain feasible while pocket geometry is unusable, motivating explicit reliability control.",
        ]
    )
    Path("experiments/P0_SOTA_GENERATOR_UNIFIED_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
