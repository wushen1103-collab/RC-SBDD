import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Contrib.SA_Score import sascorer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit SA_Score contrib module is required") from exc


KEEP_POLICIES = [
    "qed",
    "qed_minus_risk",
    "rc_select",
    "pb_qed",
    "pb_qed_minus_risk",
    "pb_rc_select",
]


def make_catalog(kind):
    params = FilterCatalogParams()
    params.AddCatalog(kind)
    return FilterCatalog(params)


def first_match(catalog, mol):
    entry = catalog.GetFirstMatch(mol)
    return "" if entry is None else entry.GetDescription()


def read_first_mol(path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=True)
    for mol in supplier:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            continue
    return None


def optional_rascore(repo_root):
    rascore_root = Path(repo_root) / "external" / "RAscore"
    if not rascore_root.exists():
        return None, "external/RAscore missing"
    sys.path.insert(0, str(rascore_root))
    try:
        # RAscore was pickled against older xgboost that exposed this class.
        import xgboost.compat

        if not hasattr(xgboost.compat, "XGBoostLabelEncoder"):
            class XGBoostLabelEncoder:  # noqa: N801 - compatibility shim
                def __init__(self, *args, **kwargs):
                    pass

            xgboost.compat.XGBoostLabelEncoder = XGBoostLabelEncoder
        from RAscore import RAscore_XGB

        return RAscore_XGB.RAScorerXGB(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def descriptors_for_mol(mol, rascore, catalogs):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    sa = float(sascorer.calculateScore(mol))
    ras = np.nan
    if rascore is not None:
        try:
            ras = float(rascore.predict(smiles))
        except Exception:
            ras = np.nan
    pains = first_match(catalogs["pains"], mol)
    brenk = first_match(catalogs["brenk"], mol)
    nih = first_match(catalogs["nih"], mol)
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    tpsa = float(Descriptors.TPSA(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    lipinski_viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    veber = bool(rot <= 10 and tpsa <= 140)
    return {
        "smiles": smiles,
        "sa_score": sa,
        "sa_pass_le_6": bool(sa <= 6.0),
        "sa_pass_le_5": bool(sa <= 5.0),
        "rascore": ras,
        "rascore_pass_ge_0_5": bool(ras >= 0.5) if not math.isnan(ras) else np.nan,
        "pains_alert": pains,
        "brenk_alert": brenk,
        "nih_alert": nih,
        "pains_free": pains == "",
        "brenk_free": brenk == "",
        "nih_free": nih == "",
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotatable_bonds": rot,
        "lipinski_violations": lipinski_viol,
        "lipinski_pass_le_1": bool(lipinski_viol <= 1),
        "veber_pass": veber,
        "qed_rdkit": float(QED.qed(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "spiro_atoms": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "bridgehead_atoms": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "chiral_centers": int(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
    }


def load_selection_tables(experiments=None, policies=None, max_rows_per_group=0):
    experiments = set(experiments or [])
    policies = policies or KEEP_POLICIES
    specs = [
        ("DiffSBDD_official", "results/posebusters_dockfast_selection.csv"),
        ("DiffSBDD_official", "results/posebusters_dockfast_pb_selection.csv"),
        ("DiffSBDD_local", "results/local_t500_dockfast_selection.csv"),
        ("TargetDiff_boundary", "results/targetdiff_t250_n64_dockfast_selection.csv"),
        ("Pocket2Mol_transfer", "results/pocket2mol_crossgen_n16_ext_dockfast_selection.csv"),
        ("SYNC-Guide", "results/syncguide_t1000_n16_dockfast_selection.csv"),
        ("PocketFlow", "results/pocketflow_crossdock_n16_dockfast_selection.csv"),
        ("SGEDiff", "results/sgediff_crossdock_t50_dockfast_selection.csv"),
        ("MolPilot", "results/molpilot_crossdock_t50_n16_dockfast_selection.csv"),
    ]
    frames = []
    for experiment, path in specs:
        if experiments and experiment not in experiments:
            continue
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "policy" not in df.columns or "mol_pred" not in df.columns:
            continue
        df = df[df["policy"].isin(policies)].copy()
        if df.empty:
            continue
        df["experiment"] = experiment
        if "set" not in df.columns:
            df["set"] = experiment
        if max_rows_per_group > 0:
            sort_cols = [c for c in ["experiment", "set", "policy", "key", "qed", "risk_prob"] if c in df.columns]
            ascending = [True] * len(sort_cols)
            if "qed" in sort_cols:
                ascending[sort_cols.index("qed")] = False
            df = df.sort_values(sort_cols, ascending=ascending)
            df = df.groupby(["experiment", "set", "policy"], dropna=False, sort=True).head(max_rows_per_group).copy()
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No selection tables found")
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize(df):
    rows = []
    group_cols = ["experiment", "set", "policy"]
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": int(len(group)),
                "targets": int(group["key"].nunique()) if "key" in group.columns else int(group.get("data_id", pd.Series()).nunique()),
                "sa_mean": float(group["sa_score"].mean()),
                "sa_median": float(group["sa_score"].median()),
                "sa_pass_le_6": float(group["sa_pass_le_6"].mean()),
                "sa_pass_le_5": float(group["sa_pass_le_5"].mean()),
                "rascore_mean": float(group["rascore"].mean()) if group["rascore"].notna().any() else np.nan,
                "rascore_pass_ge_0_5": float(group["rascore_pass_ge_0_5"].mean()) if group["rascore_pass_ge_0_5"].notna().any() else np.nan,
                "pains_free": float(group["pains_free"].mean()),
                "brenk_free": float(group["brenk_free"].mean()),
                "nih_free": float(group["nih_free"].mean()),
                "lipinski_pass_le_1": float(group["lipinski_pass_le_1"].mean()),
                "veber_pass": float(group["veber_pass"].mean()),
                "qed_mean": float(group["qed_rdkit"].mean()),
                "ring_count_mean": float(group["ring_count"].mean()),
            }
        )
        if "risk_prob" in group.columns:
            row["risk_mean"] = float(group["risk_prob"].mean())
            row["risk_gt_0_5"] = float((group["risk_prob"] > 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pct(x):
    return "NA" if pd.isna(x) else f"{100*x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def write_report(summary, meta, out_md):
    keep = summary[
        (summary["experiment"].isin(["DiffSBDD_official", "Pocket2Mol_transfer", "SYNC-Guide", "PocketFlow", "SGEDiff", "MolPilot"]))
        & (summary["policy"].isin(KEEP_POLICIES))
        & (
            (summary["set"] == "fullatom_cond")
            | (summary["experiment"].isin(["Pocket2Mol_transfer", "SYNC-Guide", "PocketFlow", "SGEDiff", "MolPilot"]))
        )
    ].copy()
    order = {p: i for i, p in enumerate(KEEP_POLICIES)}
    keep["order"] = keep["policy"].map(order)
    lines = [
        "# Synthesis and Drugability Oracle Check",
        "",
        "## Protocol",
        "",
        "- Molecules: selected generated molecules from DiffSBDD official, local DiffSBDD, TargetDiff boundary, Pocket2Mol transfer, SYNC-Guide, PocketFlow, SGEDiff, and MolPilot tables.",
        "- Core synthesizability metrics: RDKit SA score and RAscore-XGB when available.",
        "- Drugability filters: PAINS, Brenk, NIH, Lipinski, and Veber.",
        "- Lower SA score is easier; higher RAscore indicates higher predicted retrosynthetic accessibility.",
        "",
        "## Optional External Models",
        "",
    ]
    for key, value in meta.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Main Selected-Policy Summary",
            "",
            "| Experiment | Policy | N | Mean SA | SA <=6 | RAscore | RAscore >=0.5 | PAINS-free | Brenk-free | Lipinski pass | Veber pass | Mean QED | Mean risk | Risk >0.5 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in keep.sort_values(["experiment", "set", "order"]).itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.policy} | {row.n} | {row.sa_mean:.4f} | {pct(row.sa_pass_le_6)} | "
            f"{f4(row.rascore_mean)} | {pct(row.rascore_pass_ge_0_5)} | {pct(row.pains_free)} | {pct(row.brenk_free)} | "
            f"{pct(row.lipinski_pass_le_1)} | {pct(row.veber_pass)} | {row.qed_mean:.4f} | {f4(getattr(row, 'risk_mean', np.nan))} | {pct(getattr(row, 'risk_gt_0_5', np.nan))} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. This table tests whether reliability control improves pocket geometry by selecting synthetically implausible or alert-heavy molecules.",
            "2. The safest manuscript claim is preservation: RC/PB+RC should not materially worsen SA, RAscore, PAINS, or basic drugability filters relative to QED/PB+QED.",
            "3. AiZynthFinder is treated as the heavyweight route-planning follow-up; SA/RAscore/filter metrics are kept as fast cross-generator synthesizability controls.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="results/synthesis_oracle_selection.csv")
    ap.add_argument("--out-summary-csv", default="results/synthesis_oracle_selection_summary.csv")
    ap.add_argument("--out-json", default="logs/synthesis_oracle_metadata.json")
    ap.add_argument("--out-md", default="experiments/SYNTHESIS_ORACLE_SELECTION.md")
    ap.add_argument("--experiments", nargs="*", default=[])
    ap.add_argument("--policies", nargs="*", default=[])
    ap.add_argument("--max-rows-per-group", type=int, default=0)
    ap.add_argument("--disable-rascore", action="store_true")
    args = ap.parse_args()

    repo_root = Path.cwd()
    if args.disable_rascore:
        rascore, rascore_error = None, "disabled by --disable-rascore"
    else:
        rascore, rascore_error = optional_rascore(repo_root)
    catalogs = {
        "pains": make_catalog(FilterCatalogParams.FilterCatalogs.PAINS),
        "brenk": make_catalog(FilterCatalogParams.FilterCatalogs.BRENK),
        "nih": make_catalog(FilterCatalogParams.FilterCatalogs.NIH),
    }
    source = load_selection_tables(args.experiments, args.policies or None, args.max_rows_per_group)
    rows = []
    failures = []
    for idx, row in source.iterrows():
        mol = read_first_mol(row.mol_pred)
        if mol is None:
            failures.append({"row": int(idx), "mol_pred": str(row.mol_pred), "error": "unreadable"})
            continue
        try:
            metrics = descriptors_for_mol(mol, rascore, catalogs)
            base = row.to_dict()
            base.update(metrics)
            rows.append(base)
        except Exception as exc:
            failures.append({"row": int(idx), "mol_pred": str(row.mol_pred), "error": str(exc)})

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = summarize(out)
    summary.to_csv(args.out_summary_csv, index=False)
    meta = {
        "RAscore-XGB": "available" if rascore is not None else f"unavailable ({rascore_error})",
        "SCScore": "unavailable (GitHub clone timed out on the remote; not used in main table)",
        "SYNC": "repository cloned; 3D model environment differs from current PyTorch/CUDA stack, so not used in main table",
        "AiZynthFinder": "not run in main table because it is route-search heavy; reserve for final top-10 candidates if needed",
        "rows_scored": int(len(out)),
        "failures": int(len(failures)),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"meta": meta, "failures": failures[:100]}, indent=2), encoding="utf-8")
    write_report(summary, meta, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
