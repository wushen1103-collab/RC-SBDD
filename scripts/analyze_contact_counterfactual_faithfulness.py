import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcsbdd.features.interaction import feature_names, featurize_interaction


RDLogger.DisableLog("rdApp.*")

ATOM_MAP = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "B": 4,
    "BR": 5,
    "CL": 6,
    "P": 7,
    "I": 8,
    "F": 9,
}

CONTACT_FEATURES = [
    "lp_min",
    "lp_p01",
    "lp_p05",
    "lp_p10",
    "lp_p25",
    "lp_p50",
    "lp_mean",
    "clash_lt_1_0_per_lig",
    "clash_lt_1_5_per_lig",
    "clash_lt_2_0_per_lig",
    "contacts_lt_3_0_per_lig",
    "contacts_lt_4_0_per_lig",
    "contacts_lt_5_0_per_lig",
    "frac_lig_contact_lt_4_0",
    "lp_radial_bin_0",
    "lp_radial_bin_1",
    "lp_radial_bin_2",
    "lp_radial_bin_3",
    "lp_radial_bin_4",
    "lp_radial_bin_5",
    "lp_radial_bin_6",
    "lp_radial_bin_7",
    "lp_radial_bin_8",
    "lp_radial_bin_9",
    "lp_radial_bin_10",
    "lp_radial_bin_11",
]


def pct(x):
    return "NA" if pd.isna(x) else f"{100 * x:.1f}%"


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def load_model(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("include_pocket_feat", True):
        raise ValueError("risk model must be geometry-only")
    names = list(payload.get("feature_names") or feature_names(include_pocket_feat=False))
    expected = list(feature_names(include_pocket_feat=False))
    if names != expected:
        raise ValueError("risk-model features do not match featurizer")
    return payload["model"], names


def norm_element(raw, atom_name):
    elem = (raw or "").strip().upper()
    if not elem:
        elem = "".join(ch for ch in atom_name if ch.isalpha())[:2].upper()
    return elem


def read_pocket_atoms(path):
    atoms = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip().upper()
            elem = norm_element(line[76:78], atom_name)
            if elem.startswith("H") or atom_name.startswith("H"):
                continue
            try:
                xyz = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            atoms.append(
                {
                    "xyz": xyz,
                    "residue": (line[21].strip(), line[22:26].strip(), line[17:20].strip()),
                    "atom_name": atom_name,
                    "element": elem,
                }
            )
    if not atoms:
        raise ValueError(f"no heavy pocket atoms parsed from {path}")
    return atoms


def atom_type(atom):
    return ATOM_MAP.get(atom.GetSymbol().upper(), 10)


def read_ligand(path):
    mol = next((m for m in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=True) if m is not None), None)
    if mol is None:
        raise ValueError(f"RDKit could not read {path}")
    Chem.SanitizeMol(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("ligand has no conformer")
    conf = mol.GetConformer()
    coords = []
    types = []
    atom_ids = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        types.append(atom_type(atom))
        atom_ids.append(atom.GetIdx())
    if not coords:
        raise ValueError("ligand has no heavy atoms")
    return np.asarray(coords, dtype=np.float32), np.asarray(types, dtype=np.int64), atom_ids


def predict(model, lig_pos, lig_type, pock_pos, names):
    item = {"lig_pos": lig_pos, "lig_atom_type": lig_type, "pock_pos": pock_pos}
    x = featurize_interaction(item, include_pocket_feat=False)
    if len(x) != len(names):
        raise ValueError("feature length mismatch")
    return float(model.predict_proba(x.reshape(1, -1))[0, 1]), x


def summarize(group):
    out = {
        "n": int(len(group)),
        "orig_risk": float(group["orig_risk"].mean()),
    }
    for col in ["residue_deleted_risk", "atom_deleted_risk", "contact_masked_risk"]:
        ok = group[col].dropna()
        out[f"{col}_n"] = int(len(ok))
        out[f"{col}_mean"] = float(ok.mean()) if len(ok) else np.nan
        out[f"{col}_delta"] = float((ok - group.loc[ok.index, "orig_risk"]).mean()) if len(ok) else np.nan
        out[f"{col}_decrease"] = float((ok < group.loc[ok.index, "orig_risk"]).mean()) if len(ok) else np.nan
    out["close_residues_mean"] = float(group["close_residues"].mean())
    out["close_lig_atoms_mean"] = float(group["close_lig_atoms"].mean())
    return out


def load_pool(path, source, max_rows, risk_cutoff):
    df = pd.read_csv(path, low_memory=False)
    if "kind" in df.columns:
        df = df[df["kind"] == "generated"].copy()
    df = df[df["risk_prob"] >= risk_cutoff].copy()
    df = df.sort_values(["risk_prob", "qed"], ascending=[False, False]).head(max_rows)
    df["source"] = source
    return df


def process_row(row, model, names, contact_cutoff, lowrisk_median):
    lig_pos, lig_type, atom_ids = read_ligand(row.mol_pred)
    atoms = read_pocket_atoms(row.mol_cond)
    pock_pos = np.vstack([a["xyz"] for a in atoms]).astype(np.float32)
    orig_risk, orig_x = predict(model, lig_pos, lig_type, pock_pos, names)
    dist = np.linalg.norm(lig_pos[:, None, :] - pock_pos[None, :, :], axis=-1)
    close_lig = np.where((dist < contact_cutoff).any(axis=1))[0]
    close_pock = np.where((dist < contact_cutoff).any(axis=0))[0]
    close_res = {atoms[int(i)]["residue"] for i in close_pock}

    if len(close_res) == 0:
        nearest = int(np.argmin(dist.min(axis=0)))
        close_res = {atoms[nearest]["residue"]}
        close_pock = np.asarray([nearest], dtype=int)
    if len(close_lig) == 0:
        close_lig = np.asarray([int(np.argmin(dist.min(axis=1)))], dtype=int)

    residue_deleted_risk = np.nan
    keep_pocket = np.asarray([a["residue"] not in close_res for a in atoms], dtype=bool)
    if keep_pocket.sum() >= 5:
        residue_deleted_risk, _ = predict(model, lig_pos, lig_type, pock_pos[keep_pocket], names)

    atom_deleted_risk = np.nan
    keep_lig = np.ones(len(lig_pos), dtype=bool)
    keep_lig[close_lig] = False
    if keep_lig.sum() >= 3:
        atom_deleted_risk, _ = predict(model, lig_pos[keep_lig], lig_type[keep_lig], pock_pos, names)

    x_masked = orig_x.copy()
    for name in CONTACT_FEATURES:
        idx = names.index(name)
        x_masked[idx] = lowrisk_median[idx]
    contact_masked_risk = float(model.predict_proba(x_masked.reshape(1, -1))[0, 1])

    return {
        "source": row.source,
        "key": row.key,
        "mol_pred": row.mol_pred,
        "mol_cond": row.mol_cond,
        "csv_risk": float(row.risk_prob),
        "orig_risk": orig_risk,
        "residue_deleted_risk": residue_deleted_risk,
        "atom_deleted_risk": atom_deleted_risk,
        "contact_masked_risk": contact_masked_risk,
        "close_residues": int(len(close_res)),
        "close_lig_atoms": int(len(close_lig)),
        "min_distance": float(dist.min()),
    }


def write_report(summary, out_md):
    lines = [
        "# Atom/Residue-Level Contact Counterfactual Faithfulness",
        "",
        "## Protocol",
        "",
        "- High-risk generated molecules are selected from the full DiffSBDD and Pocket2Mol dock_fast-labelled pools.",
        "- Residue deletion removes protein residues that contain ligand-proximal atoms under the contact cutoff, then recomputes the RC-SBDD risk model.",
        "- Atom deletion removes ligand atoms participating in the same close contacts, then recomputes risk.",
        "- Contact-feature masking replaces distance/contact/clash feature coordinates by source-specific low-risk medians and checks whether predicted risk drops.",
        "- These are computational counterfactuals for explanation faithfulness, not chemically valid molecule edits.",
        "",
        "## Summary",
        "",
        "| Source | N | Orig risk | Residue risk | Residue delta | Residue drop | Atom risk | Atom delta | Atom drop | Masked risk | Masked delta | Masked drop | Close residues | Close atoms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.n} | {f4(row.orig_risk)} | "
            f"{f4(row.residue_deleted_risk_mean)} | {f4(row.residue_deleted_risk_delta)} | {pct(row.residue_deleted_risk_decrease)} | "
            f"{f4(row.atom_deleted_risk_mean)} | {f4(row.atom_deleted_risk_delta)} | {pct(row.atom_deleted_risk_decrease)} | "
            f"{f4(row.contact_masked_risk_mean)} | {f4(row.contact_masked_risk_delta)} | {pct(row.contact_masked_risk_decrease)} | "
            f"{f4(row.close_residues_mean)} | {f4(row.close_lig_atoms_mean)} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. A faithful contact-risk explanation should show negative risk deltas after deleting or masking the high-risk contact evidence.",
            "2. Residue/atom deletion is stricter than feature masking because deleting contacts can also remove favorable contacts; use both as complementary evidence.",
            "3. The safest manuscript wording is atom/residue-level counterfactual faithfulness for the risk scorer, not a causal biochemical mechanism.",
        ]
    )
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/risk_proxy_hard_geom_logreg/model.pkl")
    ap.add_argument("--diffsbdd", default="results/dockfast_full_pool_fullatom_cond.csv")
    ap.add_argument("--pocket2mol", default="results/dockfast_full_pool_pocket2mol_n16_ext.csv")
    ap.add_argument("--max-rows-per-source", type=int, default=120)
    ap.add_argument("--risk-cutoff", type=float, default=0.5)
    ap.add_argument("--contact-cutoff", type=float, default=2.0)
    ap.add_argument("--out-csv", default="results/contact_counterfactual_faithfulness.csv")
    ap.add_argument("--out-summary-csv", default="results/contact_counterfactual_faithfulness_summary.csv")
    ap.add_argument("--out-md", default="experiments/CONTACT_COUNTERFACTUAL_FAITHFULNESS.md")
    args = ap.parse_args()

    model, names = load_model(args.model)
    pools = pd.concat(
        [
            load_pool(args.diffsbdd, "DiffSBDD", args.max_rows_per_source, args.risk_cutoff),
            load_pool(args.pocket2mol, "Pocket2Mol", args.max_rows_per_source, args.risk_cutoff),
        ],
        ignore_index=True,
        sort=False,
    )

    lowrisk_features = []
    for path in [args.diffsbdd, args.pocket2mol]:
        df = pd.read_csv(path, low_memory=False)
        if "kind" in df.columns:
            df = df[df["kind"] == "generated"].copy()
        df = df[df["risk_prob"] <= 0.20].head(40)
        for row in df.itertuples(index=False):
            try:
                lig_pos, lig_type, _ = read_ligand(row.mol_pred)
                atoms = read_pocket_atoms(row.mol_cond)
                pock_pos = np.vstack([a["xyz"] for a in atoms]).astype(np.float32)
                _, x = predict(model, lig_pos, lig_type, pock_pos, names)
                lowrisk_features.append(x)
            except Exception:
                continue
    if not lowrisk_features:
        raise RuntimeError("could not build low-risk feature medians")
    lowrisk_median = np.median(np.vstack(lowrisk_features), axis=0)

    rows = []
    failures = []
    for row in pools.itertuples(index=False):
        try:
            rows.append(process_row(row, model, names, args.contact_cutoff, lowrisk_median))
        except Exception as exc:
            failures.append({"source": getattr(row, "source", ""), "key": getattr(row, "key", ""), "error": str(exc)})
    out = pd.DataFrame(rows)
    summary = pd.DataFrame([{"source": source, **summarize(group)} for source, group in out.groupby("source", sort=True)])
    summary["failures"] = len(failures)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary.to_csv(args.out_summary_csv, index=False)
    write_report(summary, args.out_md)
    print(Path(args.out_md).read_text(encoding="utf-8"))
    if failures:
        print(f"failures={len(failures)} first={failures[:3]}")


if __name__ == "__main__":
    main()
