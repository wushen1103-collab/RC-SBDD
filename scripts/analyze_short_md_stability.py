import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import openmm as mm
from openmm import unit


RDLogger.DisableLog("rdApp.*")

ELEMENT_PARAMS = {
    "C": {"mass": 12.011, "sigma": 0.340, "epsilon": 0.36, "charge": 0.00},
    "N": {"mass": 14.007, "sigma": 0.325, "epsilon": 0.71, "charge": 0.30},
    "O": {"mass": 15.999, "sigma": 0.296, "epsilon": 0.88, "charge": -0.30},
    "S": {"mass": 32.060, "sigma": 0.356, "epsilon": 1.05, "charge": -0.10},
    "P": {"mass": 30.974, "sigma": 0.374, "epsilon": 0.84, "charge": 0.30},
    "F": {"mass": 18.998, "sigma": 0.312, "epsilon": 0.25, "charge": -0.10},
    "CL": {"mass": 35.450, "sigma": 0.347, "epsilon": 1.11, "charge": -0.10},
    "BR": {"mass": 79.904, "sigma": 0.376, "epsilon": 1.40, "charge": -0.10},
    "I": {"mass": 126.90, "sigma": 0.398, "epsilon": 1.70, "charge": -0.10},
    "DEFAULT": {"mass": 12.011, "sigma": 0.340, "epsilon": 0.36, "charge": 0.00},
}


def f4(x):
    return "NA" if pd.isna(x) else f"{x:.4f}"


def element_param(elem):
    elem = elem.strip().upper()
    return ELEMENT_PARAMS.get(elem, ELEMENT_PARAMS["DEFAULT"])


def read_ligand(path):
    mol = next((m for m in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=True) if m is not None), None)
    if mol is None:
        raise ValueError(f"could not read ligand {path}")
    Chem.SanitizeMol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()
    coords = []
    elems = []
    charges = []
    atom_map = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        idx = atom.GetIdx()
        atom_map[idx] = len(coords)
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x * 0.1, pos.y * 0.1, pos.z * 0.1])
        elem = atom.GetSymbol().upper()
        elems.append(elem)
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(q):
                q = element_param(elem)["charge"]
        except Exception:
            q = element_param(elem)["charge"]
        charges.append(q)
    bonds = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in atom_map and b in atom_map:
            bonds.append((atom_map[a], atom_map[b]))
    angles = []
    neighbors = {atom_map[a.GetIdx()]: [] for a in mol.GetAtoms() if a.GetIdx() in atom_map}
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    for center, neigh in neighbors.items():
        for i in range(len(neigh)):
            for j in range(i + 1, len(neigh)):
                angles.append((neigh[i], center, neigh[j]))
    return mol, np.asarray(coords, dtype=float), elems, np.asarray(charges, dtype=float), bonds, angles


def norm_element(raw, atom_name):
    elem = (raw or "").strip().upper()
    if not elem:
        elem = "".join(ch for ch in atom_name if ch.isalpha())[:2].upper()
    return elem


def read_pocket(path, lig_coords_nm, cutoff_angstrom):
    lig_ang = lig_coords_nm * 10.0
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
                xyz_ang = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
            except ValueError:
                continue
            dist = float(np.linalg.norm(lig_ang - xyz_ang.reshape(1, 3), axis=1).min())
            if dist <= cutoff_angstrom:
                atoms.append({"xyz_nm": xyz_ang * 0.1, "element": elem})
    if len(atoms) < 10:
        raise ValueError(f"too few pocket atoms near ligand: {len(atoms)}")
    return atoms


def angle_value(a, b, c):
    v1 = a - b
    v2 = c - b
    cosang = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def ligand_rmsd(coords, ref):
    return float(np.sqrt(np.mean(np.sum((coords - ref) ** 2, axis=1)))) * 10.0


def contact_retention(lig, pock, initial_pairs, cutoff_nm=0.4):
    if not initial_pairs:
        return np.nan
    dist = np.linalg.norm(lig[:, None, :] - pock[None, :, :], axis=-1)
    now = {(int(i), int(j)) for i, j in zip(*np.where(dist < cutoff_nm))}
    return len(initial_pairs & now) / max(len(initial_pairs), 1)


def build_system(lig_coords, lig_elems, lig_charges, bonds, angles, pocket_atoms, temperature):
    system = mm.System()
    positions = []
    lig_indices = []
    pocket_indices = []

    for coord, elem in zip(lig_coords, lig_elems):
        p = element_param(elem)
        idx = system.addParticle(p["mass"] * unit.dalton)
        lig_indices.append(idx)
        positions.append(coord)
    for atom in pocket_atoms:
        idx = system.addParticle(0.0 * unit.dalton)
        pocket_indices.append(idx)
        positions.append(atom["xyz_nm"])

    bond_force = mm.HarmonicBondForce()
    bond_force.setForceGroup(2)
    for a, b in bonds:
        r0 = float(np.linalg.norm(lig_coords[a] - lig_coords[b]))
        bond_force.addBond(a, b, r0 * unit.nanometer, 250000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)

    angle_force = mm.HarmonicAngleForce()
    angle_force.setForceGroup(2)
    for a, b, c in angles:
        theta = angle_value(lig_coords[a], lig_coords[b], lig_coords[c])
        angle_force.addAngle(a, b, c, theta * unit.radian, 500.0 * unit.kilojoule_per_mole / unit.radian**2)
    system.addForce(angle_force)

    nonbond = mm.CustomNonbondedForce(
        "4*sqrt(epsilon1*epsilon2)*((0.5*(sigma1+sigma2)/r)^12-(0.5*(sigma1+sigma2)/r)^6)"
        "+138.935456*charge1*charge2/(dielectric*r); dielectric=20.0"
    )
    nonbond.addPerParticleParameter("sigma")
    nonbond.addPerParticleParameter("epsilon")
    nonbond.addPerParticleParameter("charge")
    nonbond.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    nonbond.setCutoffDistance(1.2 * unit.nanometer)
    nonbond.setForceGroup(1)
    for elem, q in zip(lig_elems, lig_charges):
        p = element_param(elem)
        nonbond.addParticle([p["sigma"], p["epsilon"], float(q)])
    for atom in pocket_atoms:
        p = element_param(atom["element"])
        nonbond.addParticle([p["sigma"], p["epsilon"], p["charge"]])
    nonbond.addInteractionGroup(lig_indices, pocket_indices)
    system.addForce(nonbond)

    restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    restraint.addGlobalParameter("k", 20.0)
    restraint.setForceGroup(3)
    for i, coord in enumerate(lig_coords):
        restraint.addParticle(i, [float(coord[0]), float(coord[1]), float(coord[2])])
    system.addForce(restraint)

    return system, np.asarray(positions, dtype=float), lig_indices, pocket_indices


def run_one(row, args):
    mol, lig_coords, lig_elems, charges, bonds, angles = read_ligand(row.mol_pred)
    pocket_atoms = read_pocket(row.mol_cond, lig_coords, args.pocket_cutoff)
    pock_coords = np.vstack([a["xyz_nm"] for a in pocket_atoms])
    system, positions, lig_indices, pocket_indices = build_system(lig_coords, lig_elems, charges, bonds, angles, pocket_atoms, args.temperature)
    integrator = mm.LangevinMiddleIntegrator(args.temperature * unit.kelvin, 1.0 / unit.picosecond, args.timestep_fs * unit.femtosecond)
    platform = mm.Platform.getPlatformByName("CPU")
    sim = mm.Context(system, integrator, platform)
    sim.setPositions(positions * unit.nanometer)
    initial_energy = sim.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    mm.LocalEnergyMinimizer.minimize(sim, tolerance=10.0, maxIterations=args.minimize_steps)
    minimized_energy = sim.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    state = sim.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    initial_pairs = {
        (int(i), int(j))
        for i, j in zip(*np.where(np.linalg.norm(pos[lig_indices][:, None, :] - pos[pocket_indices][None, :, :], axis=-1) < 0.4))
    }
    ref_lig = pos[lig_indices].copy()
    samples = []
    energies = []
    retentions = []
    steps = max(args.steps, args.sample_interval)
    for _ in range(0, steps, args.sample_interval):
        integrator.step(args.sample_interval)
        st = sim.getState(getPositions=True, getEnergy=True, groups={1})
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        lig_now = pos[lig_indices]
        samples.append(ligand_rmsd(lig_now, ref_lig))
        energies.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        retentions.append(contact_retention(lig_now, pock_coords, initial_pairs))
    lig_final = pos[lig_indices]
    return {
        "policy": row.policy,
        "data_id": int(row.data_id) if "data_id" in row._fields else -1,
        "key": row.key,
        "mol_index": int(row.mol_index),
        "risk_prob": float(row.risk_prob),
        "qed": float(row.qed),
        "dock_pose_pass": bool(row.dock_pose_pass) if "dock_pose_pass" in row._fields else None,
        "ligand_atoms": int(len(lig_indices)),
        "pocket_atoms": int(len(pocket_indices)),
        "bonds": int(len(bonds)),
        "angles": int(len(angles)),
        "initial_interaction_kj_mol": float(initial_energy),
        "minimized_interaction_kj_mol": float(minimized_energy),
        "md_mean_interaction_kj_mol": float(np.mean(energies)),
        "md_std_interaction_kj_mol": float(np.std(energies)),
        "final_interaction_kj_mol": float(energies[-1]),
        "mean_ligand_rmsd_ang": float(np.mean(samples)),
        "max_ligand_rmsd_ang": float(np.max(samples)),
        "final_ligand_rmsd_ang": float(samples[-1]),
        "mean_contact_retention": float(np.nanmean(retentions)),
        "final_contact_retention": float(retentions[-1]),
        "stable_proxy": bool(samples[-1] <= args.rmsd_cutoff and retentions[-1] >= args.contact_retention_cutoff),
        "mol_pred": row.mol_pred,
        "mol_cond": row.mol_cond,
    }


def write_report(out, args):
    lines = [
        "# Short MD / Coarse MM-GBSA-Like Stability",
        "",
        "## Protocol",
        "",
        "- Scope: top selected PB+RC molecules only, as a P2 stability sanity check.",
        "- Dynamics engine: OpenMM CPU Langevin dynamics on a heavy-atom ligand plus fixed local pocket atoms.",
        "- Energy model: coarse MM pair interaction with Lennard-Jones plus attenuated Coulomb terms; this is MM-GBSA-like stability proxy, not full Amber/GAFF MM-GBSA.",
        f"- MD length: {args.steps * args.timestep_fs / 1000:.2f} ps; timestep={args.timestep_fs} fs; temperature={args.temperature} K.",
        "",
        "## Results",
        "",
        "| Data | Mol | Risk | QED | Init E | Min E | MD mean E | Final RMSD A | Contact retention | Stable proxy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in out.itertuples(index=False):
        lines.append(
            f"| {row.data_id} | {row.mol_index} | {f4(row.risk_prob)} | {f4(row.qed)} | "
            f"{f4(row.initial_interaction_kj_mol)} | {f4(row.minimized_interaction_kj_mol)} | "
            f"{f4(row.md_mean_interaction_kj_mol)} | {f4(row.final_ligand_rmsd_ang)} | "
            f"{f4(row.final_contact_retention)} | {row.stable_proxy} |"
        )
    stable = float(out["stable_proxy"].mean()) if len(out) else float("nan")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"1. Stable-proxy rate is {100 * stable:.1f}% under the RMSD/contact-retention thresholds.",
            "2. This P2 check is useful for reviewer reassurance, but the manuscript should not call it rigorous production MM-GBSA unless Amber/GAFF or equivalent parameters are later added.",
            "3. Use it as short-MD stability evidence for the final selected poses, not as a binding-free-energy claim.",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-csv", default="results/prospective20_pocket2mol_n128_dockfast_selection.csv")
    ap.add_argument("--policy", default="pb_rc_select")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--sample-interval", type=int, default=100)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--pocket-cutoff", type=float, default=8.0)
    ap.add_argument("--minimize-steps", type=int, default=500)
    ap.add_argument("--rmsd-cutoff", type=float, default=2.5)
    ap.add_argument("--contact-retention-cutoff", type=float, default=0.50)
    ap.add_argument("--out-csv", default="results/short_md_stability_top3.csv")
    ap.add_argument("--out-md", default="experiments/SHORT_MD_STABILITY_TOP3.md")
    args = ap.parse_args()

    df = pd.read_csv(args.selection_csv)
    df = df[df["policy"] == args.policy].copy()
    if "dock_pose_pass" in df.columns:
        df = df.sort_values(["dock_pose_pass", "risk_prob", "qed"], ascending=[False, True, False])
    else:
        df = df.sort_values(["qed", "risk_prob"], ascending=[False, True])
    df = df.groupby("data_id", sort=False).head(1).head(args.top_n)
    rows = []
    failures = []
    for row in df.itertuples(index=False):
        try:
            rows.append(run_one(row, args))
        except Exception as exc:
            failures.append({"data_id": getattr(row, "data_id", -1), "mol_index": getattr(row, "mol_index", -1), "error": str(exc)})
    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    write_report(out, args)
    print(Path(args.out_md).read_text(encoding="utf-8"))
    if failures:
        print("Failures:", failures)


if __name__ == "__main__":
    main()
