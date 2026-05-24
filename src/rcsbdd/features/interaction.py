import copy

import numpy as np

ATOM_VOCAB_SIZE = 11
RADIAL_BINS = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float32)


def feature_names(atom_vocab_size=ATOM_VOCAB_SIZE, pocket_feat_dim=44, include_pocket_feat=True):
    names = [
        "lig_n",
        "pock_n",
        "center_dist",
        "lig_radius",
        "pock_radius",
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
    ]
    names += [f"lig_atom_hist_{i}" for i in range(atom_vocab_size)]
    if include_pocket_feat:
        names += [f"pock_feat_mean_{i}" for i in range(pocket_feat_dim)]
    names += [f"lp_radial_bin_{i}" for i in range(len(RADIAL_BINS) - 1)]
    return names


def featurize_interaction(item, atom_vocab_size=ATOM_VOCAB_SIZE, include_pocket_feat=True):
    lig_pos = np.asarray(item["lig_pos"], dtype=np.float32)
    pock_pos = np.asarray(item["pock_pos"], dtype=np.float32)
    lig_atom_type = np.asarray(item["lig_atom_type"], dtype=np.int64)
    pock_feat = np.asarray(item.get("pock_feat", np.zeros((pock_pos.shape[0], 0))), dtype=np.float32)

    lig_center = lig_pos.mean(axis=0)
    pock_center = pock_pos.mean(axis=0)
    dmat = np.linalg.norm(lig_pos[:, None, :] - pock_pos[None, :, :], axis=-1)
    per_lig_min = dmat.min(axis=1)
    flat = dmat.reshape(-1)

    lig_hist = np.bincount(np.clip(lig_atom_type, 0, atom_vocab_size - 1), minlength=atom_vocab_size).astype(np.float32)
    lig_hist /= max(float(lig_hist.sum()), 1.0)
    pock_feat_mean = pock_feat.mean(axis=0).astype(np.float32) if pock_feat.size else np.asarray([], dtype=np.float32)
    radial_hist, _ = np.histogram(flat, bins=RADIAL_BINS)
    radial_hist = radial_hist.astype(np.float32) / max(float(len(flat)), 1.0)

    scalars = np.array(
        [
            lig_pos.shape[0],
            pock_pos.shape[0],
            np.linalg.norm(lig_center - pock_center),
            np.linalg.norm(lig_pos - lig_center, axis=1).mean(),
            np.linalg.norm(pock_pos - pock_center, axis=1).mean(),
            flat.min(),
            np.percentile(flat, 1),
            np.percentile(flat, 5),
            np.percentile(flat, 10),
            np.percentile(flat, 25),
            np.percentile(flat, 50),
            flat.mean(),
            (dmat < 1.0).sum() / max(lig_pos.shape[0], 1),
            (dmat < 1.5).sum() / max(lig_pos.shape[0], 1),
            (dmat < 2.0).sum() / max(lig_pos.shape[0], 1),
            (dmat < 3.0).sum() / max(lig_pos.shape[0], 1),
            (dmat < 4.0).sum() / max(lig_pos.shape[0], 1),
            (dmat < 5.0).sum() / max(lig_pos.shape[0], 1),
            (per_lig_min < 4.0).mean(),
        ],
        dtype=np.float32,
    )
    parts = [scalars, lig_hist]
    if include_pocket_feat:
        parts.append(pock_feat_mean)
    parts.append(radial_hist)
    return np.concatenate(parts).astype(np.float32)


def corrupt_item(item, mode, rng, donor=None):
    out = copy.copy(item)
    out["lig_pos"] = np.asarray(item["lig_pos"], dtype=np.float32).copy()
    out["lig_atom_type"] = np.asarray(item["lig_atom_type"], dtype=np.int64).copy()
    out["pock_pos"] = np.asarray(item["pock_pos"], dtype=np.float32).copy()
    out["pock_feat"] = np.asarray(item["pock_feat"], dtype=np.float32).copy()

    if mode == "translate":
        direction = rng.normal(size=3).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        out["lig_pos"] += direction * rng.uniform(6.0, 12.0)
    elif mode == "mild_translate":
        direction = rng.normal(size=3).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        out["lig_pos"] += direction * rng.uniform(1.0, 3.0)
    elif mode == "gaussian":
        out["lig_pos"] += rng.normal(0.0, rng.uniform(1.5, 3.0), size=out["lig_pos"].shape).astype(np.float32)
    elif mode == "mild_gaussian":
        out["lig_pos"] += rng.normal(0.0, rng.uniform(0.35, 0.9), size=out["lig_pos"].shape).astype(np.float32)
    elif mode == "clash":
        center = out["pock_pos"][rng.integers(0, out["pock_pos"].shape[0])]
        lig_rel = out["lig_pos"] - out["lig_pos"].mean(axis=0, keepdims=True)
        out["lig_pos"] = center + lig_rel * 0.25 + rng.normal(0.0, 0.25, size=out["lig_pos"].shape).astype(np.float32)
    elif mode == "rotate_pose":
        center = out["lig_pos"].mean(axis=0, keepdims=True)
        out["lig_pos"] = (out["lig_pos"] - center) @ _random_rotation(rng).T + center
    elif mode == "pocket_mismatch":
        if donor is None:
            raise ValueError("pocket_mismatch corruption requires a donor item")
        out["pock_pos"] = np.asarray(donor["pock_pos"], dtype=np.float32).copy()
        out["pock_feat"] = np.asarray(donor["pock_feat"], dtype=np.float32).copy()
    elif mode == "pocket_mismatch_aligned":
        if donor is None:
            raise ValueError("pocket_mismatch_aligned corruption requires a donor item")
        donor_pos = np.asarray(donor["pock_pos"], dtype=np.float32).copy()
        target_center = out["pock_pos"].mean(axis=0, keepdims=True)
        donor_center = donor_pos.mean(axis=0, keepdims=True)
        out["pock_pos"] = donor_pos - donor_center + target_center
        out["pock_feat"] = np.asarray(donor["pock_feat"], dtype=np.float32).copy()
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return out


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q.astype(np.float32)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q
