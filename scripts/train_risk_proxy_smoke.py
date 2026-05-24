import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rcsbdd.data.lmdb_dataset import CrossDockedLMDB
from rcsbdd.eval.risk_metrics import summarize_risk_metrics
from rcsbdd.features.interaction import corrupt_item, feature_names, featurize_interaction


def choose_indices(n, max_items, rng):
    max_items = min(max_items, n)
    return rng.choice(n, size=max_items, replace=False)


def build_examples(ds, indices, modes, rng, include_pocket_feat=True):
    x_rows, y_rows, mode_rows = [], [], []
    for idx in indices:
        item = ds[int(idx)]
        x_rows.append(featurize_interaction(item, include_pocket_feat=include_pocket_feat))
        y_rows.append(0)
        mode_rows.append("native")
        for mode in modes:
            donor = None
            if mode in {"pocket_mismatch", "pocket_mismatch_aligned"}:
                donor_idx = int(rng.integers(0, len(ds)))
                if donor_idx == int(idx):
                    donor_idx = (donor_idx + 1) % len(ds)
                donor = ds[donor_idx]
            corrupted = corrupt_item(item, mode, rng, donor=donor)
            x_rows.append(featurize_interaction(corrupted, include_pocket_feat=include_pocket_feat))
            y_rows.append(1)
            mode_rows.append(mode)
    return np.stack(x_rows), np.asarray(y_rows, dtype=np.int64), np.asarray(mode_rows)


def robust_z(values, ref):
    med = np.median(ref)
    iqr = np.percentile(ref, 75) - np.percentile(ref, 25)
    return (values - med) / max(float(iqr), 1e-6)


def heuristic_scores(x_train, y_train, x_eval, names):
    idx = {name: i for i, name in enumerate(names)}
    native = x_train[y_train == 0]
    z_center = robust_z(x_eval[:, idx["center_dist"]], native[:, idx["center_dist"]])
    z_clash = robust_z(x_eval[:, idx["clash_lt_1_5_per_lig"]], native[:, idx["clash_lt_1_5_per_lig"]])
    z_low_contact = -robust_z(x_eval[:, idx["frac_lig_contact_lt_4_0"]], native[:, idx["frac_lig_contact_lt_4_0"]])
    raw = np.maximum(0.0, z_center) + np.maximum(0.0, z_clash) + np.maximum(0.0, z_low_contact)
    return 1.0 / (1.0 + np.exp(-raw))


def mode_metrics(y, prob, modes):
    out = {"overall": summarize_risk_metrics(y, prob)}
    for mode in sorted(set(modes.tolist()) - {"native"}):
        mask = (modes == "native") | (modes == mode)
        out[mode] = summarize_risk_metrics(y[mask], prob[mask])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/if3-crossdocked2020")
    ap.add_argument("--max-train", type=int, default=4096)
    ap.add_argument("--max-val", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--modes", nargs="+", default=["translate", "gaussian", "clash", "pocket_mismatch"])
    ap.add_argument("--classifier", choices=["hgb", "logreg", "rf", "mlp"], default="hgb")
    ap.add_argument("--geometry-only", action="store_true", help="Drop LMDB-only pocket feature means so raw PDB/SDF structures can be scored consistently.")
    ap.add_argument("--hgb-max-iter", type=int, default=120)
    ap.add_argument("--out", default="logs/risk_proxy_smoke.json")
    ap.add_argument("--model-out", default="results/risk_proxy_smoke/model.pkl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train_ds = CrossDockedLMDB(Path(args.root) / "train.lmdb")
    val_ds = CrossDockedLMDB(Path(args.root) / "val.lmdb")
    train_idx = choose_indices(len(train_ds), args.max_train, rng)
    val_idx = choose_indices(len(val_ds), args.max_val, rng)

    include_pocket_feat = not args.geometry_only
    x_train, y_train, m_train = build_examples(train_ds, train_idx, args.modes, rng, include_pocket_feat=include_pocket_feat)
    x_val, y_val, m_val = build_examples(val_ds, val_idx, args.modes, rng, include_pocket_feat=include_pocket_feat)
    names = feature_names(include_pocket_feat=include_pocket_feat)

    if args.classifier == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=8, random_state=args.seed),
        )
        model.fit(x_train, y_train)
    elif args.classifier == "hgb":
        base_train_idx, calib_idx = train_test_split(
            np.arange(len(y_train)),
            test_size=0.2,
            random_state=args.seed,
            stratify=y_train,
        )
        base = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=args.hgb_max_iter,
            learning_rate=0.06,
            l2_regularization=1e-3,
            random_state=args.seed,
        )
        base.fit(x_train[base_train_idx], y_train[base_train_idx])
        model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        model.fit(x_train[calib_idx], y_train[calib_idx])
    elif args.classifier == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=24,
            random_state=args.seed,
        )
        model.fit(x_train, y_train)
    else:
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=args.seed,
            ),
        )
        model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    train_prob = model.predict_proba(x_train)[:, 1]
    heuristic_val = heuristic_scores(x_train, y_train, x_val, names)
    heuristic_train = heuristic_scores(x_train, y_train, x_train, names)

    result = {
        "seed": args.seed,
        "modes": args.modes,
        "train_native": int(len(train_idx)),
        "val_native": int(len(val_idx)),
        "train_examples": int(len(y_train)),
        "val_examples": int(len(y_val)),
        "feature_dim": int(x_train.shape[1]),
        "classifier": args.classifier,
        "geometry_only": bool(args.geometry_only),
        "model": {
            "train": mode_metrics(y_train, train_prob, m_train),
            "val": mode_metrics(y_val, val_prob, m_val),
        },
        "heuristic": {
            "train_overall_auroc": float(roc_auc_score(y_train, heuristic_train)),
            "val": mode_metrics(y_val, heuristic_val, m_val),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": model, "feature_names": names, "modes": args.modes, "include_pocket_feat": include_pocket_feat}, f)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
