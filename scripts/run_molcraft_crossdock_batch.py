import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def resolve(root, value):
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def collect_sdfs(source_dir, output_file):
    files = sorted(Path(source_dir).glob("*.sdf"))
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_file))
    records = readable = sanitizable = 0
    for source in files:
        supplier = Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False)
        for mol in supplier:
            records += 1
            if mol is None:
                continue
            readable += 1
            try:
                Chem.SanitizeMol(Chem.Mol(mol))
                sanitizable += 1
            except Exception:
                pass
            mol.SetProp("_Name", source.stem)
            writer.write(mol)
    writer.close()
    if records == 0:
        output_file.unlink(missing_ok=True)
    return {
        "sdf_files": len(files),
        "records": records,
        "readable": readable,
        "sanitizable": sanitizable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/molpilot_crossdock_t50/manifest.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--raw-out-root", default="results/molcraft_crossdock_t50_raw")
    parser.add_argument("--collect-dir", default="results/molcraft_crossdock_t50_n16")
    parser.add_argument("--log-dir", default="logs/molcraft_crossdock_t50")
    parser.add_argument("--out-csv", default="results/molcraft_crossdock_t50_n16_run_summary.csv")
    parser.add_argument("--out-json", default="logs/molcraft_crossdock_t50/run_summary.json")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    manifest = pd.read_csv(args.manifest).head(args.limit)
    rows = []
    start_all = time.perf_counter()
    for rec in manifest.itertuples(index=False):
        data_id = int(rec.data_id)
        key = str(rec.key)
        raw_dir = root / args.raw_out_root / f"data{data_id:03d}"
        collect_path = root / args.collect_dir / f"{key}_gen.sdf"
        log_path = root / args.log_dir / f"data{data_id:03d}.log"
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and collect_path.exists() and collect_path.stat().st_size > 0:
            stats = collect_sdfs(raw_dir, collect_path)
            rows.append(
                {
                    "data_id": data_id,
                    "key": key,
                    "returncode": 0,
                    "elapsed_sec": 0.0,
                    "generated_sdf": str(collect_path.relative_to(root)),
                    "skipped_existing": True,
                    **stats,
                }
            )
            continue

        command = [
            sys.executable,
            "scripts/run_molcraft_sample_for_pocket.py",
            str(resolve(root, rec.pocket_path)),
            str(resolve(root, rec.native_ligand_path)),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["WANDB_MODE"] = "offline"
        env["MOLCRAFT_OUT_DIR"] = str(raw_dir)
        env["MOLCRAFT_NUM_SAMPLES"] = str(args.num_samples)
        env["MOLCRAFT_SAMPLE_STEPS"] = str(args.sample_steps)
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("command=" + " ".join(command) + "\n")
            log.write(f"CUDA_VISIBLE_DEVICES={args.gpu}\n\n")
            try:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    timeout=args.timeout_sec,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                log.write(f"\nTIMEOUT after {args.timeout_sec} sec\n")
                code = 124
        stats = collect_sdfs(raw_dir, collect_path)
        row = {
            "data_id": data_id,
            "key": key,
            "returncode": code,
            "elapsed_sec": time.perf_counter() - start,
            "generated_sdf": str(collect_path.relative_to(root)) if collect_path.exists() else "",
            "skipped_existing": False,
            **stats,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    result_df = pd.DataFrame(rows).sort_values("data_id")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out_csv, index=False)
    summary = {
        "generator": "MolCRAFT",
        "targets_requested": int(len(manifest)),
        "targets_successful": int((result_df["returncode"] == 0).sum()),
        "targets_with_sdf": int((result_df["records"] > 0).sum()),
        "records": int(result_df["records"].sum()),
        "sanitizable": int(result_df["sanitizable"].sum()),
        "num_samples": args.num_samples,
        "sample_steps": args.sample_steps,
        "gpu": args.gpu,
        "elapsed_sec": time.perf_counter() - start_all,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path("experiments/MOLCRAFT_CROSSDOCK_T50_GENERATION.md").write_text(
        "# MolCRAFT CrossDock Generation\n\n"
        f"- Targets requested: {summary['targets_requested']}; successful: {summary['targets_successful']}.\n"
        f"- Targets with SDF: {summary['targets_with_sdf']}.\n"
        f"- Candidate budget: {summary['num_samples']} per target, {summary['sample_steps']} sampling steps.\n"
        f"- Collected records: {summary['records']}; sanitizable: {summary['sanitizable']}.\n"
        f"- Runtime: {summary['elapsed_sec']:.1f} s on GPU {summary['gpu']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
