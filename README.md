# RC-SBDD

This repository contains the lightweight reproducibility package for **RC-SBDD:
Calibrated Multi-Oracle Reliability Control for Structure-Based Molecular
Generation**.

The repository is intentionally small. It tracks code, protocol files,
leaderboard utilities, and paper source-data snapshots. It does **not** track
paper figures, compiled tables, PDFs, large CrossDocked/BindingMOAD structures,
generated SDF files, model checkpoints, GNINA binaries, or AiZynthFinder stock
files.

## What Is Included

- `src/rcsbdd`: lightweight feature, data, and evaluation utilities.
- `scripts`: experiment, analysis, benchmark-building, and smoke-test scripts.
- `paper_source_data`: small CSV snapshots used to reproduce the paper's main
  reported tables and diagnostic summaries.
- `benchmarks/RC-SBDD-Bench-v1`: benchmark card, metric definitions,
  lightweight leaderboard script, and release manifest template.
- `docs`: reproducibility guide, dataset card, failure taxonomy, and oracle
  reliability notes.

## Quick Start

```bash
conda create -n rcsbdd python=3.10 -y
conda activate rcsbdd
pip install -e .
pip install -r requirements.txt

python scripts/run_snapshot_smoke.py
python benchmarks/RC-SBDD-Bench-v1/evaluation/score_submission.py \
  --labels paper_source_data/trans_journal_master_evidence.csv \
  --submission examples/toy/toy_submission.csv \
  --out logs/toy_leaderboard.json
```

The first command checks that the source-data snapshots reproduce the headline
values used in the manuscript. Full de novo generation, redocking, and route
planning require the upstream datasets and external tools listed in
`docs/REPRODUCIBILITY.md`.

## Full Reproduction Overview

1. Download or prepare upstream structural data following
   `docs/REPRODUCIBILITY.md`.
2. Run or collect generator outputs for DiffSBDD, Pocket2Mol, PocketFlow, SYNC,
   MolPilot, and SGEDiff using the documented scripts.
3. Compute oracle labels with PoseBusters, GNINA/Vina, RDKit, and AiZynthFinder.
4. Build RC-SBDD-Bench-v1 manifests with
   `scripts/build_rcsbdd_bench_v1_release.py`.
5. Recompute paper summaries with `scripts/build_paper_tables.py`.

The paper source-data snapshots allow reviewers to verify the reported
selection, calibration, statistical-test, and runtime summaries without
downloading large molecular files.

## Repository Size Policy

Large artifacts should be stored outside GitHub and referenced by DOI or
download instructions. The `.gitignore` excludes common molecular structure,
checkpoint, result, and figure formats.

## Citation

Please cite the RC-SBDD manuscript and this repository. A Zenodo-ready metadata
file is provided in `.zenodo.json`; a DOI can be minted from the GitHub release
through Zenodo.

