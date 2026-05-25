# RC-SBDD

Reproducibility release: `v1.1.0`.

This repository contains the lightweight reproducibility package for **RC-SBDD:
Calibrated Multi-Oracle Reliability Control for Structure-Based Molecular
Generation**.

The Git repository is intentionally small. It tracks code, protocol files,
leaderboard utilities, and paper source-data snapshots. The fixed full
benchmark asset is distributed as a compressed GitHub Release attachment so
that candidate structures and labels are reproducible without expanding Git
history. Paper figures, compiled tables, checkpoints, GNINA binaries, and
AiZynthFinder stock files are not tracked.

## What Is Included

- `src/rcsbdd`: lightweight feature, data, and evaluation utilities.
- `scripts`: experiment, analysis, benchmark-building, and smoke-test scripts.
- `paper_source_data`: small CSV snapshots used to reproduce the paper's main
  reported tables and diagnostic summaries.
- `benchmarks/RC-SBDD-Bench-v1`: benchmark card, metric definitions,
  lightweight leaderboard script, checksum, and full-asset download pointer.
- `docs`: reproducibility guide, dataset card, failure taxonomy, and oracle
  reliability notes.

## Quick Start

```bash
conda create -n rcsbdd python=3.10 -y
conda activate rcsbdd
pip install -e .
pip install -r requirements.txt

python scripts/run_snapshot_smoke.py
python scripts/verify_source_data_manifest.py
python benchmarks/RC-SBDD-Bench-v1/evaluation/score_submission.py \
  --labels paper_source_data/trans_journal_master_evidence.csv \
  --submission examples/toy/toy_submission.csv \
  --out logs/toy_leaderboard.json
```

The first command checks that the source-data snapshots reproduce the headline
values used in the manuscript, including the official 100-target dock-fast
gain (`0.169`), PocketFlow direct-output gain (`0.1125`), BindingMOAD v100
gain (`0.0975`), SYNC-Guide direct-output gain (`0.030`), and DiffSBDD
target-heldout CRC violation rate (`0.0`). The
manifest verifier checks byte counts and SHA256 hashes for every lightweight
CSV snapshot. Full de novo generation, redocking, and route planning require
the upstream datasets and external tools listed in `docs/REPRODUCIBILITY.md`.

## Full Reproduction Overview

1. Download or prepare upstream structural data following
   `docs/REPRODUCIBILITY.md`.
2. Run or collect generator outputs for DiffSBDD, Pocket2Mol, PocketFlow, SYNC,
   MolPilot, and SGEDiff using the documented scripts.
3. Compute oracle labels with PoseBusters, GNINA/Vina, RDKit, and AiZynthFinder.
4. Build RC-SBDD-Bench-v1 manifests with
   `scripts/build_rcsbdd_bench_v1_release.py`.
5. Integrate the SYNC-Guide direct-output snapshot with
   `scripts/add_syncguide_positive_sota.py` when the full `results/` files are
   available.
6. Recompute paper summaries with `scripts/build_paper_tables.py`.

The paper source-data snapshots allow reviewers to verify the reported
selection, calibration, statistical-test, and runtime summaries without
downloading large molecular files.

## Full Fixed Benchmark Asset

The full frozen `RC-SBDD-Bench-v1` asset is available from release `v1.1.0`:

`https://github.com/wushen1103-collab/RC-SBDD/releases/download/v1.1.0/RC-SBDD-Bench-v1-full-20260525.tar.gz`

SHA256:

`7d03f5fc8c8c39a8df7a78ae6ede183459abdea69e3c971e3d3c99fbf93ba9fd`

The archive contains released candidate structures, pockets, label files,
official target splits, metric definitions, and checksum inventory. Large
upstream datasets and executable third-party tools remain obtained from their
original providers.

## Artifact Review Notes

- The GitHub Actions workflow runs the source-data hash check, paper snapshot
  smoke test, and toy leaderboard checker.
- The repository is DOI-ready through `.zenodo.json`; Zenodo archival may be
  added without changing the fixed GitHub Release checksum.
- Figures, compiled tables, PDFs, checkpoints, GNINA binaries, and
  AiZynthFinder stock files are intentionally excluded from Git history.

## Citation

Please cite the RC-SBDD manuscript and this repository. A Zenodo-ready metadata
file is provided in `.zenodo.json`; a DOI can be minted from the GitHub release
through Zenodo.
