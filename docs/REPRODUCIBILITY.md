# Reproducibility Guide

## Minimal Review Reproduction

The minimal path verifies the values used in the manuscript without downloading
large upstream molecular files:

```bash
pip install -e .
pip install -r requirements.txt
python scripts/verify_source_data_manifest.py
python scripts/run_snapshot_smoke.py
```

This reads the CSV files in `paper_source_data/` and checks:

- official target-level gain;
- direct SOTA/external-generation evidence;
- multi-objective selection baselines;
- missing-modality oracle reliability;
- risk-to-dock-fast calibration;
- target-heldout selective-risk summaries;
- runtime/throughput summaries.

The smoke test asserts the headline values used in the manuscript: official
100-target dock-fast gain of `0.169`, PocketFlow direct-output dock-fast gain
of `0.1125`, BindingMOAD v100 dock-fast gain of `0.0975`, SYNC-Guide
direct-output dock-fast gain of `0.030`, and DiffSBDD target-heldout CRC
violation rate of `0.0`. The manifest verifier checks the
byte count and SHA256 hash of each lightweight CSV snapshot before these values
are recomputed.

## Full Computational Reproduction

The full pipeline is larger and requires upstream resources that are not
redistributed here.

### Required External Assets

- CrossDocked2020 or the IF3/CrossDocked LMDB used by the generator scripts.
- BindingMOAD/PDB-derived pockets for the external holdout.
- Public generator checkpoints or outputs for DiffSBDD, Pocket2Mol, PocketFlow,
  SYNC, MolPilot, and SGEDiff.
- GNINA and AutoDock Vina binaries.
- RDKit, Open Babel, PoseBusters, and AiZynthFinder.
- Optional high-fidelity tools for short MD/MM-GBSA style follow-up.

### Reproduction Stages

1. **Prepare data.** Place upstream data under `data/raw/` and processed indices
   under `data/processed/`. These directories are ignored by Git.
2. **Train or load risk scorer.** Use `scripts/train_risk_proxy_smoke.py` for a
   small corruption-based smoke run, or the full training command used in the
   paper once CrossDocked LMDB files are available.
3. **Score generated candidates.** Use `scripts/score_generated_risk.py` or
   `scripts/score_manifest_sdf_risk.py`.
4. **Run oracle evaluation.** Use the PoseBusters, GNINA/Vina, and AiZynthFinder
   scripts in `scripts/`.
5. **Build benchmark release files.** Use
   `scripts/build_rcsbdd_bench_v1_release.py`.
6. **Rebuild summaries.** Use `scripts/build_paper_tables.py` and the analysis
   scripts for calibration, selective risk, missing modalities, and runtime.

## Artifact Boundary

This repository is a reproducibility entry point rather than a storage mirror.
Large generated molecule files and third-party outputs should be archived in a
separate Zenodo record or regenerated from upstream tools.

## Release and DOI Checklist

1. Confirm that `python scripts/verify_source_data_manifest.py` passes.
2. Confirm that `python scripts/run_snapshot_smoke.py` passes.
3. Push the repository and tag `v1.0.1` to GitHub.
4. Create a GitHub release from `v1.0.1`.
5. Connect the GitHub release to Zenodo and mint a DOI.
6. Add the DOI to the manuscript data-availability statement and to the README.
