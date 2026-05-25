# RC-SBDD-Bench v1

- Targets: 100
- Generators: 4
- Candidate label rows: 13189
- Candidate SDF files: 303

See `DATASET_CARD.md`, `METRICS.md`, `LICENSE`, `labels/oracle_labels.csv`, `splits/official_target_split.csv`, and `evaluation/score_submission.py`.

## Fixed Base Asset

The fixed evaluation base archive is distributed with release `v1.1.0`:

`https://github.com/wushen1103-collab/RC-SBDD/releases/download/v1.1.0/RC-SBDD-Bench-v1-full-20260525.tar.gz`

Verify the download against `FULL_ASSET_SHA256.txt`. The archive contains the
candidate structures, pocket files, full oracle labels, official split,
metric definitions, and checksum inventory; the Git tree remains lightweight.

## Addenda

- `addenda/SGEDiff-T50`: reproduced SGEDiff T50 generator-shift stress case with candidate SDFs, oracle labels, and hashes.
- Lightweight P0 audit snapshots in `../../paper_source_data`: MolCRAFT and
  MolPilot-framefix direct-output evaluations, Prospective20 route planning,
  and generator-shift calibration. These are versioned paper addenda and do
  not redefine the fixed v1 base counts above.
