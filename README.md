# Anonymous RC-SBDD reviewer code archive

Run `python -m pip install -r requirements.txt`, then `python reproduce.py`
and `python reproduce_statistics.py`.
No network, GPU, molecular regeneration, figures, or manuscript tables are required.

The executable checks file hashes, frozen logistic-model inference from 42 features,
principal PB-QED/PB-RC selection on both complete candidate matrices, and all
20 original calibrated-decision split/policy records. Results are written to
REPRODUCTION_RESULT.json. The statistics entry point recomputes target-paired
bootstrap intervals (10,000 replicates, seed 20261105), Wilcoxon tests,
paired rank-biserial effects, and separate-family FDR for the four primary
endpoint comparisons and twelve strong-control comparisons. It writes
STATISTICS_RESULT.json. Numerical CSVs are machine-readable experiment inputs
and audit records, not publication table or figure exports.

The benchmark folder contains the fixed numerical evidence matrix, target IDs,
and available official splits. Data/audit_data also include existing target-level
certification records, unified GNINA outputs, bootstrap replicates, and the new
fallback/SASA/identity/cost reconciliation records. These permit independent
inspection; reproduce.py does not claim to rerun docking or certify new policies.

Raw protein/molecule structures and generator weights are upstream resources,
not redistributed here. Full coordinate-level regeneration requires the public
CrossDocked2020, generator releases, PoseBusters, GNINA 1.3.2, and AiZynthFinder
resources described in the manuscript. The included feature extractor and
training script require RDKit, torch, and LMDB in addition to the small offline
verification environment. Original licenses continue to govern upstream data.

This is a frozen numerical-reproduction snapshot for peer review.
Reviewers can browse the files and download the complete snapshot from the
anonymous repository link supplied with the revised manuscript. The GitHub
source branch is kept separate from ongoing development. No manuscript
figures, publication tables, server credentials, or generator checkpoints are
included. Model parameters required for numerical inference are included.
