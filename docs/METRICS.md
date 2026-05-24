# Metrics

## Selection Metrics

- `dock_fast`: primary downstream computational success endpoint.
- `risk_prob`: calibrated failure probability estimated by RC-SBDD.
- `risk_gt_0_5`: fraction of selected candidates with risk probability above
  0.5.
- `qed`: molecule-level desirability proxy.
- `mol_fast`: fast molecule-level validity.
- `protein_pass` / `dock_pose_pass`: protein-pose and docking-pose checks where
  available.

## Calibration Metrics

- AUROC and AUPRC for discrimination.
- Brier score for probabilistic accuracy.
- Expected calibration error (ECE).
- Reliability curves and coverage-risk curves.

## Statistical Testing

Main comparisons are target-level paired tests:

- paired bootstrap confidence intervals;
- Wilcoxon signed-rank tests;
- Cliff's delta;
- FDR correction across reported metrics/settings.

