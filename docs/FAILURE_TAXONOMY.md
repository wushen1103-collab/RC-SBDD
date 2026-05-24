# Failure Taxonomy

The manuscript treats failure cases as part of the reliability-control claim.
The main observable failure classes are:

| Class | Observable evidence | Interpretation |
|---|---|---|
| High-risk saturation | MolPilot and SGEDiff stress rows have 100% high-risk selected candidates | the generator pool contains too few recoverable candidates |
| External-pocket difficulty | BindingMOAD holdout has lower absolute dock-fast pass than CrossDocked-derived settings | target distribution differs from the controlled setting |
| Formal guarantee boundary | Pocket2Mol target-heldout selective-risk summary is infeasible/violated in reported splits | calibration assumptions are not stable under that generator/domain |
| Missing validity sensitivity | missing-validity fusion scenarios cause the largest Brier/ECE degradation | pose/validity oracles carry critical reliability information |
| Missing geometry sensitivity | missing-geometry scenarios are the second largest degradation mode | structural interaction evidence is not replaceable by chemistry alone |

This taxonomy is diagnostic, not a wet-lab failure label.

