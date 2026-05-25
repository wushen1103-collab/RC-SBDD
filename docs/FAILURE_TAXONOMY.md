# Failure Taxonomy

The manuscript treats failure cases as part of the reliability-control claim.
The main observable failure classes are:

| Class | Observable evidence | Interpretation |
|---|---|---|
| High-risk saturation | MolPilot-framefix and SGEDiff stress rows remain nearly all high-risk after selection | the generator pool contains too few recoverable candidates |
| Route-pose mismatch | MolPilot-framefix keeps AiZynthFinder solved rate at 20% while dock-fast remains 1.5%--3.5% | route feasibility is not enough when pocket geometry fails |
| External-pocket difficulty | BindingMOAD holdout has lower absolute dock-fast pass than CrossDocked-derived settings | target distribution differs from the controlled setting |
| Formal guarantee boundary | Pocket2Mol target-heldout selective-risk summary is infeasible/violated in reported splits | calibration assumptions are not stable under that generator/domain |
| Missing validity sensitivity | missing-validity fusion scenarios cause the largest Brier/ECE degradation | pose/validity oracles carry critical reliability information |
| Missing geometry sensitivity | missing-geometry scenarios are the second largest degradation mode | structural interaction evidence is not replaceable by chemistry alone |

This taxonomy is diagnostic, not a wet-lab failure label.
