# RC-SBDD-Bench-v1 Dataset Card

## Purpose

RC-SBDD-Bench-v1 evaluates post-generation selection reliability for
structure-based molecular generation. It fixes candidate pools, target metadata,
oracle labels, official splits, and scoring scripts so that generator quality and
selector reliability can be studied separately.

## Intended Use

- Evaluate candidate-selection policies over generated molecule pools.
- Compare calibrated reliability control against single-objective and
  multi-objective selection baselines.
- Study target shift, generator shift, missing/noisy oracle evidence, and
  selective-risk behavior.

## Not Intended For

- Claiming wet-lab activity or clinical efficacy.
- Redistributing upstream datasets without respecting original licenses.
- Training a generator on hidden or proprietary target information.

## Current Paper Snapshot

- Main benchmark scope: 100 targets.
- Main release generator families: four.
- Candidate rows tracked in the manuscript: 13,189.
- Dock-fast labels tracked in the manuscript: 12,013.
- Additional direct-output addenda: MolCRAFT positive check, MolPilot-framefix
  stress audit, SGEDiff stress audit, and Prospective20 route-planning summary.

## Key Labels

- `dock_fast`: operational downstream success label.
- PoseBusters-style molecule and protein-pose checks.
- QED and SA-style molecular-quality scores.
- GNINA/Vina score-only and redocking outputs where available.
- AiZynthFinder route-planning summaries for selected candidates.
- Calibrated RC-SBDD risk scores.

## Splits

The paper reports official target-level, protein-unseen, family-unseen,
native-scaffold-unseen, generated-scaffold-unseen, direct generator-output,
BindingMOAD external, and target-heldout selective-risk settings.

## Licenses

Code is MIT licensed. Benchmark metadata and labels are intended for academic
research use. Original molecular structures, generated molecules, and third-party
oracle outputs remain governed by their upstream licenses.
