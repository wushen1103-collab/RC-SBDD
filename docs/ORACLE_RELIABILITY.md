# Oracle Reliability and Modality Contribution

The missing-modality fusion study estimates which oracle families matter most
for downstream reliability. From the paper source-data snapshot, removing
validity evidence causes the largest mean Brier degradation, followed by
geometry. Removing chemistry or the learned risk feature has a much smaller
average effect in the reported settings.

This supports the manuscript's claim that RC-SBDD is not simply a molecule-only
QED reranker. Pose/validity and structural interaction evidence are central to
reliable post-generation selection.

