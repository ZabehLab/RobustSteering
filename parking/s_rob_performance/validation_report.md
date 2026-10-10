## Material Passport

- Material: existing selected-controller performance and calibration artifacts.
- Scope: statistical re-analysis only; no new generations or model/controller fits.
- Verification Status: ANALYZED.

## Statistical Findings

- The independent unit is the selected model-task controller, not each OOD language or attack template.
- Per-task sample sizes are 4 TruthfulQA, 3 HarmBench, 3 MGSM, and 2 L-CiteEval models.
- Exact within-task permutation tests and 2,000-sample model bootstraps are used because asymptotic correlation p-values are inappropriate at these sample sizes.
- MGSM is not a strict implementation of the planned retention estimand because no English/ID H-infinity result is available.

## Warnings

- Exact p-values are necessarily coarse with only 2–4 model-level observations.
- Bootstrap intervals at these sample sizes are discrete and can appear artificially narrow for perfectly ranked samples.
- The planned mixed model is not identified well enough by the available selected-controller inventory.
- The plan's E1 decision rule cannot be applied to E2-only evidence.

## Fallacy Scan

- Simpson's paradox: checked; raw cross-task pooling is avoided.
- Ecological fallacy: checked; claims are limited to selected model-task controllers.
- Berkson's paradox: checked; selected final controllers are a filtered set, so selection bias remains possible.
- Collider bias: checked; no post-outcome covariate is controlled, but controller selection used calibration performance.
- Base-rate neglect: checked; not a diagnostic-classification analysis.
- Regression to the mean: checked; no extreme-score enrollment or pre/post claim.
- Survivorship bias: checked; only completed final runs are present, and unavailable configurations are documented.
- Look-elsewhere effect: checked; four taskwise tests and three available predictors are reported without selective significance claims.
- Garden of forking paths: checked; the attached plan was written before this re-analysis, but MGSM requires a post-plan proxy and is explicitly labeled.
- Correlation versus causation: checked; results are associational.
- Reverse causality: checked; `S_rob` is computed before held-out evaluation, but this ordering does not itself establish predictive generalization.

Coverage: 11/11 statistical fallacy types checked.

## Reproducibility Verdict

- ANALYZED: the script deterministically reconstructs the point table, statistics, and figure from repository artifacts.
- Not VERIFIED against an external reference result because the supplied mockup contains synthetic data only.
