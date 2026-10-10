# SROB Evidence Inventory

This is an internal consolidation of the repository's existing SROB analyses. It does not regenerate or modify any figure, benchmark table, calibration, or evaluation result. Exact machine-readable values and source paths are in `plots/evidence_inventory.csv`.

## Bottom line

The current repository does **not** support a broad claim that SROB predicts steering reliability across tasks. The independent sample is only two to four models per benchmark, exact tests are coarse, and the direction changes with both task and outcome definition.

There is enough material for a short appendix subsection if it is framed as an **exploratory HarmBench case study**, not a general validation claim:

- SROB is computed from calibration data before the held-out jailbreak evaluations.
- Across three Llama models, every one of the five held-out HarmBench attack templates has a positive linear SROB--H-infinity safe-response slope.
- The condition-adjusted common slope is `+0.3467` percentage points of safe-response rate per unit of SROB.
- The exact model-block permutation result is `p = 0.667` because there are only three independent models. The five attack templates are repeated outcomes, not fifteen independent SROB estimates.
- Appropriate language is therefore "directionally consistent exploratory evidence" or "SROB tracked absolute held-out safe-response rates in this HarmBench case study." It is not appropriate to write that the analysis establishes predictive validity.

## Compact HarmBench table for a possible appendix

| Held-out attack template | Slope of H-infinity safe response on SROB (percentage points / SROB) | Direction |
|---|---:|---|
| APM programmer | +0.311 | Positive |
| DNE nonresponse | +0.172 | Positive |
| Jailbreak Bot | +0.426 | Positive |
| John persona | +0.426 | Positive |
| YOJA/Nona roleplay | +0.400 | Positive |
| Condition-adjusted common slope | +0.347 | Exact model-block `p = 0.667` |

This table is preferable to another figure: it documents the repeated directional pattern while keeping the independent model count and inferential limitation visible.

## Secondary positive result

MGSM shows a separate post-hoc pattern when the outcome is H-infinity's accuracy advantage over the best comparator rather than absolute accuracy: all five language-specific slopes are positive, the condition-adjusted common slope is `+12.773`, and the exact model-block permutation value is `p = 0.167`. Absolute MGSM H-infinity accuracy has the opposite SROB slope (`-15.713`, `p = 0.333`). Because the favorable endpoint was selected after inspecting several definitions, this result can be mentioned only as exploratory and should not be combined with HarmBench as if both tested one preregistered estimand.

## Results that should not support the claim

- ID-to-OOD retention does not give consistent positive evidence: HarmBench `rho = -1.0`, Truthfulness `rho = -0.2`, MGSM proxy `rho = +0.5`, and L-CiteEval `rho = -1.0`; all exact `p >= 0.333`.
- The pooled cross-task OOD slope mixes task-native metrics and task-specific controller coordinates; it is descriptive, not a valid global regression.
- The original task-clustering result (`p = 0.0177`) describes task structure in SROB, not held-out reliability prediction. It weakens after adding the small Truthfulness runs (`p = 0.1673`).
- The outcome-filtered small-model task-clustering result (`p = 0.0201`) cannot support prediction because inclusion explicitly depends on positive H-infinity T-times-Informative change.
- Truthfulness parameter scaling is not reliability prediction and is not statistically supported (`p = 0.368`).

## Recommended appendix scope

Use one short subsection titled along the lines of **Exploratory calibration-time SROB and held-out HarmBench safety**. Include the compact six-row table above and state the three-model limitation in the same paragraph. Do not include the pooled regression, task clustering, outcome-filtered result, or broad wording such as "SROB predicts steering reliability across benchmarks."

## Sources consolidated

- `parking/gamma_star_organization/plots/srob_ood_performance_1x5_statistics.json`
- `parking/gamma_star_organization/plots/srob_ood_performance_1x5_points.csv`
- `parking/gamma_star_organization/exports/gamma_star_collaborator_bundle/final_hinf_srob_analysis.csv`
- `parking/s_rob_performance/plots/statistics.json`
- `parking/s_rob_performance/plots/retention_points.csv`
- `parking/gamma_star_organization/plots/srob_overall_statistics.json`
- `parking/gamma_star_organization/plots/srob_parameter_size_truthfulness_main_fit.json`
- `parking/gamma_star_addition/plots/srob_overall_statistics.json`
- `parking/gamma_star_addition/plots/srob_overall_txi_improved_statistics.json`
- `parking/gamma_star_addition/truthfulness_original_vs_hinf.md`
