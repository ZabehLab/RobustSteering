# s_rob_performance

## Method

- Re-analyze only existing final H-infinity results and selected-controller artifacts; no model generation, scoring, fitting, or controller synthesis is run.
- Join `final_performance_id_ood.csv` to `final_selected_calibrations.csv` by benchmark and model.
- Compute model-level retention as Spanish/English True-times-Informative for TruthfulQA, mean jailbreak/direct safe-response rate for HarmBench, and 16K/8K answer recall for L-CiteEval.
- Because MGSM has no English/ID H-infinity evaluation, substitute the equal-language mean of H-infinity OOD accuracy divided by Original OOD accuracy. This is accuracy preservation relative to the unsteered model, not strict ID-to-OOD retention.
- Read `max_norm_K` from each already-saved selected `controller.pt` and define the available intervention-magnitude baseline as `1 / max_norm_K`.
- Write the four-panel figure, exact point table, and full statistics/availability record under `plots/`.

## Variables

- Data/input: `parking/gamma_star_organization/exports/gamma_star_collaborator_bundle/final_performance_id_ood.csv` and `final_selected_calibrations.csv`, plus the selected controller files referenced by the calibration table.
- Sessions/groups: 4 TruthfulQA models, 3 HarmBench models, 3 MGSM models, and 2 L-CiteEval models.
- Labels/targets: task-specific retention, always oriented so higher is better.
- Signals/features/measures: `S_rob = 1 / gamma_star`, retention, ID performance, and `1 / max_norm_K`.
- Parameters/thresholds: 2,000 paired bootstrap resamples per task; seed 20260926; exact enumeration of all model-label permutations; `p < 0.05` as the descriptive decision threshold.
- Outputs: `plots/s_rob_performance.{pdf,png}`, `plots/retention_points.csv`, and `plots/statistics.json`.

## Statistics

- Tests/models: within-task Spearman correlation between each predictor and retention; exact two-sided permutation p-values; 2,000-sample model bootstrap confidence intervals; leave-one-model-out minimum Spearman correlation; descriptive OLS lines of retention on `log10(S_rob)` for tasks with at least three models.
- Null hypothesis: within a task, assigning the observed retention values arbitrarily to the observed model-level predictor values produces an absolute Spearman correlation at least as large as the observed value.
- Alternative hypothesis: the observed absolute Spearman correlation is unusually large relative to all model-label permutations.
- Thresholds/decision rule: exact `p < 0.05`; no task reaches this threshold. The supplied plan's stronger decision rule concerns E1 and cannot be applied because E1 is unavailable.
- What the statistic means: Spearman rho measures within-task rank agreement; its exact p-value respects the model as the independent unit. Bootstrap intervals describe sensitivity to the tiny observed model sample but are discrete and unstable at `n = 2–4`.
- Why this statistic is appropriate here: cost weights differ across tasks, so correlations are computed separately within task and raw task levels are never pooled into one correlation.

## Legends

- Panel A x axis: selected-controller `S_rob` on a log scale. Y axis: task-specific retention percentage. The dashed 100% line marks no loss relative to the denominator.
- Panel B: per-task Spearman rho, bootstrap 95% interval, independent model count, and exact permutation p-value.
- Panel C: per-task Spearman rho for `S_rob`, ID performance, and `1 / max_norm_K`; residual norm and probe accuracy are explicitly marked unavailable.
- Panel D: across-task frequency with which each available predictor's highest-ranked model is the best-retention model (Top-1) or contains it among the two highest-ranked models (Top-2).
- Color/value: blue TruthfulQA, slate HarmBench, terracotta MGSM, and purple L-CiteEval; teal emphasizes `S_rob` in the ranking panel.
- Grouping: every point is one selected model-task controller; MGSM aggregates five transfer languages within model.
- Ordering/sorting: tasks use the fixed order TruthfulQA, HarmBench, MGSM, L-CiteEval; predictors use `S_rob`, ID performance, and inverse gain norm.
- Lines/markers/labels: task-specific markers identify models in Panel A; lines are descriptive log-x fits, not pooled inference.
- Panels: existing selected-controller retention, within-task association, available predictor comparison, and a model-ranking substitute for the unavailable layer-selection use case.

## Interpretation

- The available selected-controller data do not establish `S_rob` as a statistically supported predictor of retention: exact per-task p-values are coarse and exceed 0.05, and estimated directions differ across tasks.
- The result is an E2 re-analysis, not the requested E1 intervention sweep. It therefore cannot apply the plan's preregistered E1 wording decision.
- Panel D evaluates model ranking only; it does not answer whether `S_rob` selects a controlled layer.

## Notes

- E1 is unavailable because no layer × `rho_rob` × calibration-seed grid has matched frozen OOD evaluations.
- E3 is partial: ID performance and inverse maximum gain norm are available; residual norm and linear-probe accuracy are not cached for the matched selected controllers.
- E4 is unavailable at layer level; the figure substitutes an explicitly labeled model-ranking diagnostic.
- E5 and E6 require recomputation or new evaluation and were not attempted.
- A random-intercept mixed model was not fit: with only 2–4 points per task and nearly one observation per model-task tuple, the requested model random effect is not identifiable enough to support the planned claim.

## References

- User-supplied `Srob_validation_plan.pdf`, prepared 26 Sep 2026.
- User-supplied `appendix_fig_mockup.py`, used only as a layout reference.
- `parking/gamma_star_organization` for the synchronized final controller and performance inventories.
