# Gamma Star Organization

## Purpose

This compact analysis unit collects the scalar H-infinity synthesis metadata needed to study `gamma_star` and robust steerability without copying model weights, Jacobians, gains, hidden states, prompts, generations, or judge payloads.

The paper-facing figures are intentionally narrower than the archival inventory. **Permanent unit rule:** figures include only HarmBench, Truthfulness, and MGSM. All L-CiteEval variants are excluded unless a future request explicitly opts them back in. Archival controller tables may retain L-CiteEval rows so source calculations are not destroyed.

## Final figure registry

`plots/final_report_points.csv` is the sole source for `plots/srob_ood_reliability.{pdf,png}`. Its calibration binding is explicit in `FINAL_CALIBRATIONS` inside `gamma_star_organization.py`; no run is selected heuristically.

The 34 plotted observations are:

- Truthfulness: 4 final models x Spanish OOD True percentage = 4 points.
- HarmBench: 3 final models x 5 adversarial templates = 15 points. Reliability is `100 - ASR`.
- MGSM: 3 final models x 5 reported transfer languages = 15 points.

The figure is a point inventory only. It contains no best-fit line, Spearman coefficient, sample-count annotation, ambiguity marker, or inferential claim. Color and marker identify the benchmark. Tiny horizontal offsets are applied only when points coincide exactly; the CSV retains the unmodified values.

## Compact outputs

- `plots/final_controller_grid.csv`: controller configurations only from the 12 explicitly registered final benchmark/model calibrations.
- `plots/final_selected_calibrations.csv`: exactly one selected H-infinity calibration for each of the 12 final benchmark/model pairs.
- `plots/final_report_points.csv`: explicit final-paper subset used by the scatter.
- `plots/manifest.json`: schema definitions, counts, output paths, and extraction warnings.
- `plots/srob_ood_reliability.pdf` and `.png`: vector and raster renderings of the final-only point plot.

## Variables

- `gamma_star`: minimum feasible H-infinity attenuation boundary saved by synthesis.
- `s_rob`: `1 / gamma_star`, in the saved controller coordinates.
- `ood_reliability`: benchmark-native held-out result oriented so higher is better.
- `calibration_id`: explicit calibration attempt bound to the final table row.
- `performance_source` and `selection_source`: exact provenance for the reported outcome and controller selection.

Raw `gamma_star` and `s_rob` can depend on coordinates, costs, normalization, model scale, and calibration protocol. This unit preserves those fields; it does not claim that raw values are automatically commensurate across every model and benchmark.

The output directory deliberately contains no broad exploratory CSV. Every generated CSV is safe for the final-results collaboration bundle. Regeneration also removes the deprecated broad files (`controller_grid.csv`, `selected_calibrations.csv`, `performance_long.csv`, and `analysis_candidates.csv`) if they are present.

## Reproduction

From the repository root:

```bash
python3 code/parking/gamma_star_organization/gamma_star_organization.py
```

Use `--check` to validate sources and print counts without writing outputs. The extractor uses the Python standard library for metadata extraction; the renderer uses Matplotlib with Arial typography.

To build the final-only collaborator archive after regenerating the unit:

```bash
python3 code/parking/gamma_star_organization/build_collaborator_bundle.py
```

The resulting `exports/gamma_star_collaborator_bundle.zip` contains both ID and OOD performance, the H-infinity differential/SRob analysis table, the 12 final calibration selections, the matching controller grids, a data dictionary, and a checksum manifest.

Render the controller-level structural figures with:

```bash
/home/dev/miniconda3/bin/python3.13 code/parking/gamma_star_organization/render_task_structure.py
```

- `plots/srob_parameter_size_truthfulness_main.{pdf,png}` is the main five-model Truthfulness scaling figure. Both axes are logarithmic; Qwen-2.5-32B is excluded; and no panel-title statistics are shown. Its exact source rows and log-log fit metadata are stored in matching `.csv` and `_fit.json` files.
- `plots/srob_parameter_size_truthfulness_axis_diagnostic_1x4.{pdf,png}` is the retained exploratory coordinate-system diagnostic. It shows log-log, log-linear, linear-log, and linear-linear views of the same five models, with coordinate-specific fits plus Kendall/Mann–Kendall trend statistics. Its four fit records are stored in the matching `_fit.json` file.
- `plots/srob_parameter_size.{pdf,png}` is the cross-benchmark parameter plot colored by task. Both axes are logarithmic; light dashed links identify the same model evaluated in two tasks.
- `plots/srob_task_hierarchical.{pdf,png}` is the clustering heat map, using Ward clustering on one-dimensional `log10(S_rob)` and an ordered pairwise-distance matrix.
- `plots/srob_overall.{pdf,png}` is the compact three-panel summary. Panel A shows taskwise boxplots and model logos for Truthfulness, HarmBench, and MGSM. Panel B selects a one-dimensional `log10(S_rob)` cluster count by maximum silhouette and compares the observed task-label silhouette with its exact permutation null. Panel C repeats the five-model Truthfulness log-log scaling analysis.
- `plots/srob_overall_points.csv` records the 11 controllers used by the summary. The 10 final reported task/model rows are supplemented only by the previously approved exploratory Gemma-2-2B Truthfulness row; Qwen-2.5-32B and all L-CiteEval rows remain excluded.
- `plots/srob_overall_statistics.json` records every silhouette value, the selected cluster count, the observed task-label silhouette, the exact null size, and the upper-tail permutation p-value.

## Overall-summary statistics

The clustering feature is `log10(S_rob)` only. For each candidate `k` from 2 through 6, the renderer finds the global minimum-within-cluster-SSE contiguous one-dimensional k-means partition and reports its ordinary mean silhouette. The selection rule is the `k` with the largest silhouette; no elbow is inferred by eye.

The task-separation test fixes the observed group sizes at 5 Truthfulness, 3 HarmBench, and 3 MGSM controllers. Its null hypothesis is that these task labels are exchangeable with respect to `log10(S_rob)`; its one-sided alternative is that the observed task labels have a larger silhouette than arbitrary labels with the same 5/3/3 sizes. The exact null enumerates all 9,240 labeled assignments. The plotted upper-tail p-value is the fraction whose silhouette is at least the observed value; `p < 0.05` is the descriptive significance threshold for this analysis.

## Final performance sources

- `figs/bench_table/truthfulness/truthfulqa_spanish.md`
- `figs/bench_table/harmful/harmbench_full.md`
- `figs/bench_table/mgsm/mgsm_full.md`

## Controller sources

- `benchmarks/*/cache/*/calibrations/h_infinity/**/selection.json`
- `benchmarks/*/cache/*/calibrations/h_infinity/**/grid/controllers/*.pt`
- `benchmarks/*/cache/*/calibrations/h_infinity/**/controller_diagnostics/runs/*/score.json`

# srob_ood_performance_1x5

## Method

- Load `exports/gamma_star_collaborator_bundle/final_hinf_srob_analysis.csv` and retain OOD rows with both `s_rob` and H-infinity reliability for HarmBench, Truthfulness, MGSM, and L-CiteEval.
- Panel 1 pools all retained task-native percentages; panels 2--5 separate the same observations by benchmark without aggregating conditions or models.
- Plot every axis in raw linear coordinates. In HarmBench and MGSM, fit a condition-adjusted common-slope model with a separate intercept for each attack template or transfer language; Truthfulness uses the same model with one Spanish-OOD condition. The resulting condition lines are parallel. The pooled and L-CiteEval panels retain one descriptive overall OLS line.
- Write `plots/srob_ood_performance_1x5.{pdf,png}`, its exact plotted rows in `plots/srob_ood_performance_1x5_points.csv`, and fit/test metadata in `plots/srob_ood_performance_1x5_statistics.json`.

## Variables

- Data/input: `exports/gamma_star_collaborator_bundle/final_hinf_srob_analysis.csv`.
- Records: 38 OOD observations: 15 HarmBench attack-template results, 4 Truthfulness Spanish results, 15 MGSM transfer-language results, and 4 L-CiteEval long-context results.
- X variable: robust steerability `s_rob = 1 / gamma_star`, shown without a logarithmic transform.
- Y variables: pooled benchmark-native OOD reliability in panel 1; safe-response rate for HarmBench; True percentage for Truthfulness; accuracy for MGSM; and answer recall or citation F1 for L-CiteEval.
- Outputs: `plots/srob_ood_performance_1x5.{pdf,png}`, `plots/srob_ood_performance_1x5_points.csv`, and `plots/srob_ood_performance_1x5_statistics.json`.

## Statistics

- Tests/models: HarmBench, Truthfulness, and MGSM use the raw-coordinate fixed-condition model `performance = condition intercept + beta * S_rob`. An exact two-sided model-block permutation test permutes the complete model-level `S_rob` assignments while keeping every model's condition outcomes together. The pooled and L-CiteEval panels use descriptive OLS lines without p-values.
- Null hypothesis: after accounting for condition-specific intercepts, the observed common `S_rob` slope is no larger in absolute magnitude than slopes produced by assigning the observed `S_rob` values arbitrarily to model identities.
- Alternative hypothesis: the absolute condition-adjusted common slope is unusually large relative to those model-level permutations.
- Thresholds/decision rule: `p < 0.05` is the conventional descriptive significance threshold. All model-label permutations are enumerated: 6 for each three-model HarmBench/MGSM panel and 24 for the four-model Truthfulness panel.
- What the statistic means: the fitted coefficient is the shared raw-coordinate change in performance per unit of `S_rob` after removing condition means. The p-value is the fraction of model-level permutations whose absolute common slope is at least the observed absolute slope.
- Why this statistic is appropriate here: `S_rob` varies at the model level, so permuting whole model blocks preserves the repeated attack-template or language outcomes and avoids treating them as independent controllers.

## Legends

- X axis: raw robust steerability, `S_rob`.
- Y axis: panel-specific held-out OOD percentage, with higher values better in every panel.
- Color/value: slate circles are HarmBench safe-response rates; blue squares are Truthfulness True percentages; terracotta triangles are MGSM accuracies; purple pentagons are L-CiteEval answer-recall and citation-F1 percentages.
- Grouping: panel 1 pools the four benchmarks; panels 2--5 isolate HarmBench, Truthfulness, MGSM, and L-CiteEval in that order.
- Ordering/sorting: no observations are sorted or aggregated before fitting.
- Lines/markers/labels: points are individual model-condition results. HarmBench and MGSM colors identify attack templates and transfer languages, respectively. Their parallel colored lines share the fitted condition-adjusted slope and have condition-specific intercepts. Truthfulness has one Spanish line. Pooled and L-CiteEval solid lines are descriptive overall OLS fits. Only the three hierarchical panels show exact two-sided permutation p-values.
- Panels: OOD reliability, HarmBench, Truthfulness, MGSM, and L-CiteEval.

## Interpretation

- Read each benchmark panel independently because its y axis is the task-native performance measure and its x range is adapted to the benchmark's observed `S_rob` values.
- The pooled panel reproduces the broad reliability comparison, whereas the four task panels reveal whether that pooled pattern is shared by each benchmark.

## Notes

- Rows from multiple conditions share a model-level `S_rob`; exact inference therefore permutes complete model blocks rather than individual rows.
- HarmBench and MGSM have only three independent models, so their exact permutation tests contain six possible assignments and necessarily have coarse p-value resolution.
- L-CiteEval places answer recall and citation F1 on one percentage axis, matching the pooled reliability inventory but combining two distinct task metrics.
- This explicitly requested four-benchmark figure opts L-CiteEval into this output only; it does not change the unit's canonical three-benchmark paper-facing figure.

## References

- `render_srob_figure.py` and `plots/srob_ood_reliability.{pdf,png}` for the earlier pooled scatter.
- `exports/gamma_star_collaborator_bundle/data_dictionary.md` for bundle field definitions.
