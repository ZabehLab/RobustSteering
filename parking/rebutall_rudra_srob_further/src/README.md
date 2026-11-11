# analyze

## Method

Use the five requested TruthfulQA models. Join each saved Srob point to the
corresponding selected H∞ controller and to its Original
and H∞ evaluations. Include English (ID) and Spanish-input (OOD) as separate rows
for each model.

Check the checkpoint, dataset, evaluation condition, KV-cache setting, controller
costs, and selected-controller diagnostic before calculating the table. Srob is
copied from the original figure and checked against the controller diagnostic and
the reciprocal of its saved numerical gamma-star boundary.

Compute T×I using the benchmark table's convention, True (%) × Informative (%) /
100, through the existing `build_truthfulness_comparison.metric` helper. Divide
the H∞ T×I by the Original T×I for the ratio. Subtract the Original T×I from the
H∞ T×I for the percentage-point difference. Calculate these quantities from the
saved metric means at full floating-point precision.

Run from the repository root with the existing analysis environment.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/analyze.py
```

## Variables

- The five models are GPT-2 XL, Gemma-2-2B, Llama-3-8B,
  Qwen-2.5-14B, and OLMo-2-32B.
- Each model has one selected Srob value, reused for its two evaluation conditions.
- `original_txi_pct` and `hinf_txi_pct` are T×I percentages.
- `hinf_over_original_txi` is a dimensionless ratio.
- `hinf_minus_original_txi_pp` is a difference in percentage points.
- `tables/truthfulqa_srob.csv` and `.json` contain the ten rows at full precision,
  parameter counts, gamma-star values, evaluation sample and repetition counts,
  calibration IDs, and source paths.
- `tables/truthfulqa_srob.md` contains the readable table. T×I and the derived
  columns are displayed to six decimals. Srob and γ★ retain the source CSV precision.
- `tables/source_manifest.json` records SHA-256 hashes for all 41 input files
  and the identities of the result and generation files.
- `archive/figures/srob_txi_changes.{png,pdf}` shows Srob versus the T×I ratio and difference
  for each evaluation condition. PNG exports use 600 DPI.

## Statistics

The behavioral panels fit ordinary least squares separately for each outcome and
evaluation condition, with Srob on the linear x-axis. Each fit has five models.
The displayed r is Pearson's correlation, and p is the two-sided slope test
against a zero-slope null. `tables/regressions.json` retains the full-precision
slopes, intercepts, correlations, p-values, and slope standard errors. The fits
describe these five observations.

## Legends

- The behavioral figure has English (ID) on the left and Spanish-input (OOD) on
  the right. The top row shows H∞ / Original T×I. The bottom row shows H∞ − Original
  T×I in percentage points.
- English points and fits use teal `#007C7C`. Spanish-input points and fits use
  dark red `#8B1E1E`. Direct labels identify the models.
- Gray horizontal reference lines show a ratio of one and a difference of zero.
  The two conditions share y-axis ranges within each row.

## Interpretation

A ratio above one or a positive percentage-point difference means that H∞ has a
higher evaluated T×I than the matching Original result. The behavioral panels
compare the size of that improvement with the original selected Srob values.

## Notes

The model set was explicitly selected by the user. Both evaluation conditions
were included to preserve the ID/OOD distinction. Each Srob value retains its
original model-specific cost specification.

Gemma uses the fixed `fixed_q0p1_qf0p31622777_r1` controller from the source figure.
Llama's legacy internal key is `llama31_8b`, while its actual TruthfulQA checkpoint
is `meta-llama/Meta-Llama-3-8B`. The displayed label follows that checkpoint.
The CSV preserves the sample and repetition counts of each source evaluation.

All calculations use existing saved results. The script writes only inside this
unit.

## References

- [Original five-model figure points](../../gamma_star_organization/plots/srob_parameter_size_truthfulness_main.csv)
- [Existing result path and T×I calculation helpers](../../gamma_star_addition/build_truthfulness_comparison.py)
- [Benchmark table T×I implementation](../../../figs/bench_table/bench_table.py)
- [Source manifest](../tables/source_manifest.json)

# analyze_truthful

## Method

Use the same five selected controllers and paired English (ID) and Spanish-input
(OOD) results as `analyze.py`. Read the saved `metrics.truth.mean` value through
the existing metric helper to obtain Truthful (%). Divide the H∞ percentage by
the Original percentage for the ratio. Subtract Original from H∞ for the
percentage-point difference. Retain each controller's saved Srob and γ★ values.

The figure uses the shared renderer from `analyze.py` for the same 2×2 layout,
typography, colors, reference lines, fits, and model labels.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/analyze_truthful.py
```

## Variables

- `original_truthful_pct` and `hinf_truthful_pct` are the saved True percentages.
- `hinf_over_original_truthful` is the dimensionless ratio of those percentages.
- `hinf_minus_original_truthful_pp` is their difference in percentage points.
- `tables/truthfulqa_srob_truthful.{csv,json,md}` contain the new ten-row table,
  including Srob, γ★, both behavioral percentages, ratio, difference, and provenance.
  CSV and JSON retain full floating-point precision. Markdown displays behavioral
  quantities to six decimals.
- `tables/truthful_source_manifest.json` records the source hashes and identities.
- `archive/figures/srob_truthful_changes.{png,pdf}` contain the 2×2 figure, with a 600-DPI PNG.

## Statistics

Each panel fits ordinary least squares between linear Srob and the displayed
behavioral quantity using five models. The figure shows Pearson's r and the
two-sided p-value for the zero-slope null. Full-precision fit results are saved
in `tables/truthful_regressions.json`.

## Legends

- English (ID) is the left column, in teal `#007C7C`. Spanish-input (OOD) is the
  right column, in dark red `#8B1E1E`.
- The top row shows H∞ / Original Truthful percentages, with a reference at one.
- The bottom row shows H∞ − Original Truthful percentages, with a reference at zero.
- Every x-axis shows Srob. Each row shares its y-axis range across conditions.
- Points carry model labels, and straight lines show the fitted relationships.

## Interpretation

A ratio above one or a positive difference means H∞ has a higher True percentage
than Original under that evaluation condition. Each fit describes the relationship
between the selected Srob values and the displayed change in True percentage.

## Notes

The same checkpoint, dataset, generation identity, and controller-cost checks
used by `analyze.py` run before these outputs are created. All values come from
existing saved evaluations.

## References

- [Shared source checks and plotting implementation](analyze.py)
- [Truthful percentage table](../tables/truthfulqa_srob_truthful.md)
- [Truthful source manifest](../tables/truthful_source_manifest.json)

# analyze_harmbench

## Method

Use the three HarmBench models reported in the paper, Llama-3.2-1B-Instruct,
Llama-3.2-3B-Instruct, and Llama-3.1-8B-Instruct. Join each original Srob point
to its selected H∞ controller and paired Original/H∞ results. Verify checkpoint,
controller costs, evaluation identities, sample counts, and matched behavior IDs.
Check the Srob value against its controller diagnostic and reciprocal γ★.
Reconstruct the evaluated behavior subset from the paper's saved jackknife groups
and verify the scored prompt identities and means against the cached judgments.

Convert the saved `harmbench_test_success.score` to ASR (%) by multiplying by 100.
The two conditions are direct requests (ID) and the aggregate of the five frozen
human jailbreak templates (OOD). Each template has the same number of behaviors,
so the saved aggregate is their equal-weight mean. Cross-check every ASR against
the full-sample means in the paper's HarmBench jackknife report.

Calculate H∞ ASR / Original ASR and H∞ ASR − Original ASR in percentage points.
Use the shared plotting implementation for the 2×2 comparison.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/analyze_harmbench.py
```

## Variables

- `original_asr_pct` and `hinf_asr_pct` are the saved ASR percentages.
- `hinf_over_original_asr` is their dimensionless ratio.
- `hinf_minus_original_asr_pp` is their percentage-point difference.
- `tables/harmbench_srob_asr.{csv,json,md}` contain six rows with Srob, γ★,
  both ASRs, ratio, difference, counts, checkpoint identity, and source paths.
  CSV and JSON retain full precision. Markdown shows behavioral quantities to
  six decimals.
- `tables/harmbench_source_manifest.json` records input hashes and result identities.
- `archive/figures/srob_harmbench_asr_changes.{png,pdf}` contain the figure, with a 600-DPI PNG.

## Statistics

Each panel fits ordinary least squares between linear Srob and the displayed
ASR ratio or difference using three models. Pearson's r and the two-sided p-value
against a zero-slope null are displayed. Full-precision fit results are saved in
`tables/harmbench_regressions.json`.

## Legends

- Direct (ID) is the left column, in teal `#007C7C`. Jailbreak (OOD) is the right
  column, in dark red `#8B1E1E`.
- The top row shows H∞ / Original ASR with a reference at one.
- The bottom row shows H∞ − Original ASR in percentage points with a reference at zero.
- The x-axis shows Srob on a linear scale. Both columns share y-axis ranges
  within each row. Model labels identify the points, and straight lines show fits.

## Interpretation

ASR is lower-is-better. A ratio below one or a negative difference indicates
reduced attack success under H∞. The plots compare the size of that ASR change
with each selected controller's Srob value.

## Notes

The 1B and 3B results use 240 direct requests and 1,200 jailbreak prompts each.
The 8B results use the paper's frozen subset of 120 direct requests and 600
jailbreak prompts. Original and H∞ use matching behavior identities within
each model. Plotted model labels omit the checkpoint's `-Instruct` suffix for
readability. Full checkpoint names are retained in the tables.

## References

- [Paper HarmBench ASR summary](../../../figs/bench_table/harmful/harmbench_summary.md)
- [Paper HarmBench jackknife report](../../../figs/bench_table/harmful/harmbench_jackknife.json)
- [Original Srob figure points](../../gamma_star_organization/plots/srob_overall_points.csv)
- [HarmBench source manifest](../tables/harmbench_source_manifest.json)

# analyze_mgsm

## Method

Use Qwen3-4B, Phi-4-mini-instruct, and Granite-3.3-2B-Instruct from the paper.
Compare H∞ with the saved A-LQR baseline on Chinese, French, Japanese, Swahili,
and Telugu. These five languages are all OOD evaluation conditions.

Verify selected-controller Srob and γ★ against the original figure points and
controller diagnostics. Match model checkpoints, controller settings, evaluated
question identities, and cache settings through the saved generation records.
Compute exact-answer accuracy from all 100 cached judgments per language, counting
unparseable answers as incorrect. Check the result means against the paper's
language and overall jackknife reports.

Average each method's accuracy equally over the five languages. Compute the
ratio of those mean accuracies and their percentage-point difference, using
A-LQR as the reference. The user selected the five-language average for this
comparison. The figure therefore has two panels side by side.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/analyze_mgsm.py
```

## Variables

- `alqr_accuracy_pct` and `hinf_accuracy_pct` are five-language mean accuracies.
- `hinf_over_alqr_accuracy` is their ratio. A zero A-LQR denominator is saved as
  JSON null and an empty CSV value, and is explicitly marked undefined in Markdown.
- `hinf_minus_alqr_accuracy_pp` is H∞ minus A-LQR in percentage points.
- `tables/mgsm_srob_accuracy_vs_alqr.{csv,json,md}` contain the three-model table,
  Srob, γ★, accuracies, ratio, difference, checkpoint identities, and source paths.
- `tables/mgsm_accuracy_by_language.{csv,json}` retain the 15 underlying language
  comparisons and the verified question counts.
- `tables/mgsm_source_manifest.json` records hashes and identities for all 99
  input files, including result, generation, judgment, and controller artifacts.
- `archive/figures/srob_mgsm_accuracy_vs_alqr.{png,pdf}` contain the figure, with a 600-DPI PNG.

## Statistics

The difference panel fits ordinary least squares to Srob and the percentage-point
accuracy improvement across three models. It displays Pearson's r and the
two-sided p-value against a zero-slope null.

Phi-4-mini has zero A-LQR accuracy in all five languages, so its accuracy ratio
is undefined. The ratio panel displays two finite model ratios and identifies
Phi's undefined value in text. Regression is performed only for panels with at
least three finite model values. `tables/mgsm_regressions.json` records the counts,
undefined models, and available fit results.

## Legends

- The left panel shows H∞ / A-LQR accuracy. The right panel shows H∞ − A-LQR
  accuracy in percentage points.
- Both x-axes show Srob. All points use the OOD dark red `#8B1E1E`, with model labels.
- Gray reference lines show a ratio of one and a difference of zero. The fitted
  line in the difference panel uses the same dark red.
- Typography, marker sizes, fit styling, and output resolution use the shared
  analysis plotting conventions.

## Interpretation

A ratio above one or a positive difference indicates greater mathematical
accuracy under H∞ than under A-LQR. The average is computed over the same five
languages separately for each method before taking the ratio or difference.

## Notes

The paper's deterministic evaluations use the same 100 MGSM question identities
in each language and method. Qwen's legacy result JSON files declare 250 samples,
while the saved generation and judgment records contain the paper's 100 evaluated
questions. Verified counts come from those records. The language table also
preserves the legacy declared counts for inspection.

## References

- [Paper MGSM summary](../../../figs/bench_table/mgsm/mgsm_overall.md)
- [Paper MGSM jackknife report](../../../figs/bench_table/mgsm/mgsm_jackknife.json)
- [MGSM comparison table](../tables/mgsm_srob_accuracy_vs_alqr.md)
- [MGSM source manifest](../tables/mgsm_source_manifest.json)

# build_rebuttal_excel

## Method

Combine the saved TruthfulQA True-percentage and HarmBench ASR tables into
`tables/srob_truthful_harm_rebuttal.xlsx`. Split the eight model–benchmark rows
into separate ID and OOD sheets. Each ratio cell divides H∞ by Original and
contains the saved value for immediate display in Excel.

## Variables

Both comparison sheets contain benchmark, model, metric, Srob, γ★, Original (%),
H∞ (%), and H∞ / Original. The Sources sheet preserves the source CSV names and
all original numeric strings, including their full saved precision.

## Statistics

The workbook organizes existing values and calculates the displayed ratios by
division. It adds no statistical estimates.

## Legends

ID uses teal, and OOD uses dark red. Metric labels indicate that True percentage
is higher-is-better and ASR is lower-is-better. Headers have filters, identifying
columns remain frozen while scrolling, and numeric cells support Excel calculations.

## Interpretation

A TruthfulQA ratio above one indicates a higher True percentage. A HarmBench
ratio below one indicates a lower attack success rate.

## Notes

The export requires XlsxWriter. Run `python src/build_rebuttal_excel.py` from this
unit after installing that dependency. Display formats use ten decimal places
for Srob and γ★, four for percentages, and six for ratios. Exact source strings
remain available on the Sources sheet.

## References

- [TruthfulQA input](../tables/truthfulqa_srob_truthful.csv)
- [HarmBench input](../tables/harmbench_srob_asr.csv)

# plot_truthful_harm_ratios

## Method

Plot the saved H∞ / Original ratios against Srob in two panels, TruthfulQA on the
left and HarmBench on the right. Overlay ID and OOD points within each benchmark.
Use the full-precision saved values and fit a separate ordinary least-squares
line for each benchmark and evaluation split.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/plot_truthful_harm_ratios.py
```

## Variables

TruthfulQA uses True percentages for five paper models. HarmBench uses ASR for
three paper models. Each point is one model under ID or OOD evaluation.
Outputs are `archive/figures/srob_truthful_harm_ratios.{png,pdf}` and
`tables/truthful_harm_ratio_slopes.json`.

## Statistics

Each fitted line estimates the linear relationship between Srob and H∞ / Original.
Only slopes are annotated in the figure. Full-precision slopes, intercepts,
model counts, and input hashes are retained in the JSON output.

## Legends

ID points and lines use blue `#2463EB`. OOD points and lines use dark red
`#8B1E1E`. The common legend is centered below the panels. Each benchmark has
its own linear x-axis and y-axis ranges. PNG exports use 600 DPI.

## Interpretation

For TruthfulQA, higher H∞ / Original means a greater relative True percentage.
For HarmBench, lower H∞ / Original means a lower relative attack success rate.

## Notes

The figure contains four independent fits, each using the paper-reported model
set for its benchmark and split.

## References

- [TruthfulQA values](../tables/truthfulqa_srob_truthful.json)
- [HarmBench values](../tables/harmbench_srob_asr.json)

# plot_behavior_gain_variants

## Method

Create two additional versions of the combined TruthfulQA/HarmBench plot.
Behavior gain is `(method − Original) / Original`, using H∞ in one version and
the paper's A-LQR baseline in the other. Values are dimensionless fractions.
TruthfulQA uses True percentages, and HarmBench uses ASR percentages.

Read the raw Original, H∞, and A-LQR result files matched to each saved table row.
Check model and evaluation identities, finite values, positive denominators, and
agreement with the existing tables. A-LQR results use the same result namespace
as the Original baseline for each model and condition.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/plot_behavior_gain_variants.py
```

## Variables

- `archive/figures/srob_truthful_harm_hinf_behavior_gain.{png,pdf}` use H∞ behavior gain.
- `archive/figures/srob_truthful_harm_alqr_behavior_gain.{png,pdf}` use A-LQR behavior gain.
- `tables/truthful_harm_behavior_gain.json` contains the 16 model/condition rows,
  raw percentages, both gains, exact source paths and hashes, and fitted slopes.
- The original ratio figure remains available as `archive/figures/srob_truthful_harm_ratios.{png,pdf}`.

## Statistics

Fit a separate ordinary least-squares line for each benchmark and ID/OOD split.
Each TruthfulQA fit has five models, and each HarmBench fit has three. Only slopes
are annotated. Subtracting one from the H∞ / Original ratio preserves its slope
and decreases the fitted intercept by one.

## Legends

Both variants use the same 1×2 layout and blue ID/red OOD styling as the ratio
figure. Both y-axes are labeled Behavior gain. The H∞ figure uses the original
ratio figure's y-ranges shifted down by one. The A-LQR figure's ranges contain
its own gain values. PNG exports use 600 DPI.

## Interpretation

Positive TruthfulQA gain indicates a higher True percentage than Original.
Negative HarmBench gain indicates reduced attack success relative to Original.

## Notes

The x-axis retains each model's Srob value from its selected H∞ specification
for both controller comparisons. The saved paper A-LQR evaluation supplies the
additional controller's behavioral values.

## References

- [Ratio plotting implementation](plot_truthful_harm_ratios.py)
- [Behavior gain values and sources](../tables/truthful_harm_behavior_gain.json)

# plot_behavior_advantage

## Method

Create a fourth variant using `(H∞ − A-LQR) / A-LQR`, labeled Behavior advantage
against competitor. Reuse the matched result checks from
`plot_behavior_gain_variants.py` and the shared 1×2 plotting implementation.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/plot_behavior_advantage.py
```

## Variables

TruthfulQA uses True percentages for five paper models, and HarmBench uses ASR
percentages for three paper models. `behavior_advantage` is the dimensionless
relative change from A-LQR to H∞. All 16 rows have positive A-LQR denominators.
Outputs are `archive/figures/srob_truthful_harm_behavior_advantage.{png,pdf}` and
`tables/truthful_harm_behavior_advantage.json`.

## Statistics

Fit separate ordinary least-squares lines for each benchmark and ID/OOD split.
Only slopes are annotated. Exact slopes, intercepts, point values, source paths,
and input hashes are saved in the JSON output.

## Legends

TruthfulQA is on the left, and HarmBench is on the right. The x-axis is Srob,
using the same selected-controller values as the other variants. ID is blue,
and OOD is dark red. Both y-axes read Behavior advantage against competitor.
PNG exports use 600 DPI.

## Interpretation

Positive TruthfulQA advantage indicates a higher True percentage under H∞ than
under A-LQR. Negative HarmBench advantage indicates lower attack success under
H∞ than under A-LQR.

## Notes

The comparison uses the saved paper A-LQR baseline. Values follow the requested
formula directly on each benchmark's native behavioral metric.

## References

- [Matched behavioral result loader](plot_behavior_gain_variants.py)
- [Behavior advantage values and sources](../tables/truthful_harm_behavior_advantage.json)

# report_behavior_gain

## Method

Read the saved Original and H∞ evaluations for the five TruthfulQA models and
three HarmBench models. For each cached behavioral measure, calculate
`(H∞ − Original) / Original` and regress that relative gain on the model's Srob.
The active figure contains T, T×I, and ASR panels, with ID and OOD fitted separately.
T×I uses the benchmark convention of the product of aggregate True and Informative
percentages divided by 100.

For the three MGSM models, first average each behavioral measure over Chinese,
French, Japanese, Swahili, and Telugu separately for Original and H∞. Then calculate
the relative gain. All five languages are OOD, so the ID column is blank.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rebutall_rudra_srob_further/src/report_behavior_gain.py
```

## Variables

- The active figure files are `../hinf_behavior_gain.png` and `.pdf`.
- The unit's root `../README.md` is the slope report with three benchmark tables.
- `../tables/behavior_signal_points.{csv,json}` contain 118 model/split/measure
  records with Original values, H∞ values, relative gains, and availability status.
- `../tables/behavior_signal_slopes.{csv,json}` contain the 36 benchmark/measure/split
  summaries, with model counts, slopes, and intercepts.
- `../tables/behavior_signal_sources.json` records hashes for the 65 input files.

## Statistics

Use ordinary least squares between linear Srob and relative behavioral gain.
Only slopes are displayed. A descriptive slope requires at least two finite
model pairs with a positive Original denominator. The report shows the model
count whenever it is smaller than the benchmark's full reported model set.
Rows are ordered by the largest absolute available ID/OOD slope within each
benchmark. The ranking retains the native sign and definition of every measure.

## Legends

The panels are TruthfulQA T, TruthfulQA T×I, and HarmBench ASR. Blue `#2463EB`
identifies ID, and dark red `#8B1E1E` identifies OOD. Slope labels sit along the
corresponding lines and rotate with their displayed direction. Figure fonts and
marker sizes follow the other unit plots, and the PNG uses 600 DPI.

## Interpretation

Every slope describes how the relative H∞ change varies with Srob for the
available models. T and T×I use different behavioral outcomes. ASR remains a
lower-is-better measure, so negative ASR gain indicates reduced attack success.

## Notes

The earlier H∞ gain figure used T. The active figure includes both T and T×I
explicitly. Old figures are retained under `../archive/figures`, and the existing
plotting entry points now write their historical variants there.

TruthfulQA concept relevance has three paired models, and AXBench overall has
two finite model pairs. Missing and non-finite cached values are excluded.
MGSM Spanish relevance and AXBench overall have zero Original denominators and
are marked undefined. No substitute denominator is introduced.

## References

- [Active slope report](../README.md)
- [All signal values](../tables/behavior_signal_points.csv)
- [Slope values and model counts](../tables/behavior_signal_slopes.csv)
- [Input hashes](../tables/behavior_signal_sources.json)
