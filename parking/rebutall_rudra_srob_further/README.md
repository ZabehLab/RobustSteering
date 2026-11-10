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
  parking/rebutall_rudra_srob_further/analyze.py
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
- `plots/srob_txi_changes.{png,pdf}` shows Srob versus the T×I ratio and difference
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

- [Original five-model figure points](../gamma_star_organization/plots/srob_parameter_size_truthfulness_main.csv)
- [Existing result path and T×I calculation helpers](../gamma_star_addition/build_truthfulness_comparison.py)
- [Benchmark table T×I implementation](../../figs/bench_table/bench_table.py)
- [Source manifest](tables/source_manifest.json)

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
  parking/rebutall_rudra_srob_further/analyze_truthful.py
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
- `plots/srob_truthful_changes.{png,pdf}` contain the 2×2 figure, with a 600-DPI PNG.

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
- [Truthful percentage table](tables/truthfulqa_srob_truthful.md)
- [Truthful source manifest](tables/truthful_source_manifest.json)
