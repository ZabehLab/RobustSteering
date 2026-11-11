# plot_calibration_trends

## Method

Expand the five models from the original TruthfulQA size/Srob figure into their
complete cached controller grids. Each model contributes 32 feasible, converged
controllers from eight Q/R values and four Qf/R values, giving 160 observations.
Read gamma-star and cost parameters directly from the saved controller metadata
and compute Srob as 1/gamma-star. Check controller identity, cost parameters,
feasibility, convergence, and the saved Srob diagnostic.

Match settings across models using the numerical `(Q/R, Qf/R)` pair. Each model
must contain the same complete set of 32 unique pairs. Each colored curve then
connects one setting across the five model sizes, in increasing size order.
Black points show all recorded observations at their actual model-size positions.

The original size figure used Gemma's fixed single-setting run. This analysis
uses its separately cached `txi_fluency_95_05` full grid. The other four models
use the same calibration directories as the source figure.

```bash
/home/dev/miniconda3/envs/robust-steerability/bin/python \
  parking/rudra_rebuttal_trend_of_srob/plot_calibration_trends.py
```

## Variables

- The five models are GPT-2 XL, Gemma-2-2B, Llama-3-8B, Qwen-2.5-14B, and OLMo-2-32B.
- Q/R has eight values from 0.01 to 31.622776601683793. Qf/R has four values from
  0.01 to 0.31622776601683794.
- `tables/calibration_points.{csv,json}` contain all 160 observations with model
  identities, exact sizes, costs, lambda, calibration IDs, gamma-star, Srob,
  color ranks, selected-setting flags, and controller source paths.
- `tables/matched_cost_traces.csv` contains one row per cost pair, its color,
  all five model values, and its descriptive log–log slope and intercept.
- `tables/summary.json` records model/calibration choices, counts, slope ranges,
  and SHA-256 hashes for all 171 source files.
- `plots/truthfulqa_srob_calibration_trends.{png,pdf}` contain the figure. The PNG
  is exported at 600 DPI.

## Statistics

The plotted curves use shape-preserving PCHIP interpolation through the five
recorded values in log10(size)/log10(Srob) coordinates. The interpolation preserves
the measured model values and the direction of each adjacent segment.

For the numerical trace table, fit ordinary least squares between log10(model
size in billions) and log10(Srob) separately for each of the 32 matched settings.
Each fit has five model observations. All 32 overall slopes are positive, ranging
from 0.3297931123592286 to 0.46168625202010855. The plotted connections follow
the model values, while the table's slopes describe their overall size association.

## Legends

- The x-axis shows actual model size on a logarithmic scale, labeled with model
  names and parameter counts in billions. The y-axis shows Srob on a logarithmic scale.
- Each black point is one saved controller's Srob value. Some nearby values overlap.
- Each colored curve follows the same numerical Q/R and Qf/R pair across all models.
- The right-hand 8×4 color key has Q/R on its rows and Qf/R on its columns.
  Its numbers identify setting ranks in the trace table.
- Rank the pairs by `(q_index + qf_index, q_index, qf_index)`, using ascending
  indices of the two cost ratios. Assign 32 equally spaced colors along viridis.
  Color progression is monotonic in either cost ratio with the other held fixed.
  The ordering depends only on cost settings.

## Interpretation

Following one curve compares model Srob values while holding the two cost ratios
fixed. The vertical point collections show each model's sensitivity to its full
cost grid. The shared rise at Llama, dip at Qwen, and rise at OLMo are visible
across the matched settings.

## Notes

Every model retains its existing identification and disturbance estimates.
Lambda and tuning-sample counts remain model-specific and are recorded alongside
the observations. Llama's legacy internal key is `llama31_8b`, while the actual
TruthfulQA checkpoint is `meta-llama/Meta-Llama-3-8B`.

The script reads cached controller metadata through the existing metadata-only
reader. It writes its tables and figures inside this unit.

## References

- [Original five-model source figure](../gamma_star_organization/plots/srob_parameter_size_truthfulness_main.csv)
- [Existing controller metadata reader](../gamma_star_organization/gamma_star_organization.py)
- [All calibration observations](tables/calibration_points.csv)
- [Matched cost traces](tables/matched_cost_traces.csv)
- [Source manifest and summary](tables/summary.json)
