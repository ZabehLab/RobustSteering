# figure_harm

## Method

- Load the frozen HarmBench template-level results for Llama 3.2 1B, Llama 3.2 3B, and Llama 3.1 8B.
- Summarize each model–method pair across the direct request and five jailbreak templates.
- Plot mean versus worst-case attack rejection, template-level attack success profiles, and the aggregate safety frontier.
- Reserve the third slot of the active main layout for the published
  FalseReject response-classification analysis documented below.
- Write a vector PDF and matching PNG to `plots/`.

## Variables

- Data/input: `figs/benchmark_figures/cache/harmbench_refusal_results.csv`.
- Sessions/groups: model, steering method, and jailbreak template.
- Labels/targets: Original, A-LQR, and H-infinity.
- Signals/features/measures: attack success rate, attack rejection, and safe-concept relevance.
- Parameters/thresholds: rejection is `100 - ASR`; worst-case rejection uses the maximum ASR across the six conditions.
- Outputs: `plots/figure_harm.pdf` and `plots/figure_harm.png`.
  The preserved geometric-marker version remains unchanged; the iterative
  logo-based variant is written separately as `figure_harm_logos.pdf/.png`.
  The active main figure is `figure_harm_main.pdf/.png` (template profile plus
  safety frontier), while the detached rejection panel is retained as
  `figure_harm_rejection.pdf/.png`.

## Statistics

- Tests/models: descriptive aggregation only.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: lower ASR and higher rejection/relevance are better.
- What the statistic means: mean rejection captures average robustness, while worst-case rejection captures the weakest template condition.
- Why this statistic is appropriate here: the panels jointly expose average robustness, template sensitivity, and the safety–relevance tradeoff.

## Legends

- X axis: panel-specific mean rejection, jailbreak template, or mean ASR.
- Y axis: panel-specific worst-case rejection, template ASR, or mean safe-concept relevance.
- Color/value: model family scale in the middle/right panels and steering method in the left panel.
- Grouping: results are grouped by model and method.
- Ordering/sorting: jailbreak templates use the benchmark's fixed direct-plus-five-template order.
- Lines/markers/labels: color and line style encode method; circle, triangle, and square encode the 1B, 3B, and 8B Llama models; marker area increases with model scale.
- Logo variant: the official shared Llama mark replaces geometric shapes and
  is tinted by method. Log-scaled logo size distinguishes 1B, 3B, and 8B;
  H∞ uses exact project teal `#007C7C`. The top native-color model legend uses
  the same ordered size hierarchy, while the bottom method legend uses dots.
  Original uses gray with reduced opacity; A-LQR uses a distinct muted violet
  (`#6558A6`) so it remains separable without competing with the teal focal
  series.
- Panels: rejection summary, robustness profile, and safety frontier.

## Interpretation

- The figure compares whether H-infinity improves robust refusal across model scales and jailbreak conditions while retaining safe-concept relevance.
- The key comparison is H-infinity against the strongest available non-H-infinity result.

## Notes

- This is the initial consolidated layout; panel-specific visual revisions are intentionally deferred to subsequent iterations.
- `figure_harm_logos` is the active iterative variant. It uses enlarged
  MGSM-style axis, tick, template, and legend typography without overwriting
  the preserved initial figure.
- The final organization separates the former first panel from the main
  composition. In `figure_harm_main`, a blank marginal-height spacer aligns
  the jailbreak-template axis with the frontier axis, extra negative-y room
  prevents zero-ASR Llama marks from clipping, and the x-axis label is offset
  below the two-line template labels. The compact near-equal panel footprints
  leave room for neighboring manuscript panels; explicit endpoint padding and
  nonuniform categorical spacing keep Direct, the long central labels, and
  Jailbreak Bot clear of the axis boundaries without stretching the panel. The
  first two panels now occupy roughly 70% of a three-column canvas;
  the native-color Llama scale legend and method legend remain centered over
  those two panels, not over the reserved response-disposition slot.
- `figure_harm_main_expanded` is the large-type paper-composition variant. It
  preserves the canonical data and panel widths while matching the TruthfulQA
  composite's 20 pt axis labels, 18 pt ticks, and 19 pt legends. The model and
  method legends share the top row; the six fully written template labels are
  angled, Direct attack remains black as ID, and the five adversarial/OOD
  templates use the exact project dark red `#8B1E1E`. Blank marginal-height
  rows align the first and third axes with the frontier panel's main axis.
  Regenerate only this variant with
  `python figs/figure_harm/figure_harm.py --main-expanded-only`.

## References

- `figs/benchmark_figures/scripts/plot_harmbench_results.py`.
- `figs/bench_table/harmful/`.

# false_reject_analysis

## Method

- Load the completed Original, A-LQR, and H-infinity HarmBench generations for
  Llama 3.2 1B, Llama 3.2 3B, and Llama 3.1 8B without modifying benchmark
  caches.
- Draw the same prompt identities for all three methods. The completed full
  analysis classifies every available generation: 240 direct plus 1,200
  adversarial responses per method for Llama 3.2 1B/3B, and 120 direct plus
  600 adversarial responses per method for the compact Llama 3.1 8B run. The
  earlier matched 100-direct/100-adversarial pilot remains cached separately.
- Apply verbatim the three-class `Prompt for Evaluation to Compute USR` from
  Appendix G of FalseReject: Direct Refusal, Safe Partial Compliance, or Full
  Compliance. No project-specific response class or rubric criterion is added.
- Use pinned `gpt-4o-mini-2024-07-18` at temperature zero as the judge. This is
  the sole implementation difference from FalseReject, which used
  Claude-3.5-Sonnet for response classification.
- Save the complete sampling manifest, filled API request, verbatim API
  response, parsed label, token usage, judge identifiers, and summary so every
  classification can be audited and reproduced.

## Variables

- Data/input: the 18 completed generation artifacts under
  `benchmarks/harmful/cache/<model>/evaluations/kv_cache_off/generations/` for
  three models, three methods, and direct/human-jailbreak conditions.
- Sessions/groups: steering method, direct versus adversarial regime, model,
  and jailbreak template.
- Labels/targets: FalseReject's exact Direct Refusal, Safe Partial Compliance,
  and Full Compliance classes.
- Signals/features/measures: class proportions and FalseReject Useful Safety
  Rate for toxic prompts, defined as `(Direct Refusal + Safe Partial
  Compliance) / total`.
- Parameters/thresholds: 10,800 total full-analysis judgments;
  `gpt-4o-mini-2024-07-18`; temperature zero; concurrency 500. Pilot sampling
  used seed `20260924` and 100 matched prompts per method per regime.
- Outputs: ignored local artifacts under `cache/false_reject_pilot/`:
  `manifest.json`, `sample_manifest.jsonl`, `raw_api_responses.jsonl`,
  `parsed_judgments.jsonl`, `summary.json`, and `summary.csv`.
  The completed full-analysis path writes the same schema under
  `cache/false_reject_full/` and reused matching pilot judgments without
  resending them.

## Statistics

- Tests/models: descriptive FalseReject class proportions and toxic-prompt
  Useful Safety Rate. The final table reports Direct and every jailbreak
  template separately with ten-group behavior-clustered jackknife standard
  errors; deleting one group removes the same behavior identities from the
  corresponding condition for all methods. No hypothesis test is performed.
- Null hypothesis: not applicable because this analysis does not perform a
  significance test.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: Direct Refusal and Safe Partial Compliance both
  count toward toxic-prompt Useful Safety Rate; Full Compliance does not.
- What the statistic means: the three proportions distinguish blunt refusal,
  constructive safe engagement, and full fulfillment; Useful Safety Rate is
  their published safety/helpfulness summary for toxic prompts.
- Why this statistic is appropriate here: every HarmBench request in this
  analysis is harmful, and the published FalseReject metric was explicitly
  defined for separating useful safe behavior from harmful full compliance on
  toxic prompts.

## Legends

- X axis: Direct Refusal, Safe Partial Compliance, and Full Compliance.
- Y axis: regime-balanced response proportion from 0% to 100%.
- Color/value: muted purple is A-LQR; project teal `#007C7C` is H-infinity.
- Grouping: each unfilled box summarizes the three model-level values; Direct
  and Adversarial regimes are first weighted equally within each model.
- Ordering/sorting: Direct Refusal, Safe Partial Compliance, then Full
  Compliance; paired boxes use A-LQR followed by H-infinity.
- Lines/markers: each gray segment connects the same model under A-LQR and
  H-infinity. Points sit at the exact method positions with no jitter. Box
  borders, medians, whiskers, and points use the established method colors;
  the boxes have no fill.
- Panels: this compact comparison fills the third slot in `figure_harm_main`
  and is also exported independently as `figure_harm_refusal_composition`.

## Interpretation

- The analysis determines whether H-infinity's lower attack success rate reflects
  a shift toward direct refusals, constructive safe partial compliance, or
  both, using an existing published analysis rather than a custom rubric.
- The key comparison is the response-class composition of H-infinity against
  Original and A-LQR under matched prompt identities.
- In the completed full analysis, regime-balanced toxic-prompt Useful Safety
  Rate for H-infinity was 97.13% on Llama 3.2 1B, 93.63% on Llama 3.2 3B, and
  93.67% on Llama 3.1 8B. The corresponding best-baseline rates were 92.75%,
  86.21%, and 86.50%. H-infinity therefore retained the pilot advantage in all
  three model strata.

## Notes

- The FalseReject rubric and examples are stored verbatim in the script and
  copied into the run manifest with a SHA-256 digest.
- Raw API responses remain local ignored cache artifacts because they contain
  evaluated harmful prompts and full model outputs.
- The completed `--scope full` run used the same script, rubric, parser, and
  cache schema for all 10,800 available outputs. The 600-call pilot remains a
  separate diagnostic cache and is not substituted into the final statistics.
- The condition-level and summary Markdown, TeX, and one-page portrait PDF
  tables are under `figs/bench_table/harmful/harmbench_false_reject_full.*`
  and `figs/bench_table/harmful/harmbench_false_reject_summary.*`.
- All 10,800 full-analysis judgments produced exactly one parseable
  FalseReject label; the cache retains the requested and returned judge model,
  termination metadata, token usage, and verbatim response.

## References

- Faisal et al., *FalseReject: A Resource for Improving Contextual Safety and
  Mitigating Over-Refusals in LLMs via Structured Reasoning*, COLM 2025,
  Appendix G: <https://openreview.net/pdf?id=1w9Hay7tvm>.
