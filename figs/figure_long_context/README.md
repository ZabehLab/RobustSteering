# Qwen2.5-3B-Instruct context-length transfer

## Method

- Read the finalized L-CiteEval full-results table.
- Plot the Qwen2.5-3B-Instruct results at matched 8K and long-context 16K conditions.
- Render answer recall and citation F1 as separate panels with one trajectory per method.
- Write a vector PDF and matching PNG to `plots/`.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`.
- Sessions/groups: context length and steering method for Qwen2.5-3B-Instruct.
- Labels/targets: Original, S-PID, A-LQR, and H∞.
- Signals/features/measures: answer recall (%) and citation F1 (%).
- Parameters/thresholds: 8K is the matched condition; 16K is the long-context condition.
- Outputs: `plots/figure_long_context_qwen2_5_3b.pdf` and `plots/figure_long_context_qwen2_5_3b.png`.

## Statistics

- Tests/models: descriptive visualization of the finalized full-sample means.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher answer recall and higher citation F1 are better.
- What the statistic means: answer recall measures recovered gold-answer content; citation F1 balances citation precision and recall.
- Why this statistic is appropriate here: the two measures expose answer retention and citation quality under context-length shift without conflating them.

## Legends

- X axis: 8K matched and 16K long-context conditions.
- Y axis: answer recall (%) or citation F1 (%), depending on panel.
- Color/value: method identity; H∞ uses the exact project teal `#007C7C`.
- Grouping: one line per method.
- Ordering/sorting: 8K precedes 16K.
- Lines/markers/labels: each method has a stable color, line style, and marker; the official Qwen logo identifies the model.
- Panels: answer recall and citation F1.

## Interpretation

- The figure shows which controllers retain answer content and citation quality when the same questions move from 8K to 16K context.
- The key comparison is H∞ against Original, S-PID, and A-LQR at each context length.

## Notes

- Error bars are intentionally omitted because the requested figure compares the finalized means; uncertainty remains documented in the source table.
- Typography follows the repository-wide Arial standard.

## References

- `figs/bench_table/lciteeval/lciteeval_full.md`.
- `figs/logos/qwen_transparent.png`.

# Llama-3.2-1B-Instruct context-length transfer

## Method

- Read the finalized L-CiteEval full-results table.
- Plot the Llama-3.2-1B-Instruct results at matched 8K and long-context 16K conditions.
- Render answer recall and citation F1 as separate panels with one trajectory per method.
- Write a vector PDF and matching PNG to `plots/`.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`.
- Sessions/groups: context length and steering method for Llama-3.2-1B-Instruct.
- Labels/targets: Original, A-LQR, and H∞.
- Signals/features/measures: answer recall (%) and citation F1 (%).
- Parameters/thresholds: 8K is the matched condition; 16K is the long-context condition.
- Outputs: `plots/figure_long_context_llama3_2_1b.pdf` and `plots/figure_long_context_llama3_2_1b.png`.

## Statistics

- Tests/models: descriptive visualization of the finalized full-sample means.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher answer recall and higher citation F1 are better.
- What the statistic means: answer recall measures recovered gold-answer content; citation F1 balances citation precision and recall.
- Why this statistic is appropriate here: the two measures expose answer retention and citation quality under context-length shift without conflating them.

## Legends

- X axis: 8K matched and 16K long-context conditions.
- Y axis: answer recall (%) or citation F1 (%), depending on panel.
- Color/value: method identity; H∞ uses the exact project teal `#007C7C`.
- Grouping: one line per method.
- Ordering/sorting: 8K precedes 16K.
- Lines/markers/labels: each method has a stable color, line style, and marker; the official Meta/Llama logo identifies the model.
- Panels: answer recall and citation F1.

## Interpretation

- The figure shows which controllers retain answer content and citation quality when the same questions move from 8K to 16K context.
- The key comparison is H∞ against Original and A-LQR at each context length.

## Notes

- S-PID is absent because the finalized Llama-3.2-1B table contains no S-PID row.
- Error bars are intentionally omitted because the requested figure compares the finalized means; uncertainty remains documented in the source table.
- Typography follows the repository-wide Arial standard.

## References

- `figs/bench_table/lciteeval/lciteeval_full.md`.
- `figs/logos/llama_transparent.png`.

# Qwen long-context and residual-interpretation composite

## Method

- Replot the finalized Qwen2.5-3B-Instruct answer-recall and citation-F1 trajectories for matched 8K and long-context 16K evaluation.
- Replot the GPT-2 and intracranial-neural context-length analyses from their cached median and interquartile-range summaries.
- Place the supplied LLM and biological-network illustrations between the corresponding quantitative panels without changing either asset's aspect ratio.
- Write a vector PDF and a 600-DPI review PNG to `plots/`.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`; the GPT-2 and neural Panel-E CSVs under `figs/figure_brain/ref/original_notebook_rerun/results/`.
- Sessions/groups: Qwen context condition and steering method; residual prediction horizon and context length $k\in\{1,2,4,8\}$.
- Labels/targets: Original, S-PID, A-LQR, and H∞ for L-CiteEval; GPT-2 hidden-state and neural residual RMS for residual interpretation.
- Signals/features/measures: answer recall (%), citation F1 (%), and median residual RMS with interquartile range in z units.
- Parameters/thresholds: Qwen uses matched 8K and shifted 16K contexts; GPT-2 uses 1, 2, 4, and 8-token horizons; neural data use 20–700 ms horizons.
- Outputs: `plots/figure_l_side_composite.pdf` and `plots/figure_l_side_composite.png`.

## Statistics

- Tests/models: descriptive visualization of finalized means and cached median/IQR summaries; no new fitted model or inferential test.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher answer recall and citation F1 are better; lower residual RMS indicates more predictable state dynamics.
- What the statistic means: residual RMS measures the typical magnitude of the local-linear prediction error, while the shaded band shows its interquartile range.
- Why this statistic is appropriate here: it places long-context task transfer beside the previously established finite-horizon predictability analysis without recomputing either experiment.

## Legends

- X axis: Qwen context condition or prediction horizon, depending on panel.
- Y axis: answer recall, citation F1, or residual RMS, depending on panel.
- Color/value: method identity in the Qwen panels; ordered grayscale encodes context length in the residual panels; the dotted black line is signal SD.
- Grouping: one line per steering method or context length.
- Ordering/sorting: matched precedes long-context; horizons increase left to right; grayscale darkens with longer context.
- Lines/markers/labels: H∞ uses project teal `#007C7C`; the two supplied network illustrations are displayed at their native aspect ratios.
- Panels: Qwen long-context evaluation, LLM residual interpretation, and biological residual interpretation.

## Interpretation

- The composite juxtaposes Qwen's long-context task transfer with the finite-horizon residual structure measured in GPT-2 and biological neural dynamics.
- It is a visual comparison of parallel empirical patterns, not a claim that the three systems or timescales are identical.

## Notes

- The original Qwen- and Llama-specific outputs remain unchanged.
- The source notebooks computed signal-SD reference curves but did not export them in their median/IQR CSVs; the composite retains the reference values shown in the original rendered outputs.

## References

- `figs/figure_brain/ref/original_notebook_rerun/source/final_figure_gpt2.ipynb`.
- `figs/figure_brain/ref/original_notebook_rerun/source/final_figure_brain.ipynb`.
- `assets/llm_network.png` and `assets/biological_network.png`.

# Qwen2.5-3B-Instruct logo-marker variant

## Method

- Reuse the finalized Qwen context-transfer values and panel geometry from the primary figure.
- Replace generic geometric point markers with the Qwen logo tinted by method.
- Preserve the primary marker figure as a separate output.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`.
- Sessions/groups: context length and steering method for Qwen2.5-3B-Instruct.
- Labels/targets: Original, S-PID, A-LQR, and H∞.
- Signals/features/measures: answer recall (%) and citation F1 (%).
- Parameters/thresholds: 8K is matched; 16K is long-context.
- Outputs: `plots/figure_long_context_qwen2_5_3b_logos.pdf` and `.png`.

## Statistics

- Tests/models: descriptive visualization of finalized means; no inferential test.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher is better for both measures.
- What the statistic means: the panels report answer-content recovery and citation support quality.
- Why this statistic is appropriate here: the visual change affects encoding only, not the reported statistics.

## Legends

- X axis: 8K matched and 16K long-context.
- Y axis: answer recall (%) or citation F1 (%).
- Color/value: method identity; H∞ uses saturated teal.
- Grouping: one trajectory per method.
- Ordering/sorting: 8K precedes 16K.
- Lines/markers/labels: tinted Qwen logos replace geometric markers.
- Panels: answer recall and citation F1.

## Interpretation

- Read identically to the primary Qwen figure; this is an alternate visual encoding.

## Notes

- The primary geometric-marker output is retained unchanged.

## References

- `figs/bench_table/lciteeval/lciteeval_full.md`.
- `figs/logos/qwen_transparent.png`.

# Llama-3.2-1B-Instruct logo-marker variant

## Method

- Reuse the finalized Llama context-transfer values and panel geometry from the primary figure.
- Replace generic geometric point markers with the Meta/Llama logo tinted by method.
- Preserve the primary marker figure as a separate output.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`.
- Sessions/groups: context length and steering method for Llama-3.2-1B-Instruct.
- Labels/targets: Original, A-LQR, and H∞.
- Signals/features/measures: answer recall (%) and citation F1 (%).
- Parameters/thresholds: 8K is matched; 16K is long-context.
- Outputs: `plots/figure_long_context_llama3_2_1b_logos.pdf` and `.png`.

## Statistics

- Tests/models: descriptive visualization of finalized means; no inferential test.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher is better for both measures.
- What the statistic means: the panels report answer-content recovery and citation support quality.
- Why this statistic is appropriate here: the visual change affects encoding only, not the reported statistics.

## Legends

- X axis: 8K matched and 16K long-context.
- Y axis: answer recall (%) or citation F1 (%).
- Color/value: method identity; H∞ uses saturated teal.
- Grouping: one trajectory per method.
- Ordering/sorting: 8K precedes 16K.
- Lines/markers/labels: tinted Meta/Llama logos replace geometric markers.
- Panels: answer recall and citation F1.

## Interpretation

- Read identically to the primary Llama figure; this is an alternate visual encoding.

## Notes

- The primary geometric-marker output is retained unchanged.

## References

- `figs/bench_table/lciteeval/lciteeval_full.md`.
- `figs/logos/llama_transparent.png`.
