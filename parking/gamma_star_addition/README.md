# Gamma Star Addition

This isolated analysis unit extends the frozen `gamma_star_organization` S-rob
overall figure with the nine additional sub-billion-parameter TruthfulQA runs.
The original unit and its generated files are not modified. Copies of its
published S-rob overall outputs are retained under `ref/` for direct comparison.

## Figure contents

- The task-distribution panel contains 14 Truthfulness, 3 HarmBench, and 3 MGSM
  controllers. Individual observations are shown as dots without model labels
  or logos so that the enlarged Truthfulness group remains legible.
- The middle panels retain the one-dimensional clustering and exact labeled
  task-assignment null analysis. The null uses the actual 14/3/3 group sizes.
- The Truthfulness scaling panel shows all 14 models from 14M to 32B parameters.
  Every observation is labeled and marked with its model-family organization
  logo.

The nine additions are Pythia-14M, Pythia-31M, DistilGPT-2, GPT-2 Small,
SmolLM2-135M, Pythia-160M, GPT-2 Medium, Qwen-2.5-0.5B, and GPT-2 Large. Their
analysis-only points use the lambda-first rerun isolated under calibration ID
`lambda_txi_n200_r1`; this unit does not register them as canonical benchmark
models or modify any paper-facing benchmark table.

## Inputs and outputs

- Static calibration selection input:
  `exports/gamma_star_collaborator_bundle/final_selected_calibrations.csv`
- Frozen original figure references: `ref/srob_overall_original.*`
- Reproducible outputs: `plots/srob_overall.pdf`, `plots/srob_overall.png`,
  `plots/srob_overall_points.csv`, and `plots/srob_overall_statistics.json`
- T×I-filtered variant: `plots/srob_overall_txi_improved.{pdf,png}`. It retains
  all five pre-existing Truthfulness models and filters only the nine new model
  additions, keeping a new model exactly when evaluated H∞ mean T×I is greater
  than evaluated Original mean T×I. The decision record is stored in
  `plots/truthfulness_txi_improvement_filter.json`; matching point and statistic
  files use the `srob_overall_txi_improved_` prefix.
- Analysis-only steering comparison: `truthfulness_original_vs_hinf.md`, with
  exact source provenance in `truthfulness_result_sources.json`. This table is
  deliberately not connected to `figs/bench_table/`.

Run from the Code repository root:

```bash
/home/dev/miniconda3/bin/python3.13 \
  parking/gamma_star_addition/render_task_structure.py

/home/dev/miniconda3/bin/python3.13 \
  parking/gamma_star_addition/build_truthfulness_comparison.py
```

## Logo provenance

- OpenAI, Qwen, Meta, Google, and AI2 assets are copied into this self-contained
  unit from the repository's established shared logo library.
- Hugging Face assets come from the official Hugging Face brand asset release:
  <https://huggingface.co/brand>.
- The EleutherAI mark comes from the official EleutherAI GitHub organization
  avatar; `eleutherai_transparent.png` is a transparent, plot-ready derivative
  of that asset: <https://github.com/EleutherAI.png>. The vector reference is
  retained as `eleutherai_logo.svg`.

All figure typography is Arial. H-infinity highlights use the repository teal
`#007C7C`.
