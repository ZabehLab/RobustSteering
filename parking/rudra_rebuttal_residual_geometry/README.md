# HarmBench residual geometry

Compare empirical, isotropic and randomly rotated disturbance covariances with the H∞ objective fixed. All experiment-specific logic lives here; the package and source Mega/Giga artifacts are read only.

The source is `parking/rudra_rebuttal_mached_calib/cache/qwen25_05b_instruct`: Mega's frozen Qwen2.5-0.5B-Instruct calibration and Giga's held-out responses and official HarmBench judgments. Costs remain Q/R = 0.01 and Qf/R = 0.01, with the original shared 8D states, dynamics and setpoints. Each geometry uses the original synthesis procedure with a 1% deployment margin above its own γ★. There is no recalibration or grid search.

| Geometry | Construction at each layer |
| --- | --- |
| Empirical | Original covariance and original disturbance factor |
| Isotropic | trace(C) / 8 times the identity |
| Randomly rotated | U C Uᵀ with one seeded orthogonal rotation shared across layers |

All active geometries have the same per-layer trace: the expected squared norm of centered residuals. Rotation preserves eigenvalues while changing orientation. The rotated arm uses one predefined seed in this pilot, not a selected best/worst rotation.

## Sample and caches

The covariance-alignment analysis retains its original 50 held-out behaviors. Performance generation and official HarmBench ASR scoring are complete for 200 matched held-out behaviors, each with its direct prompt and exactly one jailbreak counterpart (40 per each of five templates). The original 50-case prefix is unchanged. No held-out behavior overlaps calibration.

**Current display:** `geometry_performance_2x2` and `geometry_performance.csv` still show the 50-case pilot, with diagonal omitted. The 200-case official ASR results are in `cache/qwen25_05b_instruct/asr_performance_n200.json`. Additional fluency/AXBench grading was stopped at the user's request; partial assistant annotations were not applied. The pilot uses cached paper-judge quality ratings for Original/LQR and assistant-audit ratings for H∞; it is a mixed-source diagnostic.

Model-indexed outputs are under `cache/qwen25_05b_instruct/`: source hashes, sample manifests, frozen reduced fit, covariance variants, controllers and synthesis checks, OOD hidden activations and residuals, per-block responses, official classifier judgments, and worker logs. The empirical responses and judgments are copied with source paths; large Jacobian caches are not duplicated. New responses and scores never write into Giga. Ten-case generation blocks and per-prompt activation reuse support extension without repeating completed work.

## Figures

- `covariance_alignment.png` / `.pdf`: OOD residual covariance alignment for Empirical, Isotropic and Rotated, using the existing 50-behavior analysis. Diagonal is hidden; its cached data and original results remain preserved.
- `geometry_performance_2x2.png` / `.pdf` and `geometry_performance.csv`: the 50-case pilot's ID/OOD ASR and AXBench overall, with Original, LQR (8D), Empirical, Isotropic and Rotated. No uncertainty bars are displayed; each panel has its own y range.
- Superseded figures and original pilot tables are under `archive/pilot_n050/`.

Covariance alignment is trace(C_variant C_OOD) / (‖C_variant‖F ‖C_OOD‖F). Values near one indicate proportional covariance matrices, not necessarily equal disturbance magnitude. Alignment brackets retain the original paired behavior-bootstrap comparisons and their original three-contrast correction; statistics were not recalculated after hiding diagonal.

OOD residuals use unsteered prompt-end activations, projected into the frozen calibration coordinates and evaluated with the frozen dynamics and reference. This matches the calibration residual definition. The unit-local collector captures all layer boundaries without computing unused vocabulary logits; a tiny-model check verifies exact agreement with the package collector. No response replay or Jacobian recomputation is required. Inputs are not truncated. Device and token lengths are saved with activations.

## Run or extend

From the repository root with the `robust-steerability` environment:

```bash
python -m unittest parking.rudra_rebuttal_residual_geometry.test_geometry
# Resume performance only; completed ten-case blocks and judgments are reused:
python -m parking.rudra_rebuttal_residual_geometry.pipeline run --count 200 --performance-only --devices cuda:0 cuda:1
# Read/rebuild the official ASR summary without generating or grading quality:
python -m parking.rudra_rebuttal_residual_geometry.performance --count 200
# Replot the existing 50-case alignment and existing pilot quality summary:
python -m parking.rudra_rebuttal_residual_geometry.pipeline report --count 50
```

The launcher waits for GPUs without other experiments' compute processes. Generation preserves the source's greedy decoding, 512-token cap and disabled KV cache. New judgments use the same pinned official HarmBench test classifier at batch size one. All inference is local. Tests cover trace/spectrum preservation, factor reconstruction, covariance alignment, nested paired sampling and hidden-state collection.
