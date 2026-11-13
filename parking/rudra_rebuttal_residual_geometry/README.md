# HarmBench residual geometry

Compare empirical, isotropic, diagonal and randomly rotated disturbance covariances with the H∞ objective fixed. All experiment-specific logic lives here; the package and source Mega/Giga artifacts are read only.

The source is `parking/rudra_rebuttal_mached_calib/cache/qwen25_05b_instruct`: Mega's frozen Qwen2.5-0.5B-Instruct calibration and Giga's held-out responses and official HarmBench judgments. Costs remain Q/R = 0.01 and Qf/R = 0.01, with the original shared 8D states, dynamics and setpoints. Each geometry uses the original synthesis procedure with a 1% deployment margin above its own γ★. There is no recalibration or grid search.

| Geometry | Construction at each layer |
| --- | --- |
| Empirical | Original covariance and original disturbance factor |
| Isotropic | trace(C) / 8 times the identity |
| Diagonal | Diagonal of C |
| Randomly rotated | U C Uᵀ with one seeded orthogonal rotation shared across layers |

All four have the same per-layer trace: the expected squared norm of centered residuals. Rotation preserves eigenvalues while changing orientation. The rotated arm uses one predefined seed in this pilot, not a selected best/worst rotation.

## Sample and caches

The first run uses 50 held-out behavior identities, each with its direct prompt and exactly one jailbreak counterpart. The OOD pool contains exactly ten prompts from each of the five templates. Seeded behavior ordering and shuffled five-template assignments give nested balanced prefixes: increasing to 100 retains the first 50 cases and adds 50. No held-out behavior overlaps the 50 calibration cases.

Model-indexed outputs are under `cache/qwen25_05b_instruct/`: source hashes, sample manifests, frozen reduced fit, covariance variants, controllers and synthesis checks, OOD hidden activations and residuals, per-block responses, official classifier judgments, and worker logs. The empirical responses and judgments are copied with source paths; large Jacobian caches are not duplicated. New responses and scores never write into Giga. Ten-case generation blocks and per-prompt activation reuse support extension without repeating completed work.

## Figures

- `geometry_behavior.png` / `.pdf`: grouped ID/OOD box plots of 2,000 paired behavior-bootstrap ASR estimates for each covariance choice. Dots show observed ASR. Boxes show sampling uncertainty, **not independent experimental runs**. The same bootstrap draws are used for every arm and condition.
- `covariance_alignment.png` / `.pdf`: transformer layer versus trace(C_variant C_OOD) / (‖C_variant‖F ‖C_OOD‖F). All four calibration hypotheses are compared with the same held-out OOD residual covariance. Values near one indicate proportional covariance matrices, not necessarily equal disturbance magnitude.
- `results.csv`: observed ID/OOD ASR, jackknife standard errors, and each controller's γ★ and deployed γ.

OOD residuals use unsteered prompt-end activations, projected into the frozen calibration coordinates and evaluated with the frozen dynamics and reference. This matches the calibration residual definition. The unit-local collector captures all layer boundaries without computing unused vocabulary logits; a tiny-model check verifies exact agreement with the package collector. No response replay or Jacobian recomputation is required. Inputs are not truncated. Device and token lengths are saved with activations.

## Run or extend

From the repository root with the `robust-steerability` environment:

```bash
python -m unittest parking.rudra_rebuttal_residual_geometry.test_geometry
python -m parking.rudra_rebuttal_residual_geometry.pipeline run --count 50 --devices cuda:0 cuda:1
# Extend, preserving completed cases:
python -m parking.rudra_rebuttal_residual_geometry.pipeline run --count 100 --devices cuda:0 cuda:1
```

The launcher waits for GPUs without other experiments' compute processes. Generation preserves the source's greedy decoding, 512-token cap and disabled KV cache. New judgments use the same pinned official HarmBench test classifier at batch size one. All inference is local. Tests cover trace/spectrum preservation, factor reconstruction, covariance alignment, nested paired sampling and hidden-state collection.
