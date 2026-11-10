# HarmBench retrospective A-LQR calibration audit

## Method

This unit checks where the submitted A-LQR setting falls in its own calibration grid. It uses the original full-width A-LQR on the 12 cost pairs and 50 direct validation prompts used to calibrate H∞ for each of Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, and Llama-3.1-8B-Instruct. The paper's H∞ sweeps are reconstructed from cached rubric scores and checked against their selection manifests.

## Variables

The grid crosses $Q/R$ in `{0.01, 0.1, 1, 10}` with $Q_f/R$ in `{0.01, 0.1, 0.316227766}`. The submitted A-LQR setting was `(0.1, 0.1)` for all three models, with `R = 1` and setpoint multiplier `1.5` throughout.

## Statistics

Each cell averages the per-response harmonic mean of the AXBench concept relevance, instruction relevance, and fluency scores. A response scores zero when any component is zero. The descriptive comparison reports the best grid score minus the submitted score and the submitted cell's rank once all cells for that model are scored.

## Legends

The top row shows the paper's H∞ reference grids. The bottom row shows A-LQR at identical coordinates and on the same 0–2 color scale. The teal outline marks the submitted A-LQR cost on both rows. Teal dots mark the maximum in each complete panel, including ties. Hatching marks cells with missing validation scores.

## Interpretation

The existing caches cover 36 of 36 H∞ cells and 0 of 36 A-LQR cells. The available A-LQR responses were produced at the submitted cost on the direct test split. Their behavior IDs, prompt IDs, and prompt texts have zero overlap with the validation inputs for each model. The submitted A-LQR cost therefore has no measured validation score, rank, or gap in this audit. A complete behavioral sweep requires 600 A-LQR validation responses per model, or 1,800 in total, followed by AXBench scoring.

## Notes

Run from the repository root with the project environment active.

```bash
python -m parking.rudra_rebuttal_alqr_retrospective.audit
```

The script reads saved responses and scores. It writes the PNG and PDF comparison, `grid.csv`, and `prompt_split_audit.csv` to `cache/`. To load a scored A-LQR cell, place its response JSON in `cache/<model>/grid/generations/<grid_id>.json` and the three per-response rubric files in `cache/<model>/grid/scores/<rubric>/<grid_id>.json`. The generation identity must include `method`, `model`, `parameters`, and `evaluated_model_kv_cache`. The script checks these against the archived benchmark and matches the prompt IDs, behavior IDs, and input texts to the corresponding H∞ cell.

## References

- `benchmarks/harmful/cache/<model>/calibrations/h_infinity/selected/` contains the paper's calibration protocol, grid responses, and rubric scores.
- `benchmarks/harmful/cache/<model>/calibrations/alqr/selected/selection.json` records the submitted A-LQR cost.
- `benchmarks/harmful/cache/<model>/evaluations/kv_cache_off/` contains the held-out A-LQR responses and scores used to audit the split.
