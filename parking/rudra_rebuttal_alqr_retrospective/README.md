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

The original caches cover all 36 H∞ cells. The original A-LQR responses were produced at the submitted cost on the direct test split. Their behavior IDs, prompt IDs, and prompt texts have zero overlap with the validation inputs for each model. The retrospective sweep produces 600 A-LQR validation responses per model, or 1,800 in total, and scores them with the paper's three AXBench rubrics.

## Notes

Run from the repository root with the project environment active. The sweep uses the available GPUs for independent grid shards, starts AXBench scoring as cells finish, and resumes from saved gains, response batches, and judge batches.

```bash
python -m parking.rudra_rebuttal_alqr_retrospective.sweep run --devices auto
```

The sweep reuses the paper's pinned BF16 models, nominal dynamics, refusal direction, and formatted inputs. It uses greedy decoding with KV caching off and the paper's 512-token cap. Generation uses batches of 32 for 1B and 16 for 3B and 8B. The archived 8B calibration used batches of 32.

The renderer can also run independently on the saved scores.

```bash
python -m parking.rudra_rebuttal_alqr_retrospective.audit
```

Each model's `cache/<model>/grid/` holds its controller gains, responses, and rubric scores. `cache/logs/` holds the run plan and worker logs. The renderer writes the PNG and PDF comparison, `grid.csv`, and `prompt_split_audit.csv` to `cache/`. It verifies the response identities and matches the prompt IDs, behavior IDs, and input texts to the corresponding H∞ cell.

## References

- `benchmarks/harmful/cache/<model>/calibrations/h_infinity/selected/` contains the paper's calibration protocol, grid responses, and rubric scores.
- `benchmarks/harmful/cache/<model>/calibrations/alqr/selected/selection.json` records the submitted A-LQR cost.
- `benchmarks/harmful/cache/<model>/evaluations/kv_cache_off/` contains the held-out A-LQR responses and scores used to audit the split.
