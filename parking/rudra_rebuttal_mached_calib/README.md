# Matched HarmBench calibration

A-LQR (Full-width), A-LQR (8D), and H∞ each search the paper's 12 cost settings on
the same direct calibration prompts within each profile. The 8D arms share the
fitted state, reference, and intervention. The full-width arm uses the paper's
A-LQR policy. The winner in each arm maximizes AXBench Overall using the paper's
tie-breakers.

The unit reuses the paper's fitting data and, when available, the model's saved
dynamics and 8D fit. Otherwise it prepares the 50 safe and 50 harmful direction
records and 50 Jacobians inside the model's unit cache. The exact archived prompt
texts are shared across checkpoints, with each model's own chat formatting.

## Run and compare

Run from the code repository with the project's Python environment. Set
`--model` to the checkpoint you want to compare.

```bash
python -m parking.rudra_rebuttal_mached_calib.pipeline run \
  --model smollm2_360m_instruct --profile mini --devices auto
python -m parking.rudra_rebuttal_mached_calib.pipeline compare --profile mini
```

The mini profile uses the first 20 archived direct validation prompts in each
calibration cell and the first 20 archived held-out behaviors for evaluation.
Every model uses the same prompts and one direct request plus all five archived
jailbreak variants per evaluation behavior. The `mega` profile uses 50 direct
validation prompts per cell and the same 20 held-out behaviors as `mini`. It
reuses matching mini generations and judge scores for the first 20 prompts, and
matching evaluation shards when the selected controller setting is the same.
The `full` profile uses 50 calibration prompts per cell and evaluates all 240
behaviors with its own selected controllers. It also reports the 220 behaviors
outside mini separately. The profiles share fitted artifacts and controller
gains and save their generations, scores, and selections separately.
`--devices auto` uses the visible GPUs for independent generation, Jacobian, and
classifier workers. AXBench judges run on calibration responses and the official
HarmBench test classifier supplies the evaluation ASRs.

`giga` loads a completed Mega selection and evaluates Original and its three
frozen controllers on all 240 archived behaviors. Each has one direct prompt
and five jailbreak variants. Matching Mega generations and ASR scores are
reused, and AXBench Overall is scored on held-out responses. The final table
reports means ± ten-group behavior-clustered jackknife standard errors.

```bash
python -m parking.rudra_rebuttal_mached_calib.pipeline run \
  --model qwen25_05b_instruct --profile giga --devices auto
```

## Model cache

```text
cache/<model>/
  artifacts/                   Frozen prompts, direction, dynamics, and reduced fit
  calibration/controllers/    Shared per-cell gains
  calibration/{mini,mega,full}/ Profile-specific generations, scores, and winners
  evaluation/{mini,mega}/      20-behavior screening shards, results.json, response.md
  evaluation/full/             20 screening and 220 holdout behaviors, results.json, response.md
  evaluation/giga/             240 behaviors, Mega provenance, and the four-arm report
  logs/                        Run manifests, worker assignments, and GPU logs
  checks/                      Implementation smoke checks
```

Each response file contains the 36 calibration cells and the three selected
controllers' held-out results in the two tables from the R1 draft. Generation
resumes from completed batches. The model comparison lists completed runs and
both jailbreak ASR gaps against H∞.

Render the calibration heatmaps from saved scores with
`python -m parking.rudra_rebuttal_mached_calib.calibration_heatmaps --model qwen25_05b_instruct --profile mini`.
The PNG and PDF are saved alongside `grid.json`.

## Implementation checks

```bash
python -m unittest parking.rudra_rebuttal_mached_calib.test_pipeline
python -m parking.rudra_rebuttal_mached_calib.smoke --devices auto
```

The smoke check uses a neutral prompt and a two-token cap. It exercises the real
controller hooks on the cached Llama 3B fit.
