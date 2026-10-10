# s_rob_performance

## Caption

**Existing-results assessment of robust steerability and held-out retention.** (A) Selected-controller robust steerability versus task-specific retention for TruthfulQA, HarmBench, MGSM, and L-CiteEval; fits are computed separately by task because controller costs differ across benchmarks. TruthfulQA uses Spanish/English True-times-Informative, HarmBench uses mean jailbreak/direct safe-response rate, and L-CiteEval uses 16K/8K answer recall. MGSM lacks an English/ID H-infinity evaluation and therefore uses the mean language-wise ratio of H-infinity to Original OOD accuracy, shown as an accuracy-preservation proxy rather than strict ID-to-OOD retention. (B) Within-task Spearman correlations with 2,000-sample model bootstrap intervals and exact two-sided model-label permutation p-values. (C) Comparison with the available ID-performance and inverse-gain-norm baselines; matched residual norms and probe accuracies were not cached. (D) Exploratory model-ranking analogue reporting whether each predictor's top-ranked or two highest-ranked models include the best-retention model. The requested within-model intervention sweep and controlled-layer selection analysis require configuration-level OOD evaluations that are not present in the existing results.

## Panel Notes

- A: One point per selected model-task controller; no raw-value correlation is pooled across tasks.
- B: Model is the independent resampling and permutation unit; intervals are unstable at `n = 2–4`.
- C: Cross-task dots show separate within-task correlations, not a pooled predictor coefficient.
- D: Model selection is a substitute diagnostic and is not the requested layer-selection experiment.

## Checks

- Visual encodings checked against: `s_rob_performance.py`, `README.md`, and `plots/retention_points.csv`.
- Statistics checked against: `plots/statistics.json`.
- Remaining uncertainty: strict MGSM ID-to-OOD retention, residual-norm and probe baselines, the E1 configuration sweep, and E4 layer selection are unavailable from the current cached results.
