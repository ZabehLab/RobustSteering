# Gamma sweep

Qwen2.5-0.5B-Instruct on HarmBench, with the Giga dynamics, eight-dimensional state, costs, setpoint, and decoding held fixed. The sweep changes only the prescribed attenuation level, γ = γ★ / ρ. The nominal LQR (8D) endpoint is loaded directly at ρ = 0; the cached H∞ endpoint uses its actual deployment value ρ = 1 / 1.01.

Every point uses the same seeded 100 direct behaviors and one jailbreak per behavior, balanced across the five templates (20 each). Endpoints reuse Giga generations and official classifier scores on this subset. Their subset ASRs need not equal the full Giga table. No calibration is repeated.

Run from the repository root with the benchmark Python environment:

```bash
python -m parking.rudra_rebuttal_gamma_sweep.sweep coarse
```

This evaluates ρ = 0.2, 0.4, 0.6, 0.8 on both GPUs. Later, `fine` adds 0.1, 0.3, 0.5, 0.7, 0.9, retaining the completed points. Both commands resume missing generations and scores. `prepare` checks synthesis and freezes inputs without generation; `plot` refreshes the figure from completed scores.

All run inputs, controller gains, generations, scores, and logs are under `cache/qwen25_05b_instruct/`. The CSV is `tables/gamma_sweep.csv`; PNG and PDF are `plots/gamma_sweep.*`. The plot shows measured ASR percentages joined across completed points: black for ID/direct, dark red for OOD/jailbreak.

```bash
python -m unittest parking.rudra_rebuttal_gamma_sweep.test_sweep
```
