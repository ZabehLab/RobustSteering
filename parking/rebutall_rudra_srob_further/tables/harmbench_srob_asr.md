# HarmBench Srob and attack success rate

The three paper-reported models. Jailbreak ASR averages the five frozen templates.
ASR is lower-is-better. Ratios below one and negative differences indicate reduced attack success.
Srob and γ★ retain source precision. ASR and derived columns are displayed to six decimals.
The CSV and JSON retain full floating-point precision.

| Model | Condition | Srob | γ★ | Original ASR (%) | H∞ ASR (%) | H∞ / Original | H∞ − Original (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | Direct (ID) | 7.687859597672181 | 0.13007521629333496 | 10.000000 | 2.916667 | 0.291667 | -7.083333 |
| Llama-3.2-1B | Jailbreak (OOD) | 7.687859597672181 | 0.13007521629333496 | 5.583333 | 0.916667 | 0.164179 | -4.666667 |
| Llama-3.2-3B | Direct (ID) | 2.404990825688073 | 0.415802001953125 | 16.666667 | 8.333333 | 0.500000 | -8.333333 |
| Llama-3.2-3B | Jailbreak (OOD) | 2.404990825688073 | 0.415802001953125 | 11.750000 | 3.666667 | 0.312057 | -8.083333 |
| Llama-3.1-8B | Direct (ID) | 3.487044249994804 | 0.2867758274078369 | 15.833333 | 3.333333 | 0.210526 | -12.500000 |
| Llama-3.1-8B | Jailbreak (OOD) | 3.487044249994804 | 0.2867758274078369 | 14.666667 | 0.500000 | 0.034091 | -14.166667 |
