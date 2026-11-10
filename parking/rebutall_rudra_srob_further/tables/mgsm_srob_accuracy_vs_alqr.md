# MGSM Srob and accuracy relative to A-LQR

Accuracy is the equal-weight mean across Chinese, French, Japanese, Swahili, and Telugu (all OOD).
Srob and γ★ retain source precision. Behavioral columns are displayed to six decimals.
Phi-4-mini's ratio is undefined because its A-LQR accuracy is zero.

| Model | Srob | γ★ | A-LQR accuracy (%) | H∞ accuracy (%) | H∞ / A-LQR | H∞ − A-LQR (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.09030505350843292 | 11.073577404022217 | 22.400000 | 32.200000 | 1.437500 | +9.800000 |
| Phi-4-mini | 0.20093436811344256 | 4.976749420166016 | 0.000000 | 35.200000 | Undefined (A-LQR = 0%) | +35.200000 |
| Granite-3.3-2B | 0.9083298683298683 | 1.100921630859375 | 18.800000 | 21.200000 | 1.127660 | +2.400000 |
