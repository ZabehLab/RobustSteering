# TruthfulQA Srob and Truthful (%)

Truthful (%) uses the saved TruthfulQA True metric mean.
Srob and γ★ retain the source CSV precision. Behavioral columns are displayed to six decimals.
The CSV and JSON retain the full saved floating-point precision.

| Model | Condition | Srob | γ★ | Original Truthful (%) | H∞ Truthful (%) | H∞ / Original | H∞ − Original (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 XL | English (ID) | 0.03242640805761803 | 30.839061737060547 | 28.886169 | 46.511628 | 1.610169 | +17.625459 |
| GPT-2 XL | Spanish-input (OOD) | 0.03242640805761803 | 30.839061737060547 | 66.095471 | 84.210526 | 1.274074 | +18.115055 |
| Gemma-2-2B | English (ID) | 0.02550682473994837 | 39.205193519592285 | 49.791922 | 70.110159 | 1.408063 | +20.318237 |
| Gemma-2-2B | Spanish-input (OOD) | 0.02550682473994837 | 39.205193519592285 | 51.407589 | 77.454100 | 1.506667 | +26.046512 |
| Llama-3-8B | English (ID) | 0.20607235592808135 | 4.852664470672607 | 49.277846 | 76.499388 | 1.552409 | +27.221542 |
| Llama-3-8B | Spanish-input (OOD) | 0.20607235592808135 | 4.852664470672607 | 42.864137 | 68.543452 | 1.599086 | +25.679315 |
| Qwen-2.5-14B | English (ID) | 0.01394882293720001 | 71.69063687324524 | 55.887393 | 85.312118 | 1.526500 | +29.424725 |
| Qwen-2.5-14B | Spanish-input (OOD) | 0.01394882293720001 | 71.69063687324524 | 68.127295 | 88.739290 | 1.302551 | +20.611995 |
| OLMo-2-32B | English (ID) | 0.38892416111272093 | 2.571195363998413 | 56.915545 | 92.656059 | 1.627957 | +35.740514 |
| OLMo-2-32B | Spanish-input (OOD) | 0.38892416111272093 | 2.571195363998413 | 62.545900 | 95.593635 | 1.528376 | +33.047736 |
