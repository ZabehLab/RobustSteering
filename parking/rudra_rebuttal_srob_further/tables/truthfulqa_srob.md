# TruthfulQA Srob and T×I

The five paper-figure models. T×I is True (%) × Informative (%) / 100.
Srob and γ★ retain the source CSV precision. T×I, ratios, and differences are displayed to six decimals.
The CSV and JSON retain the full saved floating-point precision.

| Model | Condition | Srob | γ★ | Original T×I (%) | H∞ T×I (%) | H∞ / Original | H∞ − Original (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 XL | English (ID) | 0.03242640805761803 | 30.839061737060547 | 27.224419 | 35.182602 | 1.292318 | +7.958184 |
| GPT-2 XL | Spanish-input (OOD) | 0.03242640805761803 | 30.839061737060547 | 7.604620 | 18.140823 | 2.385500 | +10.536204 |
| Gemma-2-2B | English (ID) | 0.02550682473994837 | 39.205193519592285 | 47.476018 | 55.624731 | 1.171638 | +8.148713 |
| Gemma-2-2B | Spanish-input (OOD) | 0.02550682473994837 | 39.205193519592285 | 44.435788 | 49.695764 | 1.118373 | +5.259976 |
| Llama-3-8B | English (ID) | 0.20607235592808135 | 4.852664470672607 | 47.794082 | 61.424233 | 1.285185 | +13.630151 |
| Llama-3-8B | Spanish-input (OOD) | 0.20607235592808135 | 4.852664470672607 | 40.513692 | 55.287802 | 1.364670 | +14.774110 |
| Qwen-2.5-14B | English (ID) | 0.01394882293720001 | 71.69063687324524 | 53.917311 | 76.269242 | 1.414559 | +22.351931 |
| Qwen-2.5-14B | Spanish-input (OOD) | 0.01394882293720001 | 71.69063687324524 | 66.292778 | 82.569900 | 1.245534 | +16.277122 |
| OLMo-2-32B | English (ID) | 0.38892416111272093 | 2.571195363998413 | 56.218904 | 86.305093 | 1.535161 | +30.086189 |
| OLMo-2-32B | Spanish-input (OOD) | 0.38892416111272093 | 2.571195363998413 | 61.014788 | 91.498437 | 1.499611 | +30.483648 |
