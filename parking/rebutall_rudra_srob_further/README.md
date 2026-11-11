# H∞ behavior gain

Behavior gain = (H∞ − Original) / Original.

![T, T×I, and ASR behavior gain](hinf_behavior_gain.png)

[PDF](hinf_behavior_gain.pdf) · [Signal values](tables/behavior_signal_points.csv) · [Slope data](tables/behavior_signal_slopes.csv) · [Methods](src/README.md#report_behavior_gain)

Rows are ordered by the largest absolute available slope within each benchmark. Model counts are shown when a slope uses fewer than the full reported model set.

## TruthfulQA

| Measure | ID slope | OOD slope |
|---|---:|---:|
| I | +0.184827 | -0.660872 |
| T | +0.301277 | +0.563590 |
| T×I | +0.562044 | -0.260335 |
| AXBench overall | +0.356899 (n=2) | -0.287642 (n=2) |
| Instruction relevance | -0.186760 | +0.247167 |
| Concept relevance | +0.205741 (n=3) | +0.242123 (n=3) |
| Fluency | -0.195625 | -0.029475 |

## HarmBench

| Measure | ID slope | OOD slope |
|---|---:|---:|
| AXBench overall | -0.049343 | -0.101908 |
| Instruction relevance | -0.045191 | -0.063500 |
| Fluency | -0.015146 | -0.040243 |
| ASR | -0.022963 | -0.011461 |
| Safe-concept relevance | -0.006338 | -0.007565 |

## MGSM

| Measure | ID slope | OOD slope |
|---|---:|---:|
| Fluency | — | +0.267605 |
| Accuracy | — | +0.232865 |
| Instruction relevance | — | +0.093692 |
| Answer parse rate | — | +0.009151 |
| Spanish relevance | — | Undefined |
| AXBench overall | — | Undefined |

MGSM uses the five-language OOD average and has no evaluated ID condition. Its Spanish relevance and AXBench overall gains are undefined because Original is zero.

TruthfulQA concept relevance has paired values for three models. AXBench overall has two finite model pairs. Missing or non-finite values are excluded, with counts shown above.

Earlier figure variants are retained in [archive/figures](archive/figures). Run `python src/report_behavior_gain.py` from this unit to rebuild the active outputs.
