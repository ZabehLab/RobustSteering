# H∞ behavior gain

Behavior gain = (H∞ − Original) / Original.

![TruthfulQA T and HarmBench ASR behavior gain](hinf_behavior_gain.png)

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

| Measure | ID slope | John | DNE | Jailbreak Bot | YOJA/Nona | APM | Mean OOD slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| AXBench overall | -0.049343 | -0.089615 | -0.111637 | -0.101447 | -0.100271 | -0.107209 | -0.102036 |
| Instruction relevance | -0.045191 | -0.054574 | -0.071739 | -0.065842 | -0.056177 | -0.071080 | -0.063882 |
| Fluency | -0.015146 | -0.036215 | -0.039957 | -0.043504 | -0.037732 | -0.043971 | -0.040276 |
| ASR | -0.022963 | -0.009925 | +0.006971 | -0.017606 | -0.030154 | -0.007208 | -0.011584 |
| Safe-concept relevance | -0.006338 | -0.008925 | -0.007484 | -0.006341 | -0.008016 | -0.007048 | -0.007563 |

## MGSM (OOD)

| Measure | Chinese | French | Japanese | Swahili | Telugu | Mean OOD slope |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | +0.419362 | +0.039946 | +0.262445 | +0.225965 | -0.335666 | +0.122411 |
| Fluency | +0.228143 | +0.296534 | +0.378349 | +0.246996 | +0.107089 | +0.251422 |
| Instruction relevance | +0.040467 | +0.013478 | +0.156566 | -0.250170 | +0.300734 | +0.052215 |
| Answer parse rate | +0.000000 | +0.000000 | +0.007858 | +0.015333 | +0.022743 | +0.009187 |
| Spanish relevance | Undefined | Undefined | Undefined | Undefined | Undefined | Undefined |
| AXBench overall | Undefined | Undefined | Undefined | Undefined | Undefined | Undefined |

HarmBench's five named jailbreak columns are OOD. Mean OOD slope is the arithmetic mean of the five condition-specific slopes, for both HarmBench and MGSM. It is undefined if any condition's slope is undefined. The plotted HarmBench OOD fit still uses the pooled jailbreak results.

MGSM columns show slopes fitted separately for each OOD language. Its Spanish relevance and AXBench overall gains are undefined because Original is zero.

TruthfulQA concept relevance has paired values for three models. AXBench overall has two finite model pairs. Missing or non-finite values are excluded, with counts shown above.

Earlier figure variants are retained in [archive/figures](archive/figures). Run `python src/report_behavior_gain.py` from this unit to rebuild the active outputs.
