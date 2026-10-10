# Figure Language

Two-panel MGSM language-shift figure derived directly from the finalized MGSM
table used for paper reporting.

- **Left:** five-language macro accuracy versus the literal product of AXBench
  instruction relevance and fluency (`IR × Fluency`, range 0–4).
- **Right:** per-language accuracy with the fixed order Chinese, French,
  Japanese, Swahili, and Telugu.
- **Encoding:** model family is encoded by the Qwen, Microsoft/Phi, or
  IBM/Granite logo; method is encoded by color. H∞ uses the exact project teal
  `#007C7C`, and A-LQR uses the standardized truthfulness color `#687DA3`.
- **Typography:** Arial, following the repository-wide figure style.

The script reads `../bench_table/mgsm/mgsm_full.md`, whose finalized comparison
contains Qwen3-4B, Phi-4-mini, and Granite-3.3-2B with Original, A-LQR, and H∞.
Historical Llama and S-PID rows from the earlier transfer-figure cache are not
used.

Run from the repository root:

```bash
python figs/figure_language/figure_language.py
```

Outputs are written to `plots/figure_language.pdf` and
`plots/figure_language.png` within this unit.

The preservation-focused `figure_language_delta` output reports the magnitude of
accuracy change relative to Original, `|method accuracy - Original accuracy|`.
Original is the zero line, and smaller values indicate stronger preservation.
The aggregate panel averages the five per-language magnitudes and connects A-LQR
and H∞ for each model family. The language panel uses bars grouped first by
language and then by model family; each tightly spaced pair contains A-LQR and H∞.
Method is encoded by bar color. Black circles and a dashed connector mark each
pair's endpoints, while one native company logo is centered at a fixed offset of
8% of the shared y-range above the taller bar. A-LQR uses a deliberately light
neutral gray and H∞ uses the exact project teal. Original markers are omitted; a
solid zero line marks exact preservation.

The retained alternative `figure_language_relative` leaves the aggregate panel
unchanged and expresses only the language-panel bars as `100 × method accuracy /
Original accuracy`. Thus 100% is unchanged from Original, 50% is half the
Original accuracy, and 200% is twice the Original accuracy. A solid 100% line
marks the Original reference. Company logos occupy one fixed row below the zero
baseline so their vertical position does not depend on the bar height. This
variant's right panel displays Chinese, French, Japanese, and Swahili. Its left
aggregate remains the full five-language average, including Telugu. Telugu is
omitted only from the relative bars because its 6% Original accuracy turns a
modest absolute increase into a 316.7% ratio that dominates the relative scale.
The relative panel labels only nonnegative ticks; its small negative margin
exists solely to hold the model-logo row.

`figure_language_relative_expanded` is a presentation-polish variant of the
relative figure. It preserves the panel geometry, data, typography, axes, and
method legend exactly, while grouping the model-logo header into compact
logo–name pairs (matching the TruthfulQA composite), scaling only the model
logos in the `IR × fluency` panel by 1.8×, and strengthening the paired
transfer-language connectors. Its model and method legends share one top row,
left- and right-aligned respectively, and their text and all tick labels match
the axis-label size. Regenerate only this variant with
`python figs/figure_language/figure_language.py --relative-expanded-only`.
