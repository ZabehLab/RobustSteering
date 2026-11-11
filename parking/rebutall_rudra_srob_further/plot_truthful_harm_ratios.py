#!/usr/bin/env python3
"""Plot ID and OOD H-infinity/Original ratios for TruthfulQA and HarmBench."""

from __future__ import annotations

import hashlib
import json

from analyze import PLOTS, TABLES, configure_plot_style, fit_record, save_figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


TASKS = (
    ("TruthfulQA", "truthfulqa_srob_truthful.json", "hinf_over_original_truthful", 5),
    ("HarmBench", "harmbench_srob_asr.json", "hinf_over_original_asr", 3),
)
COLORS = {"ID": "#2463EB", "OOD": "#8B1E1E"}


def main() -> None:
    configure_plot_style()
    PLOTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fit_results = {}
    sources = {}
    for ax, (benchmark, filename, outcome, expected_models) in zip(axes, TASKS, strict=True):
        source_path = TABLES / filename
        content = source_path.read_bytes()
        rows = json.loads(content)
        sources[filename] = hashlib.sha256(content).hexdigest()
        fit_results[benchmark] = {}
        for index, (split, color) in enumerate(COLORS.items()):
            selected = [row for row in rows if row["condition"].endswith(f"({split})")]
            if len(selected) != expected_models:
                raise ValueError(f"Expected {expected_models} paper models for {benchmark}, {split}")
            x = np.array([row["srob"] for row in selected])
            y = np.array([row[outcome] for row in selected])
            fit = fit_record(x, y)
            curve_x = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(curve_x, fit["intercept"] + fit["slope"] * curve_x,
                    color=color, linewidth=1.5, alpha=0.8, zorder=1)
            ax.scatter(x, y, s=48, color=color, zorder=3)
            ax.text(0.97, 0.96 - index * 0.09, f"{split} slope = {fit['slope']:.3f}",
                    transform=ax.transAxes, ha="right", va="top", color=color, fontsize=11)
            fit_results[benchmark][split] = {
                "n_models": len(selected), "slope": fit["slope"], "intercept": fit["intercept"],
            }
        ax.set(title=benchmark, xlabel=r"$S_{\mathrm{rob}}$", ylabel="H∞ / Original")
    axes[0].set(xlim=(-0.007, 0.46), ylim=(1.1, 1.85), xticks=[0, 0.1, 0.2, 0.3, 0.4])
    axes[1].set(xlim=(1.9, 8.5), ylim=(-0.02, 0.62), xticks=[2, 3, 4, 5, 6, 7, 8])
    handles = [Line2D([0], [0], color=color, marker="o", linewidth=1.5, markersize=6, label=split)
               for split, color in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.28)
    save_figure(fig, "srob_truthful_harm_ratios")
    output = {"x": "Srob", "y": "H∞ / Original", "sources_sha256": sources, "fits": fit_results}
    (TABLES / "truthful_harm_ratio_slopes.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(fit_results, indent=2))


if __name__ == "__main__":
    main()
