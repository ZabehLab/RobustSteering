"""Render paper-style heatmaps from a cached matched-calibration grid."""

from __future__ import annotations

import argparse

from figs.calibration_plots.calibration_plots import INK, TEAL, Metric, annotation, setup_style, short_number
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from .inputs import MODEL_SPECS, read_json
from .pipeline import CACHE
from .reports import LABELS, VALIDATION_ASR
from .workers import CALIBRATION_PROMPTS, METHODS, generation_root

METRICS = (
    Metric(VALIDATION_ASR, "Validation ASR (%)", "down", scale=100, decimals=0),
    Metric("axbench_fluency", "Fluency"),
    Metric("axbench_overall", "Calibration score", decimals=3),
    Metric("axbench_instruction_relevance", "Instruction relevance"),
)


def draw_panel(ax, values, qs, qfs, metric, norm, selected, show_y):
    cmap = "Greys" if metric.direction == "up" else "Greys_r"
    ax.imshow(values, cmap=cmap, norm=norm, aspect=0.72, interpolation="nearest")
    ax.set_title(f"{metric.title} {'↑' if metric.direction == 'up' else '↓'}", pad=5)
    ax.set_xticks(range(len(qfs)), [short_number(value) for value in qfs])
    ax.set_yticks(range(len(qs)), [short_number(value) for value in qs] if show_y else [])
    ax.set_xlabel(r"$Q_f/R$", labelpad=3)
    if show_y:
        ax.set_ylabel(r"$Q/R$", labelpad=3)
    ax.tick_params(axis="both", length=0, labelsize=7.3, pad=2)
    ax.set_xticks(np.arange(-0.5, len(qfs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(qs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.05)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.9)

    for (i, j), value in np.ndenumerate(values):
        finite = np.isfinite(value)
        tone = norm(value) if metric.direction == "up" else 1.0 - norm(value)
        ax.text(j, i, annotation(value, metric.decimals) if finite else "—",
                ha="center", va="center", fontsize=7, color="white" if finite and tone > 0.62 else INK)
    if selected:
        i, j = selected
        ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                               edgecolor=TEAL, linewidth=2.3, zorder=4))
        ax.scatter(j + 0.29, i - 0.27, marker="*", s=34, facecolor=TEAL,
                   edgecolor="white", linewidth=0.35, zorder=5)


def render(model: str, profile: str):
    root = CACHE / model
    directory = generation_root(root, "calibration", profile)
    grid = read_json(directory / "grid.json")
    selection_file = directory / "selection.json"
    selection = read_json(selection_file) if selection_file.exists() else {}
    configs = [cell["configuration"] for cell in grid[METHODS[0]]]
    qs = sorted({config["q_over_r"] for config in configs})
    qfs = sorted({config["q_final_over_r"] for config in configs})
    values = np.full((len(METHODS), len(METRICS), len(qs), len(qfs)), np.nan)
    selected_cells = {}
    for row, method in enumerate(METHODS):
        winner = selection.get("winners", {}).get(method)
        for cell in grid[method]:
            if cell["status"] != "complete":
                continue
            config = cell["configuration"]
            i, j = qs.index(config["q_over_r"]), qfs.index(config["q_final_over_r"])
            values[row, :, i, j] = [cell[metric.key] * metric.scale for metric in METRICS]
            if winner and config == winner["configuration"]:
                selected_cells[method] = (i, j)

    setup_style()
    fig, axes = plt.subplots(len(METHODS), len(METRICS), figsize=(7.0, 6.2),
                             gridspec_kw={"left": 0.14, "right": 0.995, "bottom": 0.07,
                                          "top": 0.91, "wspace": 0.07, "hspace": 0.45})
    for column, metric in enumerate(METRICS):
        finite = values[:, column][np.isfinite(values[:, column])]
        norm = Normalize(0, max(float(finite.max()), 1e-6) if finite.size else 1)
        for row, method in enumerate(METHODS):
            draw_panel(axes[row, column], values[row, column], qs, qfs, metric, norm,
                       selected_cells.get(method), show_y=column == 0)
    for row, method in enumerate(METHODS):
        axis = axes[row, 0]
        axis.text(-0.42, 0.5, LABELS[method], transform=axis.transAxes, ha="center", va="center",
                  rotation=90, fontsize=8.8, fontweight="bold")
    count = selection.get("prompts_per_cell", CALIBRATION_PROMPTS[profile])
    fig.suptitle(f"{MODEL_SPECS[model].label} HarmBench calibration", y=0.99, fontsize=11, fontweight="bold")
    fig.text(0.56, 0.95, f"{count} validation prompts per cell · Teal marks the selected setting",
             ha="center", fontsize=8)
    output = directory / "calibration_heatmaps"
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--profile", choices=CALIBRATION_PROMPTS, default="mini")
    args = parser.parse_args()
    print(render(args.model, args.profile))


if __name__ == "__main__":
    main()
