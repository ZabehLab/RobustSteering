#!/usr/bin/env python3
"""Render the title-free GPT-2 across-layers panel for the final figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "ref" / "original_notebook_rerun" / "results" /
          "final_llmfit_panelF_state_size.csv")
OUTPUT = HERE / "plots" / "Final Figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "pdf.fonttype": 42,
})


def main():
    table = pd.read_csv(SOURCE)
    layers = table[table["med_resid"].notna()].copy()

    fig, axis = plt.subplots(figsize=(3.6, 3.1))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(layers)))
    label_positions = {
        7: dict(xytext=(-7, 6), ha="right", va="bottom"),
        8: dict(xytext=(5, -8), ha="left", va="top"),
        10: dict(xytext=(5, -8), ha="left", va="top"),
        11: dict(xytext=(-7, 6), ha="right", va="bottom"),
    }
    for (_, row), color in zip(layers.iterrows(), colors):
        layer = int(row["layer"])
        axis.scatter(layer, row["med_resid"], s=125, color=color, zorder=3)
        position = label_positions.get(
            layer, dict(xytext=(5, 4), ha="left", va="bottom")
        )
        axis.annotate(f"L{layer}", (layer, row["med_resid"]),
                      fontsize=10, textcoords="offset points", **position)

    # Padding keeps every layer label, including L11 and L12, inside the box.
    axis.set_xlim(0.25, 13.65)
    axis.set_ylim(0.176, 0.303)
    axis.set_xlabel("layer (fixed 128 dims)", fontsize=12)
    axis.set_ylabel("median residual RMS (z)", fontsize=12)
    axis.tick_params(axis="both", labelsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.20, top=0.97)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"figure_state_size_across_layers.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
