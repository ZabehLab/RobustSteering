#!/usr/bin/env python3
"""Render the GPT-2 Spanish-shift panel for the final composite figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "ref" / "original_notebook_rerun" / "results" /
          "final_llmfit_panelD_error_shrinkage.csv")
OUTPUT = HERE / "plots" / "Final Figures"
ID_COLOR = "#c9c9c9"
OOD_COLOR = "#8B1E1E"


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
    table = table[table["split"].isin(["ID", "OOD-es"])].copy()
    table["split"] = table["split"].replace({"OOD-es": "OOD"})
    pivot = table.pivot(index="horizon_tok", columns="split", values="med_lin").sort_index()

    # This gives the plotting axis the same physical width and height as the
    # Domain-shift axis in the 5 x 4 inch figure_stim_context composition.
    fig, axis = plt.subplots(figsize=(1.50, 4.0))
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.17, top=0.91)

    x = np.arange(len(pivot))
    width = 0.36
    axis.bar(x - width / 2, pivot["ID"], width, color=ID_COLOR)
    axis.bar(x + width / 2, pivot["OOD"], width, color=OOD_COLOR)
    axis.set_xticks(x, [str(value) for value in pivot.index], fontsize=7)
    axis.set_ylim(0, table["med_lin"].max() * 1.14)
    axis.set_xlabel("Horizon (tokens)", fontsize=8)
    axis.set_ylabel("Linear residual RMS (z)", fontsize=8)
    axis.set_title("Spanish shift", fontsize=9, weight="bold")
    axis.tick_params(axis="both", labelsize=7)
    axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"figure_spanish_shift.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
