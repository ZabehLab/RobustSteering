#!/usr/bin/env python3
"""Render the title-free across-leads state-size panel for the final figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "ref" / "original_notebook_rerun" / "results" /
          "final_panelF_state_size_theta.csv")
OUTPUT = HERE / "plots" / "Final Figures"

AREA_ORDER = [
    "dorsolateral prefrontal",
    "ventrolateral prefrontal",
    "premotor / dorsomedial frontal",
    "sensorimotor",
    "temporal / peri-insular",
    "posterior temporal",
    "parieto-occipital",
]
AREA_COLORS = dict(zip(AREA_ORDER, [
    "#1f4ea1", "#5b8bd0", "#2e9c5c", "#8bc34a",
    "#e0a13c", "#c1272d", "#7b4fa6",
]))

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
    table = pd.read_csv(SOURCE, index_col=0)
    leads = table[table["area"].notna()].copy()

    fig, axis = plt.subplots(figsize=(3.6, 3.1))
    label_positions = {
        "LAO": dict(xytext=(-9, 1), ha="right", va="bottom"),
        "LAF": dict(xytext=(7, 8), ha="left", va="bottom"),
        "RVF": dict(xytext=(-8, 0), ha="right", va="center"),
        "LPO": dict(xytext=(7, 7), ha="left", va="bottom"),
        "RAT": dict(xytext=(7, -4), ha="left", va="top"),
        "LVF": dict(xytext=(6, 6)),
    }
    for lead, row in leads.iterrows():
        axis.scatter(row["n_ch"], row["med_resid"], s=125,
                     color=AREA_COLORS[row["area"]], zorder=3)
        position = label_positions.get(lead, dict(xytext=(7, 6)))
        axis.annotate(lead, (row["n_ch"], row["med_resid"]),
                      fontsize=10, textcoords="offset points",
                      **position)

    # Deliberately padded so edge labels such as RMF, LVF, and LPT remain
    # entirely inside the plotting box.
    axis.set_xlim(4.35, 16.45)
    axis.set_ylim(0.46, 1.035)
    axis.set_xlabel("channels in lead", fontsize=12)
    axis.set_ylabel("median residual RMS (z)", fontsize=12)
    axis.tick_params(axis="both", labelsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.20, top=0.97)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"figure_state_size_across_leads.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
