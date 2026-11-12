#!/usr/bin/env python3
"""Plot HarmBench Srob against TruthfulQA gain using the chosen model proxies."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


UNIT = Path(__file__).resolve().parent
SOURCE = UNIT.parent / "rudra_rebuttal_srob_further/tables"
PAIRS = (("Llama-3.2-1B", "GPT-2 XL"),
         ("Llama-3.2-3B", "Gemma-2-2B"),
         ("Llama-3.1-8B", "Llama-3-8B"))


def main() -> None:
    harmful = json.loads((SOURCE / "harmbench_srob_asr.json").read_text())
    truthful = json.loads((SOURCE / "behavior_signal_points.json").read_text())
    rows, fits = [], []
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.labelsize": 12,
                         "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(4.3, 3.8))
    for ax, split, color in zip(axes, ("ID", "OOD"), ("black", "#8B1E1E"), strict=True):
        selected = []
        for source_model, target_model in PAIRS:
            srob_values = {row["srob"] for row in harmful if row["model"] == source_model}
            if len(srob_values) != 1:
                raise ValueError(f"Expected one HarmBench Srob for {source_model}")
            target, = [row for row in truthful if row["benchmark"] == "TruthfulQA"
                       and row["measure"] == "T" and row["model"] == target_model and row["split"] == split]
            original, hinf = target["original"], target["hinf"]
            if original <= 0:
                raise ValueError(f"Undefined truthfulness gain for {target_model}")
            point = {"srob_task": "HarmBench", "srob_model": source_model,
                     "behavior_task": "TruthfulQA", "behavior_model": target_model,
                     "pairing": "user-defined model proxy", "split": split,
                     "srob": srob_values.pop(), "original_truthful_pct": original,
                     "hinf_truthful_pct": hinf, "truthfulness_gain": (hinf - original) / original}
            selected.append(point)
        rows.extend(selected)
        x = np.array([row["srob"] for row in selected])
        y = np.array([row["truthfulness_gain"] for row in selected])
        fit = linregress(x, y)
        fits.append({"split": split, "model_pairs": len(selected),
                     "slope": float(fit.slope), "intercept": float(fit.intercept)})
        curve_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(curve_x, fit.intercept + fit.slope * curve_x, color=color, linewidth=1.8)
        ax.scatter(x, y, color="black", s=9, zorder=3)
        ax.set(title=split, xlabel=r"HarmBench $S_{\mathrm{rob}}$", ylabel="Truthfulness gain",
               xlim=(1.9, 8.5), xticks=[2, 5, 8])
        ax.title.set_color(color)
        ax.title.set_fontweight("bold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.margins(y=0.20)
    fig.suptitle("TruthfulQA", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.23, top=0.84, wspace=0.75)
    plots, tables = UNIT / "plots", UNIT / "tables"
    plots.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    for stem, records in (("cross_task_points", rows), ("cross_task_slopes", fits)):
        with (tables / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    for suffix in ("png", "pdf"):
        fig.savefig(plots / f"cross_task_srob_truthfulness_gain.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
