#!/usr/bin/env python3
"""Render pooled reliability and task-specific OOD performance against SRob."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


TASKS = {
    "harmful": {
        "title": "HarmBench",
        "color": "#5B6573",
        "marker": "o",
        "ylabel": "Safe response\nrate (%)",
    },
    "truthfulness": {
        "title": "Truthfulness",
        "color": "#687DA3",
        "marker": "s",
        "ylabel": "Truthfulness (%)",
    },
    "mgsm": {
        "title": "MGSM",
        "color": "#C27D5A",
        "marker": "^",
        "ylabel": "Accuracy (%)",
    },
    "lciteeval": {
        "title": "L-CiteEval",
        "color": "#7B70B6",
        "marker": "p",
        "ylabel": "Recall / citation\nF1 (%)",
    },
}

CONDITION_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]
SHORT_CONDITION = {
    "APM programmer": "APM",
    "DNE nonresponse": "DNE",
    "Jailbreak Bot": "Jailbreak",
    "John persona": "John",
    "YOJA/Nona roleplay": "YOJA/Nona",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["distribution"] == "ood"
            and row["benchmark"] in TASKS
            and row["s_rob"]
            and row["hinf_reliability"]
        ]
    return rows


def ols_summary(rows: list[dict[str, str]]) -> dict[str, float | int]:
    x = np.asarray([float(row["s_rob"]) for row in rows], dtype=float)
    y = np.asarray([float(row["hinf_reliability"]) for row in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "n": int(x.size),
        "ols_slope_raw_coordinates": float(slope),
        "ols_intercept_raw_coordinates": float(intercept),
    }


def condition_adjusted_fit(rows: list[dict[str, str]]) -> dict[str, object]:
    models = sorted({row["model_key"] for row in rows})
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    x_by_model = {
        model: float(next(row["s_rob"] for row in rows if row["model_key"] == model))
        for model in models
    }

    def common_slope(permuted_x: dict[str, float]) -> float:
        centered_x = []
        centered_y = []
        for condition in conditions:
            selected = [row for row in rows if row["condition"] == condition]
            group_x = np.asarray([permuted_x[row["model_key"]] for row in selected], dtype=float)
            group_y = np.asarray([float(row["hinf_reliability"]) for row in selected], dtype=float)
            centered_x.extend(group_x - group_x.mean())
            centered_y.extend(group_y - group_y.mean())
        centered_x_array = np.asarray(centered_x, dtype=float)
        centered_y_array = np.asarray(centered_y, dtype=float)
        return float(
            centered_x_array @ centered_y_array
            / (centered_x_array @ centered_x_array)
        )

    observed_slope = common_slope(x_by_model)
    mean_x = float(np.mean(list(x_by_model.values())))
    intercepts = {}
    for condition in conditions:
        group_y = np.asarray(
            [float(row["hinf_reliability"]) for row in rows if row["condition"] == condition],
            dtype=float,
        )
        intercepts[condition] = float(group_y.mean() - observed_slope * mean_x)

    x_values = tuple(x_by_model[model] for model in models)
    permutation_slopes = [
        common_slope(dict(zip(models, permutation)))
        for permutation in itertools.permutations(x_values)
    ]
    two_sided_p = float(
        sum(abs(slope) >= abs(observed_slope) - 1e-12 for slope in permutation_slopes)
        / len(permutation_slopes)
    )
    return {
        "model_count": len(models),
        "condition_count": len(conditions),
        "observation_count": len(rows),
        "common_slope_raw_coordinates": observed_slope,
        "condition_intercepts": intercepts,
        "exact_model_block_permutation_count": len(permutation_slopes),
        "exact_two_sided_p_value": two_sided_p,
    }


def padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    span = float(values.max() - values.min())
    pad = span * fraction if span else max(abs(float(values[0])) * fraction, 0.05)
    return float(values.min() - pad), float(values.max() + pad)


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    title: str,
    ylabel: str,
    color: str | None,
    pooled: bool,
    stratify_conditions: bool = False,
) -> dict[str, object]:
    if pooled:
        for benchmark, style in TASKS.items():
            selected = [row for row in rows if row["benchmark"] == benchmark]
            ax.scatter(
                [float(row["s_rob"]) for row in selected],
                [float(row["hinf_reliability"]) for row in selected],
                s=38,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.92,
                zorder=3,
            )
        line_color = "#222222"
    elif stratify_conditions:
        conditions = list(dict.fromkeys(row["condition"] for row in rows))
        hierarchical_summary = condition_adjusted_fit(rows)
        for condition, condition_color in zip(conditions, CONDITION_COLORS):
            selected = [row for row in rows if row["condition"] == condition]
            x_group = np.asarray([float(row["s_rob"]) for row in selected], dtype=float)
            y_group = np.asarray([float(row["hinf_reliability"]) for row in selected], dtype=float)
            x_line = np.linspace(float(x_group.min()), float(x_group.max()), 100)
            y_line = (
                hierarchical_summary["condition_intercepts"][condition]
                + hierarchical_summary["common_slope_raw_coordinates"] * x_line
            )
            ax.plot(x_line, y_line, color=condition_color, linewidth=1.55, zorder=2)
            ax.scatter(
                x_group,
                y_group,
                s=40,
                color=condition_color,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.95,
                zorder=3,
                label=SHORT_CONDITION.get(condition, condition),
            )
        line_color = None
    else:
        assert color is not None
        ax.scatter(
            [float(row["s_rob"]) for row in rows],
            [float(row["hinf_reliability"]) for row in rows],
            s=43,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            zorder=3,
        )
        line_color = color

    x = np.asarray([float(row["s_rob"]) for row in rows], dtype=float)
    y = np.asarray([float(row["hinf_reliability"]) for row in rows], dtype=float)
    if stratify_conditions:
        summary: dict[str, object] = hierarchical_summary
    else:
        summary = ols_summary(rows)
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        y_line = summary["ols_intercept_raw_coordinates"] + summary["ols_slope_raw_coordinates"] * x_line
        ax.plot(x_line, y_line, color=line_color, linewidth=1.65, zorder=2)

    ax.set_xlim(*padded_limits(x))
    if pooled:
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 50, 100])
    else:
        ax.set_ylim(*padded_limits(y, fraction=0.12))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=7)
    ax.set_xlabel(r"$S_{\mathrm{rob}}$")
    ax.set_ylabel(ylabel)
    if stratify_conditions:
        ax.text(
            0.05,
            0.95,
            rf"$p={summary['exact_two_sided_p_value']:.3g}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        )
    ax.grid(True, color="#D9DDE1", linewidth=0.65, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(width=0.9, length=4.0)
    if stratify_conditions and summary["condition_count"] > 1:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
            ncol=2,
            frameon=False,
            fontsize=6.5,
            columnspacing=0.65,
            handlelength=1.2,
            handletextpad=0.3,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise ValueError("No OOD rows with SRob and H-infinity reliability were found.")

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15.8, 3.35),
        gridspec_kw={"width_ratios": [1.25, 1, 1, 1, 1]},
    )

    statistics: dict[str, dict[str, float | int]] = {}
    statistics["pooled"] = draw_panel(
        axes[0], rows, "OOD reliability", "OOD reliability (%)", None, pooled=True
    )
    for ax, (benchmark, style) in zip(axes[1:], TASKS.items()):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        statistics[benchmark] = draw_panel(
            ax,
            selected,
            style["title"],
            style["ylabel"],
            style["color"],
            pooled=False,
            stratify_conditions=benchmark in {"harmful", "truthfulness", "mgsm"},
        )

    handles = [
        Line2D(
            [0], [0],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6.6,
            label=style["title"],
        )
        for style in TASKS.values()
    ]
    axes[0].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.39),
        ncol=2,
        frameon=False,
        fontsize=8.2,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.87, bottom=0.34, wspace=0.56)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with args.output_prefix.with_name(args.output_prefix.name + "_points.csv").open("w", newline="") as handle:
        fieldnames = [
            "benchmark", "model_key", "model_label", "condition_axis", "condition",
            "outcome_metric", "s_rob", "hinf_reliability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])
    with args.output_prefix.with_name(args.output_prefix.name + "_statistics.json").open("w") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
