#!/usr/bin/env python3
"""Render controller-level task clustering and parameter-scaling figures."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kendalltau, linregress


UNIT = Path(__file__).resolve().parent
INPUT = UNIT / "exports/gamma_star_collaborator_bundle/final_selected_calibrations.csv"
PLOTS = UNIT / "plots"
ROOT = UNIT.parents[1]

STYLES = {
    "harmful": ("HarmBench", "#5B6573", "o"),
    "truthfulness": ("Truthfulness", "#687DA3", "s"),
    "mgsm": ("MGSM", "#C27D5A", "^"),
}

SHORT_MODEL = {
    "pythia_14m": "Pythia-14M",
    "pythia_31m": "Pythia-31M",
    "distilgpt2": "DistilGPT-2",
    "gpt2_small": "GPT-2 Small",
    "smollm2_135m": "SmolLM2-135M",
    "pythia_160m": "Pythia-160M",
    "gpt2_medium": "GPT-2 Medium",
    "qwen25_05b": "Qwen-2.5-0.5B",
    "gpt2_large": "GPT-2 Large",
    "llama32_1b": "Llama-3.2-1B",
    "llama32_3b": "Llama-3.2-3B",
    "llama31_8b": "Llama-3.1-8B",
    "gpt2_xl": "GPT-2 XL",
    "qwen25_14b": "Qwen-2.5-14B",
    "olmo2_32b": "OLMo-2-32B",
    "qwen3_4b": "Qwen3-4B",
    "phi4_mini": "Phi-4-mini",
    "granite33_2b": "Granite-3.3-2B",
    "qwen25_3b": "Qwen-2.5-3B",
    "gemma2_2b": "Gemma-2-2B",
    "qwen25_32b": "Qwen-2.5-32B",
}

TRUTHFULNESS_EXTRA_CALIBRATIONS = {
    "pythia_14m": "lambda_txi_n200_r1",
    "pythia_31m": "lambda_txi_n200_r1",
    "distilgpt2": "lambda_txi_n200_r1",
    "gpt2_small": "lambda_txi_n200_r1",
    "smollm2_135m": "lambda_txi_n200_r1",
    "pythia_160m": "lambda_txi_n200_r1",
    "gpt2_medium": "lambda_txi_n200_r1",
    "qwen25_05b": "lambda_txi_n200_r1",
    "gpt2_large": "lambda_txi_n200_r1",
    "gemma2_2b": "fixed_q0p1_qf0p31622777_r1",
    "qwen25_32b": "selected",
}

TRUTHFULNESS_NEW_MODEL_KEYS = (
    "pythia_14m",
    "pythia_31m",
    "distilgpt2",
    "gpt2_small",
    "smollm2_135m",
    "pythia_160m",
    "gpt2_medium",
    "qwen25_05b",
    "gpt2_large",
)

LOGOS = {
    "pythia_14m": "eleutherai_transparent.png",
    "pythia_31m": "eleutherai_transparent.png",
    "pythia_160m": "eleutherai_transparent.png",
    "distilgpt2": "huggingface_official.png",
    "gpt2_small": "openai_transparent.png",
    "smollm2_135m": "huggingface_official.png",
    "gpt2_medium": "openai_transparent.png",
    "gpt2_large": "openai_transparent.png",
    "qwen25_05b": "qwen_transparent.png",
    "gpt2_xl": "openai_transparent.png",
    "llama31_8b": "llama_transparent.png",
    "llama32_1b": "llama_transparent.png",
    "llama32_3b": "llama_transparent.png",
    "qwen25_14b": "qwen_transparent.png",
    "qwen25_32b": "qwen_transparent.png",
    "qwen3_4b": "qwen_transparent.png",
    "olmo2_32b": "ai2_transparent.png",
    "gemma2_2b": "google_official.png",
    "granite33_2b": "ibm_transparent.png",
    "phi4_mini": "microsoft_transparent.png",
}


def load_rows() -> list[dict[str, object]]:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        if row["benchmark"] not in STYLES:
            continue
        output.append({
            **row,
            "parameter_billions": float(row["parameter_count"]) / 1e9,
            "s_rob_value": float(row["s_rob"]),
            "log_s_rob": np.log10(float(row["s_rob"])),
            "short_model": SHORT_MODEL[row["model_key"]],
            "task_label": STYLES[row["benchmark"]][0],
        })
    if len(output) != 10:
        raise RuntimeError(f"expected 10 figure-eligible controllers, found {len(output)}")
    return output


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })


def task_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker=marker, linestyle="none", markersize=7.5,
               markerfacecolor=color, markeredgecolor="white", label=label)
        for label, color, marker in STYLES.values()
    ]


def render_clustering(rows: list[dict[str, object]]) -> None:
    values = np.array([[row["log_s_rob"]] for row in rows], dtype=float)
    tree = linkage(values, method="ward", metric="euclidean")

    fig = plt.figure(figsize=(7.5, 7.0))
    grid = fig.add_gridspec(3, 2, width_ratios=[0.18, 1], height_ratios=[0.32, 0.035, 1],
                            left=0.22, right=0.89, bottom=0.20, top=0.96, wspace=0.02, hspace=0.03)
    ax_dend = fig.add_subplot(grid[0, 1])
    dend = dendrogram(tree, ax=ax_dend, no_labels=True, color_threshold=0,
                      above_threshold_color="#4D4D4D", link_color_func=lambda _: "#4D4D4D")
    order = dend["leaves"]
    ordered = [rows[index] for index in order]
    ax_dend.set_ylabel("Ward distance")
    ax_dend.set_xticks([])
    ax_dend.spines[["top", "right", "bottom"]].set_visible(False)

    ax_top = fig.add_subplot(grid[1, 1])
    task_order = list(STYLES)
    task_index = {task: index for index, task in enumerate(task_order)}
    strip = np.array([[task_index[row["benchmark"]] for row in ordered]])
    ax_top.imshow(strip, aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([STYLES[task][1] for task in task_order]), vmin=0, vmax=len(task_order)-1)
    ax_top.set_axis_off()

    ax_matrix = fig.add_subplot(grid[2, 1])
    ordered_values = np.array([row["log_s_rob"] for row in ordered])
    distances = np.abs(ordered_values[:, None] - ordered_values[None, :])
    image = ax_matrix.imshow(distances, cmap="Greys", interpolation="nearest", vmin=0)
    xlabels = [row["short_model"] for row in ordered]
    ylabels = [f"{row['task_label']} · {row['short_model']}" for row in ordered]
    ax_matrix.set_xticks(range(len(xlabels)), xlabels, rotation=48, ha="right", rotation_mode="anchor")
    ax_matrix.set_yticks(range(len(ylabels)), ylabels)
    ax_matrix.tick_params(length=0)
    for tick, row in zip(ax_matrix.get_xticklabels(), ordered):
        tick.set_color(STYLES[row["benchmark"]][1])
    for tick, row in zip(ax_matrix.get_yticklabels(), ordered):
        tick.set_color(STYLES[row["benchmark"]][1])
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.91, 0.28, 0.022, 0.38])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(r"Pairwise distance $|\Delta\log_{10}S_{\mathrm{rob}}|$")
    colorbar.outline.set_visible(False)
    fig.legend(handles=task_legend(), loc="upper center", bbox_to_anchor=(0.56, 1.015),
               ncol=4, frameon=False, columnspacing=1.4, handletextpad=0.45)
    prefix = PLOTS / "srob_task_hierarchical"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_task_lanes(rows: list[dict[str, object]]) -> None:
    """Show the raw one-dimensional task organization without a dendrogram."""
    task_order = ["harmful", "truthfulness", "mgsm"]
    y_positions = {task: len(task_order) - 1 - index for index, task in enumerate(task_order)}
    fig, ax = plt.subplots(figsize=(7.7, 4.7))

    for task in task_order:
        selected = sorted((row for row in rows if row["benchmark"] == task), key=lambda row: row["s_rob_value"])
        y = y_positions[task]
        color = STYLES[task][1]
        values = np.array([row["s_rob_value"] for row in selected])
        ax.axhspan(y - 0.31, y + 0.31, color=color, alpha=0.045, zorder=0)
        ax.plot([values.min(), values.max()], [y, y], color=color, linewidth=9,
                alpha=0.22, solid_capstyle="round", zorder=1)
        median = float(np.median(values))
        ax.plot([median, median], [y - 0.20, y + 0.20], color=color,
                linewidth=2.4, solid_capstyle="round", zorder=2)

        offsets = np.linspace(-0.11, 0.11, len(selected)) if len(selected) > 1 else np.array([0.0])
        for index, (row, offset) in enumerate(zip(selected, offsets)):
            _, _, marker = STYLES[task]
            point_y = y + float(offset)
            ax.scatter(row["s_rob_value"], point_y, s=82, marker=marker,
                       facecolor=color, edgecolor="white", linewidth=0.85, zorder=3)
            vertical = 9 if index % 2 == 0 else -11
            valign = "bottom" if vertical > 0 else "top"
            ax.annotate(row["short_model"], (row["s_rob_value"], point_y),
                        xytext=(0, vertical), textcoords="offset points",
                        ha="center", va=valign, fontsize=8.3, color="black")

    ax.set_xscale("log")
    ax.set_xlim(0.009, 15.5)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                  ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel(r"Robust steerability $S_{\mathrm{rob}}=1/\gamma^\star$ (log scale)")
    ax.set_yticks(
        [y_positions[task] for task in task_order],
        [f"{STYLES[task][0]}  ($n={sum(row['benchmark'] == task for row in rows)}$)" for task in task_order],
    )
    for tick, task in zip(ax.get_yticklabels(), task_order):
        tick.set_color(STYLES[task][1])
        tick.set_fontweight("bold")
    ax.set_ylim(-0.55, 3.55)
    ax.grid(True, axis="x", which="major", color="#D9DDE1", linewidth=0.7, alpha=0.8)
    ax.grid(False, axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.text(0.01, -0.47, "Thick segment: observed task range     Short vertical line: median",
            fontsize=8.7, color="#4D4D4D", ha="left", va="bottom")
    fig.subplots_adjust(left=0.20, right=0.985, top=0.96, bottom=0.19)
    prefix = PLOTS / "srob_task_clusters"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_parameter_scaling(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.9))

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_key"]].append(row)
    for group in by_model.values():
        if len(group) > 1:
            group = sorted(group, key=lambda row: row["s_rob_value"])
            ax.plot([row["parameter_billions"] for row in group],
                    [row["s_rob_value"] for row in group],
                    color="#B8BDC3", linestyle="--", linewidth=1.1, zorder=1)

    annotation_offsets = [(5, 6), (5, -12), (-5, 7), (-5, -12)]
    special_offsets = {
        ("truthfulness", "qwen25_14b"): (-5, 9),
        ("truthfulness", "olmo2_32b"): (-5, 9),
        ("mgsm", "qwen3_4b"): (-5, 9),
    }
    task_counts = defaultdict(int)
    for row in rows:
        label, color, marker = STYLES[row["benchmark"]]
        ax.scatter(row["parameter_billions"], row["s_rob_value"], s=76, marker=marker,
                   facecolor=color, edgecolor="white", linewidth=0.8, zorder=3)
        index = task_counts[row["benchmark"]]
        task_counts[row["benchmark"]] += 1
        dx, dy = special_offsets.get(
            (row["benchmark"], row["model_key"]),
            annotation_offsets[index % len(annotation_offsets)],
        )
        align = "left" if dx > 0 else "right"
        ax.annotate(row["short_model"], (row["parameter_billions"], row["s_rob_value"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8.2,
                    ha=align, va="center", color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameter count (billions)")
    ax.set_ylabel(r"Robust steerability $S_{\mathrm{rob}}=1/\gamma^\star$")
    ax.set_xticks([1, 2, 3, 4, 8, 14, 32], ["1", "2", "3", "4", "8", "14", "32"])
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10], ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"])
    ax.grid(True, which="major", color="#D9DDE1", linewidth=0.65, alpha=0.75)
    ax.grid(False, which="minor")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=task_legend(), loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, frameon=False, columnspacing=1.25, handletextpad=0.4)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.24)
    prefix = PLOTS / "srob_parameter_size"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def truthfulness_extended_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [dict(row, paper_status="final_reported") for row in rows if row["benchmark"] == "truthfulness"]
    sys.path.insert(0, str(UNIT))
    from gamma_star_organization import extract_calibrations
    _, all_selections, _ = extract_calibrations()
    parameter_fallback = {
        row["model_key"]: row["parameter_count"]
        for row in all_selections if row.get("parameter_count")
    }
    for row in all_selections:
        expected = TRUTHFULNESS_EXTRA_CALIBRATIONS.get(row["model_key"])
        if row["benchmark"] != "truthfulness" or row["calibration_id"] != expected:
            continue
        # A lambda-first run contains an intermediate lambda-sweep selection
        # and the final Q/Qf selection.  Only the latter defines the controller
        # represented in the SROB figure.
        if "/lambda_sweep/" in row["selection_path"]:
            continue
        parameters = row["parameter_count"] or parameter_fallback[row["model_key"]]
        selected.append({
            **row,
            "parameter_billions": float(parameters) / 1e9,
            "s_rob_value": float(row["s_rob"]),
            "log_s_rob": np.log10(float(row["s_rob"])),
            "short_model": SHORT_MODEL[row["model_key"]],
            "task_label": "Truthfulness",
            "paper_status": "exploratory_exception",
        })
    expected = 6 + 9
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} Truthfulness controllers, found {len(selected)}"
        )
    return selected


def add_logo(ax, x: float, y: float, model_key: str, target_pixels: float = 13) -> None:
    logo = LOGOS.get(model_key)
    if logo is None:
        ax.scatter([x], [y], s=20, facecolor="white", edgecolor="black",
                   linewidth=0.9, zorder=4)
        return
    path = UNIT / "assets/logos" / logo
    image = Image.open(path).convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    array = np.asarray(image)
    zoom = target_pixels / max(image.size)
    artist = AnnotationBbox(OffsetImage(array, zoom=zoom), (x, y),
                            frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=4)
    ax.add_artist(artist)


def render_truthfulness_main(rows: list[dict[str, object]]) -> None:
    selected = [row for row in truthfulness_extended_rows(rows)
                if row["model_key"] != "qwen25_32b"]
    fig, ax = plt.subplots(figsize=(3.27, 2.25))

    log_parameters = np.log10([row["parameter_billions"] for row in selected])
    log_s_rob = np.log10([row["s_rob_value"] for row in selected])
    fit = linregress(log_parameters, log_s_rob)
    x_limits = (0.01, 62)
    fit_x = np.logspace(np.log10(x_limits[0]), np.log10(x_limits[1]), 200)
    fit_y = 10 ** (fit.intercept + fit.slope * np.log10(fit_x))
    ax.plot(fit_x, fit_y, color="black", linewidth=1.35, alpha=0.82,
            zorder=1, label="Log–log linear fit")

    for row in selected:
        x, y = row["parameter_billions"], row["s_rob_value"]
        add_logo(ax, x, y, row["model_key"])
        ax.annotate(row["short_model"], (x, y), xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.2, color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(0.0105, 0.55)
    ax.set_xlabel("Model parameter count (B; log scale)", fontsize=10.5)
    ax.set_ylabel(r"Robustness $S_{\mathrm{rob}}$ (log scale)", fontsize=10.5)
    ax.set_xticks(
        [0.01, 0.03, 0.1, 0.3, 1, 3, 8, 14, 32],
        ["0.01", "0.03", "0.1", "0.3", "1", "3", "8", "14", "32"],
    )
    ax.set_yticks([0.01, 0.03, 0.1, 0.3], ["0.01", "0.03", "0.1", "0.3"])
    ax.grid(True, which="major", color="#D9DDE1", linewidth=0.65, alpha=0.75)
    ax.grid(False, which="minor")
    ax.spines[["top", "right"]].set_visible(False)
    label_x = 12.0
    label_y = 10 ** (fit.intercept + fit.slope * np.log10(label_x))
    ax.text(label_x, label_y, f"slope = {fit.slope:.2f}",
            ha="center", va="bottom", fontsize=7.2, color="black",
            rotation=10)
    ax.text(label_x, label_y, rf"$R^2$ = {fit.rvalue ** 2:.2f}",
            ha="center", va="top", fontsize=7.2, color="black",
            rotation=10)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.98, bottom=0.18)
    prefix = PLOTS / "srob_parameter_size_truthfulness_main"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    source_fields = ["model_key", "model_label", "paper_status", "calibration_id",
                     "parameter_billions", "gamma_star", "s_rob_value", "selection_path"]
    with (PLOTS / "srob_parameter_size_truthfulness_main.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=source_fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda row: row["parameter_billions"]))

    fit_statistics = {
        "n": len(selected),
        "fit": "ordinary least squares on log10-transformed parameter_billions and s_rob_value",
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_value": fit.rvalue,
        "r_squared": fit.rvalue ** 2,
        "two_sided_slope_p_value": fit.pvalue,
        "slope_standard_error": fit.stderr,
    }
    with (PLOTS / "srob_parameter_size_truthfulness_main_fit.json").open("w") as handle:
        json.dump(fit_statistics, handle, indent=2)
        handle.write("\n")


def render_truthfulness_extended_variant(
        selected: list[dict[str, object]], *, metric: str, log_x: bool = True, log_y: bool,
        y_label: str, y_limits: tuple[float, float], y_ticks: list[float],
        y_tick_labels: list[str], output_name: str) -> None:
    """Render a metric variant without changing the canonical extended plot."""
    fig, ax = plt.subplots(figsize=(3.27, 2.25))
    x_limits = (1.15, 62) if log_x else (0, 52)
    parameters = np.asarray([float(row["parameter_billions"]) for row in selected])
    values = np.asarray([float(row[metric]) for row in selected])
    fit_x_values = np.log10(parameters) if log_x else parameters
    fit_y_values = np.log10(values) if log_y else values
    fit = linregress(fit_x_values, fit_y_values)
    fit_x = (np.logspace(np.log10(x_limits[0]), np.log10(x_limits[1]), 200)
             if log_x else np.linspace(x_limits[0], x_limits[1], 200))
    fit_x_coordinates = np.log10(fit_x) if log_x else fit_x
    fitted_coordinates = fit.intercept + fit.slope * fit_x_coordinates
    fit_y = 10 ** fitted_coordinates if log_y else fitted_coordinates
    ax.plot(fit_x, fit_y, color="black", linewidth=1.35, alpha=0.82, zorder=1)

    log_x_linear_y_label_layout = {
        "gpt2_xl": (9, 15, "left"),
        "gemma2_2b": (9, 0, "left"),
        "qwen25_14b": (-9, 18, "right"),
        "qwen25_32b": (9, 4, "left"),
    }
    fully_linear_label_layout = {
        "gpt2_xl": (9, 20, "left"),
        "gemma2_2b": (9, 0, "left"),
        "qwen25_14b": (9, 0, "left"),
        "qwen25_32b": (9, 4, "left"),
    }
    for row, y in zip(selected, values):
        x = float(row["parameter_billions"])
        add_logo(ax, x, y, row["model_key"])
        if log_y:
            layout = (9, 0, "left")
        elif log_x:
            layout = log_x_linear_y_label_layout.get(row["model_key"], (9, 0, "left"))
        else:
            layout = fully_linear_label_layout.get(row["model_key"], (9, 0, "left"))
        offset, align = layout[:2], layout[2]
        ax.annotate(row["short_model"], (x, y), xytext=offset,
                    textcoords="offset points", ha=align, va="center",
                    fontsize=7.2, color="black")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Model parameter count (B; log scale)" if log_x
                  else "Model parameter count (B)", fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    if log_x:
        ax.set_xticks([1.5, 2, 3, 4, 8, 14, 32], ["1.5", "2", "3", "4", "8", "14", "32"])
    else:
        ax.set_xticks([0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", "50"])
    ax.set_yticks(y_ticks, y_tick_labels)
    ax.grid(True, which="major", color="#D9DDE1", linewidth=0.65, alpha=0.75)
    ax.grid(False, which="minor")
    ax.spines[["top", "right"]].set_visible(False)

    label_x = 12.0 if log_x else 22.0
    label_x_coordinate = np.log10(label_x) if log_x else label_x
    label_coordinate = fit.intercept + fit.slope * label_x_coordinate
    label_y = 10 ** label_coordinate if log_y else label_coordinate
    rotation = -10 if fit.slope < 0 else 10
    slope_text = f"{fit.slope:.3f}" if abs(fit.slope) < 0.01 else f"{fit.slope:.2f}"
    ax.text(label_x, label_y, f"slope = {slope_text}",
            ha="center", va="bottom", fontsize=7.2, color="black", rotation=rotation)
    ax.text(label_x, label_y, rf"$R^2$ = {fit.rvalue ** 2:.2f}",
            ha="center", va="top", fontsize=7.2, color="black", rotation=rotation)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.98, bottom=0.18)
    prefix = PLOTS / output_name
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    fit_statistics = {
        "n": len(selected),
        "x_transform": "log10(parameter_billions)" if log_x else "identity",
        "y_metric": metric,
        "y_transform": "log10" if log_y else "identity",
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_value": fit.rvalue,
        "r_squared": fit.rvalue ** 2,
        "two_sided_slope_p_value": fit.pvalue,
        "slope_standard_error": fit.stderr,
    }
    with (PLOTS / f"{output_name}_fit.json").open("w") as handle:
        json.dump(fit_statistics, handle, indent=2)
        handle.write("\n")


def render_truthfulness_extended_variants(rows: list[dict[str, object]]) -> None:
    selected = truthfulness_extended_rows(rows)
    render_truthfulness_extended_variant(
        selected, metric="gamma_star", log_x=True, log_y=True,
        y_label=r"Gamma star $\gamma^\star$ (log scale)",
        y_limits=(2.0, 100.0), y_ticks=[3, 10, 30, 100],
        y_tick_labels=["3", "10", "30", "100"],
        output_name="srob_parameter_size_truthfulness_extended_gamma_star",
    )
    render_truthfulness_extended_variant(
        selected, metric="s_rob_value", log_x=True, log_y=False,
        y_label=r"Robustness $S_{\mathrm{rob}}$",
        y_limits=(0.0, 0.43), y_ticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        y_tick_labels=["0", "0.1", "0.2", "0.3", "0.4"],
        output_name="srob_parameter_size_truthfulness_extended_linear",
    )
    render_truthfulness_extended_variant(
        selected, metric="s_rob_value", log_x=False, log_y=False,
        y_label=r"Robustness $S_{\mathrm{rob}}$",
        y_limits=(0.0, 0.43), y_ticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        y_tick_labels=["0", "0.1", "0.2", "0.3", "0.4"],
        output_name="srob_parameter_size_truthfulness_extended_fully_linear",
    )


def render_truthfulness_axis_diagnostic(rows: list[dict[str, object]]) -> None:
    selected = [row for row in truthfulness_extended_rows(rows)
                if row["model_key"] != "qwen25_32b"]
    fig, axes = plt.subplots(1, 4, figsize=(13.08, 2.25))
    panel_specs = [
        {
            "ax": axes[0], "log_x": True, "log_y": True,
            "x_limits": (1.15, 62), "y_limits": (0.0105, 0.55),
            "x_ticks": ([1.5, 2, 3, 4, 8, 14, 32], ["1.5", "2", "3", "4", "8", "14", "32"]),
            "y_ticks": ([0.01, 0.03, 0.1, 0.3], ["0.01", "0.03", "0.1", "0.3"]),
            "x_label": "Model parameter count (B; log scale)",
            "y_label": r"Robustness $S_{\mathrm{rob}}$ (log scale)",
            "label_x": 12.0,
        },
        {
            "ax": axes[1], "log_x": True, "log_y": False,
            "x_limits": (1.15, 62), "y_limits": (0.0, 0.43),
            "x_ticks": ([1.5, 2, 3, 4, 8, 14, 32], ["1.5", "2", "3", "4", "8", "14", "32"]),
            "y_ticks": ([0.0, 0.1, 0.2, 0.3, 0.4], ["0", "0.1", "0.2", "0.3", "0.4"]),
            "x_label": "Model parameter count (B; log scale)",
            "y_label": r"Robustness $S_{\mathrm{rob}}$",
            "label_x": 27.0,
        },
        {
            "ax": axes[2], "log_x": False, "log_y": True,
            "x_limits": (0, 42), "y_limits": (0.0105, 0.55),
            "x_ticks": ([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40"]),
            "y_ticks": ([0.01, 0.03, 0.1, 0.3], ["0.01", "0.03", "0.1", "0.3"]),
            "x_label": "Model parameter count (B)",
            "y_label": r"Robustness $S_{\mathrm{rob}}$ (log scale)",
            "label_x": 22.0,
        },
        {
            "ax": axes[3], "log_x": False, "log_y": False,
            "x_limits": (0, 45), "y_limits": (0.0, 0.49),
            "x_ticks": ([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40"]),
            "y_ticks": ([0.0, 0.1, 0.2, 0.3, 0.4], ["0", "0.1", "0.2", "0.3", "0.4"]),
            "x_label": "Model parameter count (B)",
            "y_label": r"Robustness $S_{\mathrm{rob}}$",
            "label_x": 24.0,
        },
    ]
    fit_records = []
    fully_linear_layout = {
        "gpt2_xl": (9, 20, "left"),
        "gemma2_2b": (9, 0, "left"),
        "qwen25_14b": (9, 0, "left"),
    }
    log_x_linear_y_layout = {
        "gpt2_xl": (9, 15, "left"),
        "gemma2_2b": (9, 0, "left"),
        "qwen25_14b": (-9, 18, "right"),
    }
    linear_x_log_y_layout = {
        "gpt2_xl": (9, 12, "left"),
        "gemma2_2b": (9, -5, "left"),
        "qwen25_14b": (9, 0, "left"),
    }

    parameters = np.asarray([float(row["parameter_billions"]) for row in selected])
    s_rob = np.asarray([float(row["s_rob_value"]) for row in selected])
    trend_test = kendalltau(parameters, s_rob, alternative="greater", method="exact")
    total_pairs = len(selected) * (len(selected) - 1) // 2
    positive_pairs = int(sum(
        (parameters[j] - parameters[i]) * (s_rob[j] - s_rob[i]) > 0
        for i in range(len(selected)) for j in range(i + 1, len(selected))
    ))
    negative_pairs = total_pairs - positive_pairs
    for spec in panel_specs:
        ax = spec["ax"]
        fit_x_values = np.log10(parameters) if spec["log_x"] else parameters
        fit_y_values = np.log10(s_rob) if spec["log_y"] else s_rob
        fit = linregress(fit_x_values, fit_y_values)
        if spec["log_x"]:
            fit_x = np.logspace(np.log10(spec["x_limits"][0]),
                                np.log10(spec["x_limits"][1]), 200)
            fit_coordinates = np.log10(fit_x)
        else:
            fit_x = np.linspace(spec["x_limits"][0], spec["x_limits"][1], 200)
            fit_coordinates = fit_x
        fitted_y_coordinates = fit.intercept + fit.slope * fit_coordinates
        fit_y = 10 ** fitted_y_coordinates if spec["log_y"] else fitted_y_coordinates
        ax.plot(fit_x, fit_y, color="black", linewidth=1.35, alpha=0.82, zorder=1)

        for row, y in zip(selected, s_rob):
            x = float(row["parameter_billions"])
            add_logo(ax, x, y, row["model_key"])
            if spec["log_x"] and spec["log_y"]:
                layout = (9, 0, "left")
            elif spec["log_x"]:
                layout = log_x_linear_y_layout.get(row["model_key"], (9, 0, "left"))
            elif spec["log_y"]:
                layout = linear_x_log_y_layout.get(row["model_key"], (9, 0, "left"))
            else:
                layout = fully_linear_layout.get(row["model_key"], (9, 0, "left"))
            ax.annotate(row["short_model"], (x, y), xytext=layout[:2],
                        textcoords="offset points", ha=layout[2], va="center",
                        fontsize=7.2, color="black")

        if spec["log_x"]:
            ax.set_xscale("log")
        if spec["log_y"]:
            ax.set_yscale("log")
        ax.set_xlim(*spec["x_limits"])
        ax.set_ylim(*spec["y_limits"])
        ax.set_xticks(*spec["x_ticks"])
        ax.set_yticks(*spec["y_ticks"])
        ax.set_xlabel(spec["x_label"], fontsize=10.5)
        ax.set_ylabel(spec["y_label"], fontsize=10.5)
        ax.set_title(
            rf"Kendall $\tau$ = {trend_test.statistic:.2f}; "
            rf"Mann–Kendall $p_+$ = {trend_test.pvalue:.3f}",
                     fontsize=8.2, color="black", pad=5)
        ax.grid(True, which="major", color="#D9DDE1", linewidth=0.65, alpha=0.75)
        ax.grid(False, which="minor")
        ax.spines[["top", "right"]].set_visible(False)

        label_coordinate = (np.log10(spec["label_x"]) if spec["log_x"]
                            else spec["label_x"])
        label_y_coordinate = fit.intercept + fit.slope * label_coordinate
        label_y = 10 ** label_y_coordinate if spec["log_y"] else label_y_coordinate
        slope_text = f"{fit.slope:.3f}" if abs(fit.slope) < 0.01 else f"{fit.slope:.2f}"
        ax.text(spec["label_x"], label_y, f"slope = {slope_text}",
                ha="center", va="bottom", fontsize=7.2, color="black", rotation=10)
        ax.text(spec["label_x"], label_y, rf"$R^2$ = {fit.rvalue ** 2:.2f}",
                ha="center", va="top", fontsize=7.2, color="black", rotation=10)
        fit_records.append({
            "coordinates": ("log-log" if spec["log_x"] and spec["log_y"] else
                            "log-linear" if spec["log_x"] else
                            "linear-log" if spec["log_y"] else "linear-linear"),
            "n": len(selected), "excluded_model": "qwen25_32b",
            "slope": fit.slope, "intercept": fit.intercept,
            "r_value": fit.rvalue, "r_squared": fit.rvalue ** 2,
            "two_sided_slope_p_value": fit.pvalue,
            "slope_standard_error": fit.stderr,
            "kendall_positive_pairs": positive_pairs,
            "kendall_negative_pairs": negative_pairs,
            "kendall_total_pairs": total_pairs,
            "positive_pair_probability": positive_pairs / total_pairs,
            "kendall_tau": float(trend_test.statistic),
            "one_sided_mann_kendall_positive_trend_p_value": float(trend_test.pvalue),
        })

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.20, wspace=0.34)
    prefix = PLOTS / "srob_parameter_size_truthfulness_axis_diagnostic_1x4"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    with (PLOTS / "srob_parameter_size_truthfulness_axis_diagnostic_1x4_fit.json").open("w") as handle:
        json.dump(fit_records, handle, indent=2)
        handle.write("\n")


def silhouette_score_1d(values: np.ndarray, labels: np.ndarray) -> float:
    """Return the ordinary mean silhouette using absolute 1-D distance."""
    scores = []
    for index, value in enumerate(values):
        own = labels == labels[index]
        own[index] = False
        if not own.any():
            scores.append(0.0)
            continue
        a = float(np.abs(values[own] - value).mean())
        b = min(
            float(np.abs(values[labels == label] - value).mean())
            for label in np.unique(labels) if label != labels[index]
        )
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def exact_task_label_null(values: np.ndarray, group_sizes: tuple[int, int, int]) -> np.ndarray:
    """Enumerate the exact labeled null while evaluating silhouettes in batches."""
    n = len(values)
    first_size, second_size, third_size = group_sizes
    if first_size + second_size + third_size != n:
        raise ValueError("task group sizes do not sum to the number of observations")
    distances = np.abs(values[:, None] - values[None, :])
    all_indices = tuple(range(n))
    chunks: list[np.ndarray] = []
    pending: list[np.ndarray] = []

    def evaluate(label_rows: list[np.ndarray]) -> np.ndarray:
        labels = np.stack(label_rows)
        means = []
        for label, size in enumerate(group_sizes):
            membership = labels == label
            means.append((membership.astype(float) @ distances.T) / size)
        own = np.choose(labels, means)
        own_denominators = np.choose(labels, [size - 1 for size in group_sizes])
        own = own * np.choose(labels, group_sizes) / own_denominators
        other = np.empty_like(own)
        for label in range(3):
            candidates = [means[other_label] for other_label in range(3) if other_label != label]
            mask = labels == label
            other[mask] = np.minimum(candidates[0], candidates[1])[mask]
        denominator = np.maximum(own, other)
        scores = np.divide(other - own, denominator, out=np.zeros_like(own), where=denominator > 0)
        return scores.mean(axis=1)

    for first_indices in itertools.combinations(all_indices, first_size):
        first = set(first_indices)
        remainder = tuple(index for index in all_indices if index not in first)
        for second_indices in itertools.combinations(remainder, second_size):
            labels = np.full(n, 2, dtype=np.int8)
            labels[list(first_indices)] = 0
            labels[list(second_indices)] = 1
            pending.append(labels)
            if len(pending) == 4096:
                chunks.append(evaluate(pending))
                pending.clear()
    if pending:
        chunks.append(evaluate(pending))
    return np.concatenate(chunks)


def optimal_1d_kmeans_labels(values: np.ndarray, cluster_count: int) -> np.ndarray:
    """Find the global minimum-SSE contiguous partition for one-dimensional k-means."""
    order = np.argsort(values)
    sorted_values = values[order]
    best_sse = np.inf
    best_labels = None
    for cuts in itertools.combinations(range(1, len(values)), cluster_count - 1):
        boundaries = (0, *cuts, len(values))
        labels = np.empty(len(values), dtype=int)
        sse = 0.0
        for label, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            segment = sorted_values[start:stop]
            sse += float(np.square(segment - segment.mean()).sum())
            labels[order[start:stop]] = label
        if sse < best_sse:
            best_sse = sse
            best_labels = labels
    return best_labels


def truthfulness_txi_improvement_filter() -> tuple[set[str], list[dict[str, object]]]:
    """Return new model keys whose H-infinity T×I exceeds Original T×I."""
    sys.path.insert(0, str(UNIT))
    from build_truthfulness_comparison import MODELS, metric, result_path

    bases = {model: base for model, base in MODELS}
    included = set()
    records = []
    for model_key in TRUTHFULNESS_NEW_MODEL_KEYS:
        model = SHORT_MODEL[model_key]
        base = bases[model]
        original = json.loads(result_path(model, base, "original").read_text())
        h_infinity = json.loads(result_path(model, base, "h_infinity").read_text())
        original_txi = float(metric(original, "txi")[0])
        h_infinity_txi = float(metric(h_infinity, "txi")[0])
        improvement = h_infinity_txi - original_txi
        passes = improvement > 0.0
        if passes:
            included.add(model_key)
        records.append({
            "model_key": model_key,
            "model": model,
            "original_txi": original_txi,
            "h_infinity_txi": h_infinity_txi,
            "h_infinity_minus_original_txi": improvement,
            "included": passes,
        })
    return included, records


def render_overall_summary(
    rows: list[dict[str, object]],
    *,
    output_name: str = "srob_overall",
    included_new_model_keys: set[str] | None = None,
) -> None:
    """Render the task distribution, separation diagnostics, and Truthfulness scaling."""
    truthfulness = [
        row for row in truthfulness_extended_rows(rows)
        if row["model_key"] != "qwen25_32b"
    ]
    if included_new_model_keys is not None:
        truthfulness = [
            row for row in truthfulness
            if row["model_key"] not in TRUTHFULNESS_NEW_MODEL_KEYS
            or row["model_key"] in included_new_model_keys
        ]
    combined = [
        dict(row, paper_status="final_reported")
        for row in rows if row["benchmark"] != "truthfulness"
    ] + truthfulness
    task_order = ["truthfulness", "harmful", "mgsm"]
    task_names = {"truthfulness": "Truthful", "harmful": "Harmful", "mgsm": "MGSM"}
    log_s_rob = np.asarray([float(row["log_s_rob"]) for row in combined])

    cluster_counts = np.arange(2, 7)
    cluster_silhouettes = []
    for cluster_count in cluster_counts:
        labels = optimal_1d_kmeans_labels(log_s_rob, int(cluster_count))
        cluster_silhouettes.append(silhouette_score_1d(log_s_rob, labels))

    task_labels = np.asarray([task_order.index(row["benchmark"]) for row in combined])
    observed_silhouette = silhouette_score_1d(log_s_rob, task_labels)
    group_sizes = tuple(sum(row["benchmark"] == task for row in combined) for task in task_order)
    null_silhouettes = exact_task_label_null(log_s_rob, group_sizes)
    exact_p_value = float(np.mean(null_silhouettes >= observed_silhouette))

    fig = plt.figure(figsize=(10.6, 3.35))
    grid = fig.add_gridspec(
        1, 3, width_ratios=[0.68, 0.50, 1.62], wspace=0.48,
        left=0.065, right=0.992, bottom=0.19, top=0.91,
    )
    ax_task = fig.add_subplot(grid[0, 0])
    for position, task in enumerate(task_order, start=1):
        selected = [row for row in combined if row["benchmark"] == task]
        values = np.asarray([float(row["s_rob_value"]) for row in selected])
        ax_task.boxplot(
            [values], positions=[position], widths=0.25, patch_artist=True, showfliers=False,
            boxprops={"facecolor": "#D5D8DA", "edgecolor": "black", "linewidth": 1.45},
            medianprops={"color": "black", "linewidth": 2.0},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
        )
        offsets = np.linspace(-0.105, 0.105, len(selected)) if len(selected) > 1 else [0.0]
        for row, offset in zip(sorted(selected, key=lambda item: item["s_rob_value"]), offsets):
            x = float(position) + float(offset)
            y = float(row["s_rob_value"])
            ax_task.scatter(
                x, y, s=28, marker=STYLES[task][2], facecolor=STYLES[task][1],
                edgecolor="white", linewidth=0.65, alpha=0.92, zorder=4,
            )
    ax_task.set_yscale("log")
    ax_task.set_ylim(0.008, 40)
    ax_task.set_xlim(0.72, 3.62)
    ax_task.set_ylabel(r"Robustness $S_{\mathrm{rob}}$ (log scale)")
    ax_task.set_xticks(range(1, 4), [task_names[task] for task in task_order])
    ax_task.grid(axis="y", which="major", color="#DDE1E4", linewidth=0.7)
    ax_task.spines[["top", "right"]].set_visible(False)

    middle = grid[0, 1].subgridspec(2, 1, hspace=0.72)
    ax_clusters = fig.add_subplot(middle[0, 0])
    selected_cluster_count = int(cluster_counts[np.argmax(cluster_silhouettes)])
    selected_cluster_score = float(np.max(cluster_silhouettes))
    ax_clusters.bar(
        cluster_counts, cluster_silhouettes, width=0.7,
        color=["#007C7C" if value == selected_cluster_count else "#AEB4B9"
               for value in cluster_counts],
        edgecolor="white", linewidth=0.7,
    )
    ax_clusters.set_xlim(1.45, 6.55)
    ax_clusters.set_xticks(cluster_counts)
    ax_clusters.set_xlabel("Number of clusters, k")
    ax_clusters.set_ylabel("Silhouette score")
    ax_clusters.set_ylim(0, 0.68)
    ax_clusters.grid(axis="y", color="#E2E5E8", linewidth=0.7)
    ax_clusters.set_axisbelow(True)
    ax_clusters.spines[["top", "right"]].set_visible(False)
    ax_clusters.text(
        selected_cluster_count, selected_cluster_score + 0.02,
        f"{selected_cluster_score:.2f}",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#007C7C",
    )

    ax_null = fig.add_subplot(middle[1, 0])
    ax_null.hist(
        null_silhouettes, bins=28, density=True, color="#CFD3D6",
        edgecolor="white", linewidth=0.5,
    )
    ax_null.axvline(observed_silhouette, color="#007C7C", linewidth=2.5)
    ax_null.set_xlim(-0.31, 0.40)
    ax_null.set_ylim(0, 8)
    ax_null.set_xlabel("Silhouette score")
    ax_null.set_ylabel("Density")
    ax_null.spines[["top", "right"]].set_visible(False)
    ax_null.text(
        -0.29, 7.45, "Task-label shuffle\ndistribution",
        ha="left", va="top", fontsize=7.3, color="black",
    )
    ax_null.text(
        observed_silhouette + 0.018, 4.25,
        f"Task-label\nobservation\np = {exact_p_value:.3f}",
        ha="left", va="center", fontsize=7.5,
        color="#007C7C", fontweight="bold",
    )

    ax_scale = fig.add_subplot(grid[0, 2])
    parameters = np.asarray([float(row["parameter_billions"]) for row in truthfulness])
    truth_s_rob = np.asarray([float(row["s_rob_value"]) for row in truthfulness])
    fit = linregress(np.log10(parameters), np.log10(truth_s_rob))
    # The filtered variant no longer contains the very small Pythia points.
    # Start its scaling axis just before the smallest retained model rather
    # than inheriting the full figure's otherwise empty 0.01--0.1B decade.
    scale_x_min = 0.01 if included_new_model_keys is None else 0.10
    fit_x = np.logspace(np.log10(scale_x_min), np.log10(62), 200)
    fit_y = 10 ** (fit.intercept + fit.slope * np.log10(fit_x))
    ax_scale.plot(fit_x, fit_y, color="black", linewidth=1.35, alpha=0.82, zorder=1)
    label_layout = {
        "pythia_14m": (8, 0, "left"),
        "pythia_31m": (7, 9, "left"),
        "pythia_160m": (7, -10, "left"),
        "distilgpt2": (-8, 8, "right"),
        "gpt2_small": (8, 6, "left"),
        "smollm2_135m": (8, -4, "left"),
        "gpt2_medium": (-8, -8, "right"),
        "qwen25_05b": (8, 7, "left"),
        "gpt2_large": (8, -6, "left"),
        "gpt2_xl": (-8, 8, "right"),
        "gemma2_2b": (8, -8, "left"),
        "llama31_8b": (8, 0, "left"),
        "qwen25_14b": (-8, 8, "right"),
        "olmo2_32b": (-8, 9, "right"),
    }
    for row in truthfulness:
        x = float(row["parameter_billions"])
        y = float(row["s_rob_value"])
        add_logo(ax_scale, x, y, row["model_key"], target_pixels=15)
        dx, dy, alignment = label_layout[row["model_key"]]
        ax_scale.annotate(
            row["short_model"], (x, y), xytext=(dx, dy), textcoords="offset points",
            ha=alignment, va="center", fontsize=6.5, color="black",
        )
    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")
    ax_scale.set_xlim(scale_x_min, 62)
    scale_y_max = 40 if included_new_model_keys is None else 1.0
    ax_scale.set_ylim(0.008, scale_y_max)
    ax_scale.set_xlabel("Model parameter count (B; log scale)", fontsize=10.5)
    ax_scale.set_ylabel(r"Robustness $S_{\mathrm{rob}}$ (log scale)", fontsize=10.5)
    x_ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 8, 14, 32]
    x_tick_labels = ["0.01", "0.03", "0.1", "0.3", "1", "3", "8", "14", "32"]
    visible_ticks = [
        (tick, label) for tick, label in zip(x_ticks, x_tick_labels)
        if tick >= scale_x_min
    ]
    ax_scale.set_xticks(
        [tick for tick, _ in visible_ticks],
        [label for _, label in visible_ticks],
    )
    y_ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    y_tick_labels = ["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30"]
    visible_y_ticks = [
        (tick, label) for tick, label in zip(y_ticks, y_tick_labels)
        if tick <= scale_y_max
    ]
    ax_scale.set_yticks(
        [tick for tick, _ in visible_y_ticks],
        [label for _, label in visible_y_ticks],
    )
    ax_scale.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_scale.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_scale.grid(True, which="major", color="#D9DDE1", linewidth=0.65, alpha=0.75)
    ax_scale.grid(False, which="minor")
    ax_scale.spines[["top", "right"]].set_visible(False)
    label_x = 3.7
    label_y = 10 ** (fit.intercept + fit.slope * np.log10(label_x))
    fig.canvas.draw()
    angle_start = ax_scale.transData.transform(
        (8.0, 10 ** (fit.intercept + fit.slope * np.log10(8.0)))
    )
    angle_stop = ax_scale.transData.transform(
        (20.0, 10 ** (fit.intercept + fit.slope * np.log10(20.0)))
    )
    line_angle = float(np.degrees(np.arctan2(
        angle_stop[1] - angle_start[1], angle_stop[0] - angle_start[0]
    )))
    ax_scale.text(label_x, label_y, f"slope = {fit.slope:.2f}", ha="center", va="bottom",
                  fontsize=7.2, color="black", rotation=line_angle, rotation_mode="anchor")
    ax_scale.text(label_x, label_y, rf"$R^2$ = {fit.rvalue ** 2:.2f}", ha="center", va="top",
                  fontsize=7.2, color="black", rotation=line_angle, rotation_mode="anchor")

    prefix = PLOTS / output_name
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    with (PLOTS / f"{output_name}_points.csv").open("w", newline="") as handle:
        fields = ["benchmark", "model_key", "short_model", "parameter_billions",
                  "gamma_star", "s_rob_value", "paper_status"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(combined, key=lambda row: (task_order.index(row["benchmark"]),
                                                           row["s_rob_value"])))
    statistics = {
        "feature": "log10(s_rob)",
        "cluster_method": "global minimum-SSE one-dimensional k-means",
        "n": len(combined),
        "task_group_sizes": {task: sum(row["benchmark"] == task for row in combined)
                             for task in task_order},
        "silhouette_by_k": {str(k): score for k, score in zip(cluster_counts, cluster_silhouettes)},
        "selected_k_by_maximum_silhouette": int(cluster_counts[np.argmax(cluster_silhouettes)]),
        "observed_task_label_silhouette": observed_silhouette,
        "permutation_null": (
            "exact enumeration of all labeled "
            + "/".join(str(size) for size in group_sizes)
            + " assignments"
        ),
        "permutation_count": len(null_silhouettes),
        "exact_upper_tail_p_value": exact_p_value,
    }
    if included_new_model_keys is not None:
        statistics["new_truthfulness_model_filter"] = {
                "criterion": "evaluated H-infinity mean T×I > evaluated Original mean T×I",
                "applies_only_to": list(TRUTHFULNESS_NEW_MODEL_KEYS),
                "included": sorted(included_new_model_keys),
        }
    with (PLOTS / f"{output_name}_statistics.json").open("w") as handle:
        json.dump(statistics, handle, indent=2)
        handle.write("\n")


def main() -> None:
    configure()
    rows = load_rows()
    render_overall_summary(rows)
    included, records = truthfulness_txi_improvement_filter()
    with (PLOTS / "truthfulness_txi_improvement_filter.json").open("w") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    render_overall_summary(
        rows,
        output_name="srob_overall_txi_improved",
        included_new_model_keys=included,
    )


if __name__ == "__main__":
    main()
