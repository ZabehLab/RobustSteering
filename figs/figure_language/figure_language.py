#!/usr/bin/env python3
"""Render the MGSM language-shift figure with model logos and method colors."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    HPacker,
    OffsetImage,
    TextArea,
)
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = ROOT / "figs" / "bench_table" / "mgsm" / "mgsm_full.md"
LOGO_DIR = ROOT / "figs" / "logos"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

LANGUAGES = ["Chinese", "French", "Japanese", "Swahili", "Telugu"]
MODELS = ["Qwen3-4B", "Phi-4-mini", "Granite-3.3-2B"]
METHODS = ["Original", "A-LQR", "H-infinity"]
COMPARISON_METHODS = ["A-LQR", "H-infinity"]

METHOD_LABELS = {
    "Original": "Original",
    "A-LQR": "A-LQR",
    "H-infinity": r"$H_\infty$ (ours)",
}

# H-infinity uses the project teal; baselines are deliberately quieter.
METHOD_COLORS = {
    "Original": "#72777F",
    "A-LQR": "#B8BEC2",
    "H-infinity": "#007C7C",
}

MODEL_LOGOS = {
    "Qwen3-4B": LOGO_DIR / "qwen_transparent.png",
    "Phi-4-mini": LOGO_DIR / "microsoft_transparent.png",
    "Granite-3.3-2B": LOGO_DIR / "ibm_transparent.png",
}

INK = "#000000"
MUTED = "#000000"
GRID = "#E2E5E8"
SPINE = "#AEB4BA"


def configure_style(font_scale: float = 1.0) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 11.0 * font_scale,
            "axes.labelsize": 14.0 * font_scale,
            "axes.titlesize": 14.5 * font_scale,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in SOURCE_DATA.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "±" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7:
            continue
        model, language, method, accuracy, _, instruction_relevance, fluency = cells
        canonical_method = "H-infinity" if method.startswith("H∞") else method
        if model not in MODELS or canonical_method not in METHODS:
            continue

        def mean(value: str) -> float:
            return float(value.split("±", maxsplit=1)[0].strip())

        rows.append(
            {
                "model": model,
                "language": language,
                "method": canonical_method,
                "accuracy": mean(accuracy),
                "instruction_relevance": mean(instruction_relevance),
                "fluency": mean(fluency),
            }
        )
    expected = len(MODELS) * len(METHODS) * len(LANGUAGES)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} final-table rows, found {len(rows)}")
    return rows


def tinted_logo(path: Path, color: str, *, full_alpha: bool = False) -> np.ndarray:
    """Use the logo alpha/luminance as a mask and tint it with a method color."""
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image, dtype=np.float32) / 255.0
    alpha = array[..., 3]
    luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    # Transparent assets may contain dark or colored logos. Preserve their shape,
    # including antialiased edges, without placing them inside generic markers.
    mask = alpha if full_alpha else alpha * np.clip(1.25 - luminance, 0.28, 1.0)
    rgb = np.array(mpl.colors.to_rgb(color), dtype=np.float32)
    output = np.empty_like(array)
    output[..., :3] = rgb
    output[..., 3] = mask
    return output


def add_logo(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    model: str,
    method: str,
    zoom: float,
    full_alpha: bool = False,
) -> None:
    image = tinted_logo(
        MODEL_LOGOS[model],
        METHOD_COLORS[method],
        full_alpha=full_alpha,
    )
    artist = AnnotationBbox(
        OffsetImage(image, zoom=zoom, interpolation="lanczos"),
        (x, y),
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        zorder=5 if method == "H-infinity" else 4,
    )
    ax.add_artist(artist)


def add_native_logo(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    model: str,
    zoom: float,
) -> None:
    image = np.asarray(Image.open(MODEL_LOGOS[model]).convert("RGBA"))
    if model == "Granite-3.3-2B":
        image = tinted_logo(MODEL_LOGOS[model], "#0F62FE")
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(image, zoom=zoom, interpolation="lanczos"),
            (x, y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
            zorder=6,
        )
    )


def style_axis(ax: plt.Axes, font_scale: float = 1.0) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(
        length=3.5 * font_scale,
        width=0.8,
        labelsize=10.5 * font_scale,
    )


def macro_summary(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]))].append(row)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        summary[key] = {
            "accuracy": float(np.mean([float(row["accuracy"]) for row in values])),
            # Both AXBench components are on 0--2 scales, so the literal product
            # occupies 0--4. Averaging per-language products avoids inventing
            # response-level data that are absent from the frozen figure source.
            "ir_x_fluency": float(
                np.mean(
                    [
                        float(row["instruction_relevance"]) * float(row["fluency"])
                        for row in values
                    ]
                )
            ),
        }
    return summary


def add_legends(
    fig: plt.Figure,
    *,
    methods: list[str] = METHODS,
    dot_handles: bool = False,
    font_scale: float = 1.0,
    legend_y: float = 0.008,
    grouped_model_header: bool = False,
) -> None:
    model_handles = []
    for model in MODELS:
        image = tinted_logo(MODEL_LOGOS[model], INK)
        model_handles.append(
            Line2D([], [], linestyle="none", marker="", label=model)
        )

    # Matplotlib legends cannot natively use arbitrary logo images cleanly, so
    # position compact, untinted model logos and names directly above the panels.
    if grouped_model_header:
        legend_bounds = [0.075, 0.90, 0.60, 0.085]
    else:
        legend_bounds = (
            [0.25, 0.90, 0.54, 0.085]
            if font_scale > 1.0
            else [0.32, 0.91, 0.36, 0.06]
        )
    legend_ax = fig.add_axes(legend_bounds)
    legend_ax.axis("off")
    model_groups = []
    for model in MODELS:
        # The legend shows the official company artwork in its native colors;
        # only the data marks are recolored to encode method identity.
        image = np.asarray(Image.open(MODEL_LOGOS[model]).convert("RGBA"))
        # The shared IBM SVG/PNG is a currentColor asset stored in black; render
        # its official IBM-blue variant in the company-logo legend.
        if model == "Granite-3.3-2B":
            image = tinted_logo(MODEL_LOGOS[model], "#0F62FE")
        zoom = {
            "Qwen3-4B": 0.022,
            "Phi-4-mini": 0.020,
            "Granite-3.3-2B": 0.026,
        }[model]
        if grouped_model_header:
            zoom *= 1.25
        if grouped_model_header:
            model_groups.append(
                HPacker(
                    children=[
                        OffsetImage(image, zoom=zoom),
                        TextArea(
                            model,
                            textprops={
                                "fontsize": 14.0 * font_scale,
                                "fontfamily": "Arial",
                                "color": INK,
                            },
                        ),
                    ],
                    align="center",
                    pad=0,
                    sep=4,
                )
            )
        else:
            x = {"Qwen3-4B": 0.02, "Phi-4-mini": 0.38, "Granite-3.3-2B": 0.72}[model]
            legend_ax.add_artist(
                AnnotationBbox(
                    OffsetImage(image, zoom=zoom),
                    (x, 0.5),
                    xycoords="axes fraction",
                    frameon=False,
                    pad=0,
                )
            )
            legend_ax.text(
                x + 0.07,
                0.5,
                model,
                va="center",
                ha="left",
                fontsize=10.4 * font_scale,
            )

    if grouped_model_header:
        packed = HPacker(children=model_groups, align="center", pad=0, sep=24)
        legend_ax.add_artist(
            AnchoredOffsetbox(
                loc="center left",
                child=packed,
                pad=0,
                frameon=False,
                bbox_to_anchor=(0.0, 0.5),
                bbox_transform=legend_ax.transAxes,
                borderpad=0,
            )
        )

    if dot_handles:
        method_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=9.5 if method == "H-infinity" else 7.5,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="none",
                label=METHOD_LABELS[method],
            )
            for method in methods
        ]
    else:
        method_handles = [
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                linewidth=4.5 if method == "H-infinity" else 3.2,
                solid_capstyle="round",
                label=METHOD_LABELS[method],
            )
            for method in methods
        ]
    if grouped_model_header:
        fig.legend(
            handles=method_handles,
            loc="center right",
            bbox_to_anchor=(0.988, 0.9425),
            ncol=len(methods),
            frameon=False,
            handlelength=0.7,
            columnspacing=1.5,
            handletextpad=0.25,
            fontsize=14.0 * font_scale,
        )
    else:
        fig.legend(
            handles=method_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(methods),
            frameon=False,
            handlelength=1.7,
            columnspacing=2.0,
            handletextpad=0.55,
            fontsize=11.2 * font_scale,
        )


def accuracy_change(method_accuracy: float, original_accuracy: float) -> float:
    """Absolute accuracy change relative to the same model's Original row."""
    return abs(method_accuracy - original_accuracy)


def render_accuracy_change(
    rows: list[dict[str, object]],
    *,
    relative_right: bool = False,
    expanded: bool = False,
) -> None:
    font_scale = 1.5
    configure_style(font_scale)
    lookup = {
        (str(row["model"]), str(row["method"]), str(row["language"])): row
        for row in rows
    }
    summary = macro_summary(rows)
    display_languages = LANGUAGES[:-1] if relative_right else LANGUAGES
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 3.6) if expanded else (12.0, 5.8),
        gridspec_kw={
            "wspace": 0.32 if expanded else 0.24,
            "width_ratios": [0.72, 2.1],
        },
    )
    left, right = axes

    for model in MODELS:
        original_by_language = {
            language: float(lookup[(model, "Original", language)]["accuracy"])
            for language in LANGUAGES
        }
        method_points: dict[str, tuple[float, float]] = {}
        for method in COMPARISON_METHODS:
            values = [
                accuracy_change(
                    float(lookup[(model, method, language)]["accuracy"]),
                    original_by_language[language],
                )
                for language in LANGUAGES
            ]
            method_points[method] = (
                summary[(model, method)]["ir_x_fluency"],
                float(np.mean(values)),
            )
        left.plot(
            [method_points[method][0] for method in COMPARISON_METHODS],
            [method_points[method][1] for method in COMPARISON_METHODS],
            color="#9DA3A6",
            linewidth=1.2,
            linestyle=(0, (3, 2)),
            zorder=2,
        )
        for method in COMPARISON_METHODS:
            zoom = {
                "Qwen3-4B": 0.020,
                "Phi-4-mini": 0.018,
                "Granite-3.3-2B": 0.023,
            }[model]
            zoom *= 1.10 if method == "H-infinity" else 0.94
            if expanded:
                zoom *= 1.8
            add_logo(
                left,
                method_points[method][0],
                method_points[method][1],
                model=model,
                method=method,
                zoom=zoom,
                full_alpha=relative_right and method == "A-LQR",
            )

    x = np.arange(len(display_languages), dtype=float)
    model_offsets = {"Qwen3-4B": -0.29, "Phi-4-mini": 0.0, "Granite-3.3-2B": 0.29}
    bar_width = 0.105
    method_offsets = {"A-LQR": -bar_width / 2, "H-infinity": bar_width / 2}
    comparison_ymax = 76.0
    right_ymax = 120.0 if relative_right else comparison_ymax
    logo_offset = 0.08 * right_ymax
    for language_index, language in enumerate(display_languages):
        for model in MODELS:
            original = float(lookup[(model, "Original", language)]["accuracy"])
            pair: dict[str, tuple[float, float]] = {}
            for method in COMPARISON_METHODS:
                method_accuracy = float(
                    lookup[(model, method, language)]["accuracy"]
                )
                value = (
                    100.0 * method_accuracy / original
                    if relative_right
                    else accuracy_change(method_accuracy, original)
                )
                bar_x = x[language_index] + model_offsets[model] + method_offsets[method]
                pair[method] = (bar_x, value)
                right.bar(
                    bar_x,
                    value,
                    width=bar_width,
                    color=METHOD_COLORS[method],
                    edgecolor="none",
                    zorder=3,
                )
                right.scatter(
                    [bar_x],
                    [value],
                    s=20,
                    color="#000000",
                    edgecolors="none",
                    zorder=5,
                )
            right.plot(
                [pair[method][0] for method in COMPARISON_METHODS],
                [pair[method][1] for method in COMPARISON_METHODS],
                color="#000000" if expanded else "#858B8E",
                linewidth=1.5 if expanded else 1.0,
                linestyle=(0, (3, 2)),
                zorder=4,
            )
            tallest = max(value for _, value in pair.values())
            zoom = {
                "Qwen3-4B": 0.0198,
                "Phi-4-mini": 0.017325,
                "Granite-3.3-2B": 0.0231,
            }[model]
            add_native_logo(
                right,
                x[language_index] + model_offsets[model],
                -7.5 if relative_right else tallest + logo_offset,
                model=model,
                zoom=zoom,
            )

    left.set_xlabel(r"IR $\times$ fluency")
    left.set_xlim(-0.25 if expanded else -0.12, 2.5)
    if expanded:
        # Preserve the original nonnegative ticks while giving the enlarged
        # Original marker enough breathing room at the left frame.
        left.set_xticks([0, 1, 2])
    right.set_xticks(x, display_languages)
    right.set_xlabel("Transfer language (OOD)" if expanded else "Transfer language")
    right.set_xlim(-0.5, len(display_languages) - 0.5)

    left.set_ylabel(
        "$\\Delta$ Acc. from\nOriginal (%)" if expanded else r"$\Delta$ Acc. from Original (%)"
    )
    left.set_ylim(-3, 50.0 if relative_right else comparison_ymax)
    left.axhline(0.0, color=INK, linewidth=1.2, linestyle="-", zorder=2)
    left.text(
        0.04,
        0.0,
        "Original = 0",
        transform=left.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=12.75,
        color=MUTED,
    )
    style_axis(left, font_scale)
    if expanded:
        left.tick_params(labelsize=14.0 * font_scale)

    if relative_right:
        right.set_ylabel(
            "% Match of\nOriginal Acc." if expanded else "% Match of Original Acc."
        )
        right.set_ylim(-13, right_ymax)
        right.set_yticks([0, 25, 50, 75, 100])
        right.axhline(100.0, color=INK, linewidth=1.2, linestyle="-", zorder=2)
        right.text(
            0.98,
            102.5,
            "Original = 100%",
            transform=right.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=12.75,
            color=MUTED,
        )
    else:
        right.set_ylabel(r"$\Delta$ Acc. from Original (%)")
        right.set_ylim(-3, comparison_ymax)
        right.axhline(0.0, color=INK, linewidth=1.2, linestyle="-", zorder=2)
    style_axis(right, font_scale)
    if expanded:
        right.tick_params(labelsize=14.0 * font_scale)
        right.xaxis.label.set_color("#8B1E1E")
        right.xaxis.label.set_fontweight("bold")
        for label in right.get_xticklabels():
            label.set_color("#8B1E1E")
            label.set_fontweight("bold")

    add_legends(
        fig,
        methods=COMPARISON_METHODS,
        dot_handles=True,
        font_scale=font_scale,
        legend_y=0.055 if relative_right else 0.008,
        grouped_model_header=expanded,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.988,
        bottom=0.30,
        top=0.86,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_language_relative" if relative_right else "figure_language_delta"
    if expanded:
        stem += "_expanded"
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def render() -> None:
    configure_style()
    rows = read_rows()
    summary = macro_summary(rows)
    lookup = {
        (str(row["model"]), str(row["method"]), str(row["language"])): row
        for row in rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.75), gridspec_kw={"wspace": 0.28})
    left, right = axes

    # Left: model-method macro means across the five held-out languages.
    for model in MODELS:
        for method in METHODS:
            record = summary[(model, method)]
            zoom = {
                "Qwen3-4B": 0.020,
                "Phi-4-mini": 0.018,
                "Granite-3.3-2B": 0.023,
            }[model]
            add_logo(
                left,
                record["ir_x_fluency"],
                record["accuracy"],
                model=model,
                method=method,
                zoom=zoom,
            )
    left.set_xlabel(r"Instruction relevance $\times$ fluency $\rightarrow$")
    left.set_ylabel(r"Accuracy (%) $\rightarrow$")
    left.set_xlim(-0.12, 3.35)
    left.set_ylim(-2.5, 50)
    style_axis(left)

    # Right: fixed best-to-worst language order, with small deterministic offsets
    # that keep model logos legible while preserving the categorical grouping.
    x = np.arange(len(LANGUAGES), dtype=float)
    method_offsets = dict(zip(METHODS, np.linspace(-0.24, 0.24, len(METHODS))))
    model_offsets = {"Qwen3-4B": -0.055, "Phi-4-mini": 0.0, "Granite-3.3-2B": 0.055}
    for language_index, language in enumerate(LANGUAGES):
        for model in MODELS:
            for method in METHODS:
                record = lookup[(model, method, language)]
                zoom = {
                    "Qwen3-4B": 0.016,
                    "Phi-4-mini": 0.014,
                    "Granite-3.3-2B": 0.019,
                }[model]
                add_logo(
                    right,
                    x[language_index] + method_offsets[method] + model_offsets[model],
                    float(record["accuracy"]),
                    model=model,
                    method=method,
                    zoom=zoom,
                )
    right.set_xticks(x, LANGUAGES)
    right.set_xlabel("Transfer language")
    right.set_ylabel(r"Accuracy (%) $\rightarrow$")
    right.set_xlim(-0.5, len(LANGUAGES) - 0.5)
    right.set_ylim(-3, 75)
    style_axis(right)

    add_legends(fig)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.86)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "figure_language.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUTPUT_DIR / "figure_language.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative-expanded-only", action="store_true")
    args = parser.parse_args()
    rows = read_rows()
    if args.relative_expanded_only:
        render_accuracy_change(rows, relative_right=True, expanded=True)
    else:
        render()
        render_accuracy_change(rows)
        render_accuracy_change(rows, relative_right=True)
        render_accuracy_change(rows, relative_right=True, expanded=True)
