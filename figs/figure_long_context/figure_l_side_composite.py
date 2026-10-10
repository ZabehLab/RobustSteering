#!/usr/bin/env python3
"""Render the Qwen-only long-context evaluation and residual interpretation."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    HPacker,
    OffsetImage,
    TextArea,
)
from PIL import Image

from figure_long_context import (
    CONTEXTS,
    LOGO_METHOD_COLORS,
    METHOD_LABELS,
    METHOD_STYLES,
    MODEL_SPECS,
    read_results,
    tight_limits,
    tinted_logo,
)


UNIT = Path(__file__).resolve().parent
PLOTS = UNIT / "plots"
ASSETS = UNIT / "assets"
REPO = UNIT.parents[1]

GPT_RESIDUALS = (
    REPO
    / "figs/figure_brain/ref/original_notebook_rerun/results"
    / "final_llmfit_panelE_context_length.csv"
)
BRAIN_RESIDUALS = (
    REPO
    / "figs/figure_brain/ref/original_notebook_rerun/results"
    / "final_panelE_context_length_LVF_theta.csv"
)

MODEL = "Qwen2.5-3B-Instruct"
INK = "#000000"
OOD_RED = "#8B1E1E"
GRID = "#E2E5E8"
SPINE = "#AEB4BA"
CONTEXT_COLORS = ["#D6D6D6", "#ADADAD", "#7D7D7D", "#4A4A4A"]

# The source notebooks calculated these references but did not export them with
# the median/IQR CSVs. Values are retained from their original rendered outputs.
GPT_SIGNAL_SD = np.array([1.012, 1.012, 0.995, 0.989])
BRAIN_SIGNAL_SD = np.array([2.03, 2.05, 2.08, 1.76, 1.75, 1.77, 1.79, 1.80])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 36.0,
            "axes.titlesize": 38.0,
            "axes.labelsize": 40.0,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def plot_qwen_axis(
    ax: plt.Axes,
    *,
    lookup: dict[tuple[str, str], object],
    methods: tuple[str, ...],
    field: str,
    ylabel: str,
) -> None:
    values_all = [
        getattr(lookup[(context, method)], field)
        for method in methods
        for context in CONTEXTS
    ]
    ax.grid(True, axis="y", color=GRID, linewidth=0.75, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(*tight_limits(values_all))
    ax.set_xticks([0, 1], ["8K", "16K"])
    ax.tick_params(length=7.0, width=1.5, labelsize=36.0, colors="black")
    xlabels = ax.get_xticklabels()
    xlabels[0].set_color("black")
    xlabels[0].set_fontweight("normal")
    xlabels[1].set_color(OOD_RED)
    xlabels[1].set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.set_ylabel(ylabel, fontsize=40.0, fontweight="normal", labelpad=10, color="black")
    for method in methods:
        values = [getattr(lookup[(context, method)], field) for context in CONTEXTS]
        ours = method == "H-infinity"
        ax.plot(
            [0, 1],
            values,
            color=LOGO_METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=3.2 if ours else 2.0,
            solid_capstyle="round",
            zorder=4 if ours else 3,
        )
        for x, value in enumerate(values):
            add_qwen_logo_marker(ax, x=float(x), y=value, method=method)


def add_qwen_logo_marker(ax: plt.Axes, *, x: float, y: float, method: str) -> None:
    logo_path = Path(MODEL_SPECS[MODEL]["logo"])
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(
                tinted_logo(logo_path, LOGO_METHOD_COLORS[method]),
                zoom=0.037,
                interpolation="lanczos",
            ),
            (x, y),
            frameon=False,
            pad=0,
            zorder=6 if method == "H-infinity" else 5,
        )
    )


def cropped_square_logo(path: Path, side: int = 256) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image)
    visible = (pixels[..., 3] > 10) & (np.min(pixels[..., :3], axis=2) < 246)
    ys, xs = np.where(visible)
    image = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    image.thumbnail((side - 24, side - 24), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))
    return np.asarray(canvas)


def add_qwen_heading(fig: plt.Figure, left: float, right: float) -> None:
    spec = MODEL_SPECS[MODEL]
    heading = fig.add_axes([left, 0.835, right - left, 0.115])
    heading.axis("off")
    logo = OffsetImage(
        cropped_square_logo(Path(spec["logo"])),
        zoom=0.215,
        resample=True,
    )
    text = TextArea(
        str(spec["label"]),
        textprops={"fontsize": 38.0, "fontfamily": "Arial", "color": "black"},
    )
    packed = HPacker(children=[logo, text], align="center", pad=0, sep=3)
    heading.add_artist(
        AnchoredOffsetbox(
            loc="center",
            child=packed,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=heading.transAxes,
            borderpad=0,
        )
    )


def add_qwen_legend(fig: plt.Figure, left: float, right: float, methods: tuple[str, ...]) -> None:
    legend_ax = fig.add_axes([left, 0.035, right - left, 0.105])
    legend_ax.axis("off")
    positions = [(0.02, 0.50), (0.29, 0.50), (0.53, 0.50), (0.76, 0.50)]
    logo_path = Path(MODEL_SPECS[MODEL]["logo"])
    for (x, y), method in zip(positions, methods, strict=True):
        legend_ax.add_artist(
            AnnotationBbox(
                OffsetImage(
                    tinted_logo(logo_path, LOGO_METHOD_COLORS[method]),
                    zoom=0.050,
                    interpolation="lanczos",
                ),
                (x, y),
                xycoords="axes fraction",
                frameon=False,
                pad=0,
            )
        )
        legend_ax.text(
            x + 0.045,
            y,
            METHOD_LABELS[method],
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=30.0,
            color=INK,
        )


def plot_residual_axis(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    horizon_col: str,
    signal_sd: np.ndarray,
    xlabel: str,
    title: str,
    ylim: tuple[float, float],
) -> None:
    horizons = np.sort(data[horizon_col].unique())
    for k_ctx, color in zip((1, 2, 4, 8), CONTEXT_COLORS, strict=True):
        sub = data[data["k_ctx"] == k_ctx].sort_values(horizon_col)
        x = sub[horizon_col].to_numpy()
        med = sub["med"].to_numpy()
        q25 = sub["q25"].to_numpy()
        q75 = sub["q75"].to_numpy()
        ax.fill_between(x, q25, q75, color=color, alpha=0.20, linewidth=0)
        ax.plot(x, med, "o-", color=color, linewidth=1.7, markersize=3.5, label=rf"$k$={k_ctx}")
    ax.plot(horizons, signal_sd, ":", color="black", linewidth=2.0, label="signal SD")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMS (z)")
    ax.set_title("")
    ax.set_ylim(*ylim)
    if horizon_col == "horizon_tok":
        ax.set_xticks([2, 4, 6, 8])
        ax.set_yticks([0.0, 0.5, 1.0])
    else:
        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks([0.0, 1.0, 2.0])
    ax.grid(False)
    ax.tick_params(length=7.0, width=1.5, labelsize=36.0, colors="black")
    ax.xaxis.label.set_fontsize(40.0)
    ax.yaxis.label.set_fontsize(40.0)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.legend(
        title="Context length",
        loc="upper left" if horizon_col == "horizon_tok" else "lower right",
        fontsize=20.0,
        title_fontsize=20.8,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.35,
    )


def show_asset(ax: plt.Axes, path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    ax.imshow(image, interpolation="lanczos", aspect="equal")
    ax.set_axis_off()
    ax.set_anchor("C")


def render() -> None:
    setup_style()
    results = read_results()
    methods = tuple(MODEL_SPECS[MODEL]["methods"])
    lookup = {(row.context, row.method): row for row in results if row.model == MODEL}
    gpt = pd.read_csv(GPT_RESIDUALS)
    brain = pd.read_csv(BRAIN_RESIDUALS)

    fig = plt.figure(figsize=(26.4, 5.25))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.765, 4.235],
        left=0.045,
        right=0.992,
        bottom=0.29,
        top=0.78,
        wspace=0.16,
    )
    qwen_grid = outer[0, 0].subgridspec(1, 2, wspace=1.31)
    residual_groups = outer[0, 1].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.06],
        wspace=0.06,
    )
    llm_group = residual_groups[0, 0].subgridspec(
        1, 2, width_ratios=[1.24, 4.39], wspace=0.38
    )
    brain_group = residual_groups[0, 1].subgridspec(
        1, 2, width_ratios=[1.36, 4.32], wspace=0.30
    )
    ax_recall = fig.add_subplot(qwen_grid[0, 0])
    ax_citation = fig.add_subplot(qwen_grid[0, 1])
    ax_llm = fig.add_subplot(llm_group[0, 0])
    ax_gpt = fig.add_subplot(llm_group[0, 1])
    ax_bio = fig.add_subplot(brain_group[0, 0])
    ax_brain = fig.add_subplot(brain_group[0, 1])

    # Keep the image panels compact while clearing the large Truthfulness-scale
    # y labels. Moving only the image axes preserves every image's aspect ratio
    # and leaves the quantitative axes at their intended wide dimensions.
    llm_box = ax_llm.get_position()
    ax_llm.set_position([llm_box.x0 - 0.018, llm_box.y0, llm_box.width, llm_box.height])
    bio_box = ax_bio.get_position()
    ax_bio.set_position([bio_box.x0 - 0.012, bio_box.y0, bio_box.width, bio_box.height])

    plot_qwen_axis(
        ax_recall,
        lookup=lookup,
        methods=methods,
        field="answer_recall",
        ylabel="Answer\nrecall (%)",
    )
    plot_qwen_axis(
        ax_citation,
        lookup=lookup,
        methods=methods,
        field="citation_f1",
        ylabel="Citation\nF1 (%)",
    )

    show_asset(ax_llm, ASSETS / "llm_network.png")
    show_asset(ax_bio, ASSETS / "biological_network.png")
    plot_residual_axis(
        ax_gpt,
        gpt,
        horizon_col="horizon_tok",
        signal_sd=GPT_SIGNAL_SD,
        xlabel=r"Horizon $\Delta$ (tokens)",
        title="GPT-2 residuals",
        ylim=(0.0, 1.08),
    )
    plot_residual_axis(
        ax_brain,
        brain,
        horizon_col="horizon_ms",
        signal_sd=BRAIN_SIGNAL_SD,
        xlabel=r"Horizon $\Delta$ (ms)",
        title="Patient 12 residuals",
        ylim=(-0.05, 2.18),
    )

    # Group headings follow the organization of the supplied layout sketch.
    qwen_left = ax_recall.get_position().x0
    qwen_right = ax_citation.get_position().x1
    qwen_center = (qwen_left + qwen_right) / 2
    add_qwen_heading(fig, qwen_left, qwen_right)
    header_y = 0.8925
    fig.text(
        (ax_gpt.get_position().x0 + ax_gpt.get_position().x1) / 2,
        header_y,
        "GPT-2 residuals",
        ha="center",
        va="center",
        fontsize=38.0,
        fontweight="normal",
        color="black",
    )
    fig.text(
        (ax_brain.get_position().x0 + ax_brain.get_position().x1) / 2,
        header_y,
        "Patient 12 residuals",
        ha="center",
        va="center",
        fontsize=38.0,
        fontweight="normal",
        color="black",
    )

    add_qwen_legend(fig, 0.012, ax_llm.get_position().x1 - 0.008, methods)

    fig.text(0.012, 0.96, "A", fontsize=44, fontweight="bold", ha="left", va="top")
    fig.text(ax_llm.get_position().x0 - 0.022, 0.96, "B", fontsize=44, fontweight="bold", ha="left", va="top")

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = PLOTS / "figure_l_side_composite"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    render()
