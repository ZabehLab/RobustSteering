from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    HPacker,
    OffsetImage,
    TextArea,
)
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath
from PIL import Image


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
PLOTS = UNIT / "plots"
LOGOS = REPO / "figs/logos"
CATEGORY_CACHE = UNIT / "cache/truthfulqa_category_txi.csv"

TEAL = "#007C7C"
OOD_RED = "#8B1E1E"
GRAY = "#A7ADB2"
INK = "#202124"
GRID = "#E3E6E8"

MODEL_SIZES = {
    "GPT-2 XL": 1.5,
    "Llama-3-8B": 8.0,
    "Qwen-2.5-14B": 14.0,
    "OLMo-2-32B": 32.0,
}
MODEL_FAMILIES = {
    "GPT-2 XL": "GPT",
    "Llama-3-8B": "LLaMA",
    "Qwen-2.5-14B": "Qwen",
    "OLMo-2-32B": "OLMo",
}
LOGO_FILES = {
    "GPT": LOGOS / "openai_transparent.png",
    "LLaMA": LOGOS / "llama_transparent.png",
    "Qwen": LOGOS / "qwen_transparent.png",
    "OLMo": LOGOS / "ai2_transparent.png",
}
LOGO_SCALE = {"GPT": 1.00, "LLaMA": 1.00, "Qwen": 0.92, "OLMo": 1.00}

METHOD_ORDER = [
    "Original",
    "ITI",
    "ActAdd",
    "Mean-AcT",
    "Linear-AcT",
    "PID-AcT",
    "ODESteer",
    "S-PID",
    "A-LQR",
    "H∞ (ours)",
]
METHOD_COLORS = {
    "Original": "#59636D",
    "ITI": "#8E88A4",
    "ActAdd": "#B88898",
    "Mean-AcT": "#9DA6AD",
    "Linear-AcT": "#B4A06F",
    "PID-AcT": "#9A8678",
    "ODESteer": "#7C9C8D",
    "S-PID": "#B4775D",
    "A-LQR": "#687DA3",
    "H∞ (ours)": TEAL,
}
METHOD_STYLES = {
    "Original": (0, (1.0, 1.2)),
    "ITI": (0, (1.2, 1.5)),
    "ActAdd": (0, (4.0, 1.6, 1.0, 1.6)),
    "Mean-AcT": (0, (7.0, 2.2)),
    "Linear-AcT": (0, (2.0, 1.3)),
    "PID-AcT": (0, (5.5, 1.5, 1.2, 1.5)),
    "ODESteer": (0, (3.5, 1.5)),
    "S-PID": (0, (5.0, 2.0)),
    "A-LQR": "-",
    "H∞ (ours)": "-",
}


@dataclass(frozen=True)
class Result:
    split: str
    model: str
    method: str
    true: float
    informative: float
    txi: float


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def numeric(cell: str) -> float:
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell)
    if match is None:
        raise ValueError(f"No numeric prefix in {cell!r}")
    return float(match.group(1))


def read_table(path: Path, split: str) -> list[Result]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Model |"))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    records: list[Result] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        row = dict(
            zip(header, [cell.strip() for cell in line.strip("|").split("|")], strict=True)
        )
        if row["Model"] not in MODEL_SIZES:
            continue
        records.append(
            Result(
                split=split,
                model=row["Model"],
                method=row["Method"],
                true=numeric(row["True (%) ↑"]),
                informative=numeric(row["Informative (%) ↑"]),
                txi=numeric(row["T×I (%) ↑"]),
            )
        )
    return records


def square_logo(path: Path, side: int = 256) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image)
    visible = (pixels[..., 3] > 10) & (np.min(pixels[..., :3], axis=2) < 246)
    if np.any(visible):
        ys, xs = np.where(visible)
        image = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    image.thumbnail((side - 24, side - 24), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))
    return np.asarray(canvas)


def logo_zoom(size_b: float, base: float = 0.040) -> float:
    return base + 0.010 * math.log2(max(size_b, 1.0))


def tinted_logo(image: np.ndarray, color: str) -> np.ndarray:
    tinted = np.zeros_like(image)
    tinted[..., :3] = np.round(255 * np.asarray(mpl.colors.to_rgb(color))).astype(
        np.uint8
    )
    tinted[..., 3] = image[..., 3]
    return tinted


def add_logo(
    ax: plt.Axes,
    images: dict[str, np.ndarray],
    model: str,
    x: float,
    y: float,
    *,
    zoom: float | None = None,
    method: str | None = None,
    coordinates: str | mpl.transforms.Transform = "data",
) -> None:
    family = MODEL_FAMILIES[model]
    if zoom is None:
        zoom = logo_zoom(MODEL_SIZES[model])
    image_key = family if method is None else f"{family}::{method}"
    image = OffsetImage(
        images[image_key], zoom=zoom * LOGO_SCALE[family], resample=True
    )
    image.set_alpha(1.0 if method in {None, "H∞ (ours)"} else 0.70)
    transform_kwargs = {} if coordinates == "data" else {"xycoords": coordinates}
    ax.add_artist(
        AnnotationBbox(
            image,
            (x, y),
            frameon=False,
            pad=0,
            zorder=12 if method == "H∞ (ours)" else 7,
            annotation_clip=False,
            **transform_kwargs,
        )
    )


def weighted(points: list[Result], value: str) -> float:
    return sum(getattr(row, value) * MODEL_SIZES[row.model] for row in points) / sum(
        MODEL_SIZES[row.model] for row in points
    )


def draw_box_panel(
    ax: plt.Axes,
    records: list[Result],
    images: dict[str, np.ndarray],
) -> None:
    centers = np.array([0.0, 3.55])
    for center, split in zip(centers, ["ID", "OOD"], strict=True):
        split_rows = [row for row in records if row.split == split]
        by_model: dict[str, list[Result]] = defaultdict(list)
        for row in split_rows:
            by_model[row.model].append(row)
        baseline = [
            max((row for row in rows if row.method != "H∞ (ours)"), key=lambda row: row.true)
            for rows in by_model.values()
        ]
        ours = [next(row for row in rows if row.method == "H∞ (ours)") for rows in by_model.values()]
        positions = [center - 0.60, center + 0.60]
        for position, points, color in zip(
            positions, [baseline, ours], [GRAY, TEAL], strict=True
        ):
            scores = [point.true for point in points]
            ax.boxplot(
                [scores],
                positions=[position],
                widths=0.72,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                manage_ticks=False,
                boxprops={
                    "facecolor": color,
                    "edgecolor": color,
                    "alpha": 0.30,
                    "linewidth": 1.15,
                },
                medianprops={"color": INK, "linewidth": 1.25},
                whiskerprops={"color": color, "linewidth": 1.15, "alpha": 0.8},
                capprops={"color": color, "linewidth": 1.15, "alpha": 0.8},
                zorder=2,
            )
            offsets = np.linspace(-0.22, 0.22, len(points))
            for point, offset in zip(points, offsets, strict=True):
                add_logo(ax, images, point.model, position + float(offset), point.true)

    ax.set_xlim(-1.0, centers[-1] + 1.0)
    ax.set_ylim(-5, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(centers, ["English", "Spanish"], fontsize=10.0)
    ax.set_ylabel("True (%)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=10.0)
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    ax.axhline(0, color=INK, linewidth=0.8, zorder=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)


def smooth_density(values: list[float], grid: np.ndarray, bandwidth: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scaled = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    peak = float(density.max(initial=0.0))
    return density / peak if peak else density


def draw_txi_field(
    ax: plt.Axes,
    *,
    show_fill: bool = True,
    contour_alpha: float = 0.75,
    fixed_label_positions: bool = False,
) -> None:
    grid = np.linspace(0, 100, 301)
    xx, yy = np.meshgrid(grid, grid)
    field = xx * yy / 100.0
    if show_fill:
        ax.contourf(
            xx,
            yy,
            field,
            levels=np.arange(0, 101, 10),
            cmap="Greys",
            alpha=0.17,
            antialiased=True,
            zorder=-3,
        )
    contours = ax.contour(
        xx,
        yy,
        field,
        levels=[20, 40, 60, 80],
        colors="#777B80",
        linewidths=0.65,
        linestyles="--",
        alpha=contour_alpha,
        zorder=-1,
    )
    labels = ax.clabel(
        contours,
        fmt=(
            (lambda value: f"{value:.0f}")
            if fixed_label_positions
            else (lambda value: f"T×I={value:.0f}")
        ),
        fontsize=6.8,
        inline=True,
        manual=[(22.7, 88), (45.5, 88), (68.2, 88), (90.9, 88)]
        if fixed_label_positions
        else False,
    )
    if fixed_label_positions:
        for label in labels:
            label.set_alpha(contour_alpha)


def draw_frontier(
    main_ax: plt.Axes,
    top_ax: plt.Axes,
    right_ax: plt.Axes,
    records: list[Result],
    split: str,
    images: dict[str, np.ndarray],
    *,
    show_txi_fill: bool = True,
    txi_alpha: float = 0.75,
    fixed_txi_labels: bool = False,
) -> None:
    rows = [row for row in records if row.split == split]
    draw_txi_field(
        main_ax,
        show_fill=show_txi_fill,
        contour_alpha=txi_alpha,
        fixed_label_positions=fixed_txi_labels,
    )
    for method in METHOD_ORDER:
        for row in rows:
            if row.method == method:
                add_logo(
                    main_ax,
                    images,
                    row.model,
                    row.informative,
                    row.true,
                    method=row.method,
                )

    main_ax.set_xlim(0, 100)
    main_ax.set_ylim(0, 100)
    main_ax.set_xticks(np.arange(0, 101, 20))
    main_ax.set_yticks(np.arange(0, 101, 20))
    main_ax.set_xlabel("Informative (%)", fontsize=10.5)
    main_ax.set_ylabel("True (%)", fontsize=10.5)
    main_ax.tick_params(labelsize=8)
    main_ax.grid(True, color=GRID, linewidth=0.55, zorder=-2)
    for spine in main_ax.spines.values():
        spine.set_color("#AEB4BC")
        spine.set_linewidth(0.7)

    x_grid = np.linspace(0, 100, 300)
    y_grid = np.linspace(0, 100, 300)
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row.method == method]
        if not method_rows:
            continue
        color = METHOD_COLORS[method]
        primary = method == "H∞ (ours)"
        alpha = 1.0 if primary else 0.46
        linewidth = 1.9 if primary else 0.9
        x_density = smooth_density(
            [row.informative for row in method_rows], x_grid, 5.0
        )
        y_density = smooth_density([row.true for row in method_rows], y_grid, 5.0)
        top_ax.plot(
            x_grid,
            x_density,
            color=color,
            linestyle=METHOD_STYLES[method],
            linewidth=linewidth,
            alpha=alpha,
        )
        right_ax.plot(
            y_density,
            y_grid,
            color=color,
            linestyle=METHOD_STYLES[method],
            linewidth=linewidth,
            alpha=alpha,
        )
        if primary:
            top_ax.fill_between(x_grid, 0, x_density, color=color, alpha=0.10)
            right_ax.fill_betweenx(y_grid, 0, y_density, color=color, alpha=0.10)

    top_ax.set_xlim(0, 100)
    top_ax.set_ylim(0, 1.08)
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    top_ax.spines[["top", "left", "right"]].set_visible(False)
    top_ax.spines["bottom"].set_color("#AEB4BC")

    right_ax.set_xlim(0, 1.08)
    right_ax.set_ylim(0, 100)
    right_ax.set_xticks([])
    right_ax.set_yticks([])
    right_ax.spines[["top", "right", "bottom"]].set_visible(False)
    right_ax.spines["left"].set_color("#AEB4BC")


def category_radar_values(
    model: str | None = None,
    metric: str = "txi_pct",
    exclude_original_from_competitor: bool = False,
) -> tuple[
    list[str], dict[str, tuple[list[float], list[float]]]
]:
    rows = list(csv.DictReader(CATEGORY_CACHE.open()))
    categories = list(dict.fromkeys(row["category"] for row in rows))
    ours_by_split: dict[tuple[str, str], float] = {}
    best_by_split: dict[tuple[str, str], float] = {}
    for split in ["ID", "OOD"]:
        for category in categories:
            relevant = [
                row for row in rows if row["split"] == split and row["category"] == category
            ]
            by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in relevant:
                by_model[row["model"]].append(row)
            ours_points: list[float] = []
            best_points: list[float] = []
            for current_model, model_rows in by_model.items():
                if model is not None and current_model != model:
                    continue
                ours = next(row for row in model_rows if row["method"] == "H∞ (ours)")
                best = max(
                    (
                        row
                        for row in model_rows
                        if row["method"] != "H∞ (ours)"
                        and not (
                            exclude_original_from_competitor
                            and row["method"] == "Original"
                        )
                    ),
                    key=lambda row: float(row[metric]),
                )
                ours_points.append(float(ours[metric]))
                best_points.append(float(best[metric]))

            ours_by_split[(split, category)] = float(np.mean(ours_points))
            best_by_split[(split, category)] = float(np.mean(best_points))

    values = {
        split: (
            [ours_by_split[(split, category)] for category in categories],
            [best_by_split[(split, category)] for category in categories],
        )
        for split in ["ID", "OOD"]
    }
    return categories, values


def best_radar_model(
    metric: str,
    *,
    exclude_original_from_competitor: bool = False,
) -> tuple[str, dict[str, float]]:
    rows = list(csv.DictReader(CATEGORY_CACHE.open()))
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["category"], row["model"])].append(row)
    margins: dict[str, list[float]] = defaultdict(list)
    for (_, _, model), model_rows in grouped.items():
        ours = next(
            float(row[metric])
            for row in model_rows
            if row["method"] == "H∞ (ours)"
        )
        competitor = max(
            float(row[metric])
            for row in model_rows
            if row["method"] != "H∞ (ours)"
            and not (
                exclude_original_from_competitor and row["method"] == "Original"
            )
        )
        margins[model].append(ours - competitor)
    mean_margins = {
        model: float(np.mean(model_margins))
        for model, model_margins in margins.items()
    }
    selected = max(mean_margins, key=mean_margins.get)
    return selected, mean_margins


def draw_radar(
    ax: plt.Axes,
    split: str,
    *,
    model: str | None = None,
    metric: str = "txi_pct",
    include_original: bool = False,
    label_fontsize: float = 8.6,
    label_fontweight: str = "normal",
    label_pad: float = 10,
    extend_label_spokes: bool = False,
    show_radial_labels: bool = False,
    tangential_category_labels: bool = False,
) -> None:
    categories, values = category_radar_values(
        model,
        metric,
        exclude_original_from_competitor=include_original,
    )
    ours, best = values[split]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.r_[angles, angles[0]]
    ours_closed = np.r_[ours, ours[0]]
    best_closed = np.r_[best, best[0]]
    if include_original:
        if model is None:
            raise ValueError("Original radar requires a concrete model")
        rows = list(csv.DictReader(CATEGORY_CACHE.open()))
        original = [
            next(
                float(row[metric])
                for row in rows
                if row["split"] == split
                and row["category"] == category
                and row["model"] == model
                and row["method"] == "Original"
            )
            for category in categories
        ]
        original_closed = np.r_[original, original[0]]
        ax.plot(angles, original_closed, color="#737B82", linewidth=1.8)
        ax.fill(angles, original_closed, color="#737B82", alpha=0.16)
        ax.scatter(angles[:-1], original, s=18, color="#737B82", zorder=3)
    ax.plot(angles, best_closed, color=GRAY, linewidth=1.7, linestyle="--")
    ax.fill(angles, best_closed, color=GRAY, alpha=0.08)
    ax.plot(angles, ours_closed, color=TEAL, linewidth=2.2)
    ax.fill(angles, ours_closed, color=TEAL, alpha=0.12)
    ax.scatter(angles[:-1], best, s=14, color=GRAY, zorder=4)
    ax.scatter(angles[:-1], ours, s=18, color=TEAL, zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    if tangential_category_labels:
        ax.set_xticklabels([])
        label_radii = [150, 120, 180, 150, 180, 120, 180, 150, 180, 120]
        for angle, category, radius in zip(
            angles[:-1], categories, label_radii, strict=True
        ):
            draw_curved_polar_text(
                ax,
                angle,
                radius,
                category,
                fontsize=label_fontsize,
                fontweight=label_fontweight,
            )
    else:
        ax.set_xticklabels(
            categories,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
        )
        ax.tick_params(axis="x", pad=label_pad)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    if show_radial_labels:
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7.5)
        ax.set_rlabel_position(72)
        ax.tick_params(axis="y", colors="#6F767D", pad=1)
    else:
        ax.set_yticklabels([])
    ax.grid(color="#C9CED3", linewidth=0.6, linestyle=":")
    ax.spines["polar"].set_color("#AEB4BC")
    if extend_label_spokes:
        for index, angle in enumerate(angles[:-1]):
            spoke_end = (
                label_radii[index] - 4 if tangential_category_labels else 111
            )
            ax.plot(
                [angle, angle],
                [100, spoke_end],
                color="#9AA0A6",
                linewidth=1.35,
                linestyle="-",
                clip_on=False,
                zorder=0,
            )


def draw_curved_polar_text(
    ax: plt.Axes,
    center_angle: float,
    radius: float,
    label: str,
    *,
    fontsize: float,
    fontweight: str,
) -> None:
    """Place Arial glyphs along an arc using their measured rendered widths."""
    font = FontProperties(family="Arial", weight=fontweight, size=fontsize)
    glyph_widths_pt: list[float] = []
    for character in label:
        if character == " ":
            glyph_widths_pt.append(fontsize * 0.34)
            continue
        bounds = TextPath((0, 0), character, prop=font, size=fontsize).get_extents()
        glyph_widths_pt.append(max(float(bounds.width), fontsize * 0.18))

    dpi = ax.figure.dpi
    glyph_widths_px = np.asarray(glyph_widths_pt) * dpi / 72.0
    gap_px = fontsize * dpi / 72.0 * 0.07
    axes_box = ax.get_window_extent()
    radius_px = 0.5 * min(axes_box.width, axes_box.height) * radius / 100.0
    total_width_px = float(glyph_widths_px.sum() + gap_px * (len(label) - 1))
    cursor_px = -total_width_px / 2.0
    offsets = []
    for width_px in glyph_widths_px:
        offsets.append((cursor_px + width_px / 2.0) / radius_px)
        cursor_px += width_px + gap_px
    offsets = np.asarray(offsets)
    # On the lower half, reverse angular progression so glyph order remains
    # left-to-right after the tangent is flipped upright.
    for character, offset in zip(label, offsets, strict=True):
        # Theta increases clockwise in these radar axes. Keep every word in
        # that direction, with its baseline/bottom facing the radar center.
        angle = center_angle + offset
        rotation = -float(np.degrees(angle))
        ax.text(
            angle,
            radius,
            character,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontproperties=font,
            color="black",
            clip_on=False,
        )


def draw_model_legend(ax: plt.Axes, images: dict[str, np.ndarray]) -> None:
    ax.set_axis_off()
    models = list(MODEL_SIZES)
    labels = ["GPT-2 XL", "LLaMA-3-8B", "Qwen-2.5-14B", "OLMo-2-32B"]
    groups = []
    for model, label in zip(models, labels, strict=True):
        family = MODEL_FAMILIES[model]
        logo = OffsetImage(
            images[family],
            zoom=0.105 * LOGO_SCALE[family],
            resample=True,
        )
        text = TextArea(
            label,
            textprops={"fontsize": 14.0, "fontfamily": "Arial", "color": INK},
        )
        groups.append(HPacker(children=[logo, text], align="center", pad=0, sep=4))
    packed = HPacker(children=groups, align="center", pad=0, sep=28)
    ax.add_artist(
        AnchoredOffsetbox(
            loc="center",
            child=packed,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
    )


def method_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            markersize=8.0 if method == "H∞ (ours)" else 5.8,
            alpha=1.0 if method == "H∞ (ours)" else 0.72,
            label=method,
        )
        for method in METHOD_ORDER
    ]


def radar_handles(
    include_original: bool = False,
    *,
    as_lines: bool = False,
) -> list[Line2D]:
    entries = []
    if include_original:
        entries.append(("Original", "#737B82", 6.5, 0.82))
    entries.extend(
        [
            ("Best competitor", GRAY, 6.5, 0.82),
            (r"H$\infty$ (ours)", TEAL, 9.0, 1.0),
        ]
    )
    if as_lines:
        line_styles = {
            "Original": "-",
            "Best competitor": "--",
            r"H$\infty$ (ours)": "-",
        }
        return [
            Line2D(
                [],
                [],
                color=color,
                linestyle=line_styles[label],
                linewidth=2.8 if "ours" in label else 2.1,
                marker="o",
                markersize=4.5,
                alpha=alpha,
                label=label,
            )
            for label, color, _size, alpha in entries
        ]
    return [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=size,
            alpha=alpha,
            label=label,
        )
        for label, color, size, alpha in entries
    ]


def emphasize_ours(legend: plt.Legend) -> None:
    for text in legend.get_texts():
        if "ours" in text.get_text():
            text.set_fontweight("bold")


def force_black_axis_text(ax: plt.Axes) -> None:
    """Keep all composite labels legible after the panels are reduced."""
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    ax.tick_params(axis="both", colors="black")
    for text in ax.texts:
        text.set_color("black")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")


def draw_composite_model_header(
    ax: plt.Axes,
    images: dict[str, np.ndarray],
    models: list[str],
    *,
    fontsize: float = 10.5,
    logo_zoom_scale: float = 1.0,
    group_separation: float = 18,
) -> None:
    ax.set_axis_off()
    groups = []
    for model in models:
        family = MODEL_FAMILIES[model]
        logo = OffsetImage(
            images[family],
            zoom=0.075 * LOGO_SCALE[family] * logo_zoom_scale,
            resample=True,
        )
        text = TextArea(
            model,
            textprops={"fontsize": fontsize, "fontfamily": "Arial", "color": "black"},
        )
        groups.append(HPacker(children=[logo, text], align="center", pad=0, sep=3))
    packed = HPacker(
        children=groups, align="center", pad=0, sep=group_separation
    )
    ax.add_artist(
        AnchoredOffsetbox(
            loc="center",
            child=packed,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
    )


def load_inputs() -> tuple[list[Result], dict[str, np.ndarray]]:
    records = read_table(
        REPO / "figs/bench_table/truthfulness/truthfulqa.md", "ID"
    ) + read_table(
        REPO / "figs/bench_table/truthfulness/truthfulqa_spanish.md", "OOD"
    )
    images = {family: square_logo(path) for family, path in LOGO_FILES.items()}
    for family in LOGO_FILES:
        for method in METHOD_ORDER:
            images[f"{family}::{method}"] = tinted_logo(
                images[family], METHOD_COLORS[method]
            )
    return records, images


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ["pdf", "png"]:
        fig.savefig(
            PLOTS / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def render_figure_a(records: list[Result], images: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 4.5))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.97)
    draw_box_panel(ax, records, images)
    ax.legend(
        handles=[
            Patch(facecolor=GRAY, alpha=0.30, label="Best competitor"),
            Patch(facecolor=TEAL, alpha=0.30, label=r"H$\infty$ (ours)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=9.0,
        handlelength=1.4,
        columnspacing=1.2,
    )
    save_figure(fig, "figure_truthful_a")


def render_figure_b(records: list[Result], images: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(9.6, 4.8))
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.20, 0.80],
        left=0.07,
        right=0.97,
        bottom=0.25,
        top=0.98,
        wspace=0.25,
        hspace=0.04,
    )
    model_ax = fig.add_subplot(outer[0, :])
    draw_model_legend(model_ax, images)

    frontier_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for index in [0, 1]:
        nested = outer[1, index].subgridspec(
            2,
            2,
            width_ratios=[8.0, 1.15],
            height_ratios=[1.10, 7.0],
            hspace=0.04,
            wspace=0.04,
        )
        top_ax = fig.add_subplot(nested[0, 0])
        main_ax = fig.add_subplot(nested[1, 0])
        right_ax = fig.add_subplot(nested[1, 1])
        frontier_axes.append((main_ax, top_ax, right_ax))

    draw_frontier(*frontier_axes[0], records, "ID", images)
    draw_frontier(*frontier_axes[1], records, "OOD", images)
    legend = fig.legend(
        handles=method_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        fontsize=8.4,
        handlelength=0.7,
        handletextpad=0.25,
        columnspacing=0.9,
    )
    emphasize_ours(legend)
    save_figure(fig, "figure_truthful_b")


def render_figure_c() -> None:
    fig = plt.figure(figsize=(8.4, 4.2))
    outer = fig.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.94,
        bottom=0.20,
        top=0.96,
        wspace=0.58,
    )
    radar_id_ax = fig.add_subplot(outer[0], projection="polar")
    radar_ood_ax = fig.add_subplot(outer[1], projection="polar")
    draw_radar(radar_id_ax, "ID")
    draw_radar(radar_ood_ax, "OOD")
    legend = fig.legend(
        handles=radar_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=10.0,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    emphasize_ours(legend)
    save_figure(fig, "figure_truthful_c")


def render_figure_c_model(model: str, stem: str) -> None:
    fig = plt.figure(figsize=(8.6, 4.8))
    outer = fig.add_gridspec(
        1,
        2,
        left=0.07,
        right=0.96,
        bottom=0.18,
        top=0.95,
        wspace=0.42,
    )
    radar_id_ax = fig.add_subplot(outer[0], projection="polar")
    radar_ood_ax = fig.add_subplot(outer[1], projection="polar")
    draw_radar(
        radar_id_ax,
        "ID",
        model=model,
        metric="true_pct",
        include_original=True,
        label_fontsize=8.0,
        label_fontweight="bold",
        label_pad=12,
        extend_label_spokes=True,
        show_radial_labels=True,
    )
    draw_radar(
        radar_ood_ax,
        "OOD",
        model=model,
        metric="true_pct",
        include_original=True,
        label_fontsize=8.0,
        label_fontweight="bold",
        label_pad=12,
        extend_label_spokes=True,
    )
    fig.text(
        0.018,
        0.57,
        "True (%)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.5,
    )
    legend = fig.legend(
        handles=radar_handles(include_original=True),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=9.5,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    emphasize_ours(legend)
    save_figure(fig, stem)


def render_figure_c_best_model() -> tuple[str, dict[str, float]]:
    model, mean_margins = best_radar_model(
        "true_pct",
        exclude_original_from_competitor=True,
    )
    render_figure_c_model(model, "figure_truthful_c_best_model")
    return model, mean_margins


def render_figure_composite(
    records: list[Result], images: dict[str, np.ndarray]
) -> None:
    """Render the final four-panel truthfulness composition at publication size."""
    fig = plt.figure(figsize=(15.8, 5.25))
    outer = fig.add_gridspec(
        2,
        5,
        height_ratios=[0.14, 0.86],
        width_ratios=[1.25, 1.25, 0.015, 1.0, 1.0],
        left=0.045,
        right=0.985,
        bottom=0.19,
        top=0.97,
        wspace=0.20,
        hspace=0.10,
    )

    frontier_header = fig.add_subplot(outer[0, 0:2])
    draw_composite_model_header(frontier_header, images, list(MODEL_SIZES))
    radar_header = fig.add_subplot(outer[0, 3:5])
    draw_composite_model_header(radar_header, images, ["GPT-2 XL"])

    frontier_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for column in [0, 1]:
        nested = outer[1, column].subgridspec(
            2,
            2,
            width_ratios=[8.0, 1.15],
            height_ratios=[1.10, 7.0],
            hspace=0.04,
            wspace=0.04,
        )
        top_ax = fig.add_subplot(nested[0, 0])
        main_ax = fig.add_subplot(nested[1, 0])
        right_ax = fig.add_subplot(nested[1, 1])
        frontier_axes.append((main_ax, top_ax, right_ax))

    for (main_ax, top_ax, right_ax), split, title in zip(
        frontier_axes,
        ["ID", "OOD"],
        ["English (ID)", "Spanish (OOD)"],
        strict=True,
    ):
        draw_frontier(main_ax, top_ax, right_ax, records, split, images)
        main_ax.set_xlabel("Informative (%)", fontsize=10.5, color="black")
        main_ax.set_ylabel("True (%)", fontsize=10.5, color="black")
        main_ax.tick_params(labelsize=9.5, colors="black")
        for contour_label in main_ax.texts:
            contour_label.set_fontsize(8.0)
        force_black_axis_text(main_ax)
        force_black_axis_text(top_ax)
        force_black_axis_text(right_ax)
        top_ax.set_title(
            title,
            fontsize=12.0,
            fontweight="bold",
            color=OOD_RED if split == "OOD" else "black",
            pad=7,
        )

    radar_grid = outer[1, 3:5].subgridspec(1, 2, wspace=0.88)
    radar_id_ax = fig.add_subplot(radar_grid[0], projection="polar")
    radar_ood_ax = fig.add_subplot(radar_grid[1], projection="polar")
    for ax, split, title, radial_labels in zip(
        [radar_id_ax, radar_ood_ax],
        ["ID", "OOD"],
        ["English (ID)", "Spanish (OOD)"],
        [True, False],
        strict=True,
    ):
        draw_radar(
            ax,
            split,
            model="GPT-2 XL",
            metric="true_pct",
            include_original=True,
            label_fontsize=9.5,
            label_fontweight="bold",
            label_pad=12,
            extend_label_spokes=True,
            show_radial_labels=radial_labels,
        )
        for radial_label in ax.get_yticklabels():
            radial_label.set_fontsize(8.8)
        force_black_axis_text(ax)
        ax.set_title(
            title,
            fontsize=12.0,
            fontweight="bold",
            color=OOD_RED if split == "OOD" else "black",
            pad=20,
        )

    frontier_bounds = frontier_header.get_position()
    frontier_legend = fig.legend(
        handles=method_handles(),
        loc="lower left",
        bbox_to_anchor=(
            frontier_bounds.x0,
            -0.005,
            frontier_bounds.width,
            0.06,
        ),
        ncol=len(METHOD_ORDER),
        fontsize=9.0,
        handlelength=0.7,
        handletextpad=0.25,
        columnspacing=0.8,
        mode="expand",
        borderaxespad=0,
        labelcolor="black",
    )
    emphasize_ours(frontier_legend)
    radar_legend = fig.legend(
        handles=radar_handles(include_original=True),
        loc="lower center",
        bbox_to_anchor=(0.79, 0.015),
        ncol=3,
        fontsize=10.0,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=1.25,
        labelcolor="black",
    )
    emphasize_ours(radar_legend)
    save_figure(fig, "figure_truthful_composite")


def render_figure_composite_bigger(
    records: list[Result], images: dict[str, np.ndarray]
) -> None:
    """Render a large-type composite without modifying the canonical version."""
    fig = plt.figure(figsize=(15.8, 5.25))
    outer = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.13, 0.11, 0.76],
        width_ratios=[2.55, 0.03, 2.14],
        left=0.045,
        right=0.985,
        bottom=0.19,
        top=0.97,
        wspace=0.15,
        hspace=0.02,
    )

    frontier_header = fig.add_subplot(outer[0, 0])
    draw_composite_model_header(
        frontier_header,
        images,
        list(MODEL_SIZES),
        fontsize=19.0,
        logo_zoom_scale=1.65,
        group_separation=22,
    )
    radar_header = fig.add_subplot(outer[0, 2])
    draw_composite_model_header(
        radar_header,
        images,
        ["GPT-2 XL"],
        fontsize=19.0,
        logo_zoom_scale=1.65,
    )

    frontier_title_grid = outer[1, 0].subgridspec(1, 2, wspace=0.42)
    radar_title_grid = outer[1, 2].subgridspec(1, 2, wspace=1.05)
    for title_grid, titles in [
        (frontier_title_grid, ["English (ID)", "Spanish (OOD)"]),
        (radar_title_grid, ["English (ID)", "Spanish (OOD)"]),
    ]:
        for column, title in enumerate(titles):
            title_ax = fig.add_subplot(title_grid[0, column])
            title_ax.set_axis_off()
            title_ax.text(
                0.5,
                0.50,
                title,
                ha="center",
                va="center",
                fontsize=22.0,
                fontweight="bold",
                color=OOD_RED if "Spanish" in title else "black",
                transform=title_ax.transAxes,
            )

    frontier_grid = outer[2, 0].subgridspec(1, 2, wspace=0.42)
    frontier_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for column in [0, 1]:
        nested = frontier_grid[0, column].subgridspec(
            2,
            2,
            width_ratios=[8.0, 1.15],
            height_ratios=[1.10, 7.0],
            hspace=0.04,
            wspace=0.04,
        )
        top_ax = fig.add_subplot(nested[0, 0])
        main_ax = fig.add_subplot(nested[1, 0])
        right_ax = fig.add_subplot(nested[1, 1])
        frontier_axes.append((main_ax, top_ax, right_ax))

    for (main_ax, top_ax, right_ax), split, title in zip(
        frontier_axes,
        ["ID", "OOD"],
        ["English (ID)", "Spanish (OOD)"],
        strict=True,
    ):
        draw_frontier(
            main_ax,
            top_ax,
            right_ax,
            records,
            split,
            images,
            show_txi_fill=False,
            txi_alpha=0.50,
            fixed_txi_labels=True,
        )
        main_ax.set_xlabel("Informative (%)", fontsize=20.0, color="black")
        main_ax.set_ylabel("True (%)", fontsize=20.0, color="black")
        main_ax.tick_params(labelsize=18.0, colors="black")
        for contour_label in main_ax.texts:
            contour_label.set_fontsize(12.0)
        force_black_axis_text(main_ax)
        force_black_axis_text(top_ax)
        force_black_axis_text(right_ax)
        top_ax.set_title("")

    radar_content = outer[2, 2].subgridspec(
        2,
        1,
        height_ratios=[0.18, 0.82],
        hspace=0,
    )
    radar_grid = radar_content[1, 0].subgridspec(1, 2, wspace=1.05)
    radar_id_ax = fig.add_subplot(radar_grid[0], projection="polar")
    radar_ood_ax = fig.add_subplot(radar_grid[1], projection="polar")
    for ax, split, title in zip(
        [radar_id_ax, radar_ood_ax],
        ["ID", "OOD"],
        ["English (ID)", "Spanish (OOD)"],
        strict=True,
    ):
        draw_radar(
            ax,
            split,
            model="GPT-2 XL",
            metric="true_pct",
            include_original=True,
            label_fontsize=20.0,
            label_fontweight="normal",
            extend_label_spokes=True,
            show_radial_labels=False,
            tangential_category_labels=True,
        )
        force_black_axis_text(ax)
        ax.set_title("")

    frontier_bounds = frontier_header.get_position()
    frontier_legend = fig.legend(
        handles=method_handles(),
        loc="upper left",
        bbox_to_anchor=(
            frontier_bounds.x0 - 0.012,
            -0.050,
            frontier_bounds.width,
            0.09,
        ),
        ncol=5,
        fontsize=15.0,
        handlelength=0.7,
        handletextpad=0.3,
        columnspacing=0.9,
        mode="expand",
        borderaxespad=0,
        labelcolor="black",
    )
    emphasize_ours(frontier_legend)
    radar_bounds = radar_header.get_position()
    radar_legend = fig.legend(
        handles=radar_handles(include_original=True, as_lines=True),
        loc="upper center",
        bbox_to_anchor=((radar_bounds.x0 + radar_bounds.x1) / 2.0, 0.0),
        ncol=3,
        fontsize=15.0,
        handlelength=2.1,
        handletextpad=0.35,
        columnspacing=1.3,
        labelcolor="black",
    )
    emphasize_ours(radar_legend)
    save_figure(fig, "figure_truthful_composite_bigger")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-bigger-only", action="store_true")
    args = parser.parse_args()
    setup_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    records, images = load_inputs()
    if args.composite_bigger_only:
        render_figure_composite_bigger(records, images)
        return
    render_figure_a(records, images)
    render_figure_b(records, images)
    render_figure_c()
    render_figure_c_best_model()
    render_figure_c_model("GPT-2 XL", "figure_truthful_c_gpt2_xl")
    render_figure_composite(records, images)
    render_figure_composite_bigger(records, images)


if __name__ == "__main__":
    main()
