from __future__ import annotations

from pathlib import Path
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image, ImageFilter

import figure_0_overall as base


UNIT = Path(__file__).resolve().parent
PLOTS = UNIT / "plots"

BOX_LINEWIDTH = 0.75
MEDIAN_LINEWIDTH = 1.05
DATA_LOGO_DARKEN = 0.80
LOGO_SCALE = 0.70
LOGO_JITTER = 0.28
BASELINE_LOGO_GRAY = "#72777F"
FONT_SCALE = 1.20
EXPORT_BOUNDS = (0.0586779513888889, 0.17116666666666663,
                 15.964165185866014, 5.524083333333334)


def darken(color: str, factor: float = DATA_LOGO_DARKEN) -> str:
    """Return a fixed multiplicative darkening of an RGB hex color."""

    rgb = np.asarray(to_rgba(color)[:3])
    return "#" + "".join(f"{round(channel * factor * 255):02X}" for channel in rgb)


def tinted_logo(image: np.ndarray, color: str, *, strengthen_strokes: bool = False) -> np.ndarray:
    """Tint a logo using its transparency and luminance as the silhouette mask.

    The older alpha-only transformation turned light and multicolour assets into
    solid blocks. This is the same mask construction used by the MGSM figure,
    including for the four-pane Microsoft mark.
    """

    source = image.astype(np.float32) / 255.0
    alpha = source[..., 3]
    luminance = (
        0.2126 * source[..., 0]
        + 0.7152 * source[..., 1]
        + 0.0722 * source[..., 2]
    )
    mask = alpha * np.clip(1.25 - luminance, 0.28, 1.0)
    if strengthen_strokes:
        mask = np.asarray(Image.fromarray(np.round(mask * 255).astype(np.uint8)).filter(
            ImageFilter.MaxFilter(3)), dtype=np.float32) / 255
    output = np.empty_like(source)
    output[..., :3] = np.asarray(to_rgba(color)[:3])
    output[..., 3] = mask
    return np.round(output * 255).astype(np.uint8)


def logo_offsets(ax: plt.Axes, points: list[dict], images: dict[str, np.ndarray]) -> tuple[float, ...]:
    """Spread nearby logos horizontally while keeping their exact score heights."""
    slots = np.linspace(-LOGO_JITTER, LOGO_JITTER, len(points)) if len(points) > 1 else [0.0]
    widths, heights = [], []
    for point in points:
        family = str(point["family"])
        visible = images[family][..., 3] > 10
        ys, xs = np.where(visible)
        zoom = LOGO_SCALE * base.logo_zoom(float(point["size_b"])) * base.LOGO_VISUAL_SCALE[family]
        pixels_per_source_pixel = zoom * ax.figure.dpi / 72
        widths.append((xs.max() - xs.min() + 1) * pixels_per_source_pixel)
        heights.append((ys.max() - ys.min() + 1) * pixels_per_source_pixel)
    y = ax.transData.transform([(0, float(point["score"])) for point in points])[:, 1]
    x_scale = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    # Include the visible logo width when reserving each series' own column.
    limits = 0.39 - np.array(widths) / (2 * x_scale)
    if np.any(limits < 0):
        raise ValueError("A logo is wider than its reserved series column")
    candidates = [tuple(np.clip(offsets, -limits, limits)) for offsets in permutations(slots)]

    def overlap(offsets: tuple[float, ...]) -> float:
        area = 0.0
        for i in range(len(points)):
            for j in range(i):
                dx = abs(offsets[i] - offsets[j]) * x_scale
                dy = abs(y[i] - y[j])
                area += max(0, (widths[i] + widths[j]) / 2 + 2 - dx) * max(
                    0, (heights[i] + heights[j]) / 2 + 2 - dy)
        return area

    return min(candidates, key=overlap)


def draw_box_panel(
    ax: plt.Axes,
    title: str,
    conditions: dict[str, dict[str, object]],
    ylabel: str,
    ymax: float,
    logo_images: dict[str, np.ndarray],
    condition_labels: list[str],
    distribution_labels: list[str],
    *,
    group_gap: float | None = None,
    ymin: float = 0.0,
) -> None:
    if group_gap is None:
        group_gap = 2.42 if len(conditions) <= 2 else 2.08
    width = 0.184
    centers = np.arange(len(conditions), dtype=float) * group_gap
    displayed_range = ymax - ymin
    ax.set_xlim(centers[0] - 0.86, centers[-1] + 0.86)
    ax.set_ylim(ymin - 0.055 * displayed_range, ymax)
    series_logo_images = {
        "baseline": {
            family: tinted_logo(image, BASELINE_LOGO_GRAY,
                                strengthen_strokes=family in {"GPT", "Granite"})
            for family, image in logo_images.items()
        },
        "ours": {
            family: tinted_logo(image, darken(base.TEAL),
                                strengthen_strokes=family in {"GPT", "Granite"})
            for family, image in logo_images.items()
        },
    }

    for center, (_, values) in zip(centers, conditions.items(), strict=True):
        positions = [center - 0.43, center + 0.43]
        for position, key, color in zip(
            positions,
            ["baseline", "ours"],
            [base.GRAY, base.TEAL],
            strict=True,
        ):
            points = list(values[key])
            scores = [float(point["score"]) for point in points]
            ax.boxplot(
                [scores],
                positions=[position],
                widths=width,
                patch_artist=True,
                showfliers=False,
                showcaps=False,
                whis=(0, 100),
                manage_ticks=False,
                boxprops={
                    "facecolor": to_rgba(color, 0.30),
                    "edgecolor": "none",
                    "linewidth": 0,
                },
                medianprops={"color": color, "linewidth": MEDIAN_LINEWIDTH},
                whiskerprops={"color": color, "linewidth": BOX_LINEWIDTH, "linestyle": "solid"},
                capprops={"color": color, "linewidth": BOX_LINEWIDTH},
                zorder=2,
            )

            offsets = logo_offsets(ax, points, series_logo_images[key])
            for point, offset in zip(points, offsets, strict=True):
                base.add_logo(
                    ax,
                    series_logo_images[key],
                    str(point["family"]),
                    position + float(offset),
                    float(point["score"]),
                    LOGO_SCALE * base.logo_zoom(float(point["size_b"])),
                )

    if ymin == 0:
        ax.set_yticks(np.linspace(0, ymax, 6))
    else:
        ax.set_yticks(np.arange(ymin, ymax + 0.1, 10))
    ax.set_ylabel(ylabel, fontsize=16, labelpad=2)
    ax.set_xticks(centers)
    ax.set_xticklabels(condition_labels, fontsize=15.5)
    ax.set_title(title, fontsize=16.5, fontweight="bold", pad=44)
    ax.tick_params(axis="y", labelsize=15, colors="black", width=0.9, length=4.5)
    ax.tick_params(
        axis="x",
        labelsize=15.5,
        length=5,
        width=0.9,
        pad=9,
        colors="black",
        direction="out",
    )
    ax.yaxis.grid(True, color="#E2E5E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_visible(True)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)
    ax.spines["left"].set_bounds(*ax.get_ylim())
    ax.spines["left"].set_position(("outward", 0))
    ax.spines["bottom"].set_bounds(*ax.get_xlim())
    ax.spines["bottom"].set_position(("outward", 0))

    for center, label in zip(centers, distribution_labels, strict=True):
        ax.text(
            center,
            1.035,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=16.5,
            fontweight="bold",
            color="#8B1E1E" if label == "OOD" else "black",
            clip_on=False,
        )


def draw_model_legend(ax: plt.Axes, logo_images: dict[str, np.ndarray]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        1.02,
        "Model family",
        ha="left",
        va="bottom",
        fontsize=14.5,
        fontweight="bold",
        color="black",
    )
    families = ["GPT", "LLaMA", "Qwen", "Phi", "Granite", "OLMo"]
    y_positions = np.linspace(0.90, 0.36, len(families))
    for family, y in zip(families, y_positions, strict=True):
        base.add_logo(ax, logo_images, family, 0.20, float(y), LOGO_SCALE * 0.084)
        ax.text(
            0.32,
            y,
            family,
            ha="left",
            va="center",
            fontsize=13.5,
            color="black",
        )


def hierarchy_heading(
    fig: plt.Figure,
    axes: list[plt.Axes],
    indices: tuple[int, ...],
    label: str,
) -> None:
    left = min(axes[index].get_position().x0 for index in indices)
    right = max(axes[index].get_position().x1 for index in indices)
    fig.text(
        (left + right) / 2,
        0.965,
        label,
        ha="center",
        va="top",
        fontsize=16.5,
        fontweight="bold",
        color="black",
    )
    fig.add_artist(Line2D([left, right], [0.905, 0.905], transform=fig.transFigure,
                          color="#B5B5B5", linewidth=0.8))


def main() -> None:
    base.setup_style()
    plt.rcParams.update(
        {
            "font.size": 11.5,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    PLOTS.mkdir(exist_ok=True)
    results = base.build_results()
    logo_images = {
        family: base.square_logo(path) for family, path in base.LOGO_FILES.items()
    }

    fig = plt.figure(figsize=(16.2, 5.85))
    grid = fig.add_gridspec(
        1,
        9,
        width_ratios=[1.48, 0.68, 1.48, 0.68, 3.65, 0.68, 1.48, 0.68, 0.92],
        left=0.055,
        right=0.985,
        bottom=0.225,
        top=0.735,
        wspace=0.0,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in [0, 2, 4, 6, 8]]

    draw_box_panel(
        axes[0],
        "Truthfulness shift",
        results["Truthfulness shift"],
        "Weighted truthful responses (%)",
        100,
        logo_images,
        ["English", "Spanish"],
        ["ID", "OOD"],
    )
    draw_box_panel(
        axes[1],
        "Adversarial shift",
        results["Adversarial shift"],
        "Weighted safe responses (%)",
        100,
        logo_images,
        ["Direct", "Adversaries"],
        ["ID", "OOD"],
        ymin=70,
    )
    draw_box_panel(
        axes[2],
        "Language shift",
        results["Language shift"],
        "Accuracy (%)",
        60,
        logo_images,
        ["Chinese", "French", "Japanese", "Swahili", "Telugu"],
        ["OOD"] * 5,
        group_gap=2.25,
    )
    # Replace repeated language labels with one centered distribution heading.
    for text in list(axes[2].texts):
        text.remove()
    axes[2].text(
        0.5,
        1.035,
        "OOD",
        transform=axes[2].transAxes,
        ha="center",
        va="bottom",
        fontsize=16.5,
        fontweight="bold",
        color="#8B1E1E",
    )
    draw_box_panel(
        axes[3],
        "Context shift",
        results["Context shift"],
        "Citation F1 (%)",
        10,
        logo_images,
        ["8K", "16K"],
        ["ID", "OOD"],
    )
    draw_model_legend(axes[4], logo_images)

    hierarchy_heading(fig, axes, (0, 1), "Parallel Steering")
    hierarchy_heading(fig, axes, (2, 3), "Orthogonal Steering")

    fig.legend(
        handles=[
            Patch(
                facecolor=to_rgba(base.GRAY, 0.30),
                edgecolor="none",
                label="Best competitor",
            ),
            Patch(
                facecolor=to_rgba(base.TEAL, 0.30),
                edgecolor="none",
                label=r"$H_\infty$ (ours)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.49, 0.045),
        ncol=2,
        fontsize=15,
        handlelength=1.2,
        handleheight=0.70,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    export_bbox = Bbox.from_bounds(*EXPORT_BOUNDS)
    label_centers = []
    for ax in axes[:4]:
        bounds = ax.yaxis.label.get_window_extent(renderer)
        label_centers.append(fig.transFigure.inverted().transform(
            ((bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2)))
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * FONT_SCALE)
    for ax in axes[:4]:
        for text in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            text.set_fontsize(text.get_fontsize() * 0.95)
    for ax, center in zip(axes[:4], label_centers, strict=True):
        ax.yaxis.label.set_verticalalignment("center")
        ax.yaxis.set_label_coords(*center, transform=fig.transFigure)
        ax.tick_params(axis="y", pad=max(0, ax.yaxis.majorTicks[0].get_pad() - 4))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes[:4]:
        label = ax.yaxis.label
        bounds = label.get_window_extent(renderer)
        tick_bounds = [text.get_window_extent(renderer) for panel in axes[:4]
                       for text in panel.get_xticklabels()]
        clearance = max([0] + [box.y1 + 4 - bounds.y0 for box in tick_bounds
                               if box.x0 < bounds.x1 and box.x1 > bounds.x0])
        if clearance > 0:
            x, y = label.get_position()
            label.set_position((x, y + clearance / fig.dpi / fig.get_figheight()))
    # Align the heading's M and the visible left edges of all model logos.
    legend_ax = axes[4]
    heading, *model_names = legend_ax.texts
    heading_width = heading.get_window_extent(renderer).width
    column_left = min(legend_ax.get_window_extent(renderer).x0,
                      (export_bbox.x1 - 0.05) * fig.dpi - heading_width)
    heading.set_x(legend_ax.transData.inverted().transform((column_left, 0))[0])
    model_logos = legend_ax.artists
    logo_widths = []
    for logo in model_logos:
        image = logo.offsetbox.get_data()
        _, xs = np.where(image[..., 3] > 10)
        bounds = logo.offsetbox.get_window_extent(renderer)
        visible_left = bounds.x0 + xs.min() / image.shape[1] * bounds.width
        logo_widths.append((xs.max() - xs.min() + 1) / image.shape[1] * bounds.width)
        anchor = legend_ax.transData.transform(logo.xy)
        logo.xy = tuple(legend_ax.transData.inverted().transform(
            (anchor[0] + column_left - visible_left, anchor[1])))
        logo.xybox = logo.xy
    name_x = legend_ax.transData.inverted().transform((column_left + max(logo_widths) + 12, 0))[0]
    for name in model_names:
        name.set_x(name_x)

    for suffix in ["pdf", "png"]:
        fig.savefig(
            PLOTS / f"figure_0_overall_boxplot_improved.{suffix}",
            bbox_inches=export_bbox,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
