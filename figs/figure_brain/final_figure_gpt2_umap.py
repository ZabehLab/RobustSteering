#!/usr/bin/env python3
"""3D UMAP of GPT-2 layer trajectories, oriented by sentiment separation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import proj3d


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "gpt2_pca_layer_trajectories_base_prompts.pt"
OUT = ROOT / "plots" / "Final Figures"

POSITIVE = np.arange(0, 5)
NEGATIVE = np.arange(5, 10)
LABELS = {
    3: "This product is\nexcellent, reliable…",
    4: "The experience was\ndelightful…",
    5: "This was a terrible\nexperience…",
    8: "Everything felt slow,\nconfusing…",
}


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self.xyz = xyz

    def draw(self, renderer):
        x2, y2, _ = proj3d.proj_transform(*self.xyz, self.axes.get_proj())
        self.xy = (x2, y2)
        super().draw(renderer)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_trajectories() -> tuple[list[np.ndarray], dict]:
    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    trajectories = [np.asarray(t, dtype=np.float32) for t in payload["trajectories"]]
    return trajectories, payload


def align_sentiment_axis(points: np.ndarray, lengths: list[int]) -> tuple[np.ndarray, dict]:
    """Rotate UMAP coordinates so axis 1 is negative-to-positive endpoint separation."""
    cuts = np.cumsum([0, *lengths])
    endpoints = np.stack([points[cuts[i + 1] - 1] for i in range(len(lengths))])
    negative_center = endpoints[NEGATIVE].mean(axis=0)
    positive_center = endpoints[POSITIVE].mean(axis=0)
    e1 = positive_center - negative_center
    e1 /= np.linalg.norm(e1)

    centered = points - points.mean(axis=0, keepdims=True)
    orthogonal = centered - np.outer(centered @ e1, e1)
    _, _, vh = np.linalg.svd(orthogonal, full_matrices=False)
    e2 = vh[0] - np.dot(vh[0], e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)
    rotation = np.column_stack([e1, e2, e3])
    aligned = centered @ rotation

    aligned_endpoints = np.stack([aligned[cuts[i + 1] - 1] for i in range(len(lengths))])
    if aligned_endpoints[POSITIVE, 0].mean() < aligned_endpoints[NEGATIVE, 0].mean():
        aligned[:, 0] *= -1
        e1 *= -1

    info = {
        "negative_endpoint_centroid": negative_center.tolist(),
        "positive_endpoint_centroid": positive_center.tolist(),
        "sentiment_axis_original_umap_coordinates": e1.tolist(),
    }
    return aligned, info


def embed(trajectories: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    lengths = [len(t) for t in trajectories]
    if len(set(lengths)) != 1:
        raise ValueError("Layer trajectories must have equal lengths for depth-wise centering")
    stacked = np.stack(trajectories, axis=0)
    # Remove the representation shared by all prompts at each transformer depth.
    # Otherwise UMAP recovers layer identity almost exclusively and obscures the
    # content-dependent (here, sentiment-dependent) geometry.
    centered_by_depth = stacked - stacked.mean(axis=0, keepdims=True)
    joined = centered_by_depth.reshape(-1, centered_by_depth.shape[-1])
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=6,
        min_dist=0.12,
        metric="cosine",
        random_state=0,
        transform_seed=0,
        init="spectral",
    )
    raw = reducer.fit_transform(joined)
    aligned, alignment = align_sentiment_axis(raw, lengths)
    cuts = np.cumsum([0, *lengths])
    split = [aligned[cuts[i] : cuts[i + 1]] for i in range(len(lengths))]
    return split, alignment


def clean_axis(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.965, 0.965, 0.965, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
        axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.72)
        axis._axinfo["grid"]["linewidth"] = 0.7
        axis.set_ticklabels([])
        axis.set_tick_params(length=0)


def plot(trajectories: list[np.ndarray], alignment: dict) -> None:
    points = np.concatenate(trajectories, axis=0)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = 0.58 * np.maximum(np.ptp(points, axis=0), 1e-12)

    fig = plt.figure(figsize=(5.0, 4.55))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(center[0] - radius[0], center[0] + radius[0])
    ax.set_ylim(center[1] - radius[1], center[1] + radius[1])
    ax.set_zlim(center[2] - radius[2], center[2] + radius[2])
    # This camera keeps the explicitly aligned sentiment axis broadside.
    ax.view_init(elev=48, azim=-80)
    clean_axis(ax)

    cmap = matplotlib.colormaps["rainbow"]
    for trajectory in trajectories:
        depth = np.linspace(0.0, 1.0, len(trajectory))
        for j in range(len(trajectory) - 1):
            seg = trajectory[j : j + 2]
            ax.plot(*seg.T, color=cmap(depth[j]), linewidth=1.15, alpha=0.82)
        ax.scatter(
            *trajectory.T,
            c=depth,
            cmap=cmap,
            s=13,
            edgecolors="none",
            depthshade=False,
            alpha=0.96,
        )

    offsets = {3: (20, 13), 4: (18, 42), 5: (18, -12), 8: (18, -42)}
    for idx, text in LABELS.items():
        point = trajectories[idx][-1]
        ax.scatter(*point, s=31, color="#9b0000", depthshade=False, zorder=10)
        ann = Annotation3D(
            text,
            point,
            xytext=offsets[idx],
            textcoords="offset points",
            fontsize=8.3,
            color="black",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.8},
            zorder=11,
        )
        ax.add_artist(ann)

    ax.set_xlabel("Negative  ←  sentiment  →  Positive", labelpad=5)
    ax.set_ylabel("UMAP 2", labelpad=3)
    ax.set_zlabel("UMAP 3", labelpad=2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "figure_gpt2_umap_sentiment_axis.png"
    pdf = OUT / "figure_gpt2_umap_sentiment_axis.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    metadata = {
        "embedding": "UMAP",
        "n_components": 3,
        "n_neighbors": 6,
        "min_dist": 0.12,
        "metric": "cosine",
        "random_state": 0,
        "projection": "orthographic",
        "box_aspect": [1, 1, 1],
        "camera_elevation_deg": 48,
        "camera_azimuth_deg": -80,
        "orientation": "UMAP coordinates rotated so axis 1 joins negative and positive endpoint centroids",
        "preprocessing": "subtract across-prompt mean hidden state independently at each transformer depth",
        "cache": str(CACHE),
        **alignment,
    }
    (OUT / "figure_gpt2_umap_sentiment_axis.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    set_style()
    raw, _ = load_trajectories()
    trajectories, alignment = embed(raw)
    plot(trajectories, alignment)


if __name__ == "__main__":
    main()
