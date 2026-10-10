#!/usr/bin/env python3
"""Recreate the GPT-2 token-trajectory PCA panel with an optimized camera.

The hidden-state cache is created once from the ten original sentiment prompts.
Subsequent styling or camera changes only read the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "gpt2_pca_layer_trajectories_base_prompts.pt"
OUT = ROOT / "plots" / "Final Figures"
MODEL = "gpt2"

PROMPTS = [
    "I absolutely loved this movie; the acting was brilliant and the story was moving.",
    "The performance was outstanding, and the atmosphere felt warm and uplifting.",
    "The service was surprisingly smooth and the staff were helpful.",
    "This product is excellent, reliable, and genuinely improved my day.",
    "The experience was delightful from start to finish and felt thoughtfully designed.",
    "This was a terrible experience, and I hated every minute of it.",
    "The product arrived broken and the support was completely useless.",
    "I felt frustrated and disappointed by the entire process and quality.",
    "Everything felt slow, confusing, and badly handled from beginning to end.",
    "The service was awful and the final result left me deeply dissatisfied.",
]

LABELS = {
    3: "This product is\nexcellent, reliable…",
    4: "The experience was\ndelightful…",
    5: "This was a terrible\nexperience…",
    8: "Everything felt slow,\nconfusing…",
}
# The annotated exemplars define the two endpoint categories shown in-panel.
POSITIVE = np.array([3, 4])
NEGATIVE = np.array([5, 8])


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_or_build_cache() -> dict:
    if CACHE.exists():
        return torch.load(CACHE, map_location="cpu", weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True)
    model.eval()
    trajectories = []
    tokens = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt")
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            # The reference panel traces the final-token representation through
            # transformer depth (embedding output + 12 GPT-2 blocks).
            trajectory = torch.stack(
                [hidden[0, -1, :].float().cpu() for hidden in output.hidden_states], dim=0
            )
            trajectories.append(trajectory)
            tokens.append(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

    payload = {
        "model": MODEL,
        "trajectory_axis": "transformer depth",
        "prompts": PROMPTS,
        "tokens": tokens,
        "trajectories": trajectories,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE)
    return payload


def pca_trajectories(payload: dict) -> tuple[list[np.ndarray], PCA]:
    raw = [np.asarray(t, dtype=np.float64) for t in payload["trajectories"]]
    lengths = [len(t) for t in raw]
    joined = np.concatenate(raw, axis=0)
    pca = PCA(n_components=3)
    projected = pca.fit_transform(joined)
    cuts = np.cumsum([0, *lengths])
    trajectories = [projected[cuts[i] : cuts[i + 1]] for i in range(len(raw))]
    return trajectories, pca


def axis_limits(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(trajectories, axis=0)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = 0.69 * np.maximum(np.ptp(points, axis=0), 1e-12)
    return center, radius


def projected_sentiment_score(
    trajectories: list[np.ndarray], elev: float, azim: float, center: np.ndarray, radius: np.ndarray
) -> tuple[float, float, float]:
    """Score how clearly the positive/negative endpoint axis appears on screen."""
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(center[0] - radius[0], center[0] + radius[0])
    ax.set_ylim(center[1] - radius[1], center[1] + radius[1])
    ax.set_zlim(center[2] - radius[2], center[2] + radius[2])
    ax.view_init(elev=elev, azim=azim)
    endpoints = np.stack([trajectory[-1] for trajectory in trajectories])
    x2, y2, _ = proj3d.proj_transform(
        endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], ax.get_proj()
    )
    xy = np.column_stack([x2, y2])
    positive = xy[POSITIVE]
    negative = xy[NEGATIVE]
    positive_center = positive.mean(axis=0)
    negative_center = negative.mean(axis=0)
    delta = positive_center - negative_center
    distance = float(np.linalg.norm(delta))
    horizontal_fraction = float(abs(delta[0]) / max(distance, 1e-12))
    within = 0.5 * (
        np.mean(np.linalg.norm(positive - positive_center, axis=1))
        + np.mean(np.linalg.norm(negative - negative_center, axis=1))
    )
    separation = distance / max(float(within), 1e-12)
    plt.close(fig)
    # Favor category separation, but only accept a nearly dead-on horizontal
    # view of the sentiment axis.
    score = separation * horizontal_fraction**12
    # Keep positive endpoints on the right so the reading direction is stable.
    if delta[0] <= 0:
        score *= 0.1
    return float(score), separation, horizontal_fraction


def choose_camera(trajectories: list[np.ndarray]) -> tuple[float, float, float, float, float]:
    # Conventional three-quarter view used by the original panel.
    return (0.0, 22.0, -58.0, 0.0, 0.0)


def clean_3d_axis(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.965, 0.965, 0.965, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
        axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.set_ticklabels([])
        axis.set_tick_params(length=0)


class Annotation3D(Annotation):
    """A 2D-offset annotation anchored to a point in a 3D axis."""

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self.xyz = xyz

    def draw(self, renderer):
        x2, y2, _ = proj3d.proj_transform(*self.xyz, self.axes.get_proj())
        self.xy = (x2, y2)
        super().draw(renderer)


def plot(trajectories: list[np.ndarray], camera: tuple[float, float, float, float, float]) -> None:
    score, elev, azim, class_separation, horizontal_fraction = camera
    center, radius = axis_limits(trajectories)
    fig = plt.figure(figsize=(5.0, 4.55))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(center[0] - radius[0], center[0] + radius[0])
    ax.set_ylim(center[1] - radius[1], center[1] + radius[1])
    ax.set_zlim(center[2] - radius[2], center[2] + radius[2])
    ax.view_init(elev=elev, azim=azim)
    clean_3d_axis(ax)

    cmap = matplotlib.colormaps["rainbow"]
    for trajectory in trajectories:
        color_positions = np.linspace(0.0, 1.0, len(trajectory))
        for j in range(len(trajectory) - 1):
            segment = trajectory[j : j + 2]
            ax.plot(
                segment[:, 0], segment[:, 1], segment[:, 2],
                color=cmap(color_positions[j]), linewidth=1.15, alpha=0.82,
            )
        ax.scatter(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=color_positions, cmap=cmap, s=12, depthshade=False,
            edgecolors="none", alpha=0.95,
        )

    # Annotations are anchored to the true final-token locations.
    annotation_layout = {
        3: ((18, -4), "left"),
        4: ((28, 34), "left"),
        5: ((-42, -2), "right"),
        8: ((-6, -26), "right"),
    }
    for idx, text in LABELS.items():
        point = trajectories[idx][-1]
        ax.scatter(*point, s=30, color="#9b0000", depthshade=False, zorder=10)
        offset, horizontal_alignment = annotation_layout[idx]
        annotation = Annotation3D(
            text,
            point,
            xytext=offset,
            textcoords="offset points",
            fontsize=9.0,
            color="black",
            ha=horizontal_alignment,
            va="center",
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.8},
            zorder=11,
        )
        ax.add_artist(annotation)

    ax.set_xlabel("PC1", labelpad=-1)
    ax.set_ylabel("PC2", labelpad=-1)
    ax.set_zlabel("PC3", labelpad=-1)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "figure_gpt2_pca_original_clean.png"
    pdf = OUT / "figure_gpt2_pca_original_clean.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    metadata = {
        "embedding": "PCA",
        "projection": "orthographic",
        "box_aspect": [1, 1, 1],
        "camera_elevation_deg": elev,
        "camera_azimuth_deg": azim,
        "camera_separation_score": score,
        "projected_endpoint_class_separation": class_separation,
        "sentiment_axis_horizontal_fraction": horizontal_fraction,
        "camera_objective": "original conventional three-quarter PCA view",
        "model": MODEL,
        "trajectory_axis": "transformer depth",
        "cache": str(CACHE),
    }
    (OUT / "figure_gpt2_pca_original_clean.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    set_style()
    payload = load_or_build_cache()
    trajectories, pca = pca_trajectories(payload)
    camera = choose_camera(trajectories)
    print("explained variance:", pca.explained_variance_ratio_)
    plot(trajectories, camera)


if __name__ == "__main__":
    main()
