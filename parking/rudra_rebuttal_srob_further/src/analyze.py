#!/usr/bin/env python3
"""Join the five paper-figure Srob points to their matched TruthfulQA results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


UNIT = Path(__file__).resolve().parents[1]
REPO = UNIT.parents[1]
FIGURE_POINTS = REPO / "parking/gamma_star_organization/plots/srob_parameter_size_truthfulness_main.csv"
TABLES = UNIT / "tables"
PLOTS = UNIT / "archive/figures"
sys.path.insert(0, str(REPO / "parking/gamma_star_addition"))
from build_truthfulness_comparison import MODELS, metric, result_path

MODEL_LABELS = {
    "gpt2_xl": "GPT-2 XL",
    "gemma2_2b": "Gemma-2-2B",
    "llama31_8b": "Llama-3-8B",
    "qwen25_14b": "Qwen-2.5-14B",
    "olmo2_32b": "OLMo-2-32B",
}
CONDITIONS = {
    "truthfulness": ("English (ID)", "id", "#007C7C"),
    "truthfulness_spanish": ("Spanish-input (OOD)", "spanish", "#8B1E1E"),
}


def load_rows() -> tuple[list[dict], dict]:
    sources = {}

    def read_json(path: Path) -> dict:
        content = path.read_bytes()
        value = json.loads(content)
        sources[str(path.relative_to(REPO))] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "identity": value.get("identity"),
        }
        return value

    with FIGURE_POINTS.open(newline="") as handle:
        points = list(csv.DictReader(handle))
    if len(points) != 5 or {point["model_key"] for point in points} != MODEL_LABELS.keys():
        raise ValueError("The source figure must contain the five requested paper models")
    sources[str(FIGURE_POINTS.relative_to(REPO))] = {
        "sha256": hashlib.sha256(FIGURE_POINTS.read_bytes()).hexdigest()
    }
    bases = dict(MODELS)
    rows = []
    for point in sorted(points, key=lambda item: float(item["parameter_billions"])):
        label = MODEL_LABELS[point["model_key"]]
        selection_path = REPO / point["selection_path"]
        selection = read_json(selection_path)
        score_path = selection_path.parent / selection["diagnostic_bundle"] / "score.json"
        score = read_json(score_path)
        gamma_star = float(point["gamma_star"])
        srob = float(point["s_rob_value"])
        if not (math.isclose(gamma_star, score["gamma_star"], rel_tol=1e-12)
                and math.isclose(srob, score["s_rob"], rel_tol=1e-12)
                and math.isclose(srob * gamma_star, 1.0, rel_tol=1e-12)):
            raise ValueError(f"Figure and controller diagnostic disagree for {label}")

        for evaluation_key, (condition, distribution, _) in CONDITIONS.items():
            results = {}
            paths = {}
            for method in ("original", "h_infinity"):
                path = result_path(label, bases[label], method).parent.parent / evaluation_key / f"{method}.json"
                result = read_json(path)
                identity = result["identity"]
                if (identity["model_id"], identity["model_revision"]) != tuple(selection["model"]):
                    raise ValueError(f"Model/checkpoint mismatch in {path}")
                if (identity["method"], identity["distribution"], identity["evaluation_key"],
                        identity["kv_cache"]) != (method, distribution, evaluation_key, False):
                    raise ValueError(f"Evaluation identity mismatch in {path}")
                results[method] = result
                paths[method] = str(path.relative_to(REPO))
            if results["original"]["identity"]["dataset"] != results["h_infinity"]["identity"]["dataset"]:
                raise ValueError(f"Dataset mismatch for {label}, {condition}")

            # Generation identities bind legacy result files to the selected costs.
            generation_dir = selection_path.parents[3] / "evaluations/kv_cache_off"
            evaluation_calibration = results["h_infinity"]["identity"].get("calibration_id")
            if evaluation_calibration:
                generation_dir = generation_dir / "calibrations" / evaluation_calibration
            generation_path = generation_dir / "generations" / evaluation_key / "h_infinity/final.json"
            generation = read_json(generation_path)
            identity = generation["identity"]
            if (generation["status"] != "complete"
                    or identity["parameters"] != selection["selected"]["parameters"]
                    or (identity["model_id"], identity["checkpoint_revision"]) != tuple(selection["model"])
                    or identity["distribution"] != distribution):
                raise ValueError(f"H∞ generation does not match the plotted controller for {label}")

            original_txi = metric(results["original"], "txi")[0]
            hinf_txi = metric(results["h_infinity"], "txi")[0]
            if not (0 < original_txi <= 100 and 0 <= hinf_txi <= 100):
                raise ValueError(f"Invalid T×I values for {label}, {condition}")
            rows.append({
                "model": label,
                "model_key": point["model_key"],
                "parameter_billions": float(point["parameter_billions"]),
                "srob": srob,
                "gamma_star": gamma_star,
                "condition": condition,
                "evaluation_key": evaluation_key,
                "original_txi_pct": original_txi,
                "hinf_txi_pct": hinf_txi,
                "hinf_over_original_txi": hinf_txi / original_txi,
                "hinf_minus_original_txi_pp": hinf_txi - original_txi,
                "original_repetitions": results["original"]["evaluation_repetitions"],
                "hinf_repetitions": results["h_infinity"]["evaluation_repetitions"],
                "original_samples_per_repetition": results["original"]["evaluation_samples_per_repetition"],
                "hinf_samples_per_repetition": results["h_infinity"]["evaluation_samples_per_repetition"],
                "calibration_id": selection["calibration_id"],
                "original_result_path": paths["original"],
                "hinf_result_path": paths["h_infinity"],
                "selection_path": point["selection_path"],
                "score_path": str(score_path.relative_to(REPO)),
                "hinf_generation_path": str(generation_path.relative_to(REPO)),
            })
    return rows, sources


def save_tables(rows: list[dict], sources: dict) -> None:
    TABLES.mkdir(exist_ok=True)
    with (TABLES / "truthfulqa_srob.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / "truthfulqa_srob.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (TABLES / "source_manifest.json").write_text(json.dumps(sources, indent=2, allow_nan=False) + "\n")
    lines = [
        "# TruthfulQA Srob and T×I",
        "",
        "The five paper-figure models. T×I is True (%) × Informative (%) / 100.",
        "Srob and γ★ retain the source CSV precision. T×I, ratios, and differences are displayed to six decimals.",
        "The CSV and JSON retain the full saved floating-point precision.",
        "",
        "| Model | Condition | Srob | γ★ | Original T×I (%) | H∞ T×I (%) | H∞ / Original | H∞ − Original (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['srob']!r} | {row['gamma_star']!r} | "
            f"{row['original_txi_pct']:.6f} | {row['hinf_txi_pct']:.6f} | "
            f"{row['hinf_over_original_txi']:.6f} | {row['hinf_minus_original_txi_pp']:+.6f} |"
        )
    (TABLES / "truthfulqa_srob.md").write_text("\n".join(lines) + "\n")


def fit_record(x, y) -> dict:
    fit = linregress(x, y)
    return {"n": len(x), "slope": float(fit.slope), "intercept": float(fit.intercept),
            "pearson_r": float(fit.rvalue), "p_two_sided": float(fit.pvalue),
            "slope_standard_error": float(fit.stderr)}


def save_figure(fig, stem: str) -> None:
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(PLOTS / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def configure_plot_style() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 11,
                         "axes.labelsize": 12, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def render_behavior_figure(
    rows: list[dict], outcomes: tuple, stem: str, label_offsets: dict | None = None,
    *, conditions: dict = CONDITIONS, x_limits: tuple = (-0.007, 0.46),
    x_ticks: tuple = (0, 0.1, 0.2, 0.3, 0.4), y_limits: tuple | None = None,
) -> dict:
    PLOTS.mkdir(exist_ok=True)
    configure_plot_style()
    statistics = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex=True, sharey="row")
    offsets = {"gpt2_xl": (10, 5), "gemma2_2b": (10, -12), "llama31_8b": (9, -12),
               "qwen25_14b": (9, 7), "olmo2_32b": (-9, 9)}
    label_offsets = label_offsets or {}
    for column, (evaluation_key, (condition, _, color)) in enumerate(conditions.items()):
        selected = [row for row in rows if row["evaluation_key"] == evaluation_key]
        x = np.array([row["srob"] for row in selected])
        axes[0, column].set_title(condition, color=color, pad=12)
        for panel_row, (outcome, ylabel, reference) in enumerate(outcomes):
            ax = axes[panel_row, column]
            y = np.array([row[outcome] for row in selected])
            fit = fit_record(x, y)
            statistics[f"{evaluation_key}_{outcome}"] = fit
            curve_x = np.linspace(float(x.min()), float(x.max()), 100)
            ax.axhline(reference, color="#c9c9c9", linewidth=1, zorder=0)
            ax.plot(curve_x, fit["intercept"] + fit["slope"] * curve_x,
                    color=color, linewidth=1.2, alpha=0.65, zorder=1)
            ax.scatter(x, y, s=52, color=color, zorder=3)
            for row in selected:
                offset = label_offsets.get(
                    (evaluation_key, outcome, row["model_key"]), offsets.get(row["model_key"], (10, 5))
                )
                ax.annotate(row["model"], (row["srob"], row[outcome]), xytext=offset,
                            textcoords="offset points", ha="right" if offset[0] < 0 else "left",
                            fontsize=10, arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#999999"})
            ax.set_xlim(x_limits)
            ax.set_xticks(x_ticks)
            ax.text(0.97, 0.04, f"r = {fit['pearson_r']:.3f}, p = {fit['p_two_sided']:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
            if column == 0:
                ax.set_ylabel(ylabel)
            if panel_row == 1:
                ax.set_xlabel(r"$S_{\mathrm{rob}}$")
    if y_limits is None:
        y_limits = ((0.90, max(row[outcomes[0][0]] for row in rows) + 0.25),
                    (-2, max(row[outcomes[1][0]] for row in rows) + 6))
    axes[0, 0].set_ylim(y_limits[0])
    axes[1, 0].set_ylim(y_limits[1])
    save_figure(fig, stem)
    return statistics


def render_figures(rows: list[dict]) -> None:
    outcomes = (
        ("hinf_over_original_txi", "H∞ T×I / Original T×I", 1.0),
        ("hinf_minus_original_txi_pp", "H∞ T×I − Original T×I (pp)", 0.0),
    )
    statistics = render_behavior_figure(rows, outcomes, "srob_txi_changes")
    (TABLES / "regressions.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")


def main() -> None:
    rows, sources = load_rows()
    save_tables(rows, sources)
    render_figures(rows)
    print((TABLES / "truthfulqa_srob.md").read_text())
    print(f"Verified {len(rows)} rows against {len(sources)} source files")


if __name__ == "__main__":
    main()
