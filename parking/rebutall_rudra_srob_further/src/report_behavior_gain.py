#!/usr/bin/env python3
"""Render the active H-infinity figure and summarize cached behavioral-signal slopes."""

from __future__ import annotations

import csv
import hashlib
import json
import math

from analyze import REPO, TABLES, UNIT, configure_plot_style
from plot_truthful_harm_ratios import COLORS

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import linregress


MEASURES = {
    "TruthfulQA": {"T": "truth", "I": "info", "T×I": "txi", "Concept relevance": "concept_relevance",
                   "Instruction relevance": "instruction_relevance", "Fluency": "fluency", "AXBench overall": "axbench_overall"},
    "HarmBench": {"ASR": "harmbench_test_success.score", "Safe-concept relevance": "axbench_concept_relevance.score",
                  "Instruction relevance": "axbench_instruction_relevance.score", "Fluency": "axbench_fluency.score",
                  "AXBench overall": "axbench_overall.score"},
    "MGSM": {"Accuracy": "mgsm_exact_match.score", "Spanish relevance": "axbench_rule_spanish.score",
             "Instruction relevance": "axbench_instruction_relevance.score", "Fluency": "axbench_fluency.score",
             "AXBench overall": "mgsm_axbench_overall.score", "Answer parse rate": "mgsm_exact_match.valid_rate"},
}
MODEL_COUNTS = {"TruthfulQA": 5, "HarmBench": 3, "MGSM": 3}
LANGUAGES = ("zh", "fr", "ja", "sw", "te")
PANELS = (("TruthfulQA", "T", (0.1, 0.85)), ("TruthfulQA", "T×I", (-0.05, 1.65)),
          ("HarmBench", "ASR", (-1.02, -0.38)))


def metric_value(result: dict, key: str) -> float | None:
    if key == "txi":
        truth = metric_value(result, "truth")
        info = metric_value(result, "info")
        return None if truth is None or info is None else truth * info / 100
    value = result["metrics"].get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def load_signals() -> tuple[list[dict], dict]:
    sources = {}

    def read_json(path):
        content = path.read_bytes()
        sources[str(path.relative_to(REPO))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    points = []

    def append_gains(benchmark, model, srob, split, original_values, hinf_values):
        for measure in MEASURES[benchmark]:
            original = original_values[measure]
            hinf = hinf_values[measure]
            gain = None
            status = "available"
            if original is None or hinf is None:
                status = "missing or non-finite cached metric"
            elif original <= 0:
                status = "zero Original denominator"
            else:
                gain = (hinf - original) / original
            points.append({"benchmark": benchmark, "model": model, "split": split, "srob": srob,
                           "measure": measure, "original": original, "hinf": hinf,
                           "behavior_gain": gain, "status": status})

    for benchmark, filename in (("TruthfulQA", "truthfulqa_srob_truthful.json"),
                                ("HarmBench", "harmbench_srob_asr.json")):
        for row in read_json(TABLES / filename):
            original = read_json(REPO / row["original_result_path"])
            hinf = read_json(REPO / row["hinf_result_path"])
            if benchmark == "TruthfulQA":
                keys = ("model_id", "model_revision", "distribution", "evaluation_key", "dataset", "kv_cache")
            else:
                keys = ("model", "condition", "evaluated_model_kv_cache")
            if (original["identity"]["method"] != "original" or hinf["identity"]["method"] != "h_infinity"
                    or any(original["identity"][key] != hinf["identity"][key] for key in keys)):
                raise ValueError(f"Paired evaluation identity mismatch for {row['model']}, {row['condition']}")
            split = "OOD" if row["condition"].endswith("(OOD)") else "ID"
            append_gains(benchmark, row["model"], row["srob"], split,
                         {name: metric_value(original, key) for name, key in MEASURES[benchmark].items()},
                         {name: metric_value(hinf, key) for name, key in MEASURES[benchmark].items()})

    for row in read_json(TABLES / "mgsm_srob_accuracy_vs_alqr.json"):
        cache_key = (REPO / row["selection_path"]).parents[3].name
        aggregates = {}
        for method in ("original", "h_infinity"):
            language_results = []
            for language in LANGUAGES:
                path = REPO / "benchmarks/mgsm/results/kv_cache_off" / cache_key / f"mgsm_{language}" / f"{method}.json"
                result = read_json(path)
                if (result["model"] != [row["model_id"], row["model_revision"]]
                        or result["method"] != method or result["input_language"] != language):
                    raise ValueError(f"MGSM result identity mismatch in {path}")
                language_results.append(result)
            aggregates[method] = {}
            for measure, key in MEASURES["MGSM"].items():
                values = [metric_value(result, key) for result in language_results]
                aggregates[method][measure] = None if None in values else sum(values) / len(values)
        append_gains("MGSM", row["model"], row["srob"], "OOD", aggregates["original"], aggregates["h_infinity"])
    return points, sources


def fit_signals(points: list[dict]) -> list[dict]:
    slopes = []
    for benchmark, measures in MEASURES.items():
        for measure in measures:
            for split in COLORS:
                selected = [point for point in points if point["benchmark"] == benchmark
                            and point["measure"] == measure and point["split"] == split
                            and point["behavior_gain"] is not None]
                slope = intercept = None
                if len(selected) >= 2:
                    fit = linregress([point["srob"] for point in selected], [point["behavior_gain"] for point in selected])
                    slope, intercept = float(fit.slope), float(fit.intercept)
                slopes.append({"benchmark": benchmark, "measure": measure, "split": split,
                               "n_models": len(selected), "expected_models": MODEL_COUNTS[benchmark],
                               "slope": slope, "intercept": intercept})
    return slopes


def render(points: list[dict], slopes: list[dict]) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    labels = []
    for panel, (ax, (benchmark, measure, y_limits)) in enumerate(zip(axes, PANELS, strict=True)):
        for split, color in COLORS.items():
            selected = [point for point in points if point["benchmark"] == benchmark
                        and point["measure"] == measure and point["split"] == split]
            if len(selected) != MODEL_COUNTS[benchmark] or any(point["behavior_gain"] is None for point in selected):
                raise ValueError(f"Incomplete main figure values for {benchmark}, {measure}, {split}")
            fit = next(row for row in slopes if (row["benchmark"], row["measure"], row["split"]) == (benchmark, measure, split))
            x = np.array([point["srob"] for point in selected])
            y = np.array([point["behavior_gain"] for point in selected])
            curve_x = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(curve_x, fit["intercept"] + fit["slope"] * curve_x, color=color,
                    linewidth=1.5, alpha=0.8, zorder=1)
            ax.scatter(x, y, s=48, color=color, zorder=3)
            fraction = 0.70 if split == "ID" else 0.35
            label_x = float(x.min() + fraction * (x.max() - x.min()))
            labels.append((ax, label_x, fit, color, float(x.min()), float(x.max())))
        ax.set(title=f"{benchmark} · {measure}", xlabel=r"$S_{\mathrm{rob}}$", ylabel=f"{measure} behavior gain", ylim=y_limits)
        if panel < 2:
            ax.set(xlim=(-0.007, 0.46), xticks=[0, 0.1, 0.2, 0.3, 0.4])
        else:
            ax.set(xlim=(1.9, 8.5), xticks=[2, 3, 4, 5, 6, 7, 8])
    handles = [Line2D([0], [0], color=color, marker="o", linewidth=1.5, markersize=6, label=split)
               for split, color in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.88, wspace=0.34)
    fig.canvas.draw()
    for ax, label_x, fit, color, x_min, x_max in labels:
        endpoints = ax.transData.transform([[x_min, fit["intercept"] + fit["slope"] * x_min],
                                          [x_max, fit["intercept"] + fit["slope"] * x_max]])
        dx, dy = endpoints[1] - endpoints[0]
        angle = np.degrees(np.arctan2(dy, dx))
        ax.text(label_x, fit["intercept"] + fit["slope"] * label_x, f"slope = {fit['slope']:.3f}",
                ha="center", va="center", color=color, rotation=angle, rotation_mode="anchor", fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.3}, zorder=5)
    for suffix in ("png", "pdf"):
        fig.savefig(UNIT / f"hinf_behavior_gain.{suffix}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(slopes: list[dict]) -> None:
    lines = ["# H∞ behavior gain", "", "Behavior gain = (H∞ − Original) / Original.", "",
             "![T, T×I, and ASR behavior gain](hinf_behavior_gain.png)", "",
             "[PDF](hinf_behavior_gain.pdf) · [Signal values](tables/behavior_signal_points.csv) · "
             "[Slope data](tables/behavior_signal_slopes.csv) · [Methods](src/README.md#report_behavior_gain)", "",
             "Rows are ordered by the largest absolute available slope within each benchmark. "
             "Model counts are shown when a slope uses fewer than the full reported model set.", ""]
    for benchmark, measures in MEASURES.items():
        lines.extend([f"## {benchmark}", "", "| Measure | ID slope | OOD slope |", "|---|---:|---:|"])
        grouped = {measure: {row["split"]: row for row in slopes if row["benchmark"] == benchmark and row["measure"] == measure}
                   for measure in measures}
        def magnitude(measure):
            return max((abs(row["slope"]) for row in grouped[measure].values() if row["slope"] is not None), default=-1)
        for measure in sorted(measures, key=magnitude, reverse=True):
            values = []
            for split in COLORS:
                row = grouped[measure][split]
                if benchmark == "MGSM" and split == "ID":
                    value = "—"
                elif row["slope"] is None:
                    value = "Undefined"
                else:
                    value = f"{row['slope']:+.6f}"
                    if row["n_models"] < row["expected_models"]:
                        value += f" (n={row['n_models']})"
                values.append(value)
            lines.append(f"| {measure} | {' | '.join(values)} |")
        lines.append("")
    lines.extend(["MGSM uses the five-language OOD average and has no evaluated ID condition. "
                  "Its Spanish relevance and AXBench overall gains are undefined because Original is zero.", "",
                  "TruthfulQA concept relevance has paired values for three models. AXBench overall has "
                  "two finite model pairs. Missing or non-finite values are excluded, with counts shown above.", "",
                  "Earlier figure variants are retained in [archive/figures](archive/figures). "
                  "Run `python src/report_behavior_gain.py` from this unit to rebuild the active outputs.", ""])
    (UNIT / "README.md").write_text("\n".join(lines))


def main() -> None:
    points, sources = load_signals()
    slopes = fit_signals(points)
    TABLES.mkdir(exist_ok=True)
    for stem, records in (("behavior_signal_points", points), ("behavior_signal_slopes", slopes)):
        with (TABLES / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        (TABLES / f"{stem}.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (TABLES / "behavior_signal_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    render(points, slopes)
    write_report(slopes)
    print((UNIT / "README.md").read_text())


if __name__ == "__main__":
    main()
