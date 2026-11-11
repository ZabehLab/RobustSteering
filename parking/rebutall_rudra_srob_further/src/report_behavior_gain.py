#!/usr/bin/env python3
"""Render the active H-infinity figure and summarize cached behavioral-signal slopes."""

from __future__ import annotations

import argparse
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
LANGUAGES = {"zh": "Chinese", "fr": "French", "ja": "Japanese", "sw": "Swahili", "te": "Telugu"}
JAILBREAKS = {0: "John", 1: "DNE", 2: "Jailbreak Bot", 3: "YOJA/Nona", 4: "APM"}
PANELS = (("TruthfulQA", "T", "ID"), ("TruthfulQA", "T", "OOD"),
          ("HarmBench", "ASR", "ID"), ("HarmBench", "ASR", "OOD"),
          ("MGSM", "Accuracy", "OOD"))


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

    def append_gains(benchmark, model, srob, split, original_values, hinf_values, language=None, template=None):
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
                           "language": language, "template": template,
                           "measure": measure, "original": original, "hinf": hinf,
                           "behavior_gain": gain, "status": status})

    harmbench_summary = read_json(REPO / "figs/bench_table/harmful/harmbench_jackknife.json")["models"]
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
            if benchmark == "HarmBench" and split == "OOD":
                cache_key = (REPO / row["selection_path"]).parents[3].name
                summary = harmbench_summary[cache_key]
                if summary["behavior_count"] != row["behavior_count"]:
                    raise ValueError(f"HarmBench behavior subset mismatch for {row['model']}")
                template_values = {"original": [], "h_infinity": []}
                for template in JAILBREAKS:
                    condition = summary["conditions"][str(template)]
                    values = {}
                    for method in template_values:
                        values[method] = {
                            measure: condition[method][key.removesuffix(".score")]["full_sample_mean"]
                            / (100 if measure == "ASR" else 1)
                            for measure, key in MEASURES["HarmBench"].items()
                        }
                        template_values[method].append(values[method])
                    append_gains("HarmBench", row["model"], row["srob"], "OOD",
                                 values["original"], values["h_infinity"], template=template)
                for method, result in (("original", original), ("h_infinity", hinf)):
                    for measure, key in MEASURES["HarmBench"].items():
                        pooled = sum(values[measure] for values in template_values[method]) / len(JAILBREAKS)
                        if not math.isclose(pooled, metric_value(result, key), rel_tol=1e-12, abs_tol=1e-12):
                            raise ValueError(f"HarmBench template means disagree with pooled {measure} for {row['model']}")

    for row in read_json(TABLES / "mgsm_srob_accuracy_vs_alqr.json"):
        cache_key = (REPO / row["selection_path"]).parents[3].name
        for language in LANGUAGES:
            results = {}
            for method in ("original", "h_infinity"):
                path = REPO / "benchmarks/mgsm/results/kv_cache_off" / cache_key / f"mgsm_{language}" / f"{method}.json"
                result = read_json(path)
                if (result["model"] != [row["model_id"], row["model_revision"]]
                        or result["method"] != method or result["input_language"] != language):
                    raise ValueError(f"MGSM result identity mismatch in {path}")
                results[method] = {measure: metric_value(result, key) for measure, key in MEASURES["MGSM"].items()}
            append_gains("MGSM", row["model"], row["srob"], "OOD", results["original"], results["h_infinity"], language)
    return points, sources


def fit_signals(points: list[dict]) -> list[dict]:
    slopes = []
    for benchmark, measures in MEASURES.items():
        conditions = [("OOD", language, None) for language in LANGUAGES] if benchmark == "MGSM" else [
            (split, None, None) for split in COLORS]
        if benchmark == "HarmBench":
            conditions.extend(("OOD", None, template) for template in JAILBREAKS)
        for measure in measures:
            for split, language, template in conditions:
                selected = [point for point in points if point["benchmark"] == benchmark
                            and point["measure"] == measure and point["split"] == split
                            and point["language"] == language
                            and point["template"] == template
                            and point["behavior_gain"] is not None]
                slope = intercept = None
                if len(selected) >= 2:
                    fit = linregress([point["srob"] for point in selected], [point["behavior_gain"] for point in selected])
                    slope, intercept = float(fit.slope), float(fit.intercept)
                slopes.append({"benchmark": benchmark, "measure": measure, "split": split,
                               "language": language, "template": template,
                               "n_models": len(selected), "expected_models": MODEL_COUNTS[benchmark],
                               "slope": slope, "intercept": intercept})
    return slopes


def mean_ood_slopes(slopes: list[dict]) -> list[dict]:
    means = []
    for benchmark in ("HarmBench", "MGSM"):
        for measure in MEASURES[benchmark]:
            conditions = [row for row in slopes if row["benchmark"] == benchmark and row["measure"] == measure
                          and row["split"] == "OOD" and (row["language"] is not None or row["template"] is not None)]
            if len(conditions) != 5:
                raise ValueError(f"Expected five OOD condition slopes for {benchmark}, {measure}")
            values = [row["slope"] for row in conditions if row["slope"] is not None]
            means.append({"benchmark": benchmark, "measure": measure, "condition_count": len(conditions),
                          "defined_condition_count": len(values),
                          "mean_ood_slope": sum(values) / 5 if len(values) == 5 else None})
    return means


def render(points: list[dict], slopes: list[dict]) -> None:
    configure_plot_style()
    red = "#8B1E1E"
    markers = ("o", "x", "+", "s", "^")
    panel_width = (25 * (0.995 - 0.028) / (5 + 4 * 0.40)) / 3
    panel_height = 4 * (0.87 - 0.30)
    positions = [0.65]
    for gap in (0.60, 0.95, 0.60, 2.10):
        positions.append(positions[-1] + panel_width + gap)
    figure_width = positions[-1] + panel_width + 1.10
    fig = plt.figure(figsize=(figure_width, 4))
    axes = [fig.add_axes([left / figure_width, 0.30, panel_width / figure_width, panel_height / 4])
            for left in positions]
    for ax, (benchmark, measure, split) in zip(axes, PANELS, strict=True):
        multiple = split == "OOD" and benchmark in ("HarmBench", "MGSM")
        conditions = JAILBREAKS if benchmark == "HarmBench" else LANGUAGES
        series = [(key, name, markers[i]) for i, (key, name) in enumerate(conditions.items())] if multiple else [
            (None, split, "o")]
        color = "black" if split == "ID" else red
        fits = []
        handles = []
        for condition, name, marker in series:
            language = condition if multiple and benchmark == "MGSM" else None
            template = condition if multiple and benchmark == "HarmBench" else None
            selected = [point for point in points if point["benchmark"] == benchmark
                        and point["measure"] == measure and point["split"] == split
                        and point["language"] == language and point["template"] == template]
            if len(selected) != MODEL_COUNTS[benchmark] or any(point["behavior_gain"] is None for point in selected):
                raise ValueError(f"Incomplete figure values for {benchmark}, {measure}, {split}, {name}")
            fit = next(row for row in slopes if (row["benchmark"], row["measure"], row["split"], row["language"], row["template"])
                       == (benchmark, measure, split, language, template))
            x = np.array([point["srob"] for point in selected])
            y = np.array([point["behavior_gain"] for point in selected])
            curve_x = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(curve_x, fit["intercept"] + fit["slope"] * curve_x,
                    color="#c9c9c9" if multiple else color, linewidth=1 if multiple else 1.8, zorder=1)
            marker_style = {"color": "black"}
            if multiple:
                marker_style = {"linewidths": 1.1, **({"color": "black"} if marker in ("x", "+") else
                                {"facecolors": "none", "edgecolors": "black"})}
            ax.scatter(x, y, s=29 if multiple else 36, marker=marker,
                       zorder=5 if multiple else 3, **marker_style)
            fits.append(fit)
            handles.append(Line2D([0], [0], color="black", marker=marker, linestyle="none",
                                  markerfacecolor="none" if multiple else "black", markersize=4, label=name))
        if multiple:
            mean_fit = {key: float(np.mean([fit[key] for fit in fits])) for key in ("slope", "intercept")}
            ax.plot(curve_x, mean_fit["intercept"] + mean_fit["slope"] * curve_x,
                    color=red, linewidth=3.2, zorder=4)
            handles.append(Line2D([0], [0], color=red, linewidth=3.2, label="Mean"))
            lower, upper = ax.dataLim.ymin, ax.dataLim.ymax
            span = upper - lower
            ax.set_ylim(lower - 0.12 * span, upper + 0.20 * span)
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.10, 1), frameon=False,
                      fontsize=6, borderpad=0.20, labelspacing=0.15,
                      handlelength=0.8, handletextpad=0.30)
        else:
            ax.margins(y=0.20)
        ax.set(title=split, xlabel=r"$S_{\mathrm{rob}}$", ylabel=f"{measure} behavior gain")
        ax.title.set_color(color)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if benchmark == "TruthfulQA":
            ax.set(xlim=(-0.007, 0.46), xticks=[0, 0.2, 0.4])
        elif benchmark == "HarmBench":
            ax.set(xlim=(1.9, 8.5), xticks=[2, 5, 8])
        else:
            ax.set(xlim=(0, 1), xticks=[0, 0.5, 1])
    for first, last, name in ((0, 1, "TruthfulQA"), (2, 3, "HarmBench"), (4, 4, "MGSM")):
        center = (positions[first] + positions[last] + panel_width) / (2 * figure_width)
        fig.text(center, 0.98, name, ha="center", va="center", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(UNIT / f"hinf_behavior_gain.{suffix}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(slopes: list[dict], ood_means: list[dict]) -> None:
    lines = ["# H∞ behavior gain", "", "Behavior gain = (H∞ − Original) / Original.", "",
             "![TruthfulQA T and HarmBench ASR behavior gain](hinf_behavior_gain.png)", "",
             "[PDF](hinf_behavior_gain.pdf) · [Signal values](tables/behavior_signal_points.csv) · "
             "[Slope data](tables/behavior_signal_slopes.csv) · [Methods](src/README.md#report_behavior_gain)", "",
             "Rows are ordered by the largest absolute available slope within each benchmark. "
             "Model counts are shown when a slope uses fewer than the full reported model set.", ""]
    for benchmark, measures in MEASURES.items():
        if benchmark == "MGSM":
            columns = [("OOD", language, None, name) for language, name in LANGUAGES.items()]
        elif benchmark == "HarmBench":
            columns = [("ID", None, None, "ID slope")] + [
                ("OOD", None, template, name) for template, name in JAILBREAKS.items()]
        else:
            columns = [(split, None, None, f"{split} slope") for split in COLORS]
        means = {row["measure"]: row["mean_ood_slope"] for row in ood_means if row["benchmark"] == benchmark}
        headers = [label for _, _, _, label in columns] + (["Mean OOD slope"] if means else [])
        heading = "MGSM (OOD)" if benchmark == "MGSM" else benchmark
        lines.extend([f"## {heading}", "", "| Measure | " + " | ".join(headers) + " |",
                      "|---|" + "---:|" * len(headers)])
        grouped = {measure: {(row["split"], row["language"], row["template"]): row for row in slopes
                             if row["benchmark"] == benchmark and row["measure"] == measure}
                   for measure in measures}
        def magnitude(measure):
            return max((abs(grouped[measure][split, language, template]["slope"])
                        for split, language, template, _ in columns
                        if grouped[measure][split, language, template]["slope"] is not None), default=-1)
        for measure in sorted(measures, key=magnitude, reverse=True):
            values = []
            for split, language, template, _ in columns:
                row = grouped[measure][split, language, template]
                if row["slope"] is None:
                    value = "Undefined"
                else:
                    value = f"{row['slope']:+.6f}"
                    if row["n_models"] < row["expected_models"]:
                        value += f" (n={row['n_models']})"
                values.append(value)
            if means:
                values.append("Undefined" if means[measure] is None else f"{means[measure]:+.6f}")
            lines.append(f"| {measure} | {' | '.join(values)} |")
        lines.append("")
    lines.extend(["HarmBench's five named jailbreak columns are OOD. Mean OOD slope is the arithmetic mean of "
                  "the five condition-specific slopes, for both HarmBench and MGSM. "
                  "It is undefined if any condition's slope is undefined. "
                  "The plotted HarmBench OOD fit still uses the pooled jailbreak results.", "",
                  "MGSM columns show slopes fitted separately for each OOD language. "
                  "Its Spanish relevance and AXBench overall gains are undefined because Original is zero.", "",
                  "TruthfulQA concept relevance has paired values for three models. AXBench overall has "
                  "two finite model pairs. Missing or non-finite values are excluded, with counts shown above.", "",
                  "Earlier figure variants are retained in [archive/figures](archive/figures). "
                  "Run `python src/report_behavior_gain.py` from this unit to rebuild the active outputs.", ""])
    (UNIT / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true", help="Update data and slope tables without rewriting the figure")
    args = parser.parse_args()
    points, sources = load_signals()
    slopes = fit_signals(points)
    ood_means = mean_ood_slopes(slopes)
    TABLES.mkdir(exist_ok=True)
    for stem, records in (("behavior_signal_points", points), ("behavior_signal_slopes", slopes),
                          ("behavior_signal_mean_ood_slopes", ood_means)):
        with (TABLES / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        (TABLES / f"{stem}.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (TABLES / "behavior_signal_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    if not args.tables_only:
        render(points, slopes)
    write_report(slopes, ood_means)
    print((UNIT / "README.md").read_text())


if __name__ == "__main__":
    main()
