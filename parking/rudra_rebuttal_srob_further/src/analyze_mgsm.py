#!/usr/bin/env python3
"""Compare MGSM accuracy with A-LQR across the five paper evaluation languages."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from analyze import PLOTS, REPO, TABLES, configure_plot_style, fit_record, save_figure

import matplotlib.pyplot as plt
import numpy as np


POINTS = REPO / "parking/gamma_star_organization/plots/srob_overall_points.csv"
CALIBRATIONS = REPO / "parking/gamma_star_organization/exports/gamma_star_collaborator_bundle/final_selected_calibrations.csv"
JACKKNIFE = REPO / "figs/bench_table/mgsm/mgsm_jackknife.json"
MODELS = {"qwen3_4b": "Qwen3-4B", "phi4_mini": "Phi-4-mini", "granite33_2b": "Granite-3.3-2B"}
LANGUAGES = {"zh": "Chinese", "fr": "French", "ja": "Japanese", "sw": "Swahili", "te": "Telugu"}


def load_mgsm_rows() -> tuple[list[dict], list[dict], dict]:
    sources = {}

    def read_source(path: Path) -> bytes:
        content = path.read_bytes()
        sources[str(path.relative_to(REPO))] = {"sha256": hashlib.sha256(content).hexdigest()}
        return content

    def read_json(path: Path) -> dict:
        value = json.loads(read_source(path))
        sources[str(path.relative_to(REPO))]["identity"] = value.get("identity")
        return value

    points = {row["model_key"]: row for row in csv.DictReader(read_source(POINTS).decode().splitlines())
              if row["benchmark"] == "mgsm"}
    calibrations = {row["model_key"]: row for row in csv.DictReader(read_source(CALIBRATIONS).decode().splitlines())
                    if row["benchmark"] == "mgsm"}
    if points.keys() != MODELS.keys():
        raise ValueError("Expected the three paper-reported MGSM Srob points")
    report = read_json(JACKKNIFE)
    language_rows = []
    averages = []
    for model_key, label in MODELS.items():
        selection_path = REPO / calibrations[model_key]["selection_path"]
        selection = read_json(selection_path)
        score_path = selection_path.parent / selection["diagnostic_bundle"] / "score.json"
        score = read_json(score_path)
        srob = float(points[model_key]["s_rob_value"])
        gamma_star = float(points[model_key]["gamma_star"])
        if not (math.isclose(srob, score["s_rob"], rel_tol=1e-12)
                and math.isclose(gamma_star, score["gamma_star"], rel_tol=1e-12)
                and math.isclose(srob * gamma_star, 1, rel_tol=1e-12)):
            raise ValueError(f"Srob point and controller diagnostic disagree for {label}")
        cache_root = selection_path.parents[3]
        cache_key = cache_root.name
        model_report = report["models"][cache_key]
        question_ids = {question_id for group in model_report["groups"] for question_id in group}
        model_rows = []
        for language_key, language_label in LANGUAGES.items():
            accuracies = {}
            prompt_ids = {}
            declared_counts = {}
            for method in ("alqr", "h_infinity"):
                result_path = REPO / "benchmarks/mgsm/results/kv_cache_off" / cache_key / f"mgsm_{language_key}" / f"{method}.json"
                result = read_json(result_path)
                generation_path = cache_root / "evaluations/kv_cache_off/generations" / f"mgsm_{language_key}" / method / "final.json"
                generation = read_json(generation_path)
                identity = generation["identity"]
                if (result["model"], result["input_language"], result["method"]) != (selection["model"], language_key, method):
                    raise ValueError(f"Result identity mismatch in {result_path}")
                if (identity["model"], identity["input_language"], identity["method"],
                        identity["evaluated_model_kv_cache"]) != (selection["model"], language_key, method, False):
                    raise ValueError(f"Generation identity mismatch in {generation_path}")
                if method == "h_infinity" and identity["parameters"] != selection["selected"]["parameters"]:
                    raise ValueError(f"H∞ settings differ from the Srob controller for {label}")
                if generation["status"] != "complete" or len(generation["repetitions"]) != 1:
                    raise ValueError(f"Expected one completed deterministic evaluation in {generation_path}")
                expected_ids = {f"{question_id}-{language_key}" for question_id in question_ids}
                generation_ids = {row["prompt_id"] for row in generation["repetitions"][0]["rows"]}
                judgments_path = cache_root / "evaluations/kv_cache_off/scores/mgsm_exact_match" / f"mgsm_{language_key}" / method / "final.json"
                judgments = read_json(judgments_path)
                prompt_ids[method] = {row["prompt_id"] for row in judgments["rows"]}
                if (generation_ids != expected_ids or prompt_ids[method] != expected_ids
                        or len(judgments["rows"]) != len(expected_ids) or judgments["status"] != "complete"):
                    raise ValueError(f"Question identities differ from the paper evaluation for {label}, {language_label}")
                # Unparseable answers retain their saved zero score in the denominator.
                accuracy = 100 * sum(float(row["score"]) for row in judgments["rows"]) / len(judgments["rows"])
                result_accuracy = 100 * float(result["metrics"]["mgsm_exact_match.score"])
                paper_accuracy = model_report["languages"][language_key][method]["mgsm_exact_match"]["full_sample_mean"]
                if not (math.isclose(accuracy, result_accuracy, abs_tol=1e-12)
                        and math.isclose(accuracy, paper_accuracy, abs_tol=1e-12)):
                    raise ValueError(f"Cached accuracy disagrees with the paper for {label}, {language_label}, {method}")
                accuracies[method] = accuracy
                declared_counts[method] = result["sample_count"]
            row = {
                "model": label, "model_key": model_key,
                "model_id": selection["model"][0], "model_revision": selection["model"][1],
                "srob": srob, "gamma_star": gamma_star,
                "language": language_label, "language_key": language_key,
                "alqr_accuracy_pct": accuracies["alqr"],
                "hinf_accuracy_pct": accuracies["h_infinity"],
                "question_count": len(question_ids),
                "alqr_declared_result_count": declared_counts["alqr"],
                "hinf_declared_result_count": declared_counts["h_infinity"],
            }
            model_rows.append(row)
            language_rows.append(row)
        alqr_accuracy = sum(row["alqr_accuracy_pct"] for row in model_rows) / len(LANGUAGES)
        hinf_accuracy = sum(row["hinf_accuracy_pct"] for row in model_rows) / len(LANGUAGES)
        for method, accuracy in (("alqr", alqr_accuracy), ("h_infinity", hinf_accuracy)):
            paper_mean = model_report["summary"][method]["mgsm_exact_match"]["full_sample_mean"]
            if not math.isclose(accuracy, paper_mean, abs_tol=1e-12):
                raise ValueError(f"Five-language mean disagrees with the paper for {label}, {method}")
        ratio = None if alqr_accuracy == 0 else hinf_accuracy / alqr_accuracy
        averages.append({
            "model": label, "model_key": model_key,
            "model_id": selection["model"][0], "model_revision": selection["model"][1],
            "srob": srob, "gamma_star": gamma_star,
            "condition": "Five-language average (OOD)", "language_count": len(LANGUAGES),
            "questions_per_language": len(question_ids),
            "alqr_accuracy_pct": alqr_accuracy, "hinf_accuracy_pct": hinf_accuracy,
            "hinf_over_alqr_accuracy": ratio,
            "hinf_minus_alqr_accuracy_pp": hinf_accuracy - alqr_accuracy,
            "ratio_status": "undefined: A-LQR accuracy is zero" if ratio is None else "defined",
            "selection_path": str(selection_path.relative_to(REPO)),
            "score_path": str(score_path.relative_to(REPO)),
        })
    return averages, language_rows, sources


def save_tables(rows: list[dict], language_rows: list[dict], sources: dict) -> None:
    TABLES.mkdir(exist_ok=True)
    for stem, records in (("mgsm_srob_accuracy_vs_alqr", rows), ("mgsm_accuracy_by_language", language_rows)):
        with (TABLES / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        (TABLES / f"{stem}.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (TABLES / "mgsm_source_manifest.json").write_text(json.dumps(sources, indent=2, allow_nan=False) + "\n")
    lines = [
        "# MGSM Srob and accuracy relative to A-LQR",
        "",
        "Accuracy is the equal-weight mean across Chinese, French, Japanese, Swahili, and Telugu (all OOD).",
        "Srob and γ★ retain source precision. Behavioral columns are displayed to six decimals.",
        "Phi-4-mini's ratio is undefined because its A-LQR accuracy is zero.",
        "",
        "| Model | Srob | γ★ | A-LQR accuracy (%) | H∞ accuracy (%) | H∞ / A-LQR | H∞ − A-LQR (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = row["hinf_over_alqr_accuracy"]
        ratio_text = "Undefined (A-LQR = 0%)" if ratio is None else f"{ratio:.6f}"
        lines.append(
            f"| {row['model']} | {row['srob']!r} | {row['gamma_star']!r} | "
            f"{row['alqr_accuracy_pct']:.6f} | {row['hinf_accuracy_pct']:.6f} | {ratio_text} | "
            f"{row['hinf_minus_alqr_accuracy_pp']:+.6f} |"
        )
    (TABLES / "mgsm_srob_accuracy_vs_alqr.md").write_text("\n".join(lines) + "\n")


def render_mgsm(rows: list[dict]) -> None:
    PLOTS.mkdir(exist_ok=True)
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    color = "#8B1E1E"
    outcomes = (("hinf_over_alqr_accuracy", "H∞ accuracy / A-LQR accuracy", 1.0, "Accuracy ratio (OOD)"),
                ("hinf_minus_alqr_accuracy_pp", "H∞ accuracy − A-LQR accuracy (pp)", 0.0, "Accuracy difference (OOD)"))
    offsets = {"qwen3_4b": (9, 8), "phi4_mini": (9, 8), "granite33_2b": (-9, 8)}
    statistics = {}
    for ax, (outcome, ylabel, reference, title) in zip(axes, outcomes, strict=True):
        selected = [row for row in rows if row[outcome] is not None]
        undefined = [row["model"] for row in rows if row[outcome] is None]
        x = np.array([row["srob"] for row in selected])
        y = np.array([row[outcome] for row in selected])
        ax.axhline(reference, color="#c9c9c9", linewidth=1, zorder=0)
        ax.scatter(x, y, s=52, color=color, zorder=3)
        for row in selected:
            offset = offsets[row["model_key"]]
            if outcome == "hinf_minus_alqr_accuracy_pp" and row["model_key"] == "granite33_2b":
                offset = (-22, 0)
            ax.annotate(row["model"], (row["srob"], row[outcome]), xytext=offset,
                        textcoords="offset points", ha="right" if offset[0] < 0 else "left",
                        fontsize=10, arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#999999"})
        fit = None
        if len(selected) >= 3:
            fit = fit_record(x, y)
            curve_x = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(curve_x, fit["intercept"] + fit["slope"] * curve_x, color=color,
                    linewidth=1.2, alpha=0.65, zorder=1)
            annotation = f"r = {fit['pearson_r']:.3f}, p = {fit['p_two_sided']:.3f}"
        else:
            annotation = f"{len(selected)} finite ratios"
        ax.text(0.97, 0.04, annotation, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        statistics[outcome] = {"n_models": len(rows), "n_finite": len(selected), "undefined_models": undefined, "fit": fit}
        ax.set(xlim=(-0.03, 1.08), xticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
               xlabel=r"$S_{\mathrm{rob}}$", ylabel=ylabel, title=title)
        ax.title.set_color(color)
    axes[0].set_ylim(0.9, 1.8)
    axes[0].text(0.03, 0.97, "Phi-4-mini: undefined (A-LQR = 0%)", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=10)
    axes[1].set_ylim(-2, max(row["hinf_minus_alqr_accuracy_pp"] for row in rows) + 7)
    save_figure(fig, "srob_mgsm_accuracy_vs_alqr")
    (TABLES / "mgsm_regressions.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")


def main() -> None:
    rows, language_rows, sources = load_mgsm_rows()
    save_tables(rows, language_rows, sources)
    render_mgsm(rows)
    print((TABLES / "mgsm_srob_accuracy_vs_alqr.md").read_text())
    print(f"Verified {len(rows)} model averages and {len(language_rows)} language rows against {len(sources)} source files")


if __name__ == "__main__":
    main()
