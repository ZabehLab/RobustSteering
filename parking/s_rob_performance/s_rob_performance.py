#!/usr/bin/env python3
"""Re-analyze existing final runs for SRob-versus-retention evidence."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr


UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
SOURCE = ROOT / "parking/gamma_star_organization/exports/gamma_star_collaborator_bundle"
PERFORMANCE = SOURCE / "final_performance_id_ood.csv"
CALIBRATIONS = SOURCE / "final_selected_calibrations.csv"
PLOTS = UNIT / "plots"

TASKS = {
    "truthfulness": ("TruthfulQA", "#687DA3", "s"),
    "harmful": ("HarmBench", "#5B6573", "o"),
    "mgsm": ("MGSM", "#C27D5A", "^"),
    "lciteeval": ("L-CiteEval", "#7B70B6", "p"),
}
MODEL_SHORT = {
    "gpt2_xl": "GPT-2 XL",
    "llama31_8b": "Llama-8B",
    "olmo2_32b": "OLMo-32B",
    "qwen25_14b": "Qwen-14B",
    "llama32_1b": "Llama-1B",
    "llama32_3b": "Llama-3B",
    "granite33_2b": "Granite-2B",
    "phi4_mini": "Phi-4-mini",
    "qwen3_4b": "Qwen3-4B",
    "qwen25_3b": "Qwen-3B",
}
PREDICTORS = {
    "s_rob": r"$S_{\mathrm{rob}}$",
    "id_performance": "ID performance",
    "inverse_gain_norm": r"$1/\|K\|$",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def result_value(
    rows: list[dict[str, str]], benchmark: str, model: str, distribution: str,
    metric: str, method: str = "H∞ (ours)", condition: str | None = None,
) -> float:
    selected = [
        row for row in rows
        if row["benchmark"] == benchmark
        and row["model_key"] == model
        and row["distribution"] == distribution
        and row["metric"] == metric
        and row["method"] == method
        and (condition is None or row["condition"] == condition)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one result for {benchmark}/{model}/{distribution}/{metric}/{method}/{condition}; "
            f"found {len(selected)}"
        )
    return float(selected[0]["value"])


def inverse_gain_norm(selection_path: str) -> float:
    controller_path = ROOT / Path(selection_path).parent / "controller.pt"
    controller = torch.load(controller_path, map_location="cpu", weights_only=False)
    norm = float(controller["diagnostics"]["max_norm_K"])
    return 1.0 / norm


def build_points() -> list[dict[str, object]]:
    performance = read_csv(PERFORMANCE)
    calibrations = read_csv(CALIBRATIONS)
    selected = {(row["benchmark"], row["model_key"]): row for row in calibrations}
    points: list[dict[str, object]] = []

    for model in sorted({row["model_key"] for row in performance if row["benchmark"] == "truthfulness"}):
        id_value = result_value(performance, "truthfulness", model, "id", "true_times_informative")
        ood_value = result_value(performance, "truthfulness", model, "ood", "true_times_informative")
        points.append({
            "benchmark": "truthfulness",
            "model_key": model,
            "id_performance": id_value,
            "ood_performance": ood_value,
            "retention": ood_value / id_value,
            "retention_definition": "Spanish T×I / English T×I",
            "is_proxy": False,
        })

    for model in sorted({row["model_key"] for row in performance if row["benchmark"] == "harmful"}):
        id_asr = result_value(performance, "harmful", model, "id", "attack_success_rate")
        conditions = sorted({
            row["condition"] for row in performance
            if row["benchmark"] == "harmful" and row["model_key"] == model
            and row["method"] == "H∞ (ours)" and row["distribution"] == "ood"
            and row["metric"] == "attack_success_rate"
        })
        ood_safe = np.mean([
            100.0 - result_value(
                performance, "harmful", model, "ood", "attack_success_rate", condition=condition
            )
            for condition in conditions
        ])
        id_safe = 100.0 - id_asr
        points.append({
            "benchmark": "harmful",
            "model_key": model,
            "id_performance": id_safe,
            "ood_performance": float(ood_safe),
            "retention": float(ood_safe / id_safe),
            "retention_definition": "mean OOD safe-response rate / direct safe-response rate",
            "is_proxy": False,
        })

    for model in sorted({row["model_key"] for row in performance if row["benchmark"] == "mgsm"}):
        conditions = sorted({
            row["condition"] for row in performance
            if row["benchmark"] == "mgsm" and row["model_key"] == model
            and row["metric"] == "accuracy"
        })
        original = np.asarray([
            result_value(performance, "mgsm", model, "ood", "accuracy", "Original", condition)
            for condition in conditions
        ])
        hinf = np.asarray([
            result_value(performance, "mgsm", model, "ood", "accuracy", "H∞ (ours)", condition)
            for condition in conditions
        ])
        points.append({
            "benchmark": "mgsm",
            "model_key": model,
            "id_performance": float(original.mean()),
            "ood_performance": float(hinf.mean()),
            "retention": float(np.mean(hinf / original)),
            "retention_definition": "mean_language(H∞ OOD accuracy / Original OOD accuracy)",
            "is_proxy": True,
        })

    for model in sorted({row["model_key"] for row in performance if row["benchmark"] == "lciteeval"}):
        id_value = result_value(performance, "lciteeval", model, "id", "answer_recall")
        ood_value = result_value(performance, "lciteeval", model, "ood", "answer_recall")
        points.append({
            "benchmark": "lciteeval",
            "model_key": model,
            "id_performance": id_value,
            "ood_performance": ood_value,
            "retention": ood_value / id_value,
            "retention_definition": "16K answer recall / 8K answer recall",
            "is_proxy": False,
        })

    for point in points:
        calibration = selected[(str(point["benchmark"]), str(point["model_key"]))]
        point["model_label"] = calibration["model_label"]
        point["s_rob"] = float(calibration["s_rob"])
        point["gamma_star"] = float(calibration["gamma_star"])
        point["inverse_gain_norm"] = inverse_gain_norm(calibration["selection_path"])
        point["selection_path"] = calibration["selection_path"]
    return points


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    observed = float(spearmanr(x, y).statistic)
    permuted = [
        float(spearmanr(x, np.asarray(permutation, dtype=float)).statistic)
        for permutation in itertools.permutations(y.tolist())
    ]
    p_value = sum(abs(value) >= abs(observed) - 1e-12 for value in permuted) / len(permuted)
    return observed, float(p_value), len(permuted)


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, samples: int = 2000, seed: int = 20260926,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        sample_x = x[indices]
        sample_y = y[indices]
        if np.unique(sample_x).size < 2 or np.unique(sample_y).size < 2:
            continue
        rho = float(spearmanr(sample_x, sample_y).statistic)
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def leave_one_out_minimum(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    values = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        values.append(float(spearmanr(x[keep], y[keep]).statistic))
    return float(min(values))


def task_statistics(points: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    output = {}
    for task in TASKS:
        selected = [point for point in points if point["benchmark"] == task]
        x = np.asarray([point["s_rob"] for point in selected], dtype=float)
        y = np.asarray([point["retention"] for point in selected], dtype=float)
        rho, p_value, permutations = exact_spearman(x, y)
        ci_low, ci_high, valid_bootstraps = bootstrap_spearman(x, y)
        fit = np.polyfit(np.log10(x), y, 1)
        predictor_rho = {}
        for predictor in PREDICTORS:
            predictor_x = np.asarray([point[predictor] for point in selected], dtype=float)
            predictor_rho[predictor] = float(spearmanr(predictor_x, y).statistic)
        output[task] = {
            "n_models": len(selected),
            "spearman_rho": rho,
            "exact_two_sided_p_value": p_value,
            "exact_permutation_count": permutations,
            "bootstrap_samples_requested": 2000,
            "bootstrap_samples_valid": valid_bootstraps,
            "bootstrap_95_ci": [ci_low, ci_high],
            "leave_one_out_minimum_rho": leave_one_out_minimum(x, y),
            "log10_srob_ols_slope": float(fit[0]),
            "log10_srob_ols_intercept": float(fit[1]),
            "predictor_spearman_rho": predictor_rho,
        }
    return output


def selection_statistics(points: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    output = {}
    for predictor in PREDICTORS:
        top1_hits = 0
        top2_hits = 0
        regrets = []
        for task in TASKS:
            selected = [point for point in points if point["benchmark"] == task]
            best = max(selected, key=lambda point: float(point["retention"]))
            ranked = sorted(selected, key=lambda point: float(point[predictor]), reverse=True)
            top1_hits += ranked[0]["model_key"] == best["model_key"]
            top2_hits += best["model_key"] in {point["model_key"] for point in ranked[:2]}
            regrets.append(float(best["retention"]) - float(ranked[0]["retention"]))
        output[predictor] = {
            "top1_hits": int(top1_hits),
            "top2_hits": int(top2_hits),
            "task_count": len(TASKS),
            "mean_retention_regret": float(np.mean(regrets)),
            "median_retention_regret": float(np.median(regrets)),
        }
    return output


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 8.7,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9DDE1", linewidth=0.65, alpha=0.72)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(width=0.9, length=4)


def render(
    points: list[dict[str, object]], statistics: dict[str, dict[str, object]],
    selection: dict[str, dict[str, object]],
) -> None:
    configure()
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.25))

    ax = axes[0]
    for task, (label, color, marker) in TASKS.items():
        selected = [point for point in points if point["benchmark"] == task]
        x = np.asarray([point["s_rob"] for point in selected], dtype=float)
        y = 100 * np.asarray([point["retention"] for point in selected], dtype=float)
        ax.scatter(x, y, s=42, marker=marker, color=color, edgecolor="white",
                   linewidth=0.6, zorder=3, label=label)
        if len(x) >= 3:
            fit = statistics[task]
            line_x = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            line_y = 100 * (
                float(fit["log10_srob_ols_intercept"])
                + float(fit["log10_srob_ols_slope"]) * np.log10(line_x)
            )
            ax.plot(line_x, line_y, color=color, linewidth=1.35, alpha=0.82)
    ax.axhline(100, color="#737B80", linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"Robust steerability $S_{\mathrm{rob}}$")
    ax.set_ylabel("Retention (%)")
    ax.set_title("A  Existing selected controllers", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=7.2,
              handletextpad=0.35, columnspacing=0.7)
    style_axis(ax)

    ax = axes[1]
    y_positions = np.arange(len(TASKS))[::-1]
    for y_position, (task, (label, color, _)) in zip(y_positions, TASKS.items()):
        stat = statistics[task]
        rho = float(stat["spearman_rho"])
        ci_low, ci_high = map(float, stat["bootstrap_95_ci"])
        ax.errorbar(
            rho, y_position,
            xerr=[[rho - ci_low], [ci_high - rho]],
            fmt="o", color=color, ecolor=color, capsize=3, markersize=6,
        )
        ax.text(1.03, y_position, f"n={stat['n_models']}, p={stat['exact_two_sided_p_value']:.3g}",
                va="center", ha="left", fontsize=7.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.08, 1.52)
    ax.set_yticks(y_positions, [value[0] for value in TASKS.values()])
    ax.set_xlabel(r"Spearman $\rho$ [bootstrap 95% CI]")
    ax.set_title("B  Within-task association", fontweight="bold", fontsize=11)
    style_axis(ax)

    ax = axes[2]
    predictor_order = ["s_rob", "id_performance", "inverse_gain_norm"]
    offsets = np.linspace(-0.18, 0.18, len(TASKS))
    for offset, (task, (label, color, marker)) in zip(offsets, TASKS.items()):
        values = [statistics[task]["predictor_spearman_rho"][name] for name in predictor_order]
        ax.scatter(np.arange(len(predictor_order)) + offset, values, s=34, marker=marker,
                   color=color, edgecolor="white", linewidth=0.5, label=label, zorder=3)
    ax.scatter([3, 4], [-0.93, -0.93], marker="x", s=38, color="#737B80")
    ax.text(3, -0.82, "not cached", ha="center", va="bottom", fontsize=6.8, rotation=90)
    ax.text(4, -0.82, "not cached", ha="center", va="bottom", fontsize=6.8, rotation=90)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks(
        np.arange(5),
        [r"$S_{\mathrm{rob}}$", "ID\nperformance", r"$1/\|K\|$", "Residual\nnorm", "Probe\naccuracy"],
    )
    ax.set_ylabel(r"Spearman $\rho$ with retention")
    ax.set_title("C  Available predictor comparison", fontweight="bold", fontsize=11)
    style_axis(ax)

    ax = axes[3]
    labels = [PREDICTORS[name] for name in predictor_order]
    top1 = [selection[name]["top1_hits"] / selection[name]["task_count"] for name in predictor_order]
    top2 = [selection[name]["top2_hits"] / selection[name]["task_count"] for name in predictor_order]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, top1, width, color=["#007C7C", "#737B80", "#AEB3B7"], label="Top-1")
    ax.bar(x + width / 2, top2, width, facecolor="white",
           edgecolor=["#007C7C", "#737B80", "#AEB3B7"], linewidth=1.2, label="Top-2")
    for index, predictor in enumerate(predictor_order):
        total = selection[predictor]["task_count"]
        ax.text(index - width / 2, top1[index] + 0.035,
                f"{selection[predictor]['top1_hits']}/{total}", ha="center", fontsize=7.2)
        ax.text(index + width / 2, top2[index] + 0.035,
                f"{selection[predictor]['top2_hits']}/{total}", ha="center", fontsize=7.2)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Task hit rate")
    ax.set_title("D  Model-ranking use case", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    style_axis(ax)

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.88, wspace=0.52)
    prefix = PLOTS / "s_rob_performance"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(
    points: list[dict[str, object]], statistics: dict[str, dict[str, object]],
    selection: dict[str, dict[str, object]],
) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    point_fields = [
        "benchmark", "model_key", "model_label", "s_rob", "gamma_star",
        "id_performance", "ood_performance", "retention", "retention_definition",
        "is_proxy", "inverse_gain_norm", "selection_path",
    ]
    with (PLOTS / "retention_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)
    payload = {
        "task_statistics": statistics,
        "model_selection_analogue": selection,
        "availability": {
            "E1_within_model_sweep": {
                "status": "unavailable",
                "reason": "No frozen OOD evaluations exist for layer × rho_rob × calibration-seed configurations.",
            },
            "E2_across_models_within_task": {
                "status": "completed_with_mgsm_proxy",
                "reason": "TruthfulQA, HarmBench, and L-CiteEval use matched ID/OOD results; MGSM lacks an ID H∞ evaluation.",
            },
            "E3_baseline_predictors": {
                "status": "partial",
                "available": ["ID performance", "1/max_norm_K"],
                "unavailable": ["residual norm", "linear-probe accuracy"],
            },
            "E4_layer_selection": {
                "status": "unavailable",
                "substitute": "Top-1/top-2 model ranking across the four E2 tasks.",
            },
            "E5_shared_cost_normalization": {
                "status": "unavailable",
                "reason": "Existing selected controllers use task-specific costs and no shared-cost recomputation is cached.",
            },
            "E6_calibration_size_sweep": {
                "status": "unavailable",
                "reason": "No matched OOD evaluations exist for the requested calibration-size sweep.",
            },
            "secondary_mixed_model": {
                "status": "not_fit",
                "reason": "Only 2–4 selected model-level observations exist per task; model random effects are not identifiable because nearly every model-task tuple has one retention value.",
            },
        },
    }
    with (PLOTS / "statistics.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    points = build_points()
    statistics = task_statistics(points)
    selection = selection_statistics(points)
    write_outputs(points, statistics, selection)
    render(points, statistics, selection)


if __name__ == "__main__":
    main()
