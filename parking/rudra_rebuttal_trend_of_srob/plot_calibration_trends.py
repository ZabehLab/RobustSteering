#!/usr/bin/env python3
"""Connect matched TruthfulQA calibration cost pairs across five model sizes."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
from scipy.interpolate import PchipInterpolator
from scipy.stats import linregress


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
TABLES = UNIT / "tables"
PLOTS = UNIT / "plots"
SOURCE_POINTS = REPO / "parking/gamma_star_organization/plots/srob_parameter_size_truthfulness_main.csv"
sys.path.insert(0, str(REPO / "parking/gamma_star_organization"))
from gamma_star_organization import load_torch_metadata

LABELS = {"gpt2_xl": "GPT-2 XL", "gemma2_2b": "Gemma-2-2B", "llama31_8b": "Llama-3-8B",
          "qwen25_14b": "Qwen-2.5-14B", "olmo2_32b": "OLMo-2-32B"}
FULL_GRID_SELECTIONS = {
    "gemma2_2b": "benchmarks/truthfulness/cache/gemma2b/calibrations/h_infinity/txi_fluency_95_05/selection.json"
}


def load_grid() -> tuple[list[dict], list[dict], dict]:
    sources = {}

    def record_source(path: Path) -> bytes:
        content = path.read_bytes()
        sources[str(path.relative_to(REPO))] = hashlib.sha256(content).hexdigest()
        return content

    source_points = list(csv.DictReader(record_source(SOURCE_POINTS).decode().splitlines()))
    if len(source_points) != 5 or {row["model_key"] for row in source_points} != LABELS.keys():
        raise ValueError("Expected the five models from the original TruthfulQA figure")
    rows = []
    models = []
    for point in sorted(source_points, key=lambda row: float(row["parameter_billions"])):
        model_key = point["model_key"]
        selection_path = REPO / FULL_GRID_SELECTIONS.get(model_key, point["selection_path"])
        selection = json.loads(record_source(selection_path))
        score_path = selection_path.parent / selection["diagnostic_bundle"] / "score.json"
        score = json.loads(record_source(score_path))
        parameter_billions = float(point["parameter_billions"])
        if not math.isclose(parameter_billions * 1e9, score["parameter_count"], rel_tol=1e-12):
            raise ValueError(f"Parameter-count mismatch for {model_key}")
        if len(selection["grid"]) != 32:
            raise ValueError(f"Expected 32 calibration settings for {model_key}")
        selected_id = selection["selected"]["configuration_id"]
        for candidate in selection["grid"]:
            controller_path = selection_path.parent / "grid/controllers" / f"{candidate['grid_id']}.pt"
            controller = load_torch_metadata(controller_path)
            record_source(controller_path)
            identity = controller["identity"]
            parameters = identity["parameters"]
            diagnostics = controller["diagnostics"]
            gamma_star = float(controller["gamma_star"])
            if (identity["model"] != selection["model"]
                    or identity["calibration_id"] != selection["calibration_id"]
                    or identity["configuration_id"] != candidate["grid_id"]):
                raise ValueError(f"Controller identity mismatch in {controller_path}")
            if not all(math.isclose(parameters[key], candidate[key], rel_tol=1e-12)
                       for key in ("q", "r", "q_final", "lambda")):
                raise ValueError(f"Grid costs differ from the saved controller in {controller_path}")
            if not (controller["feasible"] and diagnostics["bisection_converged"]
                    and math.isfinite(gamma_star) and gamma_star > 0):
                raise ValueError(f"Expected a feasible converged controller in {controller_path}")
            srob = 1.0 / gamma_star
            if not math.isclose(srob, diagnostics["robust_steerability"], rel_tol=1e-12):
                raise ValueError(f"Saved Srob diagnostic mismatch in {controller_path}")
            rows.append({
                "model_key": model_key, "model": LABELS[model_key],
                "model_id": selection["model"][0], "model_revision": selection["model"][1],
                "parameter_billions": parameter_billions, "calibration_id": selection["calibration_id"],
                "configuration_id": candidate["grid_id"], "q": parameters["q"], "r": parameters["r"],
                "q_final": parameters["q_final"], "q_over_r": parameters["q"] / parameters["r"],
                "q_final_over_r": parameters["q_final"] / parameters["r"],
                "lambda": parameters["lambda"], "gamma_star": gamma_star, "srob": srob,
                "is_selected_in_grid": candidate["grid_id"] == selected_id,
                "tuning_samples": selection["protocol"].get("tuning_samples"),
                "normalization_protocol_id": score["normalization_protocol_id"],
                "controller_path": str(controller_path.relative_to(REPO)),
                "selection_path": str(selection_path.relative_to(REPO)),
            })
        models.append({"model_key": model_key, "model": LABELS[model_key], "parameter_billions": parameter_billions,
                       "calibration_id": selection["calibration_id"], "selection_path": str(selection_path.relative_to(REPO)),
                       "original_figure_selection_path": point["selection_path"]})
    return rows, models, sources


def match_cost_pairs(rows: list[dict], models: list[dict]) -> tuple[list[dict], np.ndarray, list, list]:
    q_values = sorted({row["q_over_r"] for row in rows})
    qf_values = sorted({row["q_final_over_r"] for row in rows})
    expected_pairs = set(itertools.product(q_values, qf_values))
    if (len(q_values), len(qf_values)) != (8, 4):
        raise ValueError("Expected an 8-by-4 cost grid")
    by_model = {}
    for model in models:
        selected = [row for row in rows if row["model_key"] == model["model_key"]]
        paired = {(row["q_over_r"], row["q_final_over_r"]): row for row in selected}
        if len(selected) != 32 or paired.keys() != expected_pairs:
            raise ValueError(f"Incomplete or duplicate cost pairs for {model['model']}")
        by_model[model["model_key"]] = paired
    qi = {value: index for index, value in enumerate(q_values)}
    qfi = {value: index for index, value in enumerate(qf_values)}
    # Diagonal ordering increases in either cost ratio with the other held fixed.
    ordered_pairs = sorted(expected_pairs, key=lambda pair: (qi[pair[0]] + qfi[pair[1]], qi[pair[0]], qfi[pair[1]]))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 32))
    rank_grid = np.empty((8, 4), dtype=int)
    traces = []
    for rank, ((q, qf), color) in enumerate(zip(ordered_pairs, colors, strict=True), start=1):
        points = [by_model[model["model_key"]][(q, qf)] for model in models]
        x = np.array([row["parameter_billions"] for row in points])
        y = np.array([row["srob"] for row in points])
        fit = linregress(np.log10(x), np.log10(y))
        record = {"setting_rank": rank, "q_over_r": q, "q_final_over_r": qf, "color_hex": to_hex(color),
                  "loglog_slope": float(fit.slope), "loglog_intercept": float(fit.intercept)}
        record.update({f"{row['model_key']}_srob": row["srob"] for row in points})
        traces.append(record)
        rank_grid[qi[q], qfi[qf]] = rank - 1
        for row in points:
            row.update({"setting_rank": rank, "color_hex": to_hex(color)})
    return traces, rank_grid, q_values, qf_values


def render(rows: list[dict], models: list[dict], traces: list[dict], rank_grid: np.ndarray,
           q_values: list, qf_values: list) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.labelsize": 12,
                         "axes.titlesize": 13, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(9.6, 5.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(4.4, 1.2), wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])
    x = np.array([model["parameter_billions"] for model in models])
    dense_x = np.geomspace(float(x.min()), float(x.max()), 250)
    for trace in traces:
        y = np.array([trace[f"{model['model_key']}_srob"] for model in models])
        curve = PchipInterpolator(np.log10(x), np.log10(y))
        ax.plot(dense_x, 10 ** curve(np.log10(dense_x)), color=trace["color_hex"], linewidth=1.0, alpha=0.85)
    ax.scatter([row["parameter_billions"] for row in rows], [row["srob"] for row in rows],
               s=9, color="#202020", alpha=0.65, linewidths=0, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{model['model']}\n{model['parameter_billions']:.2f}B" for model in models], fontsize=10)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlim=(x.min() / 1.13, x.max() * 1.13),
           ylim=(min(row["srob"] for row in rows) / 1.25, max(row["srob"] for row in rows) * 1.3),
           xlabel="Model size (billions of parameters)", ylabel=r"$S_{\mathrm{rob}}$", title="TruthfulQA")
    ax.text(0.02, 0.98, "32 matched calibration settings", transform=ax.transAxes,
            ha="left", va="top", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    colors = [trace["color_hex"] for trace in traces]
    key_ax.imshow(rank_grid, origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=31.5,
                  aspect="equal", interpolation="nearest")
    key_ax.set_xticks(range(4), [f"{value:.3g}" for value in qf_values], rotation=45, ha="right", fontsize=9)
    key_ax.set_yticks(range(8), [f"{value:.3g}" for value in q_values], fontsize=9)
    key_ax.set(xlabel=r"$Q_f/R$", ylabel=r"$Q/R$", title="Cost setting rank")
    key_ax.title.set_fontsize(11)
    for i, j in itertools.product(range(8), range(4)):
        rank = rank_grid[i, j]
        key_ax.text(j, i, str(rank + 1), ha="center", va="center", fontsize=8,
                    color="white" if rank < 16 else "black")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.91)
    for suffix in ("png", "pdf"):
        fig.savefig(PLOTS / f"truthfulqa_srob_calibration_trends.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows, models, sources = load_grid()
    traces, rank_grid, q_values, qf_values = match_cost_pairs(rows, models)
    TABLES.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    for stem, records in (("calibration_points", rows), ("matched_cost_traces", traces)):
        with (TABLES / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    (TABLES / "calibration_points.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    summary = {"models": models, "model_count": len(models), "settings_per_model": len(traces),
               "point_count": len(rows), "matching_keys": ["q_over_r", "q_final_over_r"],
               "color_order": "ascending (q_index + qf_index, q_index, qf_index)",
               "positive_loglog_slopes": sum(trace["loglog_slope"] > 0 for trace in traces),
               "loglog_slope_range": [min(trace["loglog_slope"] for trace in traces), max(trace["loglog_slope"] for trace in traces)],
               "sources_sha256": sources}
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(rows, models, traces, rank_grid, q_values, qf_values)
    print(json.dumps({key: value for key, value in summary.items() if key != "sources_sha256"}, indent=2))


if __name__ == "__main__":
    main()
