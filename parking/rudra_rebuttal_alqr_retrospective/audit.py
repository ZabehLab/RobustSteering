"""Reconstruct the HarmBench calibration grids using existing responses and scores."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from figs.calibration_plots.calibration_plots import TEAL, load_json, setup_style, short_number
from robust_steerability.judges.exact import harmonic_mean


UNIT = Path(__file__).resolve().parent
PAPER_CACHE = UNIT.parents[1] / "benchmarks/harmful/cache"
OUTPUT = UNIT / "cache"
MODELS = ("llama32_1b_instruct", "llama32_3b_instruct", "llama31_8b_instruct")
RUBRICS = ("axbench_concept_relevance", "axbench_instruction_relevance", "axbench_fluency")


def rows_by_prompt(rows: list[dict], path: Path) -> dict:
    indexed = {row["prompt_id"]: row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"Duplicate prompt IDs in {path}")
    return indexed


def read_generations(path: Path) -> tuple[dict, dict]:
    data = load_json(path)
    if data["status"] != "complete" or len(data["repetitions"]) != 1:
        raise ValueError(f"Expected a complete single repetition in {path}")
    return data["identity"], rows_by_prompt(data["repetitions"][0]["rows"], path)


def read_scores(path: Path, generations: dict) -> dict[str, float]:
    data = load_json(path)
    rows = rows_by_prompt(data["rows"], path)
    if data["status"] != "complete" or rows.keys() != generations.keys():
        raise ValueError(f"Scores do not match the completed responses in {path}")
    scores = {prompt: float(row["score"]) for prompt, row in rows.items()}
    if any(not 0 <= value <= 2 for value in scores.values()):
        raise ValueError(f"Invalid AXBench score in {path}")
    return scores


def grid_score(grid: Path, filename: str, generations: dict) -> float:
    scores = [read_scores(grid / "scores" / rubric / filename, generations) for rubric in RUBRICS]
    return fmean(harmonic_mean([score[prompt] for score in scores]) for prompt in generations)


def prompt_inputs(generations: dict) -> dict:
    return {prompt: (row["behavior_id"], row["text"]) for prompt, row in generations.items()}


def audit_model(model_key: str) -> dict:
    root = PAPER_CACHE / model_key
    selection_dir = root / "calibrations"
    hinf = load_json(selection_dir / "h_infinity/selected/selection.json")
    alqr = load_json(selection_dir / "alqr/selected/selection.json")
    protocol = hinf["protocol"]
    submitted = alqr["parameters"]
    if alqr["model"] != hinf["model"] or alqr["method"] != "alqr":
        raise ValueError(f"Unexpected submitted A-LQR selection for {model_key}")
    if (submitted["r"], submitted["lambda"]) != (
        protocol["fixed_r"], protocol["fixed_setpoint_multiplier"]
    ):
        raise ValueError(f"The submitted A-LQR setting uses a different fixed cost for {model_key}")

    hinf_grid = selection_dir / "h_infinity/selected/grid"
    alqr_grid = OUTPUT / model_key / "grid"
    validation = None
    cells = []
    for cell in hinf["grid"]:
        filename = cell["grid_id"] + ".json"
        identity, generations = read_generations(hinf_grid / "generations" / filename)
        params = {key: cell[key] for key in submitted}
        if identity["model"] != hinf["model"] or any(
            identity["configuration"][key] != value for key, value in params.items()
        ):
            raise ValueError(f"H∞ generation identity differs from its manifest for {model_key}/{filename}")
        inputs = prompt_inputs(generations)
        if len(inputs) != protocol["tuning_samples"] or (validation is not None and inputs != validation):
            raise ValueError(f"H∞ calibration prompts differ across cells for {model_key}")
        validation = inputs
        hinf_score = grid_score(hinf_grid, filename, generations)
        if not math.isclose(hinf_score, cell["axbench_overall"], abs_tol=1e-12):
            raise ValueError(f"Reconstructed H∞ score differs from the manifest for {model_key}/{filename}")

        alqr_score = None
        path = alqr_grid / "generations" / filename
        score_paths = [alqr_grid / "scores" / rubric / filename for rubric in RUBRICS]
        if path.is_file() and all(p.is_file() and load_json(p)["status"] == "complete" for p in score_paths):
            identity, generations = read_generations(path)
            if (identity["method"] != "alqr" or identity["model"] != alqr["model"]
                or identity["parameters"] != params
                or identity["evaluated_model_kv_cache"] != protocol["evaluated_model_kv_cache"]
                or prompt_inputs(generations) != validation):
                raise ValueError(f"A-LQR generation identity differs from the paper grid in {path}")
            alqr_score = grid_score(alqr_grid, filename, generations)
        cells.append(dict(
            model=model_key, grid_id=cell["grid_id"], q_over_r=cell["q_over_r"],
            qf_over_r=cell["q_final_over_r"], submitted_alqr=params == submitted,
            hinf_overall=hinf_score, alqr_overall=alqr_score,
        ))

    test = root / "evaluations/kv_cache_off"
    identity, test_rows = read_generations(test / "generations/harmbench_direct/alqr/final.json")
    if identity["model"] != alqr["model"] or identity["method"] != "alqr" or identity["parameters"] != submitted:
        raise ValueError(f"Unexpected held-out A-LQR generation for {model_key}")
    test_scores = read_scores(test / "scores/axbench_overall/harmbench_direct/alqr/final.json", test_rows)
    test_inputs = prompt_inputs(test_rows)
    validation_behaviors, validation_texts = zip(*validation.values())
    test_behaviors, test_texts = zip(*test_inputs.values())
    return dict(
        cells=cells, qs=protocol["q_over_r"], qfs=protocol["q_final_over_r"],
        name=alqr["model"][0].split("/")[-1].removesuffix("-Instruct"),
        model=model_key, validation_prompts=len(validation), test_prompts=len(test_rows),
        prompt_overlap=len(validation.keys() & test_inputs.keys()),
        behavior_overlap=len(set(validation_behaviors) & set(test_behaviors)),
        text_overlap=len(set(validation_texts) & set(test_texts)),
        submitted_test_overall=fmean(test_scores.values()),
    )


def plot(models: list[dict]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.6))
    fig.subplots_adjust(left=0.18, right=0.87, bottom=0.19, top=0.88, wspace=0.33, hspace=0.58)
    cmap = matplotlib.colormaps["Greys"]
    norm = Normalize(0, 2)
    for column, model in enumerate(models):
        qs, qfs, cells = model["qs"], model["qfs"], model["cells"]
        for row, method in enumerate(("hinf_overall", "alqr_overall")):
            ax = axes[row, column]
            measured = [cell[method] for cell in cells if cell[method] is not None]
            complete = len(measured) == len(cells)
            values = np.full((len(qs), len(qfs)), np.nan)
            for cell in cells:
                x, y = qfs.index(cell["qf_over_r"]), qs.index(cell["q_over_r"])
                score = cell[method]
                if score is None:
                    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#f6f6f6",
                                           edgecolor="#d1d4d7", hatch="///", linewidth=0))
                else:
                    values[y, x] = score
                    color = "white" if norm(score) > .62 else "black"
                    ax.text(x, y, f"{score:.3f}", ha="center", va="center", color=color, fontsize=7)
                    if complete and math.isclose(score, max(measured), abs_tol=1e-12, rel_tol=0):
                        ax.scatter(x + .35, y - .32, s=11, color=TEAL, edgecolors="white",
                                   linewidths=.35, zorder=5)
                if cell["submitted_alqr"]:
                    ax.add_patch(Rectangle((x - .46, y - .46), .92, .92, fill=False,
                                           edgecolor=TEAL, linewidth=1.6, zorder=4))
            ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
            ax.set_xticks(range(len(qfs)), [short_number(value) for value in qfs])
            ax.set_yticks(range(len(qs)), [short_number(value) for value in qs])
            ax.set_xticks(np.arange(-.5, len(qfs), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(qs), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.9)
            ax.tick_params(which="both", length=0)
            ax.set_xlabel(r"$Q_f/R$")
            if column == 0:
                ax.set_ylabel(r"$Q/R$")
            if row == 0:
                ax.set_title(model["name"], pad=7)
            else:
                summary = f"{len(measured)}/{len(cells)} cells scored"
                if complete:
                    submitted = next(cell[method] for cell in cells if cell["submitted_alqr"])
                    rank = 1 + sum(score > submitted + 1e-12 for score in measured)
                    summary = f"Gap {max(measured) - submitted:.3f} · Rank {rank}/{len(cells)}"
                ax.text(.5, -.30, summary, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    for axis, label in zip(axes[:, 0], ("H∞\nPaper grid", "A-LQR\nAudit grid")):
        bounds = axis.get_position()
        fig.text(.025, (bounds.y0 + bounds.y1) / 2, label, va="center", fontweight="bold")
    fig.suptitle("HarmBench calibration on 50 direct validation prompts", y=.97, fontsize=12, fontweight="bold")
    scale = fig.add_axes([.895, .23, .017, .58])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=scale, ticks=[0, .5, 1, 1.5, 2],
                 label="AXBench Overall")
    fig.legend(handles=[
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=TEAL, linewidth=1.6, label="Submitted A-LQR cost"),
        Line2D([], [], marker="o", color=TEAL, linestyle="none", markersize=4, label="Grid maximum"),
        Rectangle((0, 0), 1, 1, facecolor="#f6f6f6", edgecolor="#d1d4d7", hatch="///", label="No cached score"),
    ], loc="lower center", bbox_to_anchor=(.5, .013), frameon=False, ncol=3)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"harmbench_alqr_retrospective.{suffix}", dpi=300)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    models = [audit_model(model) for model in MODELS]
    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT / "grid.csv", [cell for model in models for cell in model["cells"]])
    write_csv(OUTPUT / "prompt_split_audit.csv", [
        {key: model[key] for key in ("model", "validation_prompts", "test_prompts", "prompt_overlap",
                                    "behavior_overlap", "text_overlap", "submitted_test_overall")}
        for model in models
    ])
    plot(models)


if __name__ == "__main__":
    main()
