"""Selection, paired HarmBench ASR estimates, and the two rebuttal tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from figs.bench_table.harmful.harmbench_jackknife import _balanced_groups, _jackknife
from robust_steerability.benchmarks.calibration import require_nonzero_selection_metric
from robust_steerability.benchmarks.harmful_calibration import API_SCORERS
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec

from .inputs import read_json, write_json
from .workers import CONDITIONS, GIGA_METHODS, METHODS, MINI_BEHAVIORS, calibration_records, generation_path, generation_root

LABELS = {"original": "Original", "alqr": "A-LQR (Full-width)", "alqr_8d": "A-LQR (8D)", "h_infinity": "H∞"}
VALIDATION_ASR = "harmbench_validation_success"
TEST_ASR = "harmbench_test_success"


def score_map(root: Path, path: Path, scorer: str, expected: list[str]) -> dict[str, float]:
    destination = scorer_cache_path(root, path, scorer)
    payload = read_json(destination)
    rows = payload["rows"]
    if payload["status"] != "complete" or len(rows) != len(expected):
        raise ValueError(f"Incomplete {scorer} scores: {destination}")
    if any(not row.get("valid", True) or row.get("score") is None for row in rows):
        raise ValueError(f"Invalid {scorer} score in {destination}")
    scores = {row["prompt_id"]: float(row["score"]) for row in rows}
    spec = scorer_spec(scorer)
    if set(scores) != set(expected) or len(scores) != len(expected) or any(
        not spec.minimum <= value <= spec.maximum for value in scores.values()
    ):
        raise ValueError(f"Scores do not match the generated prompts: {destination}")
    return scores


def generation_rows(path: Path) -> list[dict]:
    payload = read_json(path)
    if payload["status"] != "complete":
        raise ValueError(f"Generation is incomplete: {path}")
    repetition, = payload["repetitions"]
    rows = repetition["rows"]
    if not rows or repetition["sample_count"] != len(rows):
        raise ValueError(f"Generation row count mismatch: {path}")
    return rows


def selection_key(row: dict) -> tuple:
    """The paper's HarmBench ordering, applied independently to each method."""
    return (
        -row["axbench_overall"], *(-row[scorer] for scorer in API_SCORERS),
        row[VALIDATION_ASR], row["mean_control_energy"], row["configuration"]["q"],
        row["configuration"]["q_final"],
    )


def calibrate_scores(root: Path, profile: str, tasks: list[dict], feasible: set[Path]) -> dict:
    calibration_root = generation_root(root, "calibration", profile)
    tuning = read_json(root / "artifacts/data.json")["calibration"]["tuning"]
    tuning_ids = [row["prompt_id"] for row in calibration_records(tuning, profile)]
    grid = {method: [] for method in METHODS}
    for task in tasks:
        path = generation_path(root, task)
        entry = {"configuration": task["configuration"], "status": "infeasible"}
        if path in feasible:
            rows = generation_rows(path)
            if [row["prompt_id"] for row in rows] != tuning_ids:
                raise ValueError(f"Calibration prompt split changed: {path}")
            scores = {
                key: score_map(calibration_root, path, key, tuning_ids)
                for key in (*API_SCORERS, VALIDATION_ASR)
            }
            overall = [
                {"prompt_id": pid, "score": harmonic_mean([scores[key][pid] for key in API_SCORERS])}
                for pid in tuning_ids
            ]
            write_json(
                scorer_cache_path(calibration_root, path, "axbench_overall"),
                {"status": "complete", "rows": overall},
            )
            entry.update({key: float(np.mean([values[pid] for pid in tuning_ids])) for key, values in scores.items()})
            entry.update(
                status="complete", axbench_overall=float(np.mean([row["score"] for row in overall])),
                mean_control_energy=float(np.mean([row["control_energy"] for row in rows])),
            )
            if not math.isfinite(entry["mean_control_energy"]):
                raise ValueError(f"Nonfinite control energy: {path}")
        grid[task["method"]].append(entry)
    write_json(calibration_root / "grid.json", grid)
    winners = {}
    for method, cells in grid.items():
        valid = [cell for cell in cells if cell["status"] == "complete"]
        if not valid:
            raise ValueError(f"No feasible calibration cells for {LABELS[method]}")
        require_nonzero_selection_metric(valid, "axbench_overall", context=LABELS[method])
        winners[method] = min(valid, key=selection_key)
    result = {"prompts_per_cell": len(tuning_ids), "grid": grid, "winners": winners}
    write_json(calibration_root / "selection.json", result)
    return result


def collect_test_scores(root: Path, tasks: list[dict], scorer: str = TEST_ASR) -> dict:
    methods = dict.fromkeys(task["method"] for task in tasks)
    scores = {method: {condition: {} for condition in CONDITIONS} for method in methods}
    for task in tasks:
        path = generation_path(root, task)
        ids = [row["prompt_id"] for row in generation_rows(path)]
        directory = generation_root(root, task["stage"], task["profile"])
        values = score_map(directory, path, scorer, ids)
        dest = scores[task["method"]][task["condition"]]
        if set(dest) & set(values):
            raise ValueError(f"Test score shards overlap: {path}")
        dest.update(values)
    return scores


def paired_results(evaluation: dict, scores: dict) -> dict:
    groups = _balanced_groups(evaluation["direct"])
    results = {}
    for method in METHODS:
        results[method] = {}
        for condition, records in evaluation.items():
            method_scores = scores[method][condition]
            reference = scores["h_infinity"][condition]
            expected = {row["prompt_id"] for row in records}
            if set(method_scores) != expected or set(reference) != expected:
                raise ValueError(f"Unpaired evaluation outcomes for {method}/{condition}")
            by_behavior: dict[str, list[float]] = {}
            gaps: dict[str, list[float]] = {}
            for row in records:
                pid, behavior = row["prompt_id"], row["behavior_id"]
                by_behavior.setdefault(behavior, []).append(100 * method_scores[pid])
                gaps.setdefault(behavior, []).append(100 * (method_scores[pid] - reference[pid]))
            result = {"asr_percent": _jackknife(by_behavior, groups)}
            if method != "h_infinity":
                gap = _jackknife(gaps, groups)
                center, se = gap["full_sample_mean"], gap["jackknife_standard_error"]
                gap["interval_95"] = [center - 1.96 * se, center + 1.96 * se]
                result["gap_vs_h_infinity_pp"] = gap
            results[method][condition] = result
    return results


def ratio(value: float) -> str:
    return "1/√10" if math.isclose(value, 10 ** -0.5) else f"{value:g}"


def giga_results(evaluation: dict, asr: dict, axbench: list[dict]) -> dict:
    groups = _balanced_groups(evaluation["direct"])

    def estimate(records, scores, scale=1.0):
        by_behavior: dict[str, list[float]] = {}
        for row in records:
            by_behavior.setdefault(row["behavior_id"], []).append(scale * scores[row["prompt_id"]])
        return _jackknife(by_behavior, groups)

    results = {}
    for method in GIGA_METHODS:
        result = {}
        pooled = {}
        for condition, records in evaluation.items():
            ids = {row["prompt_id"] for row in records}
            condition_asr = asr[method][condition]
            components = [scores[method][condition] for scores in axbench]
            if set(condition_asr) != ids or any(set(scores) != ids for scores in components):
                raise ValueError(f"Giga scores do not cover every {method}/{condition} prompt")
            overall = {pid: harmonic_mean([scores[pid] for scores in components]) for pid in ids}
            result[condition] = {
                "asr_percent": estimate(records, condition_asr, 100),
                "axbench_overall": estimate(records, overall),
            }
            pooled.update(overall)
        jailbreak = evaluation["human_jailbreak"]
        result["jailbreak_asr_by_template"] = {
            str(index): estimate([row for row in jailbreak if row["template_index"] == index],
                                 asr[method]["human_jailbreak"], 100)
            for index in range(5)
        }
        result["axbench_overall"] = estimate(evaluation["direct"] + jailbreak, pooled)
        results[method] = result
    return results


def giga_table(winners: dict, results: dict) -> list[str]:
    def with_se(metric):
        return f"{metric['full_sample_mean']:.2f} ± {metric['jackknife_standard_error']:.2f}"

    lines = [
        "| Controller | Selected (Q/R, Qf/R) | Direct ASR (%) | Jailbreak ASR (%) |",
        "| --- | --- | ---: | ---: |",
    ]
    for method in GIGA_METHODS:
        selected = "—"
        if method in winners:
            config = winners[method]["configuration"]
            selected = f"({ratio(config['q_over_r'])}, {ratio(config['q_final_over_r'])})"
        scores = results[method]
        lines.append(f"| {LABELS[method]} | {selected} | {with_se(scores['direct']['asr_percent'])} | "
                     f"{with_se(scores['human_jailbreak']['asr_percent'])} |")
    return lines


def response_text(label: str, profile: str, selection: dict, results: dict, behaviors: int) -> str:
    prompts = selection["prompts_per_cell"]
    lines = [
        f"# {label} — {profile} HarmBench comparison", "",
        f"To separate the effect of synthesis from tuning effort, we ran a matched-budget comparison on {label}. "
        f"A-LQR (Full-width), A-LQR (8D), and H∞ each received the same 12 cost settings and {prompts} direct "
        "HarmBench validation behaviors per setting. Table R1a gives the complete calibration grid for "
        "each controller. A-LQR (Full-width) uses the runtime implementation from the paper. The two "
        "8D arms share the basis, dynamics, control channels, reference, and intervention. "
        "A-LQR (8D) solves the nominal problem and H∞ uses disturbance-aware synthesis.", "",
        "**Table R1a. Complete calibration grid**", "",
        "| Q/R | Qf/R | A-LQR (Full-width) | A-LQR (8D) | H∞ |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    grid, winners = selection["grid"], selection["winners"]
    for row_index, base in enumerate(grid[METHODS[0]]):
        config = base["configuration"]
        values = [ratio(config["q_over_r"]), ratio(config["q_final_over_r"])]
        for method in METHODS:
            cell = grid[method][row_index]
            if cell["configuration"] != config:
                raise ValueError("Mismatched calibration grid")
            value = f"{cell['axbench_overall']:.3f}" if cell["status"] == "complete" else cell["status"]
            if config == winners[method]["configuration"]:
                value = f"**{value}**"
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    if profile == "giga":
        lines += [
            "", f"Values are mean AXBench Overall on {prompts} validation responses per cell. Bold marks each winner.",
            "", f"We evaluated Original and the three frozen Mega controllers on all {behaviors} held-out behaviors. "
            "Each behavior was presented directly and with five jailbreak variants.",
            "", "**Table R1b. Held-out HarmBench results at the selected settings**", "",
            *giga_table(winners, results),
            "", "Values are means ± ten-group delete-one-group jackknife standard errors. Each group deletion "
            "removes the direct prompt and all five jailbreak variants for the same behaviors. Jailbreak ASR pools "
            "the five variants.",
        ]
        return "\n".join(lines) + "\n"
    jailbreak_asr = {
        method: results[method]["human_jailbreak"]["asr_percent"]["full_sample_mean"]
        for method in METHODS
    }
    lines += [
        "", f"Values are mean AXBench Overall on {prompts} validation responses per cell. Bold marks each winner.",
        "", f"After freezing the selected setting for each method, we evaluated all three on the same "
        f"{behaviors} behaviors, presented directly and with five jailbreak variants each. "
        f"H∞ achieved {jailbreak_asr['h_infinity']:.2f}% jailbreak ASR versus "
        f"{jailbreak_asr['alqr_8d']:.2f}% for A-LQR (8D) and "
        f"{jailbreak_asr['alqr']:.2f}% for tuned A-LQR (Full-width). "
        "Table R1b gives the paired differences under equal tuning and their uncertainty.", "",
        "**Table R1b. Held-out HarmBench results at the selected settings**", "",
        "| Controller | Selected (Q/R, Qf/R) | Direct ASR (%) | Jailbreak ASR (%) | Jailbreak gap vs H∞ (pp) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        config = winners[method]["configuration"]
        selected = f"({ratio(config['q_over_r'])}, {ratio(config['q_final_over_r'])})"
        direct = results[method]["direct"]["asr_percent"]["full_sample_mean"]
        jailbreak = results[method]["human_jailbreak"]
        gap = jailbreak.get("gap_vs_h_infinity_pp")
        gap_text = "Reference"
        if gap is not None:
            lo, hi = gap["interval_95"]
            gap_text = f"{gap['full_sample_mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        lines.append(
            f"| {LABELS[method]} | {selected} | {direct:.2f} | "
            f"{jailbreak['asr_percent']['full_sample_mean']:.2f} | {gap_text} |"
        )
    lines += [
        "", "Lower ASR is better. Gaps subtract H∞ ASR from the A-LQR ASR. "
        "Brackets show nominal 95% paired intervals from the behavior-group jackknife.",
    ]
    if profile != "full":
        lines += ["", f"This is the {behaviors}-behavior screening result."]
    else:
        lines += ["", f"The full sample includes the {MINI_BEHAVIORS} screening behaviors. The separate "
                  f"{behaviors - MINI_BEHAVIORS}-behavior results are in results.json."]
    return "\n".join(lines) + "\n"
