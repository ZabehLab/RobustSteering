#!/usr/bin/env python3
"""Compare Srob and ASR changes for the three paper-reported HarmBench models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from analyze import REPO, TABLES, render_behavior_figure


POINTS = REPO / "parking/gamma_star_organization/plots/srob_overall_points.csv"
CALIBRATIONS = REPO / "parking/gamma_star_organization/exports/gamma_star_collaborator_bundle/final_selected_calibrations.csv"
JACKKNIFE = REPO / "figs/bench_table/harmful/harmbench_jackknife.json"
MODELS = {
    "llama32_1b": "Llama-3.2-1B",
    "llama32_3b": "Llama-3.2-3B",
    "llama31_8b": "Llama-3.1-8B",
}
CONDITIONS = {
    "harmbench_direct": ("Direct (ID)", "direct", "#007C7C"),
    "harmbench_human_jailbreak": ("Jailbreak (OOD)", "human_jailbreak", "#8B1E1E"),
}


def load_harmbench_rows() -> tuple[list[dict], dict]:
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
              if row["benchmark"] == "harmful"}
    calibrations = {row["model_key"]: row for row in csv.DictReader(read_source(CALIBRATIONS).decode().splitlines())
                    if row["benchmark"] == "harmful"}
    if points.keys() != MODELS.keys():
        raise ValueError("Expected exactly the three paper-reported HarmBench Srob points")
    jackknife = read_json(JACKKNIFE)
    rows = []
    for model_key, label in MODELS.items():
        point = points[model_key]
        calibration = calibrations[model_key]
        selection_path = REPO / calibration["selection_path"]
        selection = read_json(selection_path)
        score_path = selection_path.parent / selection["diagnostic_bundle"] / "score.json"
        score = read_json(score_path)
        srob = float(point["s_rob_value"])
        gamma_star = float(point["gamma_star"])
        if not (math.isclose(srob, score["s_rob"], rel_tol=1e-12)
                and math.isclose(gamma_star, score["gamma_star"], rel_tol=1e-12)
                and math.isclose(srob * gamma_star, 1, rel_tol=1e-12)):
            raise ValueError(f"Figure and selected-controller diagnostic disagree for {label}")
        cache_key = selection_path.parents[3].name
        dataset_path = selection_path.parents[3] / "datasets/harmbench.json"
        dataset = read_json(dataset_path)["evaluation"]
        paper_values = jackknife["models"][cache_key]
        paper_ids = {behavior_id for group in paper_values["groups"] for behavior_id in group}
        dataset = {condition: [row for row in records if row["behavior_id"] in paper_ids]
                   for condition, records in dataset.items()}
        direct_ids = {row["behavior_id"] for row in dataset["direct"]}
        if direct_ids != paper_ids or len(direct_ids) != paper_values["behavior_count"]:
            raise ValueError(f"Dataset does not contain the paper's evaluated behaviors for {label}")
        template_ids = {int(row["template_index"]) for row in dataset["human_jailbreak"]}
        if template_ids != set(range(5)):
            raise ValueError(f"Expected the five paper jailbreak templates for {label}")
        for template in template_ids:
            behavior_ids = [row["behavior_id"] for row in dataset["human_jailbreak"]
                            if int(row["template_index"]) == template]
            if set(behavior_ids) != direct_ids or len(behavior_ids) != len(direct_ids):
                raise ValueError(f"Direct and jailbreak behavior identities differ for {label}")

        for evaluation_key, (condition, condition_key, _) in CONDITIONS.items():
            results = {}
            paths = {}
            for method in ("original", "h_infinity"):
                path = REPO / "benchmarks/harmful/results/kv_cache_off" / cache_key / evaluation_key / f"{method}.json"
                result = read_json(path)
                identity = result["identity"]
                if (identity["model"], identity["method"], identity["condition"],
                        identity["evaluated_model_kv_cache"]) != (selection["model"], method, condition_key, False):
                    raise ValueError(f"Model or evaluation identity mismatch in {path}")
                if method == "h_infinity" and identity["parameters"] != selection["selected"]["parameters"]:
                    raise ValueError(f"H∞ controller settings differ from the Srob controller in {path}")
                if result["sample_count"] != len(dataset[condition_key]):
                    raise ValueError(f"Evaluation count differs from the paper dataset in {path}")
                score_cache_path = (selection_path.parents[3] / "evaluations/kv_cache_off/scores"
                                    / "harmbench_test_success" / evaluation_key / method / "final.json")
                judgments = read_json(score_cache_path)
                expected_prompt_ids = {row["prompt_id"] for row in dataset[condition_key]}
                scored_prompt_ids = {row["prompt_id"] for row in judgments["rows"]}
                if (judgments["status"] != "complete" or scored_prompt_ids != expected_prompt_ids
                        or len(judgments["rows"]) != len(expected_prompt_ids)
                        or not all(row.get("valid", True) for row in judgments["rows"])):
                    raise ValueError(f"Scored prompts differ from the paper's behavior subset in {score_cache_path}")
                scored_asr = sum(float(row["score"]) for row in judgments["rows"]) / len(judgments["rows"])
                if not math.isclose(scored_asr, result["metrics"]["harmbench_test_success.score"], rel_tol=1e-12):
                    raise ValueError(f"ASR aggregate differs from cached judgments in {path}")
                results[method] = result
                paths[method] = str(path.relative_to(REPO))
            original_asr = 100 * float(results["original"]["metrics"]["harmbench_test_success.score"])
            hinf_asr = 100 * float(results["h_infinity"]["metrics"]["harmbench_test_success.score"])
            if not (math.isfinite(original_asr) and math.isfinite(hinf_asr)
                    and 0 < original_asr <= 100 and 0 <= hinf_asr <= 100):
                raise ValueError(f"Invalid ASR percentages for {label}, {condition}")
            for method, value in (("original", original_asr), ("h_infinity", hinf_asr)):
                if condition_key == "direct":
                    reported = paper_values["conditions"]["direct"][method]["harmbench_test_success"]
                else:
                    reported = paper_values["human_jailbreak_aggregate"][method]
                if not math.isclose(value, reported["full_sample_mean"], rel_tol=1e-12):
                    raise ValueError(f"Saved ASR differs from the paper table for {label}, {condition}")
            rows.append({
                "model": label,
                "model_key": model_key,
                "model_id": selection["model"][0],
                "model_revision": selection["model"][1],
                "srob": srob,
                "gamma_star": gamma_star,
                "condition": condition,
                "evaluation_key": evaluation_key,
                "original_asr_pct": original_asr,
                "hinf_asr_pct": hinf_asr,
                "hinf_over_original_asr": hinf_asr / original_asr,
                "hinf_minus_original_asr_pp": hinf_asr - original_asr,
                "behavior_count": len(direct_ids),
                "sample_count": results["original"]["sample_count"],
                "template_count": results["original"]["sample_count"] // len(direct_ids),
                "calibration_id": selection["calibration_id"],
                "original_result_path": paths["original"],
                "hinf_result_path": paths["h_infinity"],
                "selection_path": str(selection_path.relative_to(REPO)),
                "score_path": str(score_path.relative_to(REPO)),
                "dataset_path": str(dataset_path.relative_to(REPO)),
            })
    return rows, sources


def main() -> None:
    rows, sources = load_harmbench_rows()
    TABLES.mkdir(exist_ok=True)
    with (TABLES / "harmbench_srob_asr.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / "harmbench_srob_asr.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (TABLES / "harmbench_source_manifest.json").write_text(json.dumps(sources, indent=2, allow_nan=False) + "\n")
    lines = [
        "# HarmBench Srob and attack success rate",
        "",
        "The three paper-reported models. Jailbreak ASR averages the five frozen templates.",
        "ASR is lower-is-better. Ratios below one and negative differences indicate reduced attack success.",
        "Srob and γ★ retain source precision. ASR and derived columns are displayed to six decimals.",
        "The CSV and JSON retain full floating-point precision.",
        "",
        "| Model | Condition | Srob | γ★ | Original ASR (%) | H∞ ASR (%) | H∞ / Original | H∞ − Original (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['srob']!r} | {row['gamma_star']!r} | "
            f"{row['original_asr_pct']:.6f} | {row['hinf_asr_pct']:.6f} | "
            f"{row['hinf_over_original_asr']:.6f} | {row['hinf_minus_original_asr_pp']:+.6f} |"
        )
    (TABLES / "harmbench_srob_asr.md").write_text("\n".join(lines) + "\n")
    outcomes = (
        ("hinf_over_original_asr", "H∞ ASR / Original ASR", 1.0),
        ("hinf_minus_original_asr_pp", "H∞ ASR − Original ASR (pp)", 0.0),
    )
    offsets = {"llama32_1b": (-9, 9), "llama32_3b": (10, 7), "llama31_8b": (10, -14)}
    label_offsets = {(condition, outcome, key): offset
                     for condition in CONDITIONS for outcome, _, _ in outcomes
                     for key, offset in offsets.items()}
    statistics = render_behavior_figure(
        rows, outcomes, "srob_harmbench_asr_changes", label_offsets,
        conditions=CONDITIONS, x_limits=(1.9, 8.5), x_ticks=(2, 3, 4, 5, 6, 7, 8),
        y_limits=((-0.05, 1.1), (-17, 2)),
    )
    (TABLES / "harmbench_regressions.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    print("\n".join(lines))
    print(f"Verified {len(rows)} rows against {len(sources)} source files")


if __name__ == "__main__":
    main()
