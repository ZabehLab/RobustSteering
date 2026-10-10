#!/usr/bin/env python3
"""Build the analysis-local Original versus H-infinity TruthfulQA table."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
RESULTS = REPO / "benchmarks/truthfulness/results/kv_cache_off"
OUTPUT = UNIT / "truthfulness_original_vs_hinf.md"
MANIFEST = UNIT / "truthfulness_result_sources.json"

MODELS = (
    ("Pythia-14M", "calibrations/lambda_txi_n200_r1/pythia_14m/truthfulness"),
    ("Pythia-31M", "calibrations/lambda_txi_n200_r1/pythia_31m/truthfulness"),
    ("DistilGPT-2", "calibrations/lambda_txi_n200_r1/distilgpt2/truthfulness"),
    ("GPT-2 Small", "calibrations/lambda_txi_n200_r1/gpt2_small/truthfulness"),
    ("SmolLM2-135M", "calibrations/lambda_txi_n200_r1/smollm2_135m/truthfulness"),
    ("Pythia-160M", "calibrations/lambda_txi_n200_r1/pythia_160m/truthfulness"),
    ("GPT-2 Medium", "calibrations/lambda_txi_n200_r1/gpt2_medium/truthfulness"),
    ("Qwen-2.5-0.5B", "calibrations/lambda_txi_n200_r1/qwen25_05b/truthfulness"),
    ("GPT-2 Large", "calibrations/lambda_txi_n200_r1/gpt2_large/truthfulness"),
    (
        "GPT-2 XL",
        "calibrations/txi95_fluency05_n200_r1/gpt2_xl/truthfulness",
    ),
    ("Gemma-2-2B", "gemma2b/truthfulness"),
    ("Llama-3-8B", None),
    ("Qwen-2.5-14B", "qwen14b/truthfulness"),
    (
        "OLMo-2-32B",
        "calibrations/txi95_fluency05_n200_r1/olmo2_32b_instruct/truthfulness",
    ),
)

METHODS = (("original", "Original"), ("h_infinity", "H∞ (ours)"))
METRICS = (
    ("truth", "True (%) ↑"),
    ("info", "Informative (%) ↑"),
    ("txi", "T×I (%) ↑"),
    ("instruction_relevance", "Instruction relevance (0–2) ↑"),
    ("fluency", "Fluency (0–2) ↑"),
)


def result_path(model: str, base: str | None, method: str) -> Path:
    if model == "Llama-3-8B" and method == "h_infinity":
        relative = "calibrations/txi_fluency_95_05/llama8b/truthfulness"
    elif model == "Llama-3-8B":
        relative = "llama8b/truthfulness"
    else:
        if base is None:
            raise RuntimeError(f"missing result base for {model}")
        relative = base
    return RESULTS / relative / f"{method}.json"


def jackknife_standard_error(estimates: list[float]) -> float:
    center = sum(estimates) / len(estimates)
    return math.sqrt(
        (len(estimates) - 1)
        / len(estimates)
        * sum((estimate - center) ** 2 for estimate in estimates)
    )


def metric(result: dict, key: str) -> tuple[float, float | None]:
    metrics = result["metrics"]
    if key == "txi":
        truth = metrics["truth"]
        info = metrics["info"]
        truth_mean = float(truth["mean"])
        info_mean = float(info["mean"])
        mean = truth_mean * info_mean / 100.0
        truth_loo = truth.get("leave_group_out_estimates")
        info_loo = info.get("leave_group_out_estimates")
        if truth_loo and info_loo and len(truth_loo) == len(info_loo):
            estimates = [
                float(truth_value) * float(info_value) / 100.0
                for truth_value, info_value in zip(truth_loo, info_loo, strict=True)
            ]
            standard_error = jackknife_standard_error(estimates)
        else:
            truth_se = truth.get("standard_error")
            info_se = info.get("standard_error")
            standard_error = (
                None
                if truth_se is None or info_se is None
                else math.sqrt(
                    (info_mean / 100.0) ** 2 * float(truth_se) ** 2
                    + (truth_mean / 100.0) ** 2 * float(info_se) ** 2
                )
            )
        return mean, standard_error

    value = metrics.get(key)
    if value is None:
        raise RuntimeError(f"result lacks required metric {key!r}")
    standard_error = value.get("standard_error")
    return float(value["mean"]), (
        None if standard_error is None else float(standard_error)
    )


def formatted(value: tuple[float, float | None]) -> str:
    mean, standard_error = value
    return (
        f"{mean:.2f}"
        if standard_error is None
        else f"{mean:.2f} ± {standard_error:.2f}"
    )


def main() -> None:
    rows = []
    sources = []
    for model, base in MODELS:
        for method, method_label in METHODS:
            path = result_path(model, base, method)
            if not path.exists():
                raise FileNotFoundError(path)
            payload = path.read_bytes()
            result = json.loads(payload)
            rows.append(
                {
                    "model": model,
                    "method": method_label,
                    "values": [metric(result, key) for key, _ in METRICS],
                }
            )
            sources.append(
                {
                    "model": model,
                    "method": method,
                    "path": str(path.relative_to(REPO)),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "created_at_utc": result.get("created_at_utc"),
                    "evaluation_samples_per_repetition": result.get(
                        "evaluation_samples_per_repetition"
                    ),
                    "evaluation_repetitions": result.get("evaluation_repetitions"),
                    "uncertainty": result.get("uncertainty", {}).get("method"),
                    "calibration_id": result.get("identity", {}).get("calibration_id"),
                }
            )

    headers = " | ".join(label for _, label in METRICS)
    lines = [
        "# TruthfulQA: Original versus H∞",
        "",
        "This analysis-local table is intentionally separate from the paper-facing benchmark tables.",
        "It reports the English TruthfulQA evaluation used by the Gamma Star Addition models.",
        "",
        f"| Model | Method | {headers} |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in rows:
        values = " | ".join(formatted(value) for value in row["values"])
        lines.append(f"| {row['model']} | {row['method']} | {values} |")
    lines.extend(
        [
            "",
            "## Reading the table",
            "",
            "- T×I is the product of the aggregate True and Informative percentages divided by 100; it is not a per-response intersection rate.",
            "- Values are full-sample means ± their stored standard errors. The nine new one-pass runs use five-group delete-one-group question-jackknife uncertainty.",
            "- Instruction relevance and fluency are AXBench judge scores on the 0–2 scale.",
            "- Exact source paths, hashes, run sizes, timestamps, calibration IDs, and uncertainty methods are recorded in `truthfulness_result_sources.json`.",
            "- Nothing in `figs/bench_table/` is read, regenerated, or modified by this unit.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines))
    MANIFEST.write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
