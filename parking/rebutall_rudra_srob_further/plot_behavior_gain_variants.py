#!/usr/bin/env python3
"""Plot relative behavioral changes for H-infinity and the paper A-LQR baseline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from analyze import REPO, TABLES
from plot_truthful_harm_ratios import TASKS, render_comparison


METRICS = {
    "TruthfulQA": ("truth", 1.0, "Truthful (%)", "truthful",
                   ("model_id", "model_revision", "distribution", "evaluation_key", "dataset", "kv_cache")),
    "HarmBench": ("harmbench_test_success.score", 100.0, "ASR (%)", "asr",
                  ("model", "condition", "evaluated_model_kv_cache")),
}
VARIANTS = {
    "hinf": ("H∞", ((0.1, 0.85), (-1.02, -0.38))),
    "alqr": ("A-LQR", ((-0.05, 0.7), (-0.36, 0.30))),
}


def load_points() -> tuple[list[dict], dict]:
    sources = {}

    def read_json(path: Path) -> dict | list:
        content = path.read_bytes()
        sources[str(path.relative_to(REPO))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    points = []
    for benchmark, filename, _, _ in TASKS:
        metric_key, scale, metric_label, field_name, identity_keys = METRICS[benchmark]
        for row in read_json(TABLES / filename):
            paths = {"original": REPO / row["original_result_path"],
                     "h_infinity": REPO / row["hinf_result_path"],
                     "alqr": (REPO / row["original_result_path"]).with_name("alqr.json")}
            values = {}
            original_identity = None
            for method, path in paths.items():
                result = read_json(path)
                identity = result["identity"]
                if original_identity is None:
                    original_identity = identity
                if (identity["method"] != method
                        or any(identity[key] != original_identity[key] for key in identity_keys)):
                    raise ValueError(f"Result identity mismatch in {path}")
                saved = result["metrics"][metric_key]
                value = float(saved["mean"] if isinstance(saved, dict) else saved) * scale
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f"Invalid {metric_label} in {path}")
                values[method] = value
            original = values["original"]
            if original <= 0:
                raise ValueError(f"Original denominator must be positive for {row['model']}")
            if not (math.isclose(original, row[f"original_{field_name}_pct"], rel_tol=1e-12)
                    and math.isclose(values["h_infinity"], row[f"hinf_{field_name}_pct"], rel_tol=1e-12)):
                raise ValueError(f"Saved table and raw result disagree for {row['model']}")
            points.append({
                "benchmark": benchmark, "model": row["model"], "condition": row["condition"],
                "srob": row["srob"], "gamma_star": row["gamma_star"], "metric": metric_label,
                "original_pct": original, "hinf_pct": values["h_infinity"], "alqr_pct": values["alqr"],
                "hinf_behavior_gain": (values["h_infinity"] - original) / original,
                "alqr_behavior_gain": (values["alqr"] - original) / original,
                "result_paths": {method: str(path.relative_to(REPO)) for method, path in paths.items()},
            })
    return points, sources


def main() -> None:
    points, sources = load_points()
    fits = {}
    for variant, (method_label, y_limits) in VARIANTS.items():
        benchmark_rows = {
            benchmark: [dict(row, value=row[f"{variant}_behavior_gain"])
                        for row in points if row["benchmark"] == benchmark]
            for benchmark, _, _, _ in TASKS
        }
        fits[method_label] = render_comparison(
            benchmark_rows, ylabel="Behavior gain", stem=f"srob_truthful_harm_{variant}_behavior_gain",
            y_limits=y_limits,
        )
    output = {"definition": "(method - Original) / Original", "sources_sha256": sources,
              "points": points, "fits": fits}
    (TABLES / "truthful_harm_behavior_gain.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
