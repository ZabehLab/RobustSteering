#!/usr/bin/env python3
"""Compare selected Srob values with changes in TruthfulQA True percentages."""

from __future__ import annotations

import csv
import json
import math

from analyze import REPO, TABLES, load_rows, metric, render_behavior_figure


def main() -> None:
    rows, sources = load_rows()
    for row in rows:
        for key in ("original_txi_pct", "hinf_txi_pct", "hinf_over_original_txi",
                    "hinf_minus_original_txi_pp"):
            del row[key]
        original = json.loads((REPO / row["original_result_path"]).read_text())
        hinf = json.loads((REPO / row["hinf_result_path"]).read_text())
        original_truthful = metric(original, "truth")[0]
        hinf_truthful = metric(hinf, "truth")[0]
        if not (math.isfinite(original_truthful) and math.isfinite(hinf_truthful)
                and 0 < original_truthful <= 100 and 0 <= hinf_truthful <= 100):
            raise ValueError(f"Invalid True percentages for {row['model']}, {row['condition']}")
        row.update({
            "original_truthful_pct": original_truthful,
            "hinf_truthful_pct": hinf_truthful,
            "hinf_over_original_truthful": hinf_truthful / original_truthful,
            "hinf_minus_original_truthful_pp": hinf_truthful - original_truthful,
        })

    TABLES.mkdir(exist_ok=True)
    with (TABLES / "truthfulqa_srob_truthful.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / "truthfulqa_srob_truthful.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# TruthfulQA Srob and Truthful (%)",
        "",
        "Truthful (%) uses the saved TruthfulQA True metric mean.",
        "Srob and γ★ retain the source CSV precision. Behavioral columns are displayed to six decimals.",
        "The CSV and JSON retain the full saved floating-point precision.",
        "",
        "| Model | Condition | Srob | γ★ | Original Truthful (%) | H∞ Truthful (%) | H∞ / Original | H∞ − Original (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['srob']!r} | {row['gamma_star']!r} | "
            f"{row['original_truthful_pct']:.6f} | {row['hinf_truthful_pct']:.6f} | "
            f"{row['hinf_over_original_truthful']:.6f} | {row['hinf_minus_original_truthful_pp']:+.6f} |"
        )
    (TABLES / "truthfulqa_srob_truthful.md").write_text("\n".join(lines) + "\n")
    (TABLES / "truthful_source_manifest.json").write_text(
        json.dumps(sources, indent=2, allow_nan=False) + "\n"
    )

    outcomes = (
        ("hinf_over_original_truthful", "H∞ Truthful (%) / Original Truthful (%)", 1.0),
        ("hinf_minus_original_truthful_pp", "H∞ Truthful − Original Truthful (pp)", 0.0),
    )
    label_offsets = {
        ("truthfulness", "hinf_minus_original_truthful_pp", "gpt2_xl"): (10, -12),
        ("truthfulness", "hinf_minus_original_truthful_pp", "gemma2_2b"): (10, 15),
        ("truthfulness_spanish", "hinf_over_original_truthful", "gpt2_xl"): (10, -12),
        ("truthfulness_spanish", "hinf_over_original_truthful", "olmo2_32b"): (-9, 20),
        ("truthfulness_spanish", "hinf_minus_original_truthful_pp", "gpt2_xl"): (10, -15),
        ("truthfulness_spanish", "hinf_minus_original_truthful_pp", "qwen25_14b"): (12, 24),
        ("truthfulness_spanish", "hinf_minus_original_truthful_pp", "gemma2_2b"): (10, 7),
    }
    statistics = render_behavior_figure(rows, outcomes, "srob_truthful_changes", label_offsets)
    (TABLES / "truthful_regressions.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n"
    )
    print("\n".join(lines))
    print(f"Verified {len(rows)} rows against {len(sources)} source files")


if __name__ == "__main__":
    main()
