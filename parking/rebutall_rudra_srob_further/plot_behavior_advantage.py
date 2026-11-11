#!/usr/bin/env python3
"""Plot H-infinity behavioral advantage relative to the paper A-LQR baseline."""

from __future__ import annotations

import json

from analyze import TABLES
from plot_behavior_gain_variants import load_points
from plot_truthful_harm_ratios import TASKS, render_comparison


def main() -> None:
    points, sources = load_points()
    for row in points:
        if row["alqr_pct"] <= 0:
            raise ValueError(f"A-LQR denominator must be positive for {row['model']}, {row['condition']}")
        row["behavior_advantage"] = (row["hinf_pct"] - row["alqr_pct"]) / row["alqr_pct"]
    benchmark_rows = {
        benchmark: [dict(row, value=row["behavior_advantage"])
                    for row in points if row["benchmark"] == benchmark]
        for benchmark, _, _, _ in TASKS
    }
    fits = render_comparison(
        benchmark_rows, ylabel="Behavior advantage against competitor",
        stem="srob_truthful_harm_behavior_advantage", y_limits=((-0.03, 0.62), (-1.02, -0.40)),
    )
    output = {"definition": "(H∞ - A-LQR) / A-LQR", "sources_sha256": sources,
              "points": points, "fits": fits}
    (TABLES / "truthful_harm_behavior_advantage.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
