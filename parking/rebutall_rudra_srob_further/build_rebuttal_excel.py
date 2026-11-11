#!/usr/bin/env python3
"""Organize the saved TruthfulQA and HarmBench ratios into an Excel workbook."""

from __future__ import annotations

import csv
from pathlib import Path

import xlsxwriter


UNIT = Path(__file__).resolve().parent
TABLES = UNIT / "tables"
OUTPUT = TABLES / "srob_truthful_harm_rebuttal.xlsx"
INPUTS = (
    ("TruthfulQA", "Truthful (%) ↑", "truthfulqa_srob_truthful.csv", "truthful"),
    ("HarmBench", "ASR (%) ↓", "harmbench_srob_asr.csv", "asr"),
)
HEADERS = ["Benchmark", "Model", "Metric", "Srob", "γ★", "Original (%)", "H∞ (%)", "H∞ / Original"]


def main() -> None:
    rows = {"ID": [], "OOD": []}
    for benchmark, metric, filename, measure in INPUTS:
        with (TABLES / filename).open(newline="") as handle:
            for source in csv.DictReader(handle):
                split = "OOD" if "(OOD)" in source["condition"] else "ID"
                values = [source["srob"], source["gamma_star"], source[f"original_{measure}_pct"],
                          source[f"hinf_{measure}_pct"], source[f"hinf_over_original_{measure}"]]
                rows[split].append({"benchmark": benchmark, "model": source["model"],
                                    "metric": metric, "values": values, "source": filename})

    with xlsxwriter.Workbook(OUTPUT) as workbook:
        workbook.set_properties({"title": "Srob and relative behavioral performance", "author": ""})
        base = {"font_name": "Arial", "font_size": 11, "valign": "vcenter"}
        source_header = workbook.add_format({**base, "bold": True, "bg_color": "#E8ECEF", "text_wrap": True})
        sheets = {split: workbook.add_worksheet(split) for split in rows}
        sources = workbook.add_worksheet("Sources")
        source_columns = ["Split", "Benchmark", "Model", "Source CSV", "Srob (source text)",
                          "γ★ (source text)", "Original (source text)", "H∞ (source text)", "Ratio (source text)"]
        sources.write_row(0, 0, source_columns, source_header)
        source_row = 1
        for split, records in rows.items():
            sheet = sheets[split]
            accent = "#007C7C" if split == "ID" else "#8B1E1E"
            title = workbook.add_format({**base, "font_size": 18, "bold": True, "font_color": accent})
            subtitle = workbook.add_format({**base, "font_color": "#505860"})
            header = workbook.add_format({**base, "bold": True, "font_color": "#FFFFFF", "bg_color": accent})
            note = workbook.add_format({**base, "font_size": 10, "font_color": "#505860", "text_wrap": True})
            sheet.hide_gridlines(2)
            sheet.set_tab_color(accent)
            sheet.merge_range("A1:H1", f"{split} · Srob and H∞ / Original", title)
            description = ("English TruthfulQA · Direct HarmBench requests" if split == "ID"
                           else "Spanish-input TruthfulQA · Five-template HarmBench jailbreak average")
            sheet.merge_range("A2:H2", description, subtitle)
            sheet.set_row(0, 31)
            sheet.set_row(1, 23)
            sheet.set_row(3, 8)
            sheet.set_column("A:A", 15)
            sheet.set_column("B:B", 21)
            sheet.set_column("C:C", 19)
            sheet.set_column("D:E", 19)
            sheet.set_column("F:G", 17)
            sheet.set_column("H:H", 20)
            sheet.add_table(4, 0, 4 + len(records), 7, {
                "name": f"{split}_Comparison", "style": None,
                "columns": [{"header": value, "header_format": header} for value in HEADERS],
            })
            sheet.set_row(4, 28)
            for index, row in enumerate(records, start=5):
                fill = "#F3F6F7" if index % 2 == 0 else "#FFFFFF"
                text_format = workbook.add_format({**base, "bg_color": fill})
                precise_format = workbook.add_format({**base, "bg_color": fill, "num_format": "0.0000000000"})
                percent_format = workbook.add_format({**base, "bg_color": fill, "num_format": "0.0000"})
                ratio_format = workbook.add_format({**base, "bg_color": fill, "bold": True, "num_format": "0.000000"})
                sheet.write_row(index, 0, [row["benchmark"], row["model"], row["metric"]], text_format)
                for column, value in enumerate(row["values"][:4], start=3):
                    sheet.write_number(index, column, float(value), precise_format if column < 5 else percent_format)
                sheet.write_formula(index, 7, f"=G{index + 1}/F{index + 1}", ratio_format, float(row["values"][4]))
                sheet.set_row(index, 25)
                sources.write_row(source_row, 0, [split, row["benchmark"], row["model"], row["source"], *row["values"]])
                source_row += 1
            footer_row = 6 + len(records)
            sheet.merge_range(footer_row, 0, footer_row, 7,
                              "TruthfulQA ratios above 1 indicate a higher True rate. HarmBench ratios below 1 indicate a lower ASR.", note)
            sheet.merge_range(footer_row + 1, 0, footer_row + 1, 7,
                              "Full-precision source values are retained as text on the Sources sheet.", note)
            sheet.set_row(footer_row, 25)
            sheet.freeze_panes(5, 2)
            sheet.set_landscape()
            sheet.set_paper(9)
            sheet.fit_to_pages(1, 1)
            sheet.set_margins(0.3, 0.3, 0.4, 0.4)
            sheet.print_area(0, 0, footer_row + 1, 7)
        sources.set_column("A:B", 15)
        sources.set_column("C:C", 22)
        sources.set_column("D:D", 37)
        sources.set_column("E:I", 25)
        sources.set_row(0, 30)
        sources.freeze_panes(1, 4)
        sources.autofilter(0, 0, source_row - 1, 8)
        sheets["ID"].activate()
    print(OUTPUT)


if __name__ == "__main__":
    main()
