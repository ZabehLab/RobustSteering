#!/usr/bin/env python3
"""Combine the finished behavior panels and matched-calibration trend figure."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile


UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
TOP = REPO / "parking/rebutall_rudra_srob_further/hinf_behavior_gain.pdf"
BOTTOM = REPO / "parking/rudra_rebuttal_trend_of_srob/plots/truthfulqa_srob_calibration_trends.pdf"
PLOTS = UNIT / "plots"


def page_size(path: Path) -> tuple[float, float]:
    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
    if match is None:
        raise ValueError(f"Cannot read PDF page size: {path}")
    return float(match[1]), float(match[2])


def main() -> None:
    width, top_height = page_size(TOP)
    bottom_width, bottom_height = page_size(BOTTOM)
    bottom_height *= width / bottom_width
    gap = 18
    height = top_height + gap + bottom_height
    source = rf"""\documentclass{{article}}
\usepackage[paperwidth={width}bp,paperheight={height}bp,margin=0bp]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\topskip}}{{0pt}}
\setlength{{\unitlength}}{{1bp}}
\begin{{document}}
\noindent\begin{{picture}}({width},{height})
\put(0,0){{\includegraphics[width={width}bp]{{\detokenize{{{BOTTOM}}}}}}}
\put(0,{bottom_height + gap}){{\includegraphics[width={width}bp]{{\detokenize{{{TOP}}}}}}}
\end{{picture}}
\end{{document}}
"""
    PLOTS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        (temporary / "fig_srob_extend.tex").write_text(source)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig_srob_extend.tex"],
            cwd=temporary, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
        shutil.copy2(temporary / "fig_srob_extend.pdf", PLOTS / "fig_srob_extend.pdf")
    subprocess.run(
        ["pdftoppm", "-r", "600", "-singlefile", "-png", str(PLOTS / "fig_srob_extend.pdf"),
         str(PLOTS / "fig_srob_extend")], check=True,
    )
    print(PLOTS / "fig_srob_extend.pdf")


if __name__ == "__main__":
    main()
