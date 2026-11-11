#!/usr/bin/env python3
"""Render the Srob extension from cached figure data."""
from pathlib import Path
import csv
import json
import re
import subprocess
import sys
from unittest.mock import patch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
import numpy as np
from scipy.interpolate import PchipInterpolator

UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
BEHAVIOR = REPO / 'parking/rebutall_rudra_srob_further'
TREND = REPO / 'parking/rudra_rebuttal_trend_of_srob'
PLOTS = UNIT / 'plots'
FONT = 11


def main():
    sys.path.insert(0, str(BEHAVIOR / 'src'))
    import report_behavior_gain as behavior
    points = json.loads((BEHAVIOR / 'tables/behavior_signal_points.json').read_text())
    slopes = json.loads((BEHAVIOR / 'tables/behavior_signal_slopes.json').read_text())
    # Reuse the approved top-row artists without writing to the source unit.
    with patch.object(Figure, 'savefig'), patch.object(plt, 'close'):
        behavior.render(points, slopes)
        fig = plt.gcf()
    for ax in fig.axes:
        metric = ax.get_ylabel().split(' behavior gain')[0]
        ax.set_ylabel({'T': 'Relative truthfulness gain', 'ASR': 'Relative ASR change',
                       'Accuracy': 'Relative accuracy gain'}[metric], fontsize=FONT)
        ax.xaxis.label.set_fontsize(FONT)
        ax.title.set_fontsize(FONT)
        ax.tick_params(labelsize=FONT)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(7)
    for text in fig.texts:
        text.set_fontsize(FONT)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    width, top_height = bbox.width, bbox.height
    info = subprocess.check_output(['pdfinfo', str(TREND / 'plots/truthfulqa_srob_calibration_trends.pdf')], text=True)
    size = re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info)
    if size is None:
        raise ValueError('Cannot read original trend figure dimensions')
    bottom_height = 0.60 * width * float(size[2]) / float(size[1])
    gap = 0.25
    height = top_height + gap + bottom_height
    old_width, old_height = fig.get_size_inches()
    axes_positions = [(ax, ax.get_position().bounds) for ax in fig.axes]
    text_positions = [(text, text.get_position()) for text in fig.texts]
    fig.set_size_inches(width, height)
    for ax, (left, bottom, w, h) in axes_positions:
        ax.set_position([(left * old_width - bbox.x0) / width,
                         (bottom * old_height - bbox.y0 + bottom_height + gap) / height,
                         w * old_width / width, h * old_height / height])
    for text, (x, y) in text_positions:
        text.set_position(((x * old_width - bbox.x0) / width,
                          (y * old_height - bbox.y0 + bottom_height + gap) / height))
    rows = json.loads((TREND / 'tables/calibration_points.json').read_text())
    models = json.loads((TREND / 'tables/summary.json').read_text())['models']
    with (TREND / 'tables/matched_cost_traces.csv').open() as handle:
        traces = list(csv.DictReader(handle))
    plot_height = bottom_height - 0.98
    key_width = plot_height / 2
    key_left = width - key_width - 0.25
    ax = fig.add_axes([0.67 / width, 0.80 / height,
                       (key_left - 1.15 - 0.67) / width, plot_height / height])
    key_ax = fig.add_axes([key_left / width, 0.80 / height, key_width / width, plot_height / height])
    x = np.array([model['parameter_billions'] for model in models])
    dense_x = np.geomspace(x.min(), x.max(), 250)
    for trace in traces:
        y = np.array([float(trace[f"{model['model_key']}_srob"]) for model in models])
        curve = PchipInterpolator(np.log10(x), np.log10(y))
        ax.plot(dense_x, 10 ** curve(np.log10(dense_x)), color=trace['color_hex'], linewidth=1, alpha=0.85)
    ax.scatter([row['parameter_billions'] for row in rows], [row['srob'] for row in rows],
               s=9, color='#202020', alpha=0.65, linewidths=0, zorder=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(x, [f"{model['model']}\n{model['parameter_billions']:.2f}B" for model in models], fontsize=FONT)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlim=(x.min() / 1.13, x.max() * 1.13),
           ylim=(min(row['srob'] for row in rows) / 1.25, max(row['srob'] for row in rows) * 1.3))
    ax.set_xlabel(r'Model size (billions of parameters; $\log_{10}$ scale)', fontsize=FONT)
    ax.set_ylabel(r'TruthfulQA $S_{\mathrm{rob}}$ ($\log_{10}$ scale)', fontsize=FONT)
    ax.tick_params(labelsize=FONT)
    ax.spines[['top', 'right']].set_visible(False)
    q_values = sorted({float(trace['q_over_r']) for trace in traces})
    qf_values = sorted({float(trace['q_final_over_r']) for trace in traces})
    ranks = np.empty((8, 4), dtype=int)
    for trace in traces:
        ranks[q_values.index(float(trace['q_over_r'])), qf_values.index(float(trace['q_final_over_r']))] = int(trace['setting_rank']) - 1
    key_ax.imshow(ranks, origin='lower', cmap=ListedColormap([trace['color_hex'] for trace in traces]),
                  vmin=-0.5, vmax=31.5, aspect='equal', interpolation='nearest')
    key_ax.set_xticks(range(4), [f'{value:.3g}' for value in qf_values], rotation=45, ha='right', fontsize=FONT)
    key_ax.set_yticks(range(8), [f'{value:.3g}' for value in q_values], fontsize=FONT)
    key_ax.set_xlabel(r'$Q_f/R$', fontsize=FONT)
    key_ax.set_ylabel(r'$Q/R$', fontsize=FONT)
    for i, j in np.ndindex(ranks.shape):
        rank = ranks[i, j]
        key_ax.text(j, i, str(rank + 1), ha='center', va='center', fontsize=7,
                    color='white' if rank < 16 else 'black')
    key_ax.spines[['top', 'right']].set_visible(True)
    PLOTS.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(PLOTS / f'fig_srob_extend.{suffix}', dpi=600, facecolor='white')
    plt.close(fig)
    print(PLOTS / 'fig_srob_extend.pdf')


if __name__ == '__main__':
    main()
