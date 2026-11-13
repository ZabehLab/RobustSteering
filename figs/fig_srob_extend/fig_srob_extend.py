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
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, NullLocator
import numpy as np
from scipy.interpolate import PchipInterpolator

UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
BEHAVIOR = REPO / 'parking/rudra_rebuttal_srob_further'
TREND = REPO / 'parking/rudra_rebuttal_trend_of_srob'
TRANSFER = REPO / 'parking/rudra_rebuttal_cross_task_srob'
SWEEP = REPO / 'parking/rudra_rebuttal_gamma_sweep'
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
        position = ax.get_position()
        ax.set_position([position.x0, position.y0 + .2 * position.height,
                         position.width, .8 * position.height])
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
    sweep_width = 1.60
    sweep_left = fig.axes[4].get_position().x1 * width - sweep_width
    transfer_width = fig.axes[4].get_position().width * width
    transfer_left = sweep_left - 0.75 - transfer_width
    key_left = transfer_left - 0.75 - key_width
    ax = fig.add_axes([0.67 / width, 0.80 / height,
                       (key_left - 0.85 - 0.67) / width, plot_height / height])
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
    ax.set_xticks(x, ['GPT-2 XL', 'Gemma 2B', 'Llama 8B', 'Qwen 14B', 'OLMo 32B'], fontsize=FONT)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlim=(x.min() / 1.13, x.max() * 1.13),
           ylim=(min(row['srob'] for row in rows) / 1.25, max(row['srob'] for row in rows) * 1.3))
    ax.set_xlabel('Model size (B; log scale)', fontsize=FONT)
    ax.set_ylabel(r'TruthfulQA $S_{\mathrm{rob}}$', fontsize=FONT)
    ax.set_title(r'Consistent $S_{\mathrm{rob}}$ patterns', fontsize=FONT,
                 fontweight='bold', pad=8)
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
    key_ax.set_title('Cost settings', fontsize=FONT, fontweight='bold', pad=8)
    for i, j in np.ndindex(ranks.shape):
        rank = ranks[i, j]
        key_ax.text(j, i, str(rank + 1), ha='center', va='center', fontsize=7,
                    color='white' if rank < 16 else 'black')
    key_ax.spines[['top', 'right']].set_visible(True)
    transfer_ax = fig.add_axes([transfer_left / width, 0.80 / height,
                                transfer_width / width, plot_height / height])
    with (TRANSFER / 'tables/cross_task_truthful_to_harmbench_points.csv').open() as handle:
        transfer_points = list(csv.DictReader(handle))
    with (TRANSFER / 'tables/cross_task_truthful_to_harmbench_slopes.csv').open() as handle:
        transfer_fits = {row['split']: row for row in csv.DictReader(handle)}
    handles = []
    for split, color in (('ID', 'black'), ('OOD', '#8B1E1E')):
        selected = [row for row in transfer_points if row['split'] == split]
        transfer_x = np.array([float(row['srob']) for row in selected])
        transfer_y = np.array([float(row['asr_gain']) for row in selected])
        fit = transfer_fits[split]
        line_x = np.array([transfer_x.min(), transfer_x.max()])
        transfer_ax.plot(line_x, float(fit['intercept']) + float(fit['slope']) * line_x,
                         color=color, linewidth=1.8)
        transfer_ax.scatter(transfer_x, transfer_y, s=9, color=color, zorder=3)
        handles.append(Line2D([0], [0], color=color, linewidth=1.8, marker='o',
                              markersize=3, label=split))
    transfer_ax.set_title('Cross-task\n$S_{\\mathrm{rob}}$ transfer', fontsize=FONT,
                          fontweight='bold', pad=8)
    transfer_ax.set_xlabel(r'TruthfulQA $S_{\mathrm{rob}}$', fontsize=FONT)
    transfer_ax.set_ylabel('Relative HarmBench ASR gain', fontsize=FONT)
    transfer_ax.set_xlim(0, 0.225)
    transfer_ax.set_xticks([0, 0.1, 0.2])
    transfer_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    transfer_ax.tick_params(labelsize=FONT)
    transfer_ax.margins(y=0.20)
    transfer_ax.spines[['top', 'right']].set_visible(False)
    transfer_ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=7,
                        handlelength=1.0, handletextpad=0.3, borderpad=0.1,
                        labelspacing=0.1)
    sweep_ax = fig.add_axes([sweep_left / width, 0.80 / height,
                             sweep_width / width, plot_height / height])
    with (SWEEP / 'tables/gamma_sweep.csv').open() as handle:
        sweep_points = list(csv.DictReader(handle))
    for condition, label, color in (('direct', 'ID', 'black'),
                                     ('human_jailbreak', 'OOD', '#8B1E1E')):
        selected = [row for row in sweep_points if row['condition'] == condition]
        sweep_ax.plot([float(row['rho']) for row in selected],
                       [float(row['asr_percent']) for row in selected],
                       '-o', color=color, label=label, linewidth=1.6, markersize=3, zorder=3)
    sweep_ax.set(xlim=(0., 1.025), ylim=(0, None))
    sweep_ax.set_ylabel('Attack success rate (%) ↓', fontsize=FONT)
    sweep_ax.set_xlabel(r'$\rho_{\mathrm{rob}}=\gamma^\star/\gamma$', fontsize=FONT)
    sweep_ax.set_xticks([0, .2, .4, .6, .8, 1.], ['0.00', '0.2', '0.4', '0.6', '0.8', '1.0'])
    sweep_ax.tick_params(labelsize=FONT)
    sweep_ax.get_xticklabels()[-1].set_color('#007C7C')
    sweep_ax.xaxis.set_label_coords(.5, -.20)
    sweep_ax.xaxis.label.set_verticalalignment('baseline')
    for endpoint, label, color in ((0., 'LQR', 'black'), (1., 'H∞', '#007C7C')):
        sweep_ax.text(endpoint, -.20, label, transform=sweep_ax.get_xaxis_transform(),
                       ha='center', va='baseline', fontsize=FONT, fontweight='bold', color=color)
    sweep_ax.spines[['top', 'right']].set_visible(False)
    sweep_ax.grid(axis='y', alpha=.18)
    sweep_ax.legend(loc='lower left', frameon=False, fontsize=7,
                     handlelength=1.0, handletextpad=.3, borderpad=.1, labelspacing=.1)
    # Reserve space for y-axis labels, with larger gaps between benchmarks.
    top_axes = fig.axes[:5]
    left = top_axes[0].get_position().x0
    right = top_axes[-1].get_position().x1
    gutters = np.array([0.78, 1.08, 0.78, 1.08]) / width
    panel_width = (right - left - gutters.sum()) / len(top_axes)
    for index, top_ax in enumerate(top_axes):
        position = top_ax.get_position()
        top_ax.set_position([left, position.y0, panel_width, position.height])
        if index < len(gutters):
            left += panel_width + gutters[index]
    for heading, (first, last) in zip(fig.texts, ((0, 1), (2, 3), (4, 4)), strict=True):
        group_left = top_axes[first].get_position().x0
        group_right = top_axes[last].get_position().x1
        heading.set_x((group_left + group_right) / 2)
        rule_y = heading.get_position()[1] - 0.14 / height
        fig.add_artist(Line2D([group_left, group_right], [rule_y, rule_y],
                              transform=fig.transFigure, color='#c9c9c9', linewidth=0.8))
    for label, heading, first in zip('abc', tuple(fig.texts), (0, 2, 4), strict=True):
        label_x = top_axes[first].get_position().x0 - 12 / 72 / width
        fig.text(label_x, heading.get_position()[1], label, ha='left', va='center',
                  fontsize=1.5 * FONT, fontweight='bold')
    for panel, label in ((ax, 'd'), (transfer_ax, 'e'), (sweep_ax, 'f')):
        panel.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(-12, 6),
                        textcoords='offset points', ha='left', va='bottom',
                        fontsize=1.5 * FONT, fontweight='bold', annotation_clip=False)
    # Extend only the bottom margin if D's angled labels need it; retain every axis's physical height.
    fig.canvas.draw()
    extra_bottom = max(0., .08 - fig.get_tightbbox(fig.canvas.get_renderer()).y0)
    if extra_bottom:
        positions = [(panel, panel.get_position().bounds) for panel in fig.axes]
        text_positions = [(text, text.get_position()) for text in fig.texts]
        new_height = height + extra_bottom
        fig.set_size_inches(width, new_height)
        for panel, (left, bottom, w, h) in positions:
            panel.set_position([left, (bottom * height + extra_bottom) / new_height, w, h * height / new_height])
        for text, (x_text, y_text) in text_positions:
            text.set_position((x_text, (y_text * height + extra_bottom) / new_height))
        for rule in fig.artists:
            if isinstance(rule, Line2D):
                rule.set_ydata((np.asarray(rule.get_ydata()) * height + extra_bottom) / new_height)
    PLOTS.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(PLOTS / f'fig_srob_extend.{suffix}', dpi=600, facecolor='white')
    plt.close(fig)
    print(PLOTS / 'fig_srob_extend.pdf')


if __name__ == '__main__':
    main()
