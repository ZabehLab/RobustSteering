"""ID/OOD ASR with separate overall and fluency views of the paper-judge scores."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .pipeline import ROOT, UNIT, read
from .score_store import ACTIVE_METHODS as METHODS, LABELS

TEAL, GRAY, RED = '#007C7C', '#c9c9c9', '#8B1E1E'
CONDITIONS = [('id', 'ID', GRAY), ('ood', 'OOD', RED)]
METHOD_LABELS = [LABELS[method] for method in METHODS]


def plot_metric(source, bottom_metric):
    lookup = {(r['method'], r['condition']): r for r in source}
    bottom_label = {'overall': 'AXBench overall ↑', 'fluency': 'Fluency ↑'}[bottom_metric]
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 13,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'axes.edgecolor': '#454545', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex='col', sharey=False)
        for column, (condition, label, color) in enumerate(CONDITIONS):
            selected = [lookup[method, condition] for method in METHODS]
            for row, (metric, ylabel) in enumerate([
                ('asr_pct', 'Attack success rate (%) ↓'),
                (bottom_metric, bottom_label),
            ]):
                ax = axes[row, column]
                colors = [TEAL if method == 'empirical' else color for method in METHODS]
                ax.bar(np.arange(len(METHODS)), [r[metric] for r in selected],
                       width=.40, color=colors, edgecolor='none', zorder=3)
                ax.set_xticks(np.arange(len(METHODS)), METHOD_LABELS)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
                ax.set_xlim(-.6, len(METHODS) - .4)
                ax.tick_params(axis='both', direction='out', length=3.5, width=.8, pad=5)
                ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
                ax.set_axisbelow(True)
                ax.set_ylim(0, max(r[metric] for r in selected) * 1.10)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, integer=row == 0))
                if column == 0:
                    ax.set_ylabel(ylabel)
                if row == 1:
                    ax.get_xticklabels()[METHODS.index('empirical')].set_color(TEAL)
                    ax.get_xticklabels()[METHODS.index('empirical')].set_fontweight('bold')
            axes[0, column].set_title(label, fontweight='bold', color='black' if column == 0 else RED, pad=12)
        fig.subplots_adjust(left=.10, right=.98, top=.92, bottom=.19, wspace=.16, hspace=.14)
        for extension in ('png', 'pdf'):
            fig.savefig(UNIT / f'geometry_performance_2x2_{bottom_metric}.{extension}', dpi=600)
        plt.close(fig)


def report():
    payload = read(ROOT / 'paper_axbench_n200/results.json')
    source = payload['results']
    assert payload['status'] == 'complete'
    assert payload['judge'] == 'gpt-4o-mini-2024-07-18'
    assert len(source) == 2 * len(METHODS)
    assert {(r['method'], r['condition']) for r in source} == {
        (method, condition) for method in METHODS for condition, _, _ in CONDITIONS}
    assert all(r['count'] == 200 for r in source)
    for metric in ('overall', 'fluency'):
        plot_metric(source, metric)
    print('Saved 200-case paper-judge overall and fluency variants.')


if __name__ == '__main__':
    report()
