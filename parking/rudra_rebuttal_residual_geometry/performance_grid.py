"""Separate ASR and AXBench overall in ID/OOD columns."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import ROOT, UNIT, read
from .score_store import METHODS, LABELS

TEAL, GRAY, RED = '#007C7C', '#c9c9c9', '#8B1E1E'
CONDITIONS = [('id', 'ID', GRAY), ('ood', 'OOD', RED)]
METHOD_LABELS = [LABELS[method] for method in METHODS]


def report():
    source = read(ROOT / 'diagnostics/performance_with_baselines.json')['results']
    lookup = {(r['method'], r['condition']): r for r in source}
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 13,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'axes.edgecolor': '#454545', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex='col', sharey='row')
        for column, (condition, label, color) in enumerate(CONDITIONS):
            selected = [lookup[method, condition] for method in METHODS]
            for row, (metric, ylabel) in enumerate([
                ('asr_pct', 'Attack success rate (%) ↓'),
                ('overall', 'AXBench overall ↑'),
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
                ax.set_ylim(0, 32 if row == 0 else 1.2)
                ax.set_yticks([0, 10, 20, 30] if row == 0 else [0, .4, .8, 1.2])
                if column == 0:
                    ax.set_ylabel(ylabel)
                if row == 1:
                    ax.get_xticklabels()[METHODS.index('empirical')].set_color(TEAL)
                    ax.get_xticklabels()[METHODS.index('empirical')].set_fontweight('bold')
            axes[0, column].set_title(label, fontweight='bold', color='black' if column == 0 else RED, pad=12)
        fig.subplots_adjust(left=.10, right=.98, top=.92, bottom=.19, wspace=.16, hspace=.14)
        for extension in ('png', 'pdf'):
            fig.savefig(UNIT / f'geometry_performance_2x2.{extension}', dpi=600)
        plt.close(fig)
    print('Saved ID/OOD columns with separate ASR and AXBench overall rows.')


if __name__ == '__main__':
    report()
