"""Separate ASR and AXBench overall in ID/OOD columns."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .performance_variants import CONDITIONS, export, format_axis
from .pipeline import ROOT, read
from .score_store import METHODS


def report():
    source = read(ROOT / 'diagnostics/performance_with_baselines.json')['results']
    lookup = {(r['method'], r['condition']): r for r in source}
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex='col', sharey='row')
        for column, (condition, label, color) in enumerate(CONDITIONS):
            selected = [lookup[method, condition] for method in METHODS]
            for row, (metric, ylabel) in enumerate([
                ('asr_pct', 'Attack success rate (%) ↓'),
                ('overall', 'AXBench overall ↑'),
            ]):
                ax = axes[row, column]
                ax.bar(np.arange(len(METHODS)), [r[metric] for r in selected],
                       yerr=[r[metric + '_se'] for r in selected], width=.55,
                       color=color, alpha=.7, edgecolor='none', capsize=2,
                       error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': color})
                format_axis(ax)
                ax.set_ylim(bottom=0)
                if row == 1:
                    ax.set_ylim(0, 2.1)
                if column == 0:
                    ax.set_ylabel(ylabel)
            axes[0, column].set_title(label, fontweight='bold')
        fig.tight_layout(w_pad=2, h_pad=2)
        export(fig, 'geometry_performance_2x2')
    print('Saved ID/OOD columns with separate ASR and AXBench overall rows.')


if __name__ == '__main__':
    report()
