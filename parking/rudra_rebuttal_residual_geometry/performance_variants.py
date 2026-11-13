"""Combined and simplified performance views, using saved summaries only."""

import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .performance_diagnostics import METRICS
from .pipeline import ROOT, UNIT, read
from .score_store import METHODS, LABELS

CONDITIONS = [('id', 'ID', '#62686e'), ('ood', 'OOD', '#921f25')]


def non_attack_overall(asr_pct, overall):
    """Product of the condition's non-attack rate and mean AXBench overall."""
    if not 0 <= asr_pct <= 100 or not 0 <= overall <= 2:
        raise ValueError('Expected ASR in percent and AXBench overall on the 0–2 scale')
    return (1. - asr_pct / 100.) * overall


def format_axis(ax):
    ax.set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS],
                  rotation=45, ha='right', rotation_mode='anchor')
    ax.grid(axis='y', alpha=.12)
    ax.set_axisbelow(True)


def export(fig, name):
    for extension in ('png', 'pdf'):
        fig.savefig(UNIT / f'{name}.{extension}', dpi=220)
    plt.close(fig)


def report():
    source = read(ROOT / 'diagnostics/performance_with_baselines.json')['results']
    lookup = {(r['method'], r['condition']): r for r in source}
    assert len(lookup) == 2 * len(METHODS)
    assert all(int(r['count']) == 50 for r in source)
    product = [{
        'method': row['method'], 'condition': row['condition'], 'count': row['count'],
        'asr_pct': row['asr_pct'], 'non_attack_rate': 1. - row['asr_pct'] / 100.,
        'axbench_overall': row['overall'],
        'non_attack_x_axbench_overall': non_attack_overall(row['asr_pct'], row['overall']),
        'axbench_score_source': row['axbench_score_source'],
        'asr_score_source': row['asr_score_source'],
    } for row in source]
    with (UNIT / 'geometry_performance_simplified.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(product[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(product)

    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 5, figsize=(19, 8.5), sharey='col')
        for row_index, (condition, label, color) in enumerate(CONDITIONS):
            selected = [lookup[method, condition] for method in METHODS]
            for ax, (metric, heading) in zip(axes[row_index], METRICS, strict=True):
                ax.bar(np.arange(len(METHODS)), [r[metric] for r in selected], width=.55,
                       color=color, alpha=.7, edgecolor='none', capsize=2,
                       yerr=[r[metric + '_se'] for r in selected],
                       error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': color})
                format_axis(ax)
                ax.set_title(heading, fontweight='bold')
                ax.set_ylim(bottom=0)
                if metric != 'asr_pct':
                    ax.set_ylim(0, 2.1)
                    ax.set_yticks([0, 1, 2])
            axes[row_index, 0].set_ylabel(f'{label}\nAttack success rate (%)')
            axes[row_index, 1].set_ylabel('Score (0–2)')
        fig.tight_layout(w_pad=2, h_pad=3)
        export(fig, 'geometry_performance_combined')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
        for ax, (condition, label, color) in zip(axes, CONDITIONS, strict=True):
            selected = [next(r for r in product if r['method'] == method
                             and r['condition'] == condition) for method in METHODS]
            ax.bar(np.arange(len(METHODS)), [r['non_attack_x_axbench_overall'] for r in selected],
                   width=.55, color=color, alpha=.7, edgecolor='none')
            format_axis(ax)
            ax.set_title(label, fontweight='bold')
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel('(1 − ASR) × AXBench overall')
        fig.tight_layout(w_pad=2)
        export(fig, 'geometry_performance_simplified')
    print('Saved combined 2×5 diagnostic and simplified ID/OOD product panels.', flush=True)
    for row in product:
        print(row['method'], row['condition'], f"product={row['non_attack_x_axbench_overall']:.5f}")


if __name__ == '__main__':
    report()
