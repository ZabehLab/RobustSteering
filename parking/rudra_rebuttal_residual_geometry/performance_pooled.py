"""Pool ID/OOD responses for ASR with paper-judge fluency or AXBench overall."""

import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .performance_grid import METHODS, METHOD_LABELS, TEAL, GRAY
from .pipeline import ROOT, UNIT, read, save, sha


def pooled_results():
    source = ROOT / 'paper_axbench_n200'
    metadata = read(source / 'results.json')
    assert metadata['status'] == 'complete'
    assert metadata['judge'] == 'gpt-4o-mini-2024-07-18'
    sample = read(ROOT / 'sample_n200.json')
    results, inputs = [], {}
    for method in METHODS:
        pooled = []
        for condition in ('id', 'ood'):
            path = source / 'points' / method / f'{condition}.json'
            rows = read(path)['rows']
            assert len(rows) == 200
            assert [r['prompt_id'] for r in rows] == [p[condition]['prompt_id'] for p in sample]
            assert all(r['asr_pct'] in (0, 100) and r['fluency'] in (0, 1, 2) for r in rows)
            inputs[str(path)] = sha(path)
            pooled.extend(rows)
        assert len({r['prompt_id'] for r in pooled}) == len(pooled) == 400
        attacks = sum(r['asr_pct'] == 100 for r in pooled)
        results.append({'method': method, 'count_id': 200, 'count_ood': 200,
                        'count_total': len(pooled), 'successful_attacks': attacks,
                        'asr_pct': 100 * attacks / len(pooled),
                        'fluency': sum(r['fluency'] for r in pooled) / len(pooled),
                        'overall': sum(r['overall'] for r in pooled) / len(pooled)})
    save(source / 'pooled_results.json', {
        'results': results, 'judge': metadata['judge'], 'input_sha256': inputs,
        'aggregation': 'Concatenate all 200 ID and 200 OOD response-level records per method; '
                       'ASR is the fraction classified successful; fluency and overall are means of '
                       'the response-level ratings and harmonic means, respectively; no exclusions',
    })
    with (source / 'pooled_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    return results


def plot_metric(results, right_metric):
    right_label = {'fluency': 'Fluency ↑', 'overall': 'AXBench overall ↑'}[right_metric]
    stem = 'geometry_performance_1x2_pooled' + ('_overall' if right_metric == 'overall' else '')
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'axes.edgecolor': '#454545', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(4.8, 3.8), sharey=False)
        for ax, (metric, label) in zip(axes, [('asr_pct', 'Attack success rate (%) ↓'),
                                            (right_metric, right_label)], strict=True):
            colors = [TEAL if method == 'empirical' else GRAY for method in METHODS]
            ax.bar(np.arange(len(METHODS)), [r[metric] for r in results],
                   width=.40, color=colors, edgecolor='none', zorder=3)
            ax.set_xticks(np.arange(len(METHODS)), METHOD_LABELS)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
            ax.set_xlim(-.6, len(METHODS) - .4)
            ax.tick_params(axis='both', direction='out', length=3.5, width=.8, pad=5)
            ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_ylim(0, max(r[metric] for r in results) * 1.10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, integer=metric == 'asr_pct'))
            ax.set_ylabel(label)
            tick = ax.get_xticklabels()[METHODS.index('empirical')]
            tick.set_color(TEAL)
            tick.set_fontweight('bold')
        fig.subplots_adjust(left=.17, right=.98, top=.94, bottom=.33, wspace=.55)
        for extension in ('png', 'pdf'):
            fig.savefig(UNIT / f'{stem}.{extension}', dpi=600)
        plt.close(fig)


def report():
    results = pooled_results()
    for metric in ('fluency', 'overall'):
        plot_metric(results, metric)
    print('Pooled 200 ID + 200 OOD response-level records per method:')
    for row in results:
        print(row['method'], f"ASR={row['asr_pct']:.2f}%, fluency={row['fluency']:.4f}, overall={row['overall']:.4f}")


if __name__ == '__main__':
    report()
