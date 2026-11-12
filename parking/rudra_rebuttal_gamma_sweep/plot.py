"""Plot only completed, officially scored operating points."""

import csv
from math import sqrt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robust_steerability.judges.specs import scorer_cache_path
from parking.rudra_rebuttal_mached_calib.inputs import read_json
from parking.rudra_rebuttal_mached_calib.workers import generation_path
from .sweep import CONDITIONS, COARSE, FINE, ROOT, SCORER, UNIT, evaluation_root, task_for


def collect_results():
    manifest = read_json(ROOT / 'inputs/manifest.json')
    records = read_json(ROOT / 'inputs/records.json')
    points = [('lqr', 0.)] + [(f'rho_{r:.1f}', r) for r in sorted(COARSE + FINE)]
    points += [('hinf', manifest['endpoint_rho'])]
    results = []
    for point, rho in points:
        values = []
        for condition in CONDITIONS:
            rows = []
            complete = True
            for shard in range(1 if point in ('lqr', 'hinf') else 2):
                generation = generation_path(ROOT, task_for(point, condition, shard))
                score = scorer_cache_path(evaluation_root(), generation, SCORER)
                if not score.exists() or read_json(score)['status'] != 'complete':
                    complete = False
                    break
                rows.extend(read_json(score)['rows'])
            if not complete:
                break
            expected = {r['prompt_id'] for r in records[condition]}
            if len(rows) != 100 or {r['prompt_id'] for r in rows} != expected:
                raise ValueError(f'Unmatched evaluation set: {point}/{condition}')
            if any(not r['valid'] or r['score'] not in (0, 1) for r in rows):
                raise ValueError(f'Invalid classifier result: {point}/{condition}')
            successes = sum(r['score'] for r in rows)
            proportion = successes / len(rows)
            values.append({'point': point, 'rho': rho,
                           'gamma': None if rho == 0 else manifest['gamma_star'] / rho,
                           'condition': condition, 'n': len(rows),
                           'successes': successes, 'asr_percent': 100. * successes / len(rows),
                           'asr_se_percent': 100. * sqrt(proportion * (1. - proportion) / (len(rows) - 1))})
        if len(values) == 2:
            results.extend(values)
    return results


def make_plot():
    results = collect_results()
    tables = UNIT / 'tables'
    plots = UNIT / 'plots'
    tables.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    with (tables / 'gamma_sweep.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(4.96, 4.96))
    ax.set_box_aspect(1)
    for condition, label, color in [('direct', 'ID · Direct', 'black'),
                                     ('human_jailbreak', 'OOD · Jailbreak', '#8B1E1E')]:
        subset = [r for r in results if r['condition'] == condition]
        ax.fill_between(
            [r['rho'] for r in subset],
            [max(0., r['asr_percent'] - r['asr_se_percent']) for r in subset],
            [min(100., r['asr_percent'] + r['asr_se_percent']) for r in subset],
            color=color, alpha=.12, linewidth=0,
        )
        ax.plot([r['rho'] for r in subset], [r['asr_percent'] for r in subset],
                '-o', color=color, label=label, linewidth=1.8, markersize=5, zorder=3)
    ax.set(xlabel=r'$\rho_{\mathrm{rob}}$',
           ylabel='Attack success rate (%) ↓', xlim=(-.025, 1.025), ylim=(0, None))
    ax.set_xticks([0, .2, .4, .6, .8, 1.],
                  labels=['0.00\nLQR', '0.2', '0.4', '0.6', '0.8', '1.0\nH∞'])
    ticks = ax.get_xticklabels()
    ticks[0].set_fontweight('bold')
    ticks[-1].set_fontweight('bold')
    ticks[-1].set_color('#007C7C')
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, loc='lower left')
    fig.subplots_adjust(left=.21, right=.97, top=.97, bottom=.21)
    fig.savefig(plots / 'gamma_sweep.png', dpi=600)
    fig.savefig(plots / 'gamma_sweep.pdf')
    plt.close(fig)
    print(f'Plot updated: {len(results) // 2} completed operating points.', flush=True)
    return results


if __name__ == '__main__':
    make_plot()
