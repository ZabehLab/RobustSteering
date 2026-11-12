"""Plot only completed, officially scored operating points."""

import csv
from pathlib import Path

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
            values.append({'point': point, 'rho': rho,
                           'gamma': None if rho == 0 else manifest['gamma_star'] / rho,
                           'condition': condition, 'n': len(rows),
                           'successes': successes, 'asr_percent': 100. * successes / len(rows)})
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
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for condition, label, color in [('direct', 'ID · Direct', 'black'),
                                     ('human_jailbreak', 'OOD · Jailbreak', '#8B1E1E')]:
        subset = [r for r in results if r['condition'] == condition]
        ax.plot([r['rho'] for r in subset], [r['asr_percent'] for r in subset],
                '-o', color=color, label=label, linewidth=1.8, markersize=5)
    ax.set(xlabel=r'Robustness level $\rho_{\mathrm{rob}}=\gamma^\star/\gamma$',
           ylabel='Attack success rate (%)', xlim=(-.03, 1.03), ylim=(0, None))
    ax.set_xticks([0, .2, .4, .6, .8, 1.])
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False)
    ax.set_title('Qwen2.5-0.5B-Instruct · HarmBench', fontsize=12, fontweight='bold', pad=12)
    ax.annotate('LQR (8D)', xy=(0, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -51), textcoords='offset points', ha='center', fontsize=10)
    ax.annotate('H∞', xy=(results[-1]['rho'], 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -51), textcoords='offset points', ha='center', fontsize=10)
    fig.subplots_adjust(left=.14, right=.97, top=.88, bottom=.27)
    fig.savefig(plots / 'gamma_sweep.png', dpi=600)
    fig.savefig(plots / 'gamma_sweep.pdf')
    plt.close(fig)
    print(f'Plot updated: {len(results) // 2} completed operating points.', flush=True)
    return results


if __name__ == '__main__':
    make_plot()
