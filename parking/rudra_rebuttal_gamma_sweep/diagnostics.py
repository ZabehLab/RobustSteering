"""AXBench diagnostics from cached sweep responses; never redraw the main figure."""

from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robust_steerability.benchmarks.harmful_calibration import API_SCORERS
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.openai import EVALUATION_PREAMBLE, score_generations
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec
from parking.rudra_rebuttal_mached_calib.inputs import read_json, write_json
from parking.rudra_rebuttal_mached_calib.workers import generation_path
from .sweep import (
    COARSE, FINE, CONDITIONS, ROOT, SOURCE, UNIT, file_hash, freeze_json, task_for,
)

DIAGNOSTICS = ROOT / 'diagnostics'
METRICS = (*API_SCORERS, 'axbench_overall')
FROZEN_FILES = ('plot.py', 'plots/gamma_sweep.png', 'plots/gamma_sweep.pdf',
                'tables/gamma_sweep.csv')


def response_key(row: dict) -> str:
    # These are the complete judge inputs for the three existing paper rubrics.
    data = {key: row[key] for key in ('text', 'completion', 'concept')}
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_hashes() -> dict:
    return {name: file_hash(UNIT / name) for name in FROZEN_FILES}


def paper_scores() -> tuple[dict, dict]:
    """Index exact judge inputs, retaining each endpoint's own paper ratings."""
    by_content, by_arm = {}, {}
    for path in sorted((SOURCE / 'evaluation/giga').glob('*/generations/*/*/*.json')):
        root = path.parents[3]
        generation = read_json(path)
        if generation['status'] != 'complete':
            raise ValueError(f'Incomplete Giga generations: {path}')
        scores, sources = {}, {}
        for metric in API_SCORERS:
            destination = scorer_cache_path(root, path, metric)
            payload = read_json(destination)
            spec = scorer_spec(metric)
            if (payload['status'] != 'complete' or payload['scorer']['model'] != spec.model_id
                    or payload['scorer']['rubric'] != spec.rubric
                    or payload['scorer']['evaluation_preamble'] != EVALUATION_PREAMBLE):
                raise ValueError(f'Paper judge specification mismatch: {destination}')
            scores[metric] = {r['prompt_id']: r for r in payload['rows']}
            sources[metric] = str(destination)
        for row in generation['repetitions'][0]['rows']:
            key = response_key(row)
            bundle = {'response_key': key, 'scores': {
                metric: scores[metric][row['prompt_id']] for metric in API_SCORERS},
                'sources': sources}
            by_content.setdefault(key, bundle)
            arm = (path.parent.name, row['prompt_id'])
            if arm in by_arm:
                raise ValueError(f'Duplicate Giga arm/prompt: {arm}')
            by_arm[arm] = bundle
    return by_content, by_arm


def sweep_responses() -> list:
    manifest = read_json(ROOT / 'inputs/manifest.json')
    records = read_json(ROOT / 'inputs/records.json')
    points = [('lqr', 0.)] + [(f'rho_{r:.1f}', r) for r in sorted(COARSE + FINE)]
    points.append(('hinf', manifest['endpoint_rho']))
    groups = []
    for point, rho in points:
        for condition in CONDITIONS:
            rows = []
            for shard in range(1 if point in ('lqr', 'hinf') else 2):
                path = generation_path(ROOT, task_for(point, condition, shard))
                payload = read_json(path)
                if payload['status'] != 'complete':
                    raise ValueError(f'Incomplete sweep generations: {path}')
                rows.extend(payload['repetitions'][0]['rows'])
            if (len(rows) != 100 or {r['prompt_id'] for r in rows}
                    != {r['prompt_id'] for r in records[condition]}):
                raise ValueError(f'Unmatched diagnostic sample: {point}/{condition}')
            groups.append((point, rho, condition, sorted(rows, key=lambda r: r['prompt_id'])))
    return groups


def select_cached(row: dict, point: str, by_content: dict, by_arm: dict):
    key = response_key(row)
    if point in ('lqr', 'hinf'):
        method = 'alqr_8d' if point == 'lqr' else 'h_infinity'
        bundle = by_arm[method, row['prompt_id']]
        if bundle['response_key'] != key:
            raise ValueError('Endpoint response differs from Giga')
        return bundle
    return by_content.get(key)


def summarize(rows: list[dict]) -> dict:
    values = {metric: mean(row[metric] for row in rows) for metric in API_SCORERS}
    # As in the paper: harmonic mean per response, then average over behaviors.
    values['axbench_overall'] = mean(
        harmonic_mean([row[metric] for metric in API_SCORERS]) for row in rows)
    return values


def make_plot(results: list[dict]) -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    titles = ['Concept relevance (safe refusal)', 'Instruction relevance', 'Fluency', 'AXBench overall']
    for ax, metric, title in zip(axes.flat, METRICS, titles, strict=True):
        for condition, label, color in [('direct', 'ID · Direct', 'black'),
                                       ('human_jailbreak', 'OOD · Jailbreak', '#8B1E1E')]:
            data = [r for r in results if r['condition'] == condition]
            ax.plot([r['rho'] for r in data], [r[metric] for r in data], '-o',
                    color=color, label=label, linewidth=1.5, markersize=4)
        ax.set(title=title, xlim=(0, 1.025), ylim=(0, 2.05))
        ax.set_xticks([0, .2, .4, .6, .8, 1.], labels=['0.00', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.tick_params(labelbottom=True)
        ax.get_xticklabels()[-1].set_color('#007C7C')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlabel(r'$\rho_{\mathrm{rob}}$')
        ax.xaxis.set_label_coords(.5, -.24)
        ax.xaxis.label.set_verticalalignment('baseline')
        for x, label, color in [(0., 'LQR', 'black'), (1., 'H∞', '#007C7C')]:
            ax.text(x, -.24, label, transform=ax.get_xaxis_transform(), ha='center',
                    va='baseline', fontweight='bold', color=color)
        ax.set_ylabel('Mean score (0–2)')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(UNIT / 'plots/gamma_sweep_diagnostics.png', dpi=300)
    fig.savefig(UNIT / 'plots/gamma_sweep_diagnostics.pdf')
    plt.close(fig)


def run() -> None:
    snapshot = frozen_hashes()
    try:
        by_content, by_arm = paper_scores()
        groups = sweep_responses()
        pending, reuse_counts = {}, Counter()
        for point, _, _, rows in groups:
            for row in rows:
                if select_cached(row, point, by_content, by_arm) is None:
                    key = response_key(row)
                    pending.setdefault(key, {**row, 'prompt_id': f'diagnostic-{key}'})
                    reuse_counts['new_response_slots'] += 1
                else:
                    reuse_counts['reused_response_slots'] += 1
        ordered = [pending[k] for k in sorted(pending)]
        path = DIAGNOSTICS / 'generations/unique_responses.json'
        freeze_json(path, {'status': 'complete', 'repetitions': [{'repetition': 0,
                           'sample_count': len(ordered), 'rows': ordered}]})
        write_json(DIAGNOSTICS / 'manifest.json', {
            'frozen_figure_hashes': snapshot, 'unique_new_responses': len(ordered),
            'reuse_counts': dict(reuse_counts), 'scorers': list(API_SCORERS),
            'batch_size': 20, 'concurrency': 12,
            'cache_matching': 'Exact instruction, completion, and concept; endpoints retain their own Giga scores',
        })
        print(f'Diagnostics: {dict(reuse_counts)}; scoring {len(ordered)} unique new responses.', flush=True)
        if ordered:
            score_generations([path], DIAGNOSTICS, list(API_SCORERS), concurrency=12, batch_size=20)
        new_scores = {}
        for metric in API_SCORERS:
            destination = scorer_cache_path(DIAGNOSTICS, path, metric)
            payload = read_json(destination)
            if payload['status'] != 'complete' or len(payload['rows']) != len(ordered):
                raise ValueError(f'Incomplete diagnostic scores: {destination}')
            new_scores[metric] = {row['prompt_id']: row for row in payload['rows']}
        results = []
        for point, rho, condition, rows in groups:
            scored = []
            for row in rows:
                bundle = select_cached(row, point, by_content, by_arm)
                key = response_key(row)
                if bundle is None:
                    bundle = {'sources': {metric: str(scorer_cache_path(DIAGNOSTICS, path, metric))
                                          for metric in API_SCORERS},
                              'scores': {metric: new_scores[metric][f'diagnostic-{key}']
                                         for metric in API_SCORERS}}
                values = {metric: float(bundle['scores'][metric]['score']) for metric in API_SCORERS}
                if any(v not in (0., 1., 2.) for v in values.values()):
                    raise ValueError('Invalid AXBench component rating')
                scored.append({'prompt_id': row['prompt_id'], 'response_key': key,
                               **values, 'rating_sources': bundle['sources']})
            write_json(DIAGNOSTICS / f'points/{point}/{condition}.json', {'status': 'complete', 'rows': scored})
            results.append({'point': point, 'rho': rho, 'condition': condition, 'n': len(scored),
                            **summarize(scored)})
        with (UNIT / 'tables/gamma_sweep_diagnostics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(results)
        make_plot(results)
        print('Diagnostic plot and table saved; frozen figure unchanged.', flush=True)
    finally:
        if frozen_hashes() != snapshot:
            raise RuntimeError('Frozen figure files changed during diagnostics')


if __name__ == '__main__':
    run()
