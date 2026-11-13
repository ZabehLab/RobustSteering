"""Separate five-panel diagnostic using the existing geometry responses."""

import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from robust_steerability.datasets.harmful import SAFE_CONCEPT
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.openai import score_generations
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec

from .geometry import VARIANTS
from .pipeline import ROOT, SOURCE, UNIT, SCORER, blocks, generation_path, read, save, frozen_json, sha
from .score_store import load_cases

DIAGNOSTIC = ROOT / 'diagnostics'
AX_SCORERS = ('axbench_concept_relevance', 'axbench_instruction_relevance', 'axbench_fluency')


def joined_responses(variant, condition, count):
    result = []
    for index, records in blocks(count, condition):
        path = generation_path(variant, condition, index)
        payload = read(path)
        assert payload['status'] == 'complete'
        assert [r['prompt_id'] for r in payload['rows']] == [r['prompt_id'] for r in records]
        result.extend(payload['rows'])
    return result


def diagnostic_path(variant, condition, count):
    return DIAGNOSTIC / 'generations' / variant / f'{condition}_n{count:03d}.json'


def prepare(count):
    paths = []
    for variant in VARIANTS:
        for condition in ('id', 'ood'):
            rows = [{**r, 'concept': SAFE_CONCEPT} for r in joined_responses(variant, condition, count)]
            path = diagnostic_path(variant, condition, count)
            frozen_json(path, {'status': 'complete', 'repetitions': [{'repetition': 0, 'rows': rows, 'sample_count': count}]})
            paths.append(path)
    for key in AX_SCORERS:
        lookup, metadata = {}, None
        for part in ('screening', 'holdout'):
            directory = SOURCE / 'evaluation/giga' / part / 'scores' / key
            for path in sorted(directory.glob('*/h_infinity/*.json')):
                payload = read(path)
                assert payload['status'] == 'complete'
                metadata = payload['scorer']
                assert metadata['model'] == scorer_spec(key).model_id
                source_hash = sha(path)
                for row in payload['rows']:
                    assert row['prompt_id'] not in lookup
                    lookup[row['prompt_id']] = {**row, 'reused_from': str(path), 'source_sha256': source_hash}
        for condition in ('id', 'ood'):
            path = diagnostic_path('empirical', condition, count)
            rows = read(path)['repetitions'][0]['rows']
            selected = [{**lookup[r['prompt_id']], 'item_index': i} for i, r in enumerate(rows)]
            frozen_json(scorer_cache_path(DIAGNOSTIC, path, key),
                        {'status': 'complete', 'scorer': metadata, 'rows': selected,
                         'generation_sha256': sha(path)})
    return paths


def report(count, score_source='paper_judge'):
    results = []
    for variant in VARIANTS:
        for condition in ('id', 'ood'):
            cases = load_cases(variant, condition, score_source=score_source, count=count)
            raw = {key: np.array([r[key.removeprefix('axbench_')] for r in cases], dtype=float)
                   for key in AX_SCORERS}
            raw['axbench_overall'] = np.array([harmonic_mean([raw[k][i] for k in AX_SCORERS]) for i in range(count)])
            raw['asr_pct'] = np.array([r['harmbench_attack_success'] for r in cases], dtype=float) * 100
            row = {'geometry': variant, 'condition': condition, 'count': count}
            for key, values in raw.items():
                row[key] = float(values.mean())
                row[key + '_se'] = float(values.std(ddof=1) / np.sqrt(count))
            results.append(row)
    suffix = '_assistant' if score_source == 'assistant_audit' else ''
    save(DIAGNOSTIC / f'results{suffix}_n{count:03d}.json', {
        'results': results, 'overall': 'Mean of per-response harmonic means of the three AXBench scores',
        'score_source': score_source,
        'uncertainty': 'Standard error over the paired held-out behaviors',
    })
    with (UNIT / f'geometry_diagnostics{suffix}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metrics = [('asr_pct', 'ASR ↓'), ('axbench_concept_relevance', 'Safe-concept relevance ↑'),
               ('axbench_instruction_relevance', 'Instruction relevance ↑'),
               ('axbench_fluency', 'Fluency ↑'), ('axbench_overall', 'AXBench overall ↑')]
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8))
        for ax, (metric, title) in zip(axes, metrics):
            for condition, offset, color in [('id', -.18, '#62686e'), ('ood', .18, '#921f25')]:
                selected = [next(r for r in results if r['geometry'] == v and r['condition'] == condition) for v in VARIANTS]
                ax.bar(np.arange(4) + offset, [r[metric] for r in selected], width=.32,
                       color=color, alpha=.7, edgecolor='none',
                       yerr=[r[metric + '_se'] for r in selected], capsize=2,
                       error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': color})
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(np.arange(4), ['Empirical', 'Isotropic', 'Diagonal', 'Rotated'],
                          rotation=45, ha='right', rotation_mode='anchor')
            ax.set_ylim(bottom=0)
            if metric != 'asr_pct':
                ax.set_ylim(0, 2.1)
                ax.set_yticks([0, 1, 2])
            ax.grid(axis='y', alpha=.12)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Attack success rate (%)')
        axes[1].set_ylabel('AXBench score (0–2)')
        fig.legend(handles=[Patch(facecolor='#62686e', alpha=.7, label='ID: direct'),
                            Patch(facecolor='#921f25', alpha=.7, label='OOD: pooled jailbreaks')],
                   loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .0))
        fig.tight_layout(rect=(0, .08, 1, 1), w_pad=2)
        for extension in ('png', 'pdf'):
            fig.savefig(UNIT / f'geometry_diagnostics{suffix}.{extension}', dpi=220)
        plt.close(fig)
    for row in results:
        print(row['geometry'], row['condition'], ' | '.join(f'{key}={row[key]:.3f}' for key, _ in metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'plot'))
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--scores', choices=('assistant_audit', 'paper_judge'), default='paper_judge')
    args = parser.parse_args()
    if args.command == 'run':
        paths = prepare(args.count)
        print('Reused empirical AXBench scores; scoring only the 300 new saved responses.', flush=True)
        score_generations(paths, DIAGNOSTIC, list(AX_SCORERS), concurrency=12, batch_size=20)
    report(args.count, args.scores)
