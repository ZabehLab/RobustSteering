"""Score saved geometry responses with the pinned paper AXBench judge only."""

import argparse
import csv
from math import sqrt
from statistics import mean, stdev

from robust_steerability.datasets.harmful import SAFE_CONCEPT
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.openai import EVALUATION_PREAMBLE, score_generations
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec

from .pipeline import ROOT, SOURCE, read, save, frozen_json, sha
from .score_store import ACTIVE_METHODS, COMPONENTS, load_responses

SCORERS = tuple('axbench_' + c for c in COMPONENTS)


def checked_scores(root, generation, scorer):
    path = scorer_cache_path(root, generation, scorer)
    payload = read(path)
    spec = scorer_spec(scorer)
    assert payload['status'] == 'complete'
    assert payload['scorer']['model'] == spec.model_id
    assert payload['scorer']['rubric'] == spec.rubric
    assert payload['scorer']['evaluation_preamble'] == EVALUATION_PREAMBLE
    values = {r['prompt_id']: r for r in payload['rows']}
    assert len(values) == len(payload['rows'])
    assert all(r['score'] in (0, 1, 2) for r in values.values())
    return values, {'path': str(path), 'sha256': sha(path)}


def cached_scores():
    """Index each method's own cached paper judgments with exact source inputs."""
    indexed = {}
    sources = []
    giga = SOURCE / 'evaluation/giga'
    methods = {'original': 'original', 'alqr_8d': 'lqr_8d', 'h_infinity': 'empirical'}
    for path in sorted(giga.glob('*/generations/*/*/*.json')):
        if path.parent.name in methods:
            sources.append((methods[path.parent.name], path, path.parents[3]))
    for method in ('isotropic', 'rotated'):
        for path in sorted((ROOT / 'diagnostics/generations' / method).glob('*_n050.json')):
            sources.append((method, path, ROOT / 'diagnostics'))
    for method, generation, root in sources:
        payload = read(generation)
        assert payload['status'] == 'complete'
        ratings = [checked_scores(root, generation, scorer) for scorer in SCORERS]
        for repetition in payload['repetitions']:
            for row in repetition['rows']:
                key = method, row['prompt_id']
                assert key not in indexed
                indexed[key] = {
                    'text': row['text'], 'completion': row['completion'],
                    'values': {c: r[0][row['prompt_id']]['score']
                               for c, r in zip(COMPONENTS, ratings, strict=True)},
                    'sources': [r[1] for r in ratings],
                }
    return indexed


def run(count, *, score=True):
    destination = ROOT / f'paper_axbench_n{count:03d}'
    cached = cached_scores()
    groups = {(m, c): load_responses(m, c, count)
              for m in ACTIVE_METHODS for c in ('id', 'ood')}
    paths, references = [], {}
    for (method, condition), rows in groups.items():
        missing = []
        for row in rows:
            key = method, row['prompt_id']
            if key in cached:
                old = cached[key]
                assert (old['text'], old['completion']) == (row['text'], row['completion'])
                references[key] = old
            else:
                missing.append({**row, 'concept': SAFE_CONCEPT})
        if missing:
            path = destination / 'generations' / method / f'{condition}.json'
            frozen_json(path, {'status': 'complete', 'repetitions': [
                {'repetition': 0, 'sample_count': len(missing), 'rows': missing}]})
            paths.append((method, path))
    frozen_json(destination / 'inputs.json', {
        'count_per_condition': count, 'methods': list(ACTIVE_METHODS),
        'reused_response_slots': len(references),
        'new_response_slots': sum(read(p)['repetitions'][0]['sample_count'] for _, p in paths),
        'judge': scorer_spec(SCORERS[0]).model_id, 'scorers': list(SCORERS),
        'generation_sha256': {str(p): sha(p) for _, p in paths},
        'grading': 'Pinned paper AXBench judge for every method; no assistant ratings',
    })
    print(f'Reusing {len(references)} cached paper-rated responses; '
          f'scoring {sum(read(p)["repetitions"][0]["sample_count"] for _, p in paths)} saved responses.', flush=True)
    if paths and score:
        score_generations([p for _, p in paths], destination, list(SCORERS), concurrency=12, batch_size=20)
    for method, path in paths:
        ratings = [checked_scores(destination, path, scorer) for scorer in SCORERS]
        for row in read(path)['repetitions'][0]['rows']:
            references[method, row['prompt_id']] = {
                'values': {c: r[0][row['prompt_id']]['score']
                           for c, r in zip(COMPONENTS, ratings, strict=True)},
                'sources': [r[1] for r in ratings],
            }
    results = []
    for (method, condition), rows in groups.items():
        points = []
        for row in rows:
            record = references[method, row['prompt_id']]
            values = record['values']
            points.append({'prompt_id': row['prompt_id'], **values,
                           'overall': harmonic_mean([values[c] for c in COMPONENTS]),
                           'asr_pct': 100 * row['harmbench_attack_success'],
                           'sources': record['sources']})
        save(destination / 'points' / method / f'{condition}.json', {'rows': points})
        item = {'method': method, 'condition': condition, 'count': count}
        for metric in (*COMPONENTS, 'overall', 'asr_pct'):
            values = [p[metric] for p in points]
            item[metric] = mean(values)
            item[metric + '_se'] = stdev(values) / sqrt(count)
        results.append(item)
    save(destination / 'results.json', {
        'status': 'complete', 'results': results,
        'judge': scorer_spec(SCORERS[0]).model_id,
        'overall': 'Mean of per-response harmonic means; zero if any component is zero',
    })
    with (destination / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(row['method'], row['condition'], ' '.join(f'{m}={row[m]:.4f}' for m in (*COMPONENTS, 'overall')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    run(args.count, score=not args.report_only)
