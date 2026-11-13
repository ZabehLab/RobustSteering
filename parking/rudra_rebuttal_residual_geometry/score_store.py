"""One response-level interface for plotting, with explicit grading provenance."""

import hashlib

from robust_steerability.judges.specs import scorer_cache_path

from .geometry import VARIANTS
from .pipeline import ROOT, SOURCE, SCORER, read, frozen_json, sha, blocks, generation_path

BASELINES = ('original', 'lqr_8d')
METHODS = (*BASELINES, *VARIANTS)
LABELS = {
    'original': 'Original', 'lqr_8d': 'LQR (8D)',
    'empirical': 'H∞ (Empirical)', 'isotropic': 'H∞ (Isotropic)',
    'diagonal': 'H∞ (Diagonal)', 'rotated': 'H∞ (Rotated)',
}
COMPONENTS = ('concept_relevance', 'instruction_relevance', 'fluency')


class MissingAuditScores(ValueError):
    """Responses exist, but the requested grading source does not cover them."""


def prepare_baselines(count=50):
    """Copy only the selected cached baseline cases and judgments into this unit."""
    pairs = read(ROOT / f'sample_n{count:03d}.json')
    winners = read(SOURCE / 'calibration/mega/selection.json')['winners']
    assert winners['alqr_8d']['configuration'] == winners['h_infinity']['configuration']
    assert read(ROOT / 'protocol.json')['model'] == read(SOURCE / 'artifacts/manifest.json')['model']
    for method, source_method in [('original', 'original'), ('lqr_8d', 'alqr_8d')]:
        for condition, source_condition in [('id', 'direct'), ('ood', 'human_jailbreak')]:
            generations, scores = {}, {}
            for part in ('screening', 'holdout'):
                stage = SOURCE / 'evaluation/giga' / part
                for path in sorted((stage / 'generations' / source_condition / source_method).glob('*.json')):
                    payload = read(path)
                    scoring_path = scorer_cache_path(stage, path, SCORER)
                    scoring = read(scoring_path)
                    assert payload['status'] == scoring['status'] == 'complete'
                    generation_hash, scoring_hash = sha(path), sha(scoring_path)
                    for row in payload['repetitions'][0]['rows']:
                        assert row['prompt_id'] not in generations
                        generations[row['prompt_id']] = {
                            **row, 'reused_from': str(path), 'source_sha256': generation_hash}
                    for row in scoring['rows']:
                        assert row['valid']
                        scores[row['prompt_id']] = {
                            **row, 'reused_from': str(scoring_path), 'source_sha256': scoring_hash}
            selected = []
            for pair in pairs:
                record = pair[condition]
                row = generations[record['prompt_id']]
                assert row['text'] == record['text'] and row['behavior_id'] == record['behavior_id']
                selected.append({**row, 'harmbench_judgment': scores[row['prompt_id']]})
            frozen_json(ROOT / 'baselines' / method / f'{condition}_n{count:03d}.json', {
                'status': 'complete', 'model': read(ROOT / 'protocol.json')['model'],
                'configuration': None if method == 'original' else winners['alqr_8d']['configuration'],
                'rows': selected,
            })


def load_cases(method, condition, *, score_source='assistant_audit', count=50):
    """Return matched scores without substituting a different judge for missing data."""
    if method not in METHODS or condition not in ('id', 'ood'):
        raise ValueError('Unknown method or condition')
    ids = [pair[condition]['prompt_id'] for pair in read(ROOT / f'sample_n{count:03d}.json')]
    if method in BASELINES:
        path = ROOT / 'baselines' / method / f'{condition}_n{count:03d}.json'
        payload = read(path)
        responses = payload['rows']
        asr = {r['prompt_id']: r['harmbench_judgment']['score'] for r in responses}
    else:
        responses, asr = [], {}
        for index, _ in blocks(count, condition):
            path = generation_path(method, condition, index)
            responses.extend(read(path)['rows'])
            scoring = read(scorer_cache_path(ROOT, path, SCORER))
            assert scoring['status'] == 'complete' and all(r['valid'] for r in scoring['rows'])
            asr.update({r['prompt_id']: r['score'] for r in scoring['rows']})
    assert [r['prompt_id'] for r in responses] == ids
    if score_source == 'assistant_audit':
        audit = read(ROOT / 'manual_audit/scores.json')['rows']
        additional_path = ROOT / 'manual_audit/baseline_scores.json'
        if additional_path.exists():
            audit += read(additional_path)['rows']
        scores = {r['prompt_id']: r for r in audit if r['geometry'] == method and
                  r['condition'] == condition and r['prompt_id'] in ids}
        if set(scores) != set(ids):
            raise MissingAuditScores(f'{LABELS[method]} {condition}: {len(scores)}/{count} assistant audit scores; '
                                     'responses are cached, but no fallback to the old judge is allowed')
        for response in responses:
            assert scores[response['prompt_id']]['completion_sha256'] == hashlib.sha256(
                response['completion'].encode()).hexdigest(), 'Assistant score does not match this response'
    elif score_source == 'paper_judge':
        scores = {pid: {} for pid in ids}
        for component in COMPONENTS:
            key = 'axbench_' + component
            if method in BASELINES:
                source_method = 'alqr_8d' if method == 'lqr_8d' else 'original'
                source_condition = 'direct' if condition == 'id' else 'human_jailbreak'
                paths = [p for part in ('screening', 'holdout') for p in
                         (SOURCE / 'evaluation/giga' / part / 'scores' / key / source_condition / source_method).glob('*.json')]
            else:
                paths = [ROOT / 'diagnostics/scores' / key / method / f'{condition}_n{count:03d}.json']
            values = {}
            for path in paths:
                payload = read(path)
                assert payload['status'] == 'complete'
                values.update({r['prompt_id']: r['score'] for r in payload['rows']})
            for pid in ids:
                scores[pid][component] = values[pid]
    else:
        raise ValueError('Select assistant_audit or paper_judge explicitly')
    assert all(0 <= scores[pid][component] <= 2 for pid in ids for component in COMPONENTS)
    return [{**r, **{k: scores[r['prompt_id']][k] for k in COMPONENTS},
             'harmbench_attack_success': asr[r['prompt_id']], 'score_source': score_source}
            for r in responses]
