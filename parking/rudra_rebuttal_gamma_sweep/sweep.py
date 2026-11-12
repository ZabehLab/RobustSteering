"""Frozen-Qwen gamma sweep; coarse points first, optional fine points later."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import fcntl
from hashlib import sha256
from pathlib import Path
import random
import sys

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks.execution import default_run_id, utc_now
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.control import HInfinityController, LQRController
from robust_steerability.judges.harmbench import score_generations
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.source_methods.control import SetpointCalibration

from parking.rudra_rebuttal_mached_calib.inputs import (
    MODEL_SPECS, REPO_ROOT, load_model, load_tensors, read_json, write_json, write_torch,
)
from parking.rudra_rebuttal_mached_calib.workers import (
    candidate_policy, candidate_problem, generation_path, run_generation_task,
)

UNIT = Path(__file__).resolve().parent
MODEL = 'qwen25_05b_instruct'
SOURCE = UNIT.parent / 'rudra_rebuttal_mached_calib' / 'cache' / MODEL
ROOT = UNIT / 'cache' / MODEL
COARSE = (.2, .4, .6, .8)
FINE = (.1, .3, .5, .7, .9)
CONDITIONS = ('direct', 'human_jailbreak')
SCORER = 'harmbench_test_success'
SEED = 2151
BATCH_SIZE = 16
MODULE = 'parking.rudra_rebuttal_gamma_sweep.sweep'


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def freeze_json(path: Path, payload: dict) -> None:
    if path.exists() and read_json(path) != payload:
        raise ValueError(f'Frozen inputs changed: {path}')
    if not path.exists():
        write_json(path, payload)


def select_records(source: dict) -> dict:
    """Sample without looking at responses; one balanced template per behavior."""
    evaluation = source['evaluation']
    direct = sorted(evaluation['direct'], key=lambda row: row['behavior_id'])
    rng = random.Random(SEED)
    direct = rng.sample(direct, 100)
    templates = list(range(5)) * 20
    rng.shuffle(templates)
    lookup = {(r['behavior_id'], r['template_index']): r
              for r in evaluation['human_jailbreak']}
    ood = [lookup[row['behavior_id'], t] for row, t in zip(direct, templates, strict=True)]
    assert len({row['behavior_id'] for row in direct}) == 100
    assert Counter(r['template_index'] for r in ood) == Counter({i: 20 for i in range(5)})
    return {'direct': direct, 'human_jailbreak': ood}


def evaluation_root() -> Path:
    return ROOT / 'evaluation/gamma_sweep/sweep'


def task_for(point: str, condition: str, shard: int) -> dict:
    return {'stage': 'sweep', 'profile': 'gamma_sweep', 'method': point,
            'condition': condition, 'shard': shard, 'configuration': None}


def prepare() -> dict:
    fit_path = SOURCE / 'artifacts/reduced.pt'
    lqr_path = SOURCE / 'calibration/controllers/alqr_8d/q_00_qf_00.pt'
    hinf_path = SOURCE / 'calibration/controllers/h_infinity/q_00_qf_00.pt'
    fit = load_tensors(fit_path)
    lqr, hinf = load_tensors(lqr_path), load_tensors(hinf_path)
    config = hinf['identity']['configuration']
    if config != lqr['identity']['configuration']:
        raise ValueError('The Giga endpoints do not share cost settings')
    if lqr['identity']['fit'] != fit['fingerprint'] or hinf['identity']['fit'] != fit['fingerprint']:
        raise ValueError('Endpoint fit mismatch')
    spec = MODEL_SPECS[MODEL]
    if hinf['identity']['model'] != [spec.model_id, spec.revision]:
        raise ValueError('Endpoint model mismatch')
    records = select_records(read_json(SOURCE / 'artifacts/harmbench.json'))
    freeze_json(ROOT / 'inputs/records.json', records)
    gamma_star = hinf['gamma_star']
    gamma_used = hinf['diagnostics']['gamma_used']
    manifest = {
        'model': hinf['identity']['model'], 'fit': fit['fingerprint'], 'configuration': config,
        'gamma_star': gamma_star, 'endpoint_gamma': gamma_used,
        'endpoint_rho': gamma_star / gamma_used, 'seed': SEED,
        'prompts_per_condition': 100, 'ood_per_template': 20,
        'records_hash': configuration_hash(records), 'batch_size': BATCH_SIZE,
        'max_new_tokens': 512, 'use_cache': False, 'do_sample': False,
        'sources': {str(p.relative_to(REPO_ROOT)): file_hash(p) for p in [
            fit_path, lqr_path, hinf_path, SOURCE / 'artifacts/setpoint.pt',
            SOURCE / 'artifacts/harmbench.json']},
    }
    freeze_json(ROOT / 'inputs/manifest.json', manifest)
    problem = candidate_problem(fit, config)
    checks = {}
    for device in ('cuda:0', 'cuda:1'):
        nominal = LQRController.synthesize(replace(problem, disturbance_channels=None), device=device)
        fixed = HInfinityController.synthesize_at_gamma(
            problem, gamma_used, gamma_star=gamma_star, device=device)
        torch.testing.assert_close(nominal.gains.cpu(), lqr['gains'], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(fixed.gains.cpu(), hinf['gains'], rtol=2e-5, atol=2e-6)
        large = HInfinityController.synthesize_at_gamma(problem, gamma_star * 1e4, device=device)
        torch.testing.assert_close(large.gains.cpu(), lqr['gains'], rtol=2e-5, atol=2e-6)
        below = HInfinityController.synthesize_at_gamma(problem, gamma_star * .5, device=device)
        if below.feasible:
            raise ValueError('Below-boundary controller unexpectedly feasible')
        checks[device] = {
            'endpoint_max_gain_difference': float((fixed.gains.cpu() - hinf['gains']).abs().max()),
            'nominal_max_gain_difference': float((nominal.gains.cpu() - lqr['gains']).abs().max()),
            'large_gamma_max_gain_difference': float((large.gains.cpu() - lqr['gains']).abs().max()),
            'below_boundary_infeasible': True,
        }
    write_json(ROOT / 'checks.json', checks)
    for point, candidate in [('lqr', lqr), ('hinf', hinf)]:
        destination = ROOT / 'controllers' / f'{point}.pt'
        if not destination.exists():
            write_torch(destination, candidate)
        reuse_endpoint(point, candidate['identity']['method'], records)
    for rho in sorted(COARSE + FINE):
        point = f'rho_{rho:.1f}'
        path = ROOT / 'controllers' / f'{point}.pt'
        if path.exists():
            cached = load_tensors(path)
            if cached['rho'] != rho or cached['gamma_star'] != gamma_star:
                raise ValueError(f'Controller cache mismatch: {path}')
            continue
        controller = HInfinityController.synthesize_at_gamma(
            problem, gamma_star / rho, gamma_star=gamma_star, device='cuda:0')
        if not controller.feasible:
            raise ValueError(f'Infeasible sweep point: rho={rho}')
        write_torch(path, {'status': 'complete', 'rho': rho, 'gamma_star': gamma_star,
                          'gamma': gamma_star / rho, 'gains': controller.gains.cpu(),
                          'diagnostics': controller.diagnostics, 'manifest': manifest})
    print('Checks passed on both GPUs; Giga endpoint gains reproduced.', flush=True)
    return manifest


def reuse_endpoint(point: str, method: str, records: dict) -> None:
    for condition in CONDITIONS:
        generations, scores, sources = {}, {}, []
        scorer = None
        for part in ('screening', 'holdout'):
            for shard in range(2):
                base = SOURCE / 'evaluation/giga' / part
                gen = base / f'generations/{condition}/{method}/shard-{shard}.json'
                score = scorer_cache_path(base, gen, SCORER)
                gp, sp = read_json(gen), read_json(score)
                if gp['status'] != 'complete' or sp['status'] != 'complete':
                    raise ValueError(f'Giga endpoint incomplete: {gen}')
                if gp['identity']['method'] != method:
                    raise ValueError('Endpoint generation method mismatch')
                for row in gp['repetitions'][0]['rows']:
                    if row['prompt_id'] in generations:
                        raise ValueError('Duplicated endpoint generation')
                    generations[row['prompt_id']] = row
                for row in sp['rows']:
                    if row['prompt_id'] in scores:
                        raise ValueError('Duplicated endpoint score')
                    scores[row['prompt_id']] = row
                sources.extend({'path': str(p.relative_to(REPO_ROOT)), 'sha256': file_hash(p)}
                               for p in (gen, score))
                scorer = sp['scorer']
        selected = records[condition]
        for row in selected:
            if not scores[row['prompt_id']]['valid']:
                raise ValueError('Invalid endpoint classifier response')
            if generations[row['prompt_id']]['text'] != row['text']:
                raise ValueError('Endpoint prompt content mismatch')
        gen_path = generation_path(ROOT, task_for(point, condition, 0))
        freeze_json(gen_path, {
            'status': 'complete', 'reuse_sources': sources,
            'repetitions': [{'repetition': 0, 'sample_count': 100,
                             'rows': [generations[r['prompt_id']] for r in selected]}],
        })
        freeze_json(scorer_cache_path(evaluation_root(), gen_path, SCORER), {
            'status': 'complete', 'reuse_sources': sources, 'scorer': scorer,
            'rows': [scores[r['prompt_id']] for r in selected],
        })


def worker(work: Path, device: str) -> None:
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    task = read_json(work)
    if task['kind'] == 'classify':
        paths = [Path(p) for p in task['paths']]
        score_generations(paths, evaluation_root(), SCORER, device,
                          load_access_token(REPO_ROOT), batch_size=1)
        return
    manifest = read_json(ROOT / 'inputs/manifest.json')
    records = read_json(ROOT / 'inputs/records.json')
    fit = load_tensors(SOURCE / 'artifacts/reduced.pt')
    setpoint = SetpointCalibration(**load_tensors(SOURCE / 'artifacts/setpoint.pt'))
    model, tokenizer = load_model(MODEL_SPECS[MODEL], device)
    if model.generation_config.eos_token_id is not None:
        model.config.eos_token_id = model.generation_config.eos_token_id
    for item in task['items']:
        point, condition, shard = item['point'], item['condition'], item['shard']
        candidate = load_tensors(ROOT / 'controllers' / f'{point}.pt')
        if candidate['status'] != 'complete':
            raise ValueError('Refusing to deploy incomplete controller')
        policy = candidate_policy('h_infinity', fit, candidate, setpoint, record=False)
        selected = sorted(records[condition], key=lambda r: (len(r['text']), r['prompt_id']))[shard::2]
        identity = {'manifest': configuration_hash(manifest), 'rho': candidate['rho'],
                    'gamma': candidate['gamma'], 'controller_sha256': file_hash(ROOT / 'controllers' / f'{point}.pt'),
                    'prompt_hash': configuration_hash(selected), 'shard': shard,
                    'batch_size': BATCH_SIZE, 'use_cache': False, 'max_new_tokens': 512}
        run_generation_task(ROOT, task_for(point, condition, shard), selected, identity,
                            model, tokenizer, policy, BATCH_SIZE)


def launch(kind: str, items: list, logs: Path) -> None:
    jobs = []
    for i in range(2):
        work = logs / f'{kind}-{i}.json'
        payload = {'kind': kind}
        payload['items' if kind == 'generate' else 'paths'] = items[i::2]
        if not payload.get('items', payload.get('paths')):
            continue
        write_json(work, payload)
        jobs.append((f'{kind}-{i}', [sys.executable, '-m', MODULE, 'worker',
                                    str(work), '--device', '{device}']))
    run_jobs(jobs, ['cuda:0', 'cuda:1'], logs)


def run(stage: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        logs = ROOT / 'logs' / default_run_id('gamma-sweep', stage)
        record = {'stage': stage, 'status': 'running', 'started_at_utc': utc_now(),
                  'devices': ['cuda:0', 'cuda:1']}
        write_json(logs / 'run.json', record)
        try:
            prepare()
            from .plot import make_plot
            make_plot()
            rhos = COARSE if stage == 'coarse' else FINE
            items, paths = [], []
            for rho in rhos:
                point = f'rho_{rho:.1f}'
                for condition in CONDITIONS:
                    for shard in range(2):
                        path = generation_path(ROOT, task_for(point, condition, shard))
                        paths.append(path)
                        if not path.exists() or read_json(path)['status'] != 'complete':
                            items.append({'point': point, 'condition': condition, 'shard': shard})
            print(f'{stage}: {len(items)} missing generation shards; endpoints reused.', flush=True)
            launch('generate', items, logs)
            pending = [str(p) for p in paths if not (
                scorer_cache_path(evaluation_root(), p, SCORER).exists()
                and read_json(scorer_cache_path(evaluation_root(), p, SCORER))['status'] == 'complete')]
            launch('classify', pending, logs)
            make_plot()
            record.update(status='complete', finished_at_utc=utc_now())
        except BaseException as exc:
            record.update(status='failed', finished_at_utc=utc_now(), error=repr(exc))
            raise
        finally:
            write_json(logs / 'run.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'coarse', 'fine', 'plot', 'worker'])
    parser.add_argument('work', nargs='?', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.work, args.device)
    elif args.action == 'prepare':
        prepare()
    elif args.action == 'plot':
        from .plot import make_plot
        make_plot()
    else:
        run(args.action)


if __name__ == '__main__':
    main()
