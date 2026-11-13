"""Frozen-calibration geometry ablation, isolated from the source Giga experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from robust_steerability.benchmarks.harmful_runtime import generate_completions
from robust_steerability.control import FiniteHorizonControlProblem, HInfinityController
from robust_steerability.control.h_infinity import HInfinityOptions
from robust_steerability.judges.harmbench import score_generations
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.modeling.tokenization import tokenize_prompts, RAW_TEXT_TOKENIZATION
from robust_steerability.runtime.policy import ReducedStateSetpointPolicy

from .geometry import VARIANTS, covariance_variants, covariance_factor, residual_covariance, covariance_alignment, paired_sample

UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[1]
MODEL = 'qwen25_05b_instruct'
SOURCE = REPO / 'parking/rudra_rebuttal_mached_calib/cache' / MODEL
ROOT = UNIT / 'cache' / MODEL
MODULE = 'parking.rudra_rebuttal_residual_geometry.pipeline'
SCORER = 'harmbench_test_success'
SEED = 20261112
BLOCK = 10


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor_read(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def tensor_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen input changed: {path}')
    else:
        save(path, value)


def sample(count):
    return paired_sample(read(SOURCE / 'artifacts/harmbench.json'), count, SEED)


def blocks(count, condition):
    pairs = sample(count)
    return [(i // BLOCK, [p[condition] for p in pairs[i:i + BLOCK]]) for i in range(0, count, BLOCK)]


def generation_path(variant, condition, index):
    return ROOT / 'generations' / variant / condition / f'block-{index:03d}.json'


def reuse_empirical(count):
    lookup = {}
    scorer = None
    for part in ('screening', 'holdout'):
        stage = SOURCE / 'evaluation/giga' / part
        for condition in ('direct', 'human_jailbreak'):
            for path in sorted((stage / 'generations' / condition / 'h_infinity').glob('*.json')):
                generation = read(path)
                scoring_path = scorer_cache_path(stage, path, SCORER)
                scoring = read(scoring_path)
                assert generation['status'] == scoring['status'] == 'complete'
                scorer = scoring['scorer']
                assert scorer['model'] == [scorer_spec(SCORER).model_id, scorer_spec(SCORER).revision]
                score_rows = {r['prompt_id']: r for r in scoring['rows']}
                for row in generation['repetitions'][0]['rows']:
                    key = row['prompt_id']
                    assert key not in lookup
                    assert score_rows[key]['valid']
                    lookup[key] = (row, score_rows[key], str(path), str(scoring_path))
    for condition in ('id', 'ood'):
        for index, records in blocks(count, condition):
            destination = generation_path('empirical', condition, index)
            rows, scores = [], []
            for record in records:
                row, score, source_gen, source_score = lookup[record['prompt_id']]
                assert row['text'] == record['text'] and row['behavior_id'] == record['behavior_id']
                rows.append({**row, 'reused_from': source_gen})
                scores.append({**score, 'reused_from': source_score})
            frozen_json(destination, {'status': 'complete', 'rows': rows})
            frozen_json(scorer_cache_path(ROOT, destination, SCORER),
                        {'status': 'complete', 'scorer': scorer, 'rows': scores})


def prepare(count):
    torch.set_num_threads(1)
    if count % BLOCK:
        raise ValueError('Use complete ten-behavior blocks so later extensions never replace a saved block')
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / p for p in ('artifacts/reduced.pt', 'artifacts/base/controller_diagnostics/input.pt',
                                  'artifacts/harmbench.json', 'calibration/mega/selection.json',
                                  'calibration/controllers/h_infinity/q_00_qf_00.pt')]
    frozen_json(ROOT / 'protocol.json', {
        'model': read(SOURCE / 'artifacts/manifest.json')['model'],
        'source_sha256': {str(p): sha(p) for p in paths},
        'sample_seed': SEED, 'rotation_seed': SEED + 1, 'variants': list(VARIANTS),
        'heldout_pool': '240 Giga behaviors; one balanced assigned jailbreak per behavior',
        'geometry': 'per-layer trace matched; same frozen costs and coordinates',
        'deployment_margin': .01, 'max_new_tokens': 512, 'do_sample': False, 'use_cache': False,
    })
    pairs = sample(count)
    tuning = read(SOURCE / 'artifacts/data.json')['calibration']['tuning']
    assert not {p['id']['behavior_id'] for p in pairs} & {r['behavior_id'] for r in tuning}
    frozen_json(ROOT / f'sample_n{count:03d}.json', pairs)
    fit = tensor_read(SOURCE / 'artifacts/reduced.pt')
    bundle = tensor_read(SOURCE / 'artifacts/base/controller_diagnostics/input.pt')
    cal = bundle['calibration']
    covariance = residual_covariance(cal['residuals'])
    torch.testing.assert_close(covariance, cal['disturbance_construction']['empirical_covariance'].double(), rtol=1e-5, atol=1e-8)
    variants = covariance_variants(covariance, SEED + 1)
    winner = read(SOURCE / 'calibration/mega/selection.json')['winners']['h_infinity']['configuration']
    problem = FiniteHorizonControlProblem(**fit['problem'])
    problem = replace(problem,
                      state_costs=problem.state_costs * winner['q'] / fit['settings']['q'],
                      control_costs=problem.control_costs * winner['r'] / fit['settings']['r'],
                      terminal_cost=problem.terminal_cost * winner['q_final'] / fit['settings']['q_final'])
    original = tensor_read(SOURCE / 'calibration/controllers/h_infinity/q_00_qf_00.pt')
    checks = {}
    for variant in VARIANTS:
        channels = (problem.disturbance_channels if variant == 'empirical'
                    else covariance_factor(variants[variant]).to(problem.disturbance_channels.dtype))
        factor_error = (torch.linalg.matrix_norm(channels.double() @ channels.double().transpose(-1, -2)
                                                - variants[variant])
                        / torch.linalg.matrix_norm(variants[variant]))
        # Match the package's covariance-factor diagnostic, including archived float32 SVD roundoff.
        assert factor_error.max() < 1e-4
        path = ROOT / 'controllers' / f'{variant}.pt'
        if not path.exists():
            controller = HInfinityController.synthesize(replace(problem, disturbance_channels=channels),
                                                        device='cpu', options=HInfinityOptions(**fit['options']))
            solution = controller.solution()
            if not solution.feasible:
                raise RuntimeError(f'Infeasible controller: {variant}')
            tensor_save(path, {'gains': solution.gains, 'gamma_star': solution.gamma_star,
                              'diagnostics': solution.diagnostics, 'channels': channels,
                              'covariance': variants[variant]})
        saved = tensor_read(path)
        if variant == 'empirical':
            torch.testing.assert_close(saved['gains'], original['gains'], rtol=1e-5, atol=1e-6)
            assert abs(saved['gamma_star'] - original['gamma_star']) < 1e-7
        checks[variant] = {'gamma_star': saved['gamma_star'], 'gamma_used': saved['diagnostics']['gamma_used'],
                           'factor_covariance_relative_error_max': factor_error.max().item(),
                           'trace_relative_error_max': ((variants[variant].diagonal(dim1=-2, dim2=-1).sum(-1)
                                                       / covariance.diagonal(dim1=-2, dim2=-1).sum(-1)) - 1).abs().max().item()}
    save(ROOT / 'synthesis_checks.json', {'empirical_reproduces_giga': True, 'variants': checks})
    if not (ROOT / 'frozen_fit.pt').exists():
        tensor_save(ROOT / 'frozen_fit.pt', {
            'coordinates': fit['coordinates'], 'control_channels': problem.control_channels,
            'dynamics': problem.dynamics, 'means': cal['means'], 'encoders': cal['encoders'],
            'reference_states': cal['reference_states'], 'calibration_residuals': cal['residuals'],
            'covariances': variants,
        })
    reuse_empirical(count)
    print(f'Prepared {count} paired cases; empirical controller reproduced exactly within tolerance.', flush=True)


def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id, revision = read(ROOT / 'protocol.json')['model']
    token = load_access_token(REPO) or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, token=token,
                                                dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                attn_implementation='sdpa').to(device).eval()
    if model.generation_config.eos_token_id is not None:
        model.config.eos_token_id = model.generation_config.eos_token_id
    return model, tokenizer


def collect_hidden_states(model, tokenizer, texts, batch_size=8):
    """Match cached block-boundary states, without allocating unused vocabulary logits."""
    decoder = model.model
    layers = decoder.layers
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenize_prompts(tokenizer, texts[start:start + batch_size],
                                   prompt_tokenization=RAW_TEXT_TOKENIZATION,
                                   return_tensors='pt', padding=True, truncation=False).to(model.device)
        states = [None] * (len(layers) + 1)
        def before(index):
            def capture(_module, args):
                states[index] = args[0][:, -1].detach().float().cpu()
            return capture
        def after(_module, _args, output):
            states[-1] = (output[0] if isinstance(output, tuple) else output)[:, -1].detach().float().cpu()
        handles = [layer.register_forward_pre_hook(before(i)) for i, layer in enumerate(layers)]
        handles.append(layers[-1].register_forward_hook(after))
        try:
            with torch.inference_mode():
                decoder(**encoded, return_dict=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        batches.append(torch.stack(states, dim=1))
    return torch.cat(batches)


def geometry_worker(count, device):
    path = ROOT / 'activations' / f'ood_n{count:03d}.pt'
    if not path.exists():
        records = [p['ood'] for p in sample(count)]
        previous = {}
        for old_path in sorted((ROOT / 'activations').glob('ood_n*.pt')):
            old = tensor_read(old_path)
            previous.update({key: (state, length) for key, state, length in
                             zip(old['prompt_ids'], old['hidden'], old['token_lengths'], strict=True)})
        missing = [r for r in records if r['prompt_id'] not in previous]
        started = time.perf_counter()
        if missing:
            model, tokenizer = load_model(device)
            lengths = [len(tokenizer.encode(r['model_input'])) for r in missing]
            states = collect_hidden_states(model, tokenizer, [r['model_input'] for r in missing])
            previous.update({r['prompt_id']: (state, length) for r, state, length in
                             zip(missing, states, lengths, strict=True)})
        tensor_save(path, {'prompt_ids': [r['prompt_id'] for r in records],
                          'token_lengths': [previous[r['prompt_id']][1] for r in records],
                          'hidden': torch.stack([previous[r['prompt_id']][0] for r in records]),
                          'new_forward_passes': len(missing), 'elapsed_seconds': time.perf_counter() - started,
                          'device': device,
                          'tokenization': 'same default raw tokenization as cached calibration; no truncation'})
        print(f'Saved {count} OOD prompt activations across all layers ({len(missing)} new).', flush=True)
    states = tensor_read(path)['hidden']
    fit = tensor_read(ROOT / 'frozen_fit.pt')
    reduced = torch.einsum('nld,ldr->nlr', states - fit['means'][None], fit['encoders'])
    deviations = reduced - fit['reference_states'][None]
    residuals = deviations[:, 1:] - torch.einsum('lij,nlj->nli', fit['dynamics'], deviations[:, :-1])
    covariance = residual_covariance(residuals)
    tensor_save(ROOT / 'residuals' / f'ood_n{count:03d}.pt', {'residuals': residuals, 'covariance': covariance})
    save(ROOT / f'alignment_n{count:03d}.json', {
        'count': count, 'layers': list(range(1, len(covariance) + 1)),
        'alignment': {v: covariance_alignment(c, covariance).tolist() for v, c in fit['covariances'].items()},
        'definition': 'trace(C_variant C_OOD)/(Frobenius(C_variant)*Frobenius(C_OOD))',
    })


def generate_worker(variant, count, device):
    model, tokenizer = load_model(device)
    fit, controller = tensor_read(ROOT / 'frozen_fit.pt'), tensor_read(ROOT / 'controllers' / f'{variant}.pt')
    policy = ReducedStateSetpointPolicy(HInfinityController(
        controller['gains'], fit['control_channels'], feasible=True,
        gamma_star=controller['gamma_star'], diagnostics=controller['diagnostics']), **fit['coordinates'])
    fields = ('prompt_id', 'behavior_id', 'behavior', 'context', 'functional_category', 'semantic_category', 'text')
    for condition in ('id', 'ood'):
        for index, records in blocks(count, condition):
            path = generation_path(variant, condition, index)
            if path.exists():
                saved = read(path)
                assert [r['prompt_id'] for r in saved['rows']] == [r['prompt_id'] for r in records]
                assert saved['controller_sha256'] == sha(ROOT / 'controllers' / f'{variant}.pt')
                if saved['status'] == 'complete':
                    continue
            started = time.perf_counter()
            texts, token_counts = generate_completions(
                model, tokenizer, [r['model_input'] for r in records], steering_policy=policy,
                use_cache=False, batch_size=len(records), max_new_tokens=512)
            rows = [{**{k: r[k] for k in fields}, 'completion': t, 'generated_tokens': n}
                    for r, t, n in zip(records, texts, token_counts, strict=True)]
            save(path, {'status': 'complete', 'rows': rows, 'controller_sha256': sha(ROOT / 'controllers' / f'{variant}.pt'),
                        'elapsed_seconds': time.perf_counter() - started})
            print(f'{variant}/{condition}: {index * BLOCK + len(records)}/{count}, '
                  f'{time.perf_counter() - started:.1f}s for {len(records)} responses', flush=True)


def classify_worker(variants, count, device):
    paths = [generation_path(v, c, i) for v in variants for c in ('id', 'ood') for i, _ in blocks(count, c)]
    score_generations(paths, ROOT, SCORER, device, load_access_token(REPO), batch_size=1)
    for path in paths:
        payload = read(scorer_cache_path(ROOT, path, SCORER))
        if not all(r['valid'] for r in payload['rows']):
            raise RuntimeError(f'Invalid HarmBench verdict: {path}')


def launch(tasks, devices, count):
    pending = list(tasks)
    active = []
    last_wait_notice = 0
    device_uuids = {f'cuda:{i.strip()}': u.strip() for i, u in [line.split(',') for line in
                   subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()]}
    while pending or active:
        own_pids = {str(a[0].pid) for a in active}
        foreign = set()
        for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True).splitlines():
            uuid, pid = [part.strip() for part in line.split(',')]
            if pid not in own_pids:
                foreign.add(uuid)
        free = [d for d in devices if d not in [a[1] for a in active] and device_uuids[d] not in foreign]
        if pending and not free and not active and time.time() - last_wait_notice > 60:
            print('Waiting for an idle GPU; leaving other experiments untouched.', flush=True)
            last_wait_notice = time.time()
        for device in free:
            if not pending:
                break
            kind, variants = pending.pop(0)
            name = kind + ('-' + '-'.join(variants) if variants else '')
            log_path = ROOT / 'logs' / f'{name}_n{count:03d}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open('a')
            cmd = [sys.executable, '-u', '-m', MODULE, 'worker', '--kind', kind, '--count', str(count),
                   '--device', device, '--variants', *variants]
            process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
            active.append((process, device, name, log))
            print(f'Started {name} on {device}; PID {process.pid}; log {log_path}', flush=True)
        for entry in active[:]:
            process, device, name, log = entry
            code = process.poll()
            if code is not None:
                log.close()
                active.remove(entry)
                print(f'Finished {name}: exit {code}', flush=True)
                if code:
                    # Other independent workers may still be making valid progress; do not kill them.
                    raise RuntimeError(f'{name} failed; inspect its log. Other workers are left running.')
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'report'])
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--kind', choices=['geometry', 'generate', 'classify'])
    parser.add_argument('--variants', nargs='*', default=[])
    args = parser.parse_args()
    torch.set_num_threads(4 if args.device == 'cpu' else 1)
    if args.command == 'prepare':
        prepare(args.count)
    elif args.command == 'run':
        prepare(args.count)
        save(ROOT / 'run_status.json', {'status': 'running', 'count': args.count, 'pid': os.getpid(), 'started': time.time()})
        save(ROOT / 'implementation.json', {p.name: sha(p) for p in UNIT.glob('*.py')})
        try:
            launch([('geometry', []), *[('generate', [v]) for v in VARIANTS[1:]]], args.devices, args.count)
            launch([('classify', ['isotropic', 'diagonal']), ('classify', ['rotated'])], args.devices, args.count)
            from .plots import report
            report(args.count)
        except BaseException as error:
            save(ROOT / 'run_status.json', {'status': 'failed', 'count': args.count,
                                            'error': str(error), 'finished': time.time()})
            raise
        save(ROOT / 'run_status.json', {'status': 'complete', 'count': args.count, 'finished': time.time()})
    elif args.command == 'worker':
        if args.device.startswith('cuda:'):
            torch.cuda.set_device(args.device)
        if args.kind == 'geometry':
            geometry_worker(args.count, args.device)
        elif args.kind == 'generate':
            generate_worker(args.variants[0], args.count, args.device)
        else:
            classify_worker(args.variants, args.count, args.device)
    else:
        from .plots import report
        report(args.count)


if __name__ == '__main__':
    main()
