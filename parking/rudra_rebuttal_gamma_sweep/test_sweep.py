"""Checks for fixed-gamma synthesis, frozen sampling, and staged execution."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from robust_steerability.control import (
    FiniteHorizonControlProblem, HInfinityController, HInfinityOptions, LQRController,
)


class FixedGammaTests(unittest.TestCase):
    def setUp(self):
        eye = torch.eye(2)
        self.problem = FiniteHorizonControlProblem(
            dynamics=eye.repeat(3, 1, 1) * .9,
            control_channels=eye.repeat(3, 1, 1),
            disturbance_channels=eye.repeat(3, 1, 1) * .2,
            state_costs=eye.repeat(3, 1, 1),
            control_costs=eye.repeat(3, 1, 1), terminal_cost=eye,
        )

    def test_fixed_level_reproduces_searched_controller(self):
        searched = HInfinityController.synthesize(
            self.problem, options=HInfinityOptions(deployment_margin=.01))
        fixed = HInfinityController.synthesize_at_gamma(
            self.problem, searched.diagnostics['gamma_used'], gamma_star=searched.gamma_star)
        torch.testing.assert_close(fixed.gains, searched.gains, rtol=0, atol=0)
        self.assertTrue(fixed.feasible)

    def test_large_gamma_recovers_lqr_and_roundtrips_without_claiming_a_boundary(self):
        nominal = LQRController.synthesize(replace(self.problem, disturbance_channels=None))
        fixed = HInfinityController.synthesize_at_gamma(self.problem, 1e4)
        torch.testing.assert_close(fixed.gains, nominal.gains, rtol=1e-5, atol=1e-6)
        self.assertIsNone(fixed.gamma_star)
        restored = HInfinityController.from_solution(self.problem, fixed.solution())
        torch.testing.assert_close(restored.gains, fixed.gains)

    def test_infeasible_is_not_deployable(self):
        fixed = HInfinityController.synthesize_at_gamma(self.problem, .01)
        self.assertFalse(fixed.feasible)
        with self.assertRaises(RuntimeError):
            fixed.control(0, torch.ones(1, 2))

    def test_invalid_levels_are_rejected(self):
        for gamma in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                HInfinityController.synthesize_at_gamma(self.problem, gamma)


class SweepTests(unittest.TestCase):
    def test_plot_reports_binomial_standard_error_on_the_matched_sample(self):
        from . import plot, sweep
        from parking.rudra_rebuttal_mached_calib.inputs import write_json
        from robust_steerability.judges.specs import scorer_cache_path
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'inputs/manifest.json', {'gamma_star': .3, 'endpoint_rho': 1 / 1.01})
            records = {condition: [{'prompt_id': f'p{i}'} for i in range(100)]
                       for condition in sweep.CONDITIONS}
            write_json(root / 'inputs/records.json', records)
            with patch.object(plot, 'ROOT', root), patch.object(sweep, 'ROOT', root):
                for point in ('lqr', 'hinf'):
                    for condition in sweep.CONDITIONS:
                        path = sweep.generation_path(root, sweep.task_for(point, condition, 0))
                        write_json(scorer_cache_path(sweep.evaluation_root(), path, sweep.SCORER), {
                            'status': 'complete', 'rows': [
                                {'prompt_id': f'p{i}', 'score': int(i < 10), 'valid': True}
                                for i in range(100)]})
                results = plot.collect_results()
            self.assertEqual(len(results), 4)
            for row in results:
                self.assertEqual(row['asr_percent'], 10.)
                self.assertAlmostEqual(row['asr_se_percent'], 3.)

    def test_sampling_is_matched_balanced_and_deterministic(self):
        from .sweep import select_records
        source = {'evaluation': {
            'direct': [{'behavior_id': f'b{i}', 'prompt_id': f'b{i}-direct'} for i in range(240)],
            'human_jailbreak': [
                {'behavior_id': f'b{i}', 'prompt_id': f'b{i}-human-{t}', 'template_index': t}
                for i in range(240) for t in range(5)],
        }}
        a = select_records(source)
        self.assertEqual(a, select_records(source))
        self.assertEqual(len(a['direct']), 100)
        self.assertEqual(len(a['human_jailbreak']), 100)
        self.assertEqual({r['behavior_id'] for r in a['direct']},
                         {r['behavior_id'] for r in a['human_jailbreak']})
        for t in range(5):
            self.assertEqual(sum(r['template_index'] == t for r in a['human_jailbreak']), 20)

    def test_fine_only_fills_missing_midpoints(self):
        from .sweep import COARSE, FINE
        self.assertEqual(COARSE, (.2, .4, .6, .8))
        self.assertEqual(FINE, (.1, .3, .5, .7, .9))
        self.assertEqual(len(set(COARSE + FINE)), 9)

    def test_worker_generates_matching_rows_and_resumes_without_regeneration(self):
        from . import sweep
        from parking.rudra_rebuttal_mached_calib import workers
        from parking.rudra_rebuttal_mached_calib.inputs import write_json, read_json
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = {'direct': [
                {'prompt_id': f'p{i}', 'behavior_id': f'b{i}', 'behavior': 'fixture',
                 'context': '', 'functional_category': 'standard', 'semantic_category': 'fixture',
                 'text': 'fixture', 'model_input': 'fixture'} for i in range(100)]}
            write_json(root / 'inputs/records.json', records)
            write_json(root / 'inputs/manifest.json', {'fixture': True})
            point = 'rho_0.2'
            (root / 'controllers').mkdir()
            (root / 'controllers' / f'{point}.pt').write_bytes(b'fixture')
            work = root / 'work.json'
            write_json(work, {'kind': 'generate', 'items': [
                {'point': point, 'condition': 'direct', 'shard': 0}]})
            candidate = {'status': 'complete', 'rho': .2, 'gamma': 1.5}
            def tensors(path):
                if path.name == f'{point}.pt':
                    return candidate
                return {}
            def completions(_model, _tokenizer, prompts, **kwargs):
                self.assertFalse(kwargs['use_cache'])
                self.assertEqual(kwargs['max_new_tokens'], 512)
                return ['fixture response'] * len(prompts), [2] * len(prompts)
            class Model:
                class generation_config:
                    eos_token_id = 1
                class config:
                    eos_token_id = 1
            with patch.object(sweep, 'ROOT', root), \
                 patch.object(sweep.torch.cuda, 'set_device'), \
                 patch.object(sweep, 'load_tensors', side_effect=tensors), \
                 patch.object(sweep, 'SetpointCalibration'), \
                 patch.object(sweep, 'load_model', return_value=(Model(), None)), \
                 patch.object(sweep, 'candidate_policy', return_value=None), \
                 patch.object(workers, 'generate_completions', side_effect=completions) as generate:
                sweep.worker(work, 'cuda:0')
                path = workers.generation_path(root, sweep.task_for(point, 'direct', 0))
                rows = read_json(path)['repetitions'][0]['rows']
                self.assertEqual(len(rows), 50)
                self.assertEqual(generate.call_count, 4)
                sweep.worker(work, 'cuda:0')
                self.assertEqual(generate.call_count, 4)

    def test_run_schedules_only_missing_stage_points(self):
        from . import sweep, plot
        from parking.rudra_rebuttal_mached_calib.inputs import read_json, write_json
        from robust_steerability.judges.specs import scorer_cache_path
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def launch(kind, items, logs):
                calls.append((kind, len(items)))
                if kind == 'generate':
                    for item in items:
                        path = sweep.generation_path(root, sweep.task_for(
                            item['point'], item['condition'], item['shard']))
                        write_json(path, {'status': 'complete'})
                else:
                    for item in items:
                        write_json(scorer_cache_path(sweep.evaluation_root(), Path(item), sweep.SCORER),
                                   {'status': 'complete'})
            with patch.object(sweep, 'ROOT', root), patch.object(sweep, 'prepare'), \
                 patch.object(plot, 'make_plot'), patch.object(sweep, 'launch', side_effect=launch):
                sweep.run('coarse')
                self.assertEqual(calls, [('generate', 16), ('classify', 16)])
                self.assertEqual(read_json(next((root / 'logs').glob('*/run.json')))['status'], 'complete')
                calls.clear()
                sweep.run('coarse')
                self.assertEqual(calls, [('generate', 0), ('classify', 0)])
                calls.clear()
                sweep.run('fine')
                self.assertEqual(calls, [('generate', 20), ('classify', 20)])


if __name__ == '__main__':
    unittest.main()
