"""Checks for fixed-gamma synthesis, frozen sampling, and staged execution."""

from dataclasses import replace
import unittest

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


if __name__ == '__main__':
    unittest.main()
