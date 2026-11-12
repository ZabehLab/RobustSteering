"""Checks for diagnostic score reuse and paper-consistent aggregation."""

import unittest

from .diagnostics import API_SCORERS, response_key, select_cached, summarize


class DiagnosticTests(unittest.TestCase):
    def test_cache_key_matches_all_judge_inputs(self):
        row = {'text': 'prompt', 'completion': 'response', 'concept': 'safe refusal'}
        self.assertEqual(response_key(row), response_key({**row, 'prompt_id': 'other'}))
        for key in row:
            self.assertNotEqual(response_key(row), response_key({**row, key: 'different'}))

    def test_endpoints_retain_their_own_scores(self):
        row = {'text': 'prompt', 'completion': 'response', 'concept': 'safe refusal', 'prompt_id': 'p'}
        key = response_key(row)
        own = {'response_key': key, 'score': 1}
        other = {'response_key': key, 'score': 2}
        self.assertIs(select_cached(row, 'lqr', {key: other}, {('alqr_8d', 'p'): own}), own)
        self.assertIs(select_cached(row, 'rho_0.5', {key: other}, {}), other)

    def test_overall_is_averaged_after_per_response_harmonic_mean(self):
        rows = [dict(zip(API_SCORERS, values, strict=True)) for values in [(0, 2, 2), (2, 2, 2)]]
        result = summarize(rows)
        self.assertEqual(result['axbench_overall'], 1.)
        self.assertEqual(result['axbench_fluency'], 2.)


if __name__ == '__main__':
    unittest.main()
