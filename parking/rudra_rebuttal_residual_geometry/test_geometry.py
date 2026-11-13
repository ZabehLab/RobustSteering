"""Checks for scale matching, genuine rotation, and extendable paired sampling."""

from collections import Counter
import unittest

import torch

from .geometry import covariance_variants, covariance_factor, covariance_alignment, paired_sample, residual_covariance


class GeometryTests(unittest.TestCase):
    def test_alignment_contrast_matches_plotted_covariance_statistic(self):
        from .statistics import mean_alignment_advantage
        import numpy as np
        residuals = torch.randn(50, 3, 4, generator=torch.Generator().manual_seed(82)).double()
        covariance = residual_covariance(residuals)
        variants = covariance_variants(covariance, seed=18)
        actual = mean_alignment_advantage(residuals.numpy(), np.stack([v.numpy() for v in variants.values()]))
        expected = torch.stack([(covariance_alignment(covariance, covariance) -
                                 covariance_alignment(variants[v], covariance)).mean() for v in ('isotropic', 'diagonal', 'rotated')])
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-14)
        batched = mean_alignment_advantage(np.stack([residuals.numpy()] * 2), np.stack([v.numpy() for v in variants.values()]))
        np.testing.assert_allclose(batched, np.stack([actual] * 2), atol=1e-14)

    def test_variants_preserve_trace_and_rotations_preserve_spectrum(self):
        a = torch.tensor([[3., .7, .2], [.7, 1., .1], [.2, .1, .4]], dtype=torch.float64)
        covariance = torch.stack([a, a * 2])
        variants = covariance_variants(covariance, seed=17)
        for value in variants.values():
            torch.testing.assert_close(value.diagonal(dim1=-2, dim2=-1).sum(-1),
                                       covariance.diagonal(dim1=-2, dim2=-1).sum(-1))
            self.assertGreaterEqual(torch.linalg.eigvalsh(value).min().item(), -1e-10)
        torch.testing.assert_close(torch.linalg.eigvalsh(variants['rotated']),
                                   torch.linalg.eigvalsh(covariance))
        self.assertFalse(torch.allclose(variants['rotated'], covariance))
        torch.testing.assert_close(variants['diagonal'], torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1)))
        torch.testing.assert_close(variants['isotropic'], torch.eye(3).double()[None] * covariance.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None])
        for value in variants.values():
            factor = covariance_factor(value)
            torch.testing.assert_close(factor @ factor.transpose(-1, -2), value)

    def test_alignment_is_scale_invariant_and_residual_covariance_is_centered(self):
        x = torch.tensor([[[1., 2.]], [[3., 4.]], [[2., 6.]]], dtype=torch.float64)
        covariance = residual_covariance(x)
        torch.testing.assert_close(covariance[0], torch.cov(x[:, 0].T))
        torch.testing.assert_close(covariance_alignment(covariance, covariance * 11), torch.ones(1).double())
        orthogonal = torch.diag_embed(torch.tensor([[1., 0.]])).double()
        other = torch.diag_embed(torch.tensor([[0., 1.]])).double()
        self.assertEqual(covariance_alignment(orthogonal, other).item(), 0)

    def test_sample_is_paired_balanced_and_extends_without_changing_prefix(self):
        direct = [{'behavior_id': str(i), 'prompt_id': f'{i}-id'} for i in range(120)]
        ood = [{**r, 'prompt_id': f"{r['behavior_id']}-{j}", 'template_index': j}
               for r in direct for j in range(5)]
        source = {'evaluation': {'direct': direct, 'human_jailbreak': ood}}
        small = paired_sample(source, 50, seed=14)
        large = paired_sample(source, 100, seed=14)
        self.assertEqual(small, large[:50])
        self.assertEqual(Counter(p['ood']['template_index'] for p in small), {j: 10 for j in range(5)})
        self.assertEqual(len({p['id']['behavior_id'] for p in large}), 100)
        self.assertTrue(all(p['id']['behavior_id'] == p['ood']['behavior_id'] for p in large))

    def test_performance_extension_to_200_keeps_pilot_and_drops_diagonal(self):
        from .geometry import ACTIVE_VARIANTS
        from .score_store import ACTIVE_METHODS
        direct = [{'behavior_id': str(i), 'prompt_id': f'{i}-id'} for i in range(240)]
        ood = [{**r, 'prompt_id': f"{r['behavior_id']}-{j}", 'template_index': j}
               for r in direct for j in range(5)]
        source = {'evaluation': {'direct': direct, 'human_jailbreak': ood}}
        pilot = paired_sample(source, 50, seed=20261112)
        extended = paired_sample(source, 200, seed=20261112)
        self.assertEqual(pilot, extended[:50])
        self.assertEqual(Counter(p['ood']['template_index'] for p in extended), {j: 40 for j in range(5)})
        self.assertEqual(ACTIVE_VARIANTS, ('empirical', 'isotropic', 'rotated'))
        self.assertNotIn('diagonal', ACTIVE_METHODS)

    def test_hidden_collector_matches_calibration_without_language_model_head(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from transformers.tokenization_utils_base import BatchEncoding
        from robust_steerability.experiments.calibration import collect_last_token_states
        from .pipeline import collect_hidden_states

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return BatchEncoding({'input_ids': torch.tensor([[1, 2, 3]] * len(texts)),
                                      'attention_mask': torch.ones(len(texts), 3, dtype=torch.long)})

        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval()
        tokenizer = Tokenizer()
        expected = collect_last_token_states(model, tokenizer, ['a', 'b'], max_length=4, batch_size=2)['hidden']
        actual = collect_hidden_states(model, tokenizer, ['a', 'b'], batch_size=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
