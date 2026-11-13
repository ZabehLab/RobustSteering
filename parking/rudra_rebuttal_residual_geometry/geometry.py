"""Unit-local covariance controls; the package's controller synthesis is unchanged."""

from __future__ import annotations

import random

import torch

VARIANTS = ('empirical', 'isotropic', 'diagonal', 'rotated')


def residual_covariance(residuals: torch.Tensor) -> torch.Tensor:
    if residuals.ndim != 3 or len(residuals) < 2:
        raise ValueError('Expected at least two residuals with shape (samples, layers, state)')
    centered = residuals.double() - residuals.double().mean(0, keepdim=True)
    return torch.einsum('nli,nlj->lij', centered, centered) / (len(centered) - 1)


def covariance_variants(covariance: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    covariance = covariance.double()
    dimension = covariance.shape[-1]
    eye = torch.eye(dimension, dtype=covariance.dtype)
    total = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator, dtype=covariance.dtype))
    q = q * torch.where(r.diag() >= 0, 1., -1.)[None]
    return {
        'empirical': covariance,
        'isotropic': (total / dimension)[:, None, None] * eye,
        'diagonal': torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1)),
        'rotated': q @ covariance @ q.T,
    }


def covariance_factor(covariance: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    if eigenvalues.min() < -1e-8:
        raise ValueError('Covariance is not positive semidefinite')
    return eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(-2)


def covariance_alignment(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, right = left.double(), right.double()
    denominator = torch.linalg.matrix_norm(left) * torch.linalg.matrix_norm(right)
    if (denominator == 0).any():
        raise ValueError('Alignment is undefined for zero covariance')
    return (left * right).sum((-1, -2)) / denominator


def paired_sample(source: dict, count: int, seed: int) -> list[dict]:
    direct = {r['behavior_id']: r for r in source['evaluation']['direct']}
    if count % 5 or not 5 <= count <= len(direct):
        raise ValueError('Sample count must be a multiple of five within the held-out pool')
    ood = {(r['behavior_id'], r['template_index']): r for r in source['evaluation']['human_jailbreak']}
    rng = random.Random(seed)
    identities = sorted(direct)
    rng.shuffle(identities)
    assignments = []
    for _ in range((len(identities) + 4) // 5):
        block = list(range(5))
        rng.shuffle(block)
        assignments.extend(block)
    return [{'id': direct[b], 'ood': ood[b, assignments[i]]}
            for i, b in enumerate(identities[:count])]
