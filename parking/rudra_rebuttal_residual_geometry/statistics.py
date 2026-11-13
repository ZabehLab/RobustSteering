"""Behavior-level uncertainty for mean covariance-alignment contrasts."""

import numpy as np
from scipy.stats import bootstrap

from .geometry import VARIANTS


def mean_alignment_advantage(residuals, covariances):
    """Mean over layers of empirical minus alternative covariance alignment."""
    residuals = np.asarray(residuals, dtype=float)
    centered = residuals - residuals.mean(axis=-3, keepdims=True)
    covariance = np.einsum('...nli,...nlj->...lij', centered, centered) / (centered.shape[-3] - 1)
    numerator = np.einsum('vlik,...lik->...vl', covariances, covariance)
    denominator = np.linalg.norm(covariances, axis=(-2, -1)) * np.linalg.norm(covariance, axis=(-2, -1))[..., None, :]
    alignment = (numerator / denominator).mean(axis=-1)
    return alignment[..., :1] - alignment[..., 1:]


def alignment_comparisons(residuals, covariances, seed, resamples=49999):
    """Invert two-sided BCa intervals with Bonferroni correction for three contrasts.

    Calibration hypotheses are fixed. A bootstrap case is one whole OOD
    behavior with all its layers, shared across the three comparisons.
    """
    residuals = np.asarray(residuals, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    observed = mean_alignment_advantage(residuals, covariances)

    def statistic(indices, axis=-1):
        contrast = mean_alignment_advantage(residuals[indices], covariances)
        return np.moveaxis(contrast, -1, 0)

    data = (np.arange(len(residuals)),)
    result = bootstrap(data, statistic, vectorized=True, method='BCa', batch=128,
                       n_resamples=resamples, confidence_level=1 - .05 / 3,
                       rng=np.random.default_rng(seed))
    intervals = {}
    for alpha in (.05, .01, .001):
        if alpha != .05:
            result = bootstrap(data, statistic, vectorized=True, method='BCa', batch=128,
                               n_resamples=0, confidence_level=1 - alpha / 3,
                               bootstrap_result=result)
        intervals[alpha] = result.confidence_interval
    comparisons = []
    for index, variant in enumerate(VARIANTS[1:]):
        stars = 'ns'
        for alpha, label in ((.05, '*'), (.01, '**'), (.001, '***')):
            ci = intervals[alpha]
            if ci.low[index] > 0 or ci.high[index] < 0:
                stars = label
        comparisons.append({
            'alternative': variant, 'mean_alignment_difference': float(observed[index]),
            'stars': stars,
            'simultaneous_intervals': {
                str(alpha): [float(intervals[alpha].low[index]), float(intervals[alpha].high[index])]
                for alpha in (.05, .01, .001)},
        })
    return {
        'method': 'Two-sided BCa behavior bootstrap; Bonferroni-adjusted intervals for three comparisons',
        'statistic': 'Mean across layers of empirical minus alternative covariance alignment',
        'resampling_unit': 'Whole OOD behavior; all layers and comparisons stay paired',
        'conditional_on': 'Frozen calibration covariance hypotheses and the selected model',
        'resamples': resamples, 'seed': seed,
        'star_thresholds': {'*': .05, '**': .01, '***': .001},
        'comparisons': comparisons,
    }, result.bootstrap_distribution
