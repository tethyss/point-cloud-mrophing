"""Empirical CDF transformation and optimal-transport pairing."""

from __future__ import annotations

import numpy as np
import ot
from scipy.stats import norm, rankdata


def empirical_cdf(values: np.ndarray) -> np.ndarray:
    """Convert every variable column to open-interval empirical probabilities."""

    probabilities = [
        rankdata(values[:, index], method="average") / (len(values) + 1)
        for index in range(values.shape[1])
    ]
    return np.column_stack(probabilities)


def paired_morphing_factors(
    target_cdf: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Pair target samples with independent Gaussian vectors using exact OT."""

    source = generator.standard_normal(target_cdf.shape)
    source_cdf = norm.cdf(source)
    weights = np.full(len(target_cdf), 1.0 / len(target_cdf))

    cost = ot.dist(target_cdf, source_cdf, metric="euclidean")
    coupling = ot.emd(weights, weights, cost)

    # Equal cardinality and uniform weights produce a permutation-like coupling.
    assignment = coupling.argmax(axis=1)
    return source[assignment]

