"""Experimental variograms and exponential model fitting."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import distance

from .models import VariogramModel

EPSILON = 1.0e-6
LOGGER = logging.getLogger(__name__)


def experimental_variogram(
    coordinates: np.ndarray,
    values: np.ndarray,
    lag: float,
    nlag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate omnidirectional direct variograms for each value column."""

    pair_distances = distance.pdist(coordinates)
    if not np.any(pair_distances > 0):
        raise ValueError("Landmarks must not all occupy the same location.")

    lag_centers = lag * np.arange(1, nlag + 1)
    direct_variograms = np.full((values.shape[1], nlag), np.nan)

    for variable_index in range(values.shape[1]):
        semivariance = 0.5 * distance.pdist(
            values[:, [variable_index]],
            metric="sqeuclidean",
        )
        for lag_index, center in enumerate(lag_centers):
            in_bin = np.abs(pair_distances - center) <= lag / 2
            if np.any(in_bin):
                direct_variograms[variable_index, lag_index] = np.mean(
                    semivariance[in_bin]
                )
    return lag_centers, direct_variograms


def exponential_variogram(
    distance_values: np.ndarray,
    nugget: float,
    partial_sill: float,
    range_: float,
) -> np.ndarray:
    """Evaluate an exponential variogram model."""

    return nugget + partial_sill * (
        1.0 - np.exp(-3.0 * distance_values / range_)
    )


def fit_variogram_models(
    lags: np.ndarray,
    variograms: np.ndarray,
) -> list[VariogramModel]:
    """Fit one robust exponential model per morphing-factor variable."""

    return [
        _fit_single_variogram(lags, gamma, variable_index)
        for variable_index, gamma in enumerate(variograms)
    ]


def _fit_single_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    variable_index: int,
) -> VariogramModel:
    valid = np.isfinite(gamma)
    x_values = lags[valid]
    y_values = gamma[valid]
    variance = float(np.nanmax(y_values)) if len(y_values) else 1.0

    if len(x_values) < 3 or variance <= 0:
        LOGGER.warning(
            "Variable %s has insufficient variogram bins; using a fallback model.",
            variable_index,
        )
        return VariogramModel(0.0, 1.0, float(lags[-1]))

    lower_bounds = (0.0, 0.0, max(float(np.min(x_values)) / 10, EPSILON))
    upper_bounds = (variance * 2, variance * 3, float(lags[-1]) * 3)
    initial_guess = (0.0, variance, float(np.median(x_values)))

    try:
        parameters, _ = curve_fit(
            exponential_variogram,
            x_values,
            y_values,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        LOGGER.warning(
            "Variable %s variogram fitting failed; using a fallback model.",
            variable_index,
        )
        return VariogramModel(0.0, variance, float(np.median(x_values)))

    return VariogramModel(*map(float, parameters))
