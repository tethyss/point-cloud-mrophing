"""Local thin-plate-spline mapping from MF space to original variables."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.stats import norm

from .models import Dataset
from .pairing import empirical_cdf

PROBABILITY_EPSILON = 1.0e-6


class LocalThinPlateSpline:
    """Thin-plate-spline map fitted to a local set of control points."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("TPS source and target arrays must be two-dimensional.")
        if len(source) != len(target):
            raise ValueError("TPS source and target row counts must match.")

        self.source = source
        kernel = distance.cdist(source, source)
        affine_design = np.column_stack((np.ones(len(source)), source))
        affine_size = source.shape[1] + 1
        system = np.block(
            (
                (kernel, affine_design),
                (
                    affine_design.T,
                    np.zeros((affine_size, affine_size)),
                ),
            )
        )
        right_side = np.vstack(
            (
                target,
                np.zeros((affine_size, target.shape[1])),
            )
        )
        self.parameters = np.linalg.lstsq(system, right_side, rcond=None)[0]

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Map source-space points into target logit space."""

        kernel = distance.cdist(points, self.source)
        design = np.column_stack((kernel, np.ones(len(points)), points))
        return design @ self.parameters


def logit(probabilities: np.ndarray) -> np.ndarray:
    """Map probabilities from the open unit interval to real values."""

    clipped = np.clip(
        probabilities,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


def inverse_empirical_cdf(
    probabilities: np.ndarray,
    raw_values: np.ndarray,
) -> np.ndarray:
    """Back-transform probabilities using each landmark marginal distribution."""

    output = np.empty_like(probabilities)
    for variable_index in range(raw_values.shape[1]):
        output[:, variable_index] = np.quantile(
            raw_values[:, variable_index],
            probabilities[:, variable_index],
            method="linear",
        )
    return output


def map_to_original_space(
    simulated_mf: np.ndarray,
    prediction_coordinates: np.ndarray,
    landmarks: Dataset,
    paired_mf: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Map one simulated MF realization back to the original variable space."""

    _validate_mapping_inputs(
        simulated_mf,
        prediction_coordinates,
        landmarks,
        paired_mf,
        neighbors,
    )
    neighbor_indices = _find_spatial_neighbors(
        prediction_coordinates,
        landmarks.coordinates,
        neighbors,
    )

    source_landmarks = logit(norm.cdf(paired_mf))
    target_landmarks = logit(empirical_cdf(landmarks.values))
    query_points = logit(norm.cdf(simulated_mf))
    mapped_cdf = np.empty_like(simulated_mf)

    # Grid nodes sharing the same local landmarks reuse one fitted TPS model.
    for indices, point_indices in _group_by_neighbors(neighbor_indices).items():
        landmark_indices = np.asarray(indices, dtype=int)
        grid_indices = np.asarray(point_indices, dtype=int)
        mapper = LocalThinPlateSpline(
            source_landmarks[landmark_indices],
            target_landmarks[landmark_indices],
        )
        mapped_cdf[grid_indices] = mapper.predict(query_points[grid_indices])

    clipped_cdf = np.clip(
        mapped_cdf,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return inverse_empirical_cdf(clipped_cdf, landmarks.values)


def _find_spatial_neighbors(
    prediction_coordinates: np.ndarray,
    landmark_coordinates: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    tree = cKDTree(landmark_coordinates)
    _, indices = tree.query(prediction_coordinates, k=neighbors)
    return indices[:, None] if neighbors == 1 else indices


def _group_by_neighbors(
    neighbor_indices: np.ndarray,
) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for point_index, indices in enumerate(neighbor_indices):
        groups[tuple(np.asarray(indices, dtype=int))].append(point_index)
    return groups


def _validate_mapping_inputs(
    simulated_mf: np.ndarray,
    prediction_coordinates: np.ndarray,
    landmarks: Dataset,
    paired_mf: np.ndarray,
    neighbors: int,
) -> None:
    if len(simulated_mf) != len(prediction_coordinates):
        raise ValueError("Simulated MF and prediction coordinate rows must match.")
    if paired_mf.shape != landmarks.values.shape:
        raise ValueError("Paired MF shape must match the landmark value shape.")
    if simulated_mf.shape[1] != landmarks.variable_count:
        raise ValueError("Simulated MF columns must match the dataset variables.")
    if neighbors < landmarks.variable_count + 2:
        raise ValueError("TPS neighbors must exceed the variable count plus one.")
    if neighbors > len(landmarks.coordinates):
        raise ValueError("TPS neighbors cannot exceed the landmark count.")

