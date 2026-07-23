"""Local thin-plate-spline mapping from MF space to original variables."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.special import expit
from scipy.stats import norm

from .models import Dataset
from .pairing import empirical_cdf

PROBABILITY_EPSILON = 1.0e-6
TPS_REGULARIZATION = 1.0e-8


class LocalThinPlateSpline:
    """Thin-plate-spline map fitted to a local set of control points."""

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        regularization: float = TPS_REGULARIZATION,
    ) -> None:
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("TPS source and target arrays must be two-dimensional.")
        if len(source) != len(target):
            raise ValueError("TPS source and target row counts must match.")
        if len(source) <= source.shape[1] + 1:
            raise ValueError(
                "TPS requires more control points than source dimensions plus one."
            )
        if regularization < 0:
            raise ValueError("TPS regularization must be non-negative.")

        self.source = source
        kernel = distance.cdist(source, source)
        kernel.flat[:: len(source) + 1] += regularization
        affine_design = np.column_stack((np.ones(len(source)), source))
        affine_size = source.shape[1] + 1
        system = np.block(
            [
                [kernel, affine_design],
                [
                    affine_design.T,
                    np.zeros((affine_size, affine_size)),
                ],
            ]
        )
        right_side = np.vstack(
            (
                target,
                np.zeros((affine_size, target.shape[1])),
            )
        )
        try:
            self.parameters = np.linalg.solve(system, right_side)
        except np.linalg.LinAlgError:
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
    plotting_positions = np.arange(1, len(raw_values) + 1) / (len(raw_values) + 1)
    for variable_index in range(raw_values.shape[1]):
        sorted_values = np.sort(raw_values[:, variable_index])
        output[:, variable_index] = np.interp(
            probabilities[:, variable_index],
            plotting_positions,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    return output


def map_to_original_space(
    simulated_mf: np.ndarray,
    prediction_coordinates: np.ndarray,
    landmarks: Dataset,
    paired_mf: np.ndarray,
    neighbors: int,
    marginal_correction: bool = True,
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
    for landmark_indices, grid_indices in _iter_neighbor_groups(neighbor_indices):
        mapper = LocalThinPlateSpline(
            source_landmarks[landmark_indices],
            target_landmarks[landmark_indices],
        )
        mapped_cdf[grid_indices] = mapper.predict(query_points[grid_indices])

    mapped_probabilities = np.clip(
        expit(mapped_cdf),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    if marginal_correction:
        mapped_probabilities = empirical_cdf(mapped_probabilities)
    mapped_values = inverse_empirical_cdf(mapped_probabilities, landmarks.values)
    _restore_conditioning_values(
        mapped_values,
        prediction_coordinates,
        landmarks,
    )
    return mapped_values


def _find_spatial_neighbors(
    prediction_coordinates: np.ndarray,
    landmark_coordinates: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    tree = cKDTree(landmark_coordinates)
    _, indices = tree.query(prediction_coordinates, k=neighbors)
    return indices[:, None] if neighbors == 1 else indices


def _iter_neighbor_groups(
    neighbor_indices: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    unique_neighbors, inverse = np.unique(
        neighbor_indices,
        axis=0,
        return_inverse=True,
    )
    order = np.argsort(inverse, kind="stable")
    group_sizes = np.bincount(inverse, minlength=len(unique_neighbors))
    boundaries = np.concatenate(([0], np.cumsum(group_sizes)))
    for group_index, landmark_indices in enumerate(unique_neighbors):
        yield (
            landmark_indices,
            order[boundaries[group_index] : boundaries[group_index + 1]],
        )


def _restore_conditioning_values(
    mapped_values: np.ndarray,
    prediction_coordinates: np.ndarray,
    landmarks: Dataset,
) -> None:
    """Restore exact observed values at grid nodes containing landmarks."""

    coordinate_lookup = {
        tuple(coordinate): index
        for index, coordinate in enumerate(prediction_coordinates)
    }
    for landmark_coordinate, landmark_values in zip(
        landmarks.coordinates,
        landmarks.values,
    ):
        prediction_index = coordinate_lookup.get(tuple(landmark_coordinate))
        if prediction_index is not None:
            mapped_values[prediction_index] = landmark_values


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
