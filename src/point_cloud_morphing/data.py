"""Input loading, grid inference, and landmark sampling."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .models import Dataset, Grid

LOGGER = logging.getLogger(__name__)
DEFAULT_EXCLUDED_COLUMNS = frozenset({"dem", "slope", "granites"})


def default_benchmark_input() -> Path:
    """Locate the benchmark conditioning file in known project locations."""

    candidates = (
        Path("benchmark/bmdata/Conditioning_Data_6dim_Numpy.txt"),
        Path("benchmark/Conditioning_Data_6dim_Numpy.txt"),
        Path("2022-Morphing-master/Codes/Files/Conditioning_Data_6dim_Numpy.txt"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate Conditioning_Data_6dim_Numpy.txt; pass --input explicitly."
    )


def infer_grid(coordinates: np.ndarray) -> Grid:
    """Infer and validate a complete regular grid from coordinate pairs."""

    x_values = np.unique(coordinates[:, 0])
    y_values = np.unique(coordinates[:, 1])
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("At least two unique x and y coordinates are required.")

    x_diffs = np.diff(x_values)
    y_diffs = np.diff(y_values)
    x_size = float(np.median(x_diffs))
    y_size = float(np.median(y_diffs))
    if not np.allclose(x_diffs, x_size) or not np.allclose(y_diffs, y_size):
        raise ValueError("CSV input must define a regular grid.")

    expected_nodes = len(x_values) * len(y_values)
    unique_nodes = len(np.unique(coordinates, axis=0))
    if len(coordinates) != expected_nodes or unique_nodes != expected_nodes:
        raise ValueError("CSV input must contain each regular-grid node exactly once.")

    return Grid(
        nx=len(x_values),
        ny=len(y_values),
        x_min=float(x_values[0]),
        y_min=float(y_values[0]),
        x_size=x_size,
        y_size=y_size,
    )


def load_csv_dataset(
    path: Path,
    variables: str | None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> Dataset:
    """Load a complete regular-grid dataset from CSV."""

    frame = pd.read_csv(path)
    if frame.shape[1] < 3:
        raise ValueError("CSV input must include two coordinates and at least one variable.")

    coordinate_columns = [x_column or frame.columns[0], y_column or frame.columns[1]]
    if coordinate_columns[0] == coordinate_columns[1]:
        raise ValueError("X and Y coordinate columns must be different.")

    variable_names = _resolve_variable_names(
        frame,
        variables,
        coordinate_columns,
    )
    requested_columns = coordinate_columns + variable_names
    missing = set(requested_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")

    numeric = frame[requested_columns].apply(pd.to_numeric, errors="raise")
    matrix = numeric.to_numpy()
    if numeric.isna().any().any() or not np.isfinite(matrix).all():
        raise ValueError("Input contains missing or non-finite values.")

    coordinates = numeric[coordinate_columns].to_numpy(dtype=float)
    values = numeric[variable_names].to_numpy(dtype=float)
    return Dataset(
        coordinates=coordinates,
        values=values,
        variable_names=variable_names,
        grid=infer_grid(coordinates),
    )


def load_benchmark_dataset(path: Path) -> Dataset:
    """Load the original 200 x 200, six-variable benchmark conditioning data."""

    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError("Benchmark input must have x, y, and variable columns.")

    values = data[:, 2:]
    return Dataset(
        coordinates=data[:, :2],
        values=values,
        variable_names=[f"Z{index}" for index in range(1, values.shape[1] + 1)],
        grid=Grid(200, 200, 0.0, 0.0, 1.0, 1.0),
    )


def select_landmarks(
    dataset: Dataset,
    count: int,
    generator: np.random.Generator,
) -> Dataset:
    """Sample conditioning landmarks without replacement."""

    minimum_count = dataset.variable_count + 2
    if count < minimum_count:
        raise ValueError(
            f"Landmark count must be at least {minimum_count} for TPS."
        )
    if count > len(dataset.coordinates):
        raise ValueError("Landmark count exceeds the available input samples.")

    if count == len(dataset.coordinates):
        LOGGER.info("Using all %s input samples as landmarks.", count)
        return dataset

    indices = generator.choice(len(dataset.coordinates), size=count, replace=False)
    LOGGER.info("Selected %s landmarks from %s input nodes.", count, len(dataset.coordinates))
    return Dataset(
        coordinates=dataset.coordinates[indices],
        values=dataset.values[indices],
        variable_names=dataset.variable_names,
        grid=dataset.grid,
    )


def _resolve_variable_names(
    frame: pd.DataFrame,
    variables: str | None,
    coordinate_columns: list[str],
) -> list[str]:
    """Resolve explicit or default variable columns."""

    if variables:
        names = [name.strip() for name in variables.split(",") if name.strip()]
        if not names:
            raise ValueError("--variables must contain at least one column name.")
        return names
    return [
        name
        for name in frame.columns
        if name not in coordinate_columns
        and name.lower() not in DEFAULT_EXCLUDED_COLUMNS
    ]
