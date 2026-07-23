"""Domain models shared by the SMMT workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Definition of a regular two-dimensional simulation grid."""

    nx: int
    ny: int
    x_min: float
    y_min: float
    x_size: float
    y_size: float

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("Grid dimensions must be positive.")
        if self.x_size <= 0 or self.y_size <= 0:
            raise ValueError("Grid cell sizes must be positive.")

    @property
    def node_count(self) -> int:
        """Return the total number of grid nodes."""

        return self.nx * self.ny

    @property
    def coordinates(self) -> np.ndarray:
        """Return grid coordinates in GSLIB-compatible row-major order."""

        x_values = self.x_min + self.x_size * np.arange(self.nx)
        y_values = self.y_min + self.y_size * np.arange(self.ny)
        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
        return np.column_stack((x_grid.ravel(), y_grid.ravel()))


@dataclass(frozen=True)
class Dataset:
    """Coordinates, variables, labels, and grid metadata for one dataset."""

    coordinates: np.ndarray
    values: np.ndarray
    variable_names: list[str]
    grid: Grid

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("Dataset coordinates must have shape (n_samples, 2).")
        if self.values.ndim != 2:
            raise ValueError("Dataset values must have shape (n_samples, n_variables).")
        if len(self.coordinates) != len(self.values):
            raise ValueError("Coordinate and value row counts must match.")
        if self.values.shape[1] != len(self.variable_names):
            raise ValueError("Variable names must match the value columns.")
        if not np.isfinite(self.coordinates).all() or not np.isfinite(self.values).all():
            raise ValueError("Dataset contains non-finite values.")

    @property
    def variable_count(self) -> int:
        """Return the number of simulated variables."""

        return self.values.shape[1]


@dataclass(frozen=True)
class VariogramModel:
    """Single exponential variogram model consumed by SGSIM."""

    nugget: float
    partial_sill: float
    range_: float

    def __post_init__(self) -> None:
        if self.nugget < 0 or self.partial_sill < 0:
            raise ValueError("Variogram sills must be non-negative.")
        if self.range_ <= 0:
            raise ValueError("Variogram range must be positive.")


@dataclass(frozen=True)
class PipelineConfig:
    """Validated runtime options for one SMMT execution."""

    dataset_kind: str = "csv"
    input_path: Path | None = None
    variables: str | None = None
    x_column: str | None = None
    y_column: str | None = None
    landmarks: int = 200
    pairings: int = 20
    realizations: int = 1
    neighbors: int | None = None
    lag: float | None = None
    nlag: int = 30
    seed: int = 20260713
    gslib_dir: Path = Path("gslibexe")
    output_dir: Path = Path("result/smmt")
    skip_mapping: bool = False

    def __post_init__(self) -> None:
        if self.dataset_kind not in {"csv", "benchmark"}:
            raise ValueError("Dataset kind must be 'csv' or 'benchmark'.")
        if self.landmarks < 3:
            raise ValueError("At least three landmarks are required.")
        if self.pairings < 1 or self.realizations < 1:
            raise ValueError("Pairings and realizations must be positive.")
        if self.nlag < 3:
            raise ValueError("At least three variogram lags are required.")
        if self.lag is not None and self.lag <= 0:
            raise ValueError("Variogram lag distance must be positive.")
