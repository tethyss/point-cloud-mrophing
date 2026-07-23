"""Lightweight visual diagnostics for every SMMT processing stage."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import norm

from .models import Dataset, Grid, VariogramModel
from .variogram import experimental_variogram, exponential_variogram

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)
PANELS_PER_FIGURE = 6


class DiagnosticWriter:
    """Write deterministic PNG summaries without displaying interactive windows."""

    def __init__(
        self,
        output_dir: Path,
        grid: Grid,
        variable_names: Sequence[str],
    ) -> None:
        self.output_dir = output_dir
        self.grid = grid
        self.variable_names = list(variable_names)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def input_and_landmarks(
        self,
        dataset: Dataset,
        landmarks: Dataset,
    ) -> list[Path]:
        """Visualize input variables and selected conditioning locations."""

        return self._spatial_fields(
            stage="01_input",
            coordinates=dataset.coordinates,
            values=dataset.values,
            landmarks=landmarks,
            color_label="Input value",
        )

    def optimal_transport(
        self,
        target_cdf: np.ndarray,
        paired_mf: np.ndarray,
    ) -> Path:
        """Visualize the first two dimensions of one OT pairing."""

        source_cdf = norm.cdf(paired_mf)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        if target_cdf.shape[1] >= 2:
            sample_indices = np.linspace(
                0,
                len(target_cdf) - 1,
                min(80, len(target_cdf)),
                dtype=int,
            )
            for sample_index in sample_indices:
                axes[0].plot(
                    [source_cdf[sample_index, 0], target_cdf[sample_index, 0]],
                    [source_cdf[sample_index, 1], target_cdf[sample_index, 1]],
                    color="0.75",
                    linewidth=0.5,
                    zorder=1,
                )
            axes[0].scatter(
                source_cdf[:, 0],
                source_cdf[:, 1],
                s=14,
                label="Paired MF",
                alpha=0.75,
                zorder=2,
            )
            axes[0].scatter(
                target_cdf[:, 0],
                target_cdf[:, 1],
                s=14,
                label="Target",
                alpha=0.75,
                zorder=3,
            )
            axes[0].set_xlabel(self.variable_names[0])
            axes[0].set_ylabel(self.variable_names[1])
        else:
            axes[0].scatter(
                source_cdf[:, 0],
                target_cdf[:, 0],
                s=16,
                alpha=0.8,
            )
            axes[0].set_xlabel("Paired MF CDF")
            axes[0].set_ylabel("Target CDF")
        axes[0].set_title("OT pairing in CDF space")
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.2)

        pairing_error = np.linalg.norm(source_cdf - target_cdf, axis=1)
        axes[1].hist(pairing_error, bins=min(30, max(5, len(pairing_error) // 5)))
        axes[1].set_title("Pairing distance")
        axes[1].set_xlabel("Euclidean CDF distance")
        axes[1].set_ylabel("Count")
        axes[1].grid(alpha=0.2)

        path = self.output_dir / "02_optimal_transport.png"
        self._save(figure, path)
        return path

    def average_morphing_factors(
        self,
        landmark_coordinates: np.ndarray,
        average_mf: np.ndarray,
    ) -> list[Path]:
        """Visualize spatial continuity of centroid morphing factors."""

        return self._spatial_fields(
            stage="03_average_mf",
            coordinates=landmark_coordinates,
            values=average_mf,
            landmarks=None,
            color_label="Average MF",
        )

    def variograms(
        self,
        lags: np.ndarray,
        variograms: np.ndarray,
        models: Sequence[VariogramModel],
    ) -> list[Path]:
        """Plot experimental direct variograms and fitted models."""

        paths: list[Path] = []
        model_lags = np.linspace(0, float(lags[-1]), 300)
        for batch_index, variable_indices in enumerate(
            _batches(range(len(self.variable_names))),
            start=1,
        ):
            figure, axes = _panel_figure(len(variable_indices))
            for axis, variable_index in zip(axes, variable_indices):
                axis.scatter(
                    lags,
                    variograms[variable_index],
                    s=20,
                    label="Experimental",
                )
                axis.plot(
                    model_lags,
                    exponential_variogram(
                        model_lags,
                        models[variable_index].nugget,
                        models[variable_index].partial_sill,
                        models[variable_index].range_,
                    ),
                    color="black",
                    linewidth=1.2,
                    label="Fitted",
                )
                axis.set_title(self.variable_names[variable_index])
                axis.set_xlabel("Lag distance")
                axis.set_ylabel("Semivariance")
                axis.grid(alpha=0.2)
            axes[0].legend(loc="best")
            path = self.output_dir / f"04_variograms_{batch_index:02d}.png"
            self._save(figure, path)
            paths.append(path)
        return paths

    def simulated_morphing_factors(
        self,
        realization_index: int,
        coordinates: np.ndarray,
        simulated_mf: np.ndarray,
        landmarks: Dataset,
    ) -> list[Path]:
        """Visualize one exhaustive Gaussian-space realization."""

        return self._spatial_fields(
            stage=f"05_mf_realization_{realization_index:03d}",
            coordinates=coordinates,
            values=simulated_mf,
            landmarks=landmarks,
            color_label="Simulated MF",
        )

    def mapped_realization(
        self,
        realization_index: int,
        coordinates: np.ndarray,
        mapped_values: np.ndarray,
        simulated_mf: np.ndarray,
        landmarks: Dataset,
        lags: np.ndarray,
        mf_variograms: np.ndarray,
        models: Sequence[VariogramModel],
    ) -> list[Path]:
        """Visualize mapped attributes and marginal reproduction."""

        paths = self._spatial_fields(
            stage=f"06_mapped_realization_{realization_index:03d}",
            coordinates=coordinates,
            values=mapped_values,
            landmarks=landmarks,
            color_label="Mapped value",
        )
        paths.extend(
            self._marginal_distributions(
                realization_index,
                mapped_values,
                landmarks.values,
            )
        )
        paths.append(
            self._correlation_comparison(
                realization_index,
                mapped_values,
                landmarks.values,
            )
        )
        simulated_mf_variograms = _grid_direct_variograms(
            simulated_mf,
            self.grid,
            lags,
        )
        mapped_variograms = _grid_direct_variograms(
            mapped_values,
            self.grid,
            lags,
        )
        _, landmark_variograms = experimental_variogram(
            landmarks.coordinates,
            landmarks.values,
            lag=float(lags[0]),
            nlag=len(lags),
        )
        paths.extend(
            self._variogram_reproduction(
                stage=f"09_mf_variogram_reproduction_{realization_index:03d}",
                lags=lags,
                reference=mf_variograms,
                simulated=simulated_mf_variograms,
                models=models,
                reference_label="Average MF",
            )
        )
        paths.extend(
            self._variogram_reproduction(
                stage=f"10_mapped_variogram_reproduction_{realization_index:03d}",
                lags=lags,
                reference=landmark_variograms,
                simulated=mapped_variograms,
                models=None,
                reference_label="Landmarks",
            )
        )
        self._quality_report(
            realization_index,
            simulated_mf,
            mapped_values,
            landmarks.values,
            lags,
            simulated_mf_variograms,
            mapped_variograms,
            landmark_variograms,
            models,
        )
        return paths

    def _spatial_fields(
        self,
        stage: str,
        coordinates: np.ndarray,
        values: np.ndarray,
        landmarks: Dataset | None,
        color_label: str,
    ) -> list[Path]:
        paths: list[Path] = []
        for batch_index, variable_indices in enumerate(
            _batches(range(values.shape[1])),
            start=1,
        ):
            figure, axes = _panel_figure(len(variable_indices))
            for axis, variable_index in zip(axes, variable_indices):
                if len(coordinates) == self.grid.node_count:
                    field = _values_on_grid(
                        coordinates,
                        values[:, variable_index],
                        self.grid,
                    )
                    artist = axis.imshow(
                        field,
                        origin="lower",
                        extent=_grid_extent(self.grid),
                        aspect="auto",
                        cmap="viridis",
                    )
                else:
                    artist = axis.scatter(
                        coordinates[:, 0],
                        coordinates[:, 1],
                        c=values[:, variable_index],
                        s=18,
                        cmap="viridis",
                    )
                if landmarks is not None:
                    axis.scatter(
                        landmarks.coordinates[:, 0],
                        landmarks.coordinates[:, 1],
                        facecolors="none",
                        edgecolors="white",
                        linewidths=0.6,
                        s=24,
                    )
                axis.set_title(self.variable_names[variable_index])
                axis.set_xlabel("X")
                axis.set_ylabel("Y")
                figure.colorbar(artist, ax=axis, shrink=0.78, label=color_label)
            path = self.output_dir / f"{stage}_{batch_index:02d}.png"
            self._save(figure, path)
            paths.append(path)
        return paths

    def _marginal_distributions(
        self,
        realization_index: int,
        mapped_values: np.ndarray,
        landmark_values: np.ndarray,
    ) -> list[Path]:
        paths: list[Path] = []
        for batch_index, variable_indices in enumerate(
            _batches(range(mapped_values.shape[1])),
            start=1,
        ):
            figure, axes = _panel_figure(len(variable_indices))
            for axis, variable_index in zip(axes, variable_indices):
                axis.hist(
                    mapped_values[:, variable_index],
                    bins=35,
                    density=True,
                    alpha=0.55,
                    label="Mapped",
                )
                axis.hist(
                    landmark_values[:, variable_index],
                    bins=20,
                    density=True,
                    histtype="step",
                    linewidth=1.4,
                    label="Landmarks",
                )
                axis.set_title(self.variable_names[variable_index])
                axis.set_xlabel("Value")
                axis.set_ylabel("Density")
                axis.grid(alpha=0.2)
            axes[0].legend(loc="best")
            path = self.output_dir / (
                f"07_marginals_{realization_index:03d}_{batch_index:02d}.png"
            )
            self._save(figure, path)
            paths.append(path)
        return paths

    def _correlation_comparison(
        self,
        realization_index: int,
        mapped_values: np.ndarray,
        landmark_values: np.ndarray,
    ) -> Path:
        landmark_correlation = _correlation_matrix(landmark_values)
        mapped_correlation = _correlation_matrix(mapped_values)
        difference = mapped_correlation - landmark_correlation
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        matrices = (
            (landmark_correlation, "Landmark correlation", -1.0, 1.0),
            (mapped_correlation, "Mapped correlation", -1.0, 1.0),
            (difference, "Mapped - landmark", -0.5, 0.5),
        )
        for axis, (matrix, title, lower, upper) in zip(axes, matrices):
            artist = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=lower,
                vmax=upper,
            )
            axis.set_title(title)
            axis.set_xticks(range(len(self.variable_names)))
            axis.set_yticks(range(len(self.variable_names)))
            axis.set_xticklabels(self.variable_names, rotation=90)
            axis.set_yticklabels(self.variable_names)
            figure.colorbar(artist, ax=axis, shrink=0.8)
        path = self.output_dir / (
            f"08_correlations_{realization_index:03d}.png"
        )
        self._save(figure, path)
        return path

    def _quality_report(
        self,
        realization_index: int,
        simulated_mf: np.ndarray,
        mapped_values: np.ndarray,
        landmark_values: np.ndarray,
        lags: np.ndarray,
        simulated_mf_variograms: np.ndarray,
        mapped_variograms: np.ndarray,
        landmark_variograms: np.ndarray,
        models: Sequence[VariogramModel],
    ) -> Path:
        mf_correlation = _correlation_matrix(simulated_mf)
        off_diagonal = mf_correlation - np.eye(mf_correlation.shape[0])
        landmark_correlation = _correlation_matrix(landmark_values)
        mapped_correlation = _correlation_matrix(mapped_values)
        fitted_mf_variograms = np.asarray(
            [
                exponential_variogram(
                    lags,
                    model.nugget,
                    model.partial_sill,
                    model.range_,
                )
                for model in models
            ]
        )
        report = {
            "realization": realization_index,
            "variables": self.variable_names,
            "morphing_factors": {
                "mean": simulated_mf.mean(axis=0).tolist(),
                "standard_deviation": simulated_mf.std(axis=0).tolist(),
                "maximum_absolute_cross_correlation": float(
                    np.max(np.abs(off_diagonal))
                ),
            },
            "mapped_values": {
                "mean": mapped_values.mean(axis=0).tolist(),
                "standard_deviation": mapped_values.std(axis=0).tolist(),
                "minimum": mapped_values.min(axis=0).tolist(),
                "maximum": mapped_values.max(axis=0).tolist(),
            },
            "landmarks": {
                "mean": landmark_values.mean(axis=0).tolist(),
                "standard_deviation": landmark_values.std(axis=0).tolist(),
                "minimum": landmark_values.min(axis=0).tolist(),
                "maximum": landmark_values.max(axis=0).tolist(),
            },
            "correlation_rmse": float(
                np.sqrt(
                    np.mean(
                        (mapped_correlation - landmark_correlation) ** 2
                    )
                )
            ),
            "morphing_factor_variogram_rmse": _rowwise_rmse(
                simulated_mf_variograms,
                fitted_mf_variograms,
            ).tolist(),
            "mapped_variogram_rmse": _rowwise_rmse(
                mapped_variograms,
                landmark_variograms,
            ).tolist(),
        }
        path = self.output_dir / (
            f"quality_metrics_{realization_index:03d}.json"
        )
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Saved diagnostic metrics: %s", path)
        return path

    def _variogram_reproduction(
        self,
        stage: str,
        lags: np.ndarray,
        reference: np.ndarray,
        simulated: np.ndarray,
        models: Sequence[VariogramModel] | None,
        reference_label: str,
    ) -> list[Path]:
        paths: list[Path] = []
        for batch_index, variable_indices in enumerate(
            _batches(range(len(self.variable_names))),
            start=1,
        ):
            figure, axes = _panel_figure(len(variable_indices))
            for axis, variable_index in zip(axes, variable_indices):
                axis.plot(
                    lags,
                    reference[variable_index],
                    marker="o",
                    markersize=3,
                    linewidth=1.0,
                    label=reference_label,
                )
                axis.plot(
                    lags,
                    simulated[variable_index],
                    linewidth=1.2,
                    label="Realization",
                )
                if models is not None:
                    model = models[variable_index]
                    axis.plot(
                        lags,
                        exponential_variogram(
                            lags,
                            model.nugget,
                            model.partial_sill,
                            model.range_,
                        ),
                        color="black",
                        linewidth=1.0,
                        label="Fitted model",
                    )
                axis.set_title(self.variable_names[variable_index])
                axis.set_xlabel("Lag distance")
                axis.set_ylabel("Semivariance")
                axis.grid(alpha=0.2)
            axes[0].legend(loc="best")
            path = self.output_dir / f"{stage}_{batch_index:02d}.png"
            self._save(figure, path)
            paths.append(path)
        return paths

    @staticmethod
    def _save(figure: plt.Figure, path: Path) -> None:
        figure.tight_layout()
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        LOGGER.info("Saved diagnostic: %s", path)


def _panel_figure(panel_count: int) -> tuple[plt.Figure, list[plt.Axes]]:
    columns = min(3, panel_count)
    rows = math.ceil(panel_count / columns)
    figure, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 4.1 * rows),
        squeeze=False,
    )
    axes = list(axes_array.ravel())
    for axis in axes[panel_count:]:
        axis.set_visible(False)
    return figure, axes[:panel_count]


def _batches(indices: Sequence[int] | range) -> list[list[int]]:
    values = list(indices)
    return [
        values[start : start + PANELS_PER_FIGURE]
        for start in range(0, len(values), PANELS_PER_FIGURE)
    ]


def _values_on_grid(
    coordinates: np.ndarray,
    values: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    x_indices = np.rint((coordinates[:, 0] - grid.x_min) / grid.x_size).astype(int)
    y_indices = np.rint((coordinates[:, 1] - grid.y_min) / grid.y_size).astype(int)
    field = np.full((grid.ny, grid.nx), np.nan)
    field[y_indices, x_indices] = values
    return field


def _grid_extent(grid: Grid) -> tuple[float, float, float, float]:
    return (
        grid.x_min - grid.x_size / 2,
        grid.x_min + (grid.nx - 0.5) * grid.x_size,
        grid.y_min - grid.y_size / 2,
        grid.y_min + (grid.ny - 0.5) * grid.y_size,
    )


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    correlation = np.atleast_2d(np.corrcoef(values, rowvar=False))
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _grid_direct_variograms(
    values: np.ndarray,
    grid: Grid,
    lags: np.ndarray,
) -> np.ndarray:
    fields = values.reshape(grid.ny, grid.nx, values.shape[1])
    variograms = np.full((values.shape[1], len(lags)), np.nan)
    for lag_index, lag in enumerate(lags):
        squared_differences: list[np.ndarray] = []
        x_offset = int(round(float(lag) / grid.x_size))
        y_offset = int(round(float(lag) / grid.y_size))
        if 0 < x_offset < grid.nx:
            squared_differences.append(
                (fields[:, x_offset:] - fields[:, :-x_offset]).reshape(
                    -1,
                    values.shape[1],
                )
                ** 2
            )
        if 0 < y_offset < grid.ny:
            squared_differences.append(
                (fields[y_offset:] - fields[:-y_offset]).reshape(
                    -1,
                    values.shape[1],
                )
                ** 2
            )
        if squared_differences:
            combined = np.vstack(squared_differences)
            variograms[:, lag_index] = 0.5 * np.mean(combined, axis=0)
    return variograms


def _rowwise_rmse(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    valid = np.isfinite(observed) & np.isfinite(reference)
    squared_error = np.where(valid, (observed - reference) ** 2, np.nan)
    return np.sqrt(np.nanmean(squared_error, axis=1))
