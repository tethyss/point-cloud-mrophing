"""High-level orchestration of the three-stage SMMT workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from .data import (
    default_benchmark_input,
    load_benchmark_dataset,
    load_csv_dataset,
    select_landmarks,
)
from .diagnostics import DiagnosticWriter
from .gslib import SgsimRunner
from .mapping import map_to_original_space
from .models import Dataset, PipelineConfig, VariogramModel
from .pairing import empirical_cdf, paired_morphing_factors
from .variogram import experimental_variogram, fit_variogram_models

LOGGER = logging.getLogger(__name__)


class SmmtPipeline:
    """Execute pairing, spatial simulation, and inverse morphing."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.generator = np.random.default_rng(config.seed)

    def run(self) -> list[Path]:
        """Run the configured workflow and return generated realization paths."""

        dataset = self._load_dataset()
        landmarks = self._select_landmarks(dataset)
        neighbors = self._resolve_neighbors(landmarks)
        output_dir, work_dir = self._prepare_output_directories()
        runner = SgsimRunner(
            executable=self.config.gslib_dir / "sgsim.exe",
            work_dir=work_dir,
            timeout=self.config.sgsim_timeout,
        )
        diagnostics = (
            DiagnosticWriter(
                output_dir=output_dir / "diagnostics",
                grid=dataset.grid,
                variable_names=dataset.variable_names,
            )
            if self.config.diagnostics
            else None
        )

        LOGGER.info(
            "Dataset: %s input samples, %s landmarks, %s variables, grid %sx%s.",
            len(dataset.coordinates),
            len(landmarks.coordinates),
            dataset.variable_count,
            dataset.grid.nx,
            dataset.grid.ny,
        )
        if diagnostics is not None:
            diagnostics.input_and_landmarks(dataset, landmarks)

        target_cdf, pairing_mf, lags, variograms, models = (
            self._prepare_morphing_models(
                dataset,
                landmarks,
            )
        )
        if diagnostics is not None:
            diagnostics.optimal_transport(target_cdf, pairing_mf[0])
            diagnostics.average_morphing_factors(
                landmarks.coordinates,
                pairing_mf.mean(axis=0),
            )
            diagnostics.variograms(lags, variograms, models)

        self._save_supporting_outputs(
            output_dir=output_dir,
            dataset=dataset,
            landmarks=landmarks,
            target_cdf=target_cdf,
            pairing_mf=pairing_mf,
            lags=lags,
            variograms=variograms,
            models=models,
        )
        return self._simulate_realizations(
            dataset=dataset,
            landmarks=landmarks,
            pairing_mf=pairing_mf,
            models=models,
            neighbors=neighbors,
            runner=runner,
            output_dir=output_dir,
            diagnostics=diagnostics,
            lags=lags,
            variograms=variograms,
        )

    def _load_dataset(self) -> Dataset:
        if self.config.dataset_kind == "benchmark":
            path = self.config.input_path or default_benchmark_input()
            return load_benchmark_dataset(path)
        return load_csv_dataset(
            path=self.config.input_path or Path("data.csv"),
            variables=self.config.variables,
            x_column=self.config.x_column,
            y_column=self.config.y_column,
        )

    def _select_landmarks(self, dataset: Dataset) -> Dataset:
        if self.config.landmarks is None:
            count = (
                len(dataset.coordinates)
                if self.config.dataset_kind == "benchmark"
                else min(200, len(dataset.coordinates))
            )
        else:
            count = self.config.landmarks
        return select_landmarks(dataset, count, self.generator)

    def _resolve_neighbors(self, landmarks: Dataset) -> int:
        minimum_neighbors = landmarks.variable_count + 2
        neighbors = self.config.neighbors or max(18, minimum_neighbors)
        if neighbors < minimum_neighbors:
            raise ValueError(
                f"TPS requires at least {minimum_neighbors} neighbors "
                f"for {landmarks.variable_count} variables."
            )
        if neighbors > len(landmarks.coordinates):
            raise ValueError("TPS neighbors cannot exceed the landmark count.")
        return neighbors

    def _prepare_output_directories(self) -> tuple[Path, Path]:
        output_dir = self.config.output_dir.resolve()
        work_dir = output_dir / "gslib"
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, work_dir

    def _prepare_morphing_models(
        self,
        dataset: Dataset,
        landmarks: Dataset,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[VariogramModel],
    ]:
        target_cdf = empirical_cdf(landmarks.values)
        pairing_mf = np.stack(
            [
                paired_morphing_factors(target_cdf, self.generator)
                for _ in trange(self.config.pairings, desc="OT pairings")
            ],
            axis=0,
        )

        average_mf = pairing_mf.mean(axis=0)
        lag = self._resolve_lag(dataset)
        lags, variograms = experimental_variogram(
            landmarks.coordinates,
            average_mf,
            lag,
            self.config.nlag,
        )
        models = fit_variogram_models(lags, variograms)
        LOGGER.info("Fitted %s morphing-factor variogram models.", len(models))
        return target_cdf, pairing_mf, lags, variograms, models

    def _resolve_lag(self, dataset: Dataset) -> float:
        if self.config.lag is not None:
            return self.config.lag
        grid = dataset.grid
        diagonal = np.hypot(
            (grid.nx - 1) * grid.x_size,
            (grid.ny - 1) * grid.y_size,
        )
        return max(
            grid.x_size,
            grid.y_size,
            float(diagonal) / (2 * self.config.nlag),
        )

    def _save_supporting_outputs(
        self,
        output_dir: Path,
        dataset: Dataset,
        landmarks: Dataset,
        target_cdf: np.ndarray,
        pairing_mf: np.ndarray,
        lags: np.ndarray,
        variograms: np.ndarray,
        models: list[VariogramModel],
    ) -> None:
        average_mf = pairing_mf.mean(axis=0)
        np.savez_compressed(
            output_dir / "pairings_and_variograms.npz",
            pairing_mf=pairing_mf,
            average_mf=average_mf,
            landmark_coordinates=landmarks.coordinates,
            landmark_values=landmarks.values,
            target_cdf=target_cdf,
            lags=lags,
            variograms=variograms,
        )
        save_metadata(
            output_dir / "metadata.json",
            self.config,
            dataset,
            models,
            landmark_count=len(landmarks.coordinates),
        )

    def _simulate_realizations(
        self,
        dataset: Dataset,
        landmarks: Dataset,
        pairing_mf: np.ndarray,
        models: list[VariogramModel],
        neighbors: int,
        runner: SgsimRunner,
        output_dir: Path,
        diagnostics: DiagnosticWriter | None,
        lags: np.ndarray,
        variograms: np.ndarray,
    ) -> list[Path]:
        outputs: list[Path] = []
        prediction_coordinates = dataset.grid.coordinates

        for realization_index in trange(
            self.config.realizations,
            desc="SMMT realizations",
        ):
            paired_mf = pairing_mf[realization_index]
            simulated_mf = runner.simulate(
                realization_index=realization_index,
                coordinates=landmarks.coordinates,
                morphing_factors=paired_mf,
                grid=dataset.grid,
                models=models,
                generator=self.generator,
            )
            if diagnostics is not None:
                diagnostics.simulated_morphing_factors(
                    realization_index,
                    prediction_coordinates,
                    simulated_mf,
                    landmarks,
                )
            output_path = output_dir / f"realization_{realization_index:03d}.npz"
            simulated_values = self._save_realization(
                output_path=output_path,
                prediction_coordinates=prediction_coordinates,
                simulated_mf=simulated_mf,
                landmarks=landmarks,
                paired_mf=paired_mf,
                neighbors=neighbors,
                variable_names=dataset.variable_names,
            )
            if diagnostics is not None and simulated_values is not None:
                diagnostics.mapped_realization(
                    realization_index,
                    prediction_coordinates,
                    simulated_values,
                    simulated_mf,
                    landmarks,
                    lags,
                    variograms,
                    models,
                )
            outputs.append(output_path)
            LOGGER.info("Saved realization %03d to %s.", realization_index, output_path)
        return outputs

    def _save_realization(
        self,
        output_path: Path,
        prediction_coordinates: np.ndarray,
        simulated_mf: np.ndarray,
        landmarks: Dataset,
        paired_mf: np.ndarray,
        neighbors: int,
        variable_names: list[str],
    ) -> np.ndarray | None:
        if self.config.skip_mapping:
            np.savez_compressed(
                output_path,
                coordinates=prediction_coordinates,
                morphing_factors=simulated_mf,
            )
            return None

        simulated_values = map_to_original_space(
            simulated_mf=simulated_mf,
            prediction_coordinates=prediction_coordinates,
            landmarks=landmarks,
            paired_mf=paired_mf,
            neighbors=neighbors,
            marginal_correction=self.config.marginal_correction,
        )
        np.savez_compressed(
            output_path,
            coordinates=prediction_coordinates,
            values=simulated_values,
            variable_names=np.asarray(variable_names),
        )
        return simulated_values


def save_metadata(
    path: Path,
    config: PipelineConfig,
    dataset: Dataset,
    models: list[VariogramModel],
    landmark_count: int | None = None,
) -> None:
    """Persist configuration and fitted models as human-readable JSON."""

    metadata = {
        "configuration": _json_safe(asdict(config)),
        "variables": dataset.variable_names,
        "grid": asdict(dataset.grid),
        "input_sample_count": len(dataset.coordinates),
        "landmark_count": landmark_count or len(dataset.coordinates),
        "variogram_models": [asdict(model) for model in models],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
