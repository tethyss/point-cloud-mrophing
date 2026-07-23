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
        landmarks = select_landmarks(
            dataset,
            self.config.landmarks,
            self.generator,
        )
        neighbors = self._resolve_neighbors(dataset)
        output_dir, work_dir = self._prepare_output_directories()
        runner = SgsimRunner(
            executable=self.config.gslib_dir / "sgsim.exe",
            work_dir=work_dir,
        )

        LOGGER.info(
            "Dataset: %s nodes, %s variables, grid %sx%s.",
            len(dataset.coordinates),
            dataset.variable_count,
            dataset.grid.nx,
            dataset.grid.ny,
        )
        pairing_mf, lags, variograms, models = self._prepare_morphing_models(
            dataset,
            landmarks,
        )
        self._save_supporting_outputs(
            output_dir=output_dir,
            dataset=dataset,
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

    def _resolve_neighbors(self, dataset: Dataset) -> int:
        minimum_neighbors = dataset.variable_count + 2
        neighbors = self.config.neighbors or max(18, minimum_neighbors)
        if neighbors < minimum_neighbors:
            raise ValueError(
                f"TPS requires at least {minimum_neighbors} neighbors "
                f"for {dataset.variable_count} variables."
            )
        if neighbors > self.config.landmarks:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariogramModel]]:
        target_cdf = empirical_cdf(landmarks.values)
        pairing_mf = np.stack(
            [
                paired_morphing_factors(target_cdf, self.generator)
                for _ in trange(self.config.pairings, desc="OT pairings")
            ],
            axis=0,
        )

        average_mf = pairing_mf.mean(axis=0)
        lag = self.config.lag or max(dataset.grid.x_size, dataset.grid.y_size)
        lags, variograms = experimental_variogram(
            landmarks.coordinates,
            average_mf,
            lag,
            self.config.nlag,
        )
        models = fit_variogram_models(lags, variograms)
        LOGGER.info("Fitted %s morphing-factor variogram models.", len(models))
        return pairing_mf, lags, variograms, models

    def _save_supporting_outputs(
        self,
        output_dir: Path,
        dataset: Dataset,
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
            lags=lags,
            variograms=variograms,
        )
        save_metadata(
            output_dir / "metadata.json",
            self.config,
            dataset,
            models,
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
    ) -> list[Path]:
        outputs: list[Path] = []
        prediction_coordinates = dataset.grid.coordinates

        for realization_index in trange(
            self.config.realizations,
            desc="SMMT realizations",
        ):
            paired_mf = pairing_mf[realization_index % self.config.pairings]
            simulated_mf = runner.simulate(
                realization_index=realization_index,
                coordinates=landmarks.coordinates,
                morphing_factors=paired_mf,
                grid=dataset.grid,
                models=models,
                generator=self.generator,
            )
            output_path = output_dir / f"realization_{realization_index:03d}.npz"
            self._save_realization(
                output_path=output_path,
                prediction_coordinates=prediction_coordinates,
                simulated_mf=simulated_mf,
                landmarks=landmarks,
                paired_mf=paired_mf,
                neighbors=neighbors,
                variable_names=dataset.variable_names,
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
    ) -> None:
        if self.config.skip_mapping:
            np.savez_compressed(
                output_path,
                coordinates=prediction_coordinates,
                morphing_factors=simulated_mf,
            )
            return

        simulated_values = map_to_original_space(
            simulated_mf=simulated_mf,
            prediction_coordinates=prediction_coordinates,
            landmarks=landmarks,
            paired_mf=paired_mf,
            neighbors=neighbors,
        )
        np.savez_compressed(
            output_path,
            coordinates=prediction_coordinates,
            values=simulated_values,
            variable_names=np.asarray(variable_names),
        )


def save_metadata(
    path: Path,
    config: PipelineConfig,
    dataset: Dataset,
    models: list[VariogramModel],
) -> None:
    """Persist configuration and fitted models as human-readable JSON."""

    metadata = {
        "configuration": _json_safe(asdict(config)),
        "variables": dataset.variable_names,
        "grid": asdict(dataset.grid),
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

