"""Regression tests using only the local 200 x 200 benchmark data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from point_cloud_morphing.data import load_benchmark_dataset
from point_cloud_morphing.mapping import map_to_original_space
from point_cloud_morphing.models import PipelineConfig
from point_cloud_morphing.pairing import empirical_cdf, paired_morphing_factors
from point_cloud_morphing.pipeline import SmmtPipeline
from point_cloud_morphing.variogram import (
    experimental_variogram,
    fit_variogram_models,
)


def test_benchmark_pairing_and_variogram(benchmark_data_path: Path) -> None:
    dataset = load_benchmark_dataset(benchmark_data_path)
    generator = np.random.default_rng(20260713)
    target_cdf = empirical_cdf(dataset.values)
    pairings = np.stack(
        [
            paired_morphing_factors(target_cdf, generator)
            for _ in range(4)
        ],
        axis=0,
    )

    assert dataset.coordinates.shape == (199, 2)
    assert dataset.values.shape == (199, 6)
    assert dataset.grid.node_count == 40_000
    assert pairings.shape == (4, 199, 6)
    assert len(np.unique(pairings[0], axis=0)) == len(dataset.coordinates)

    average_mf = pairings.mean(axis=0)
    lags, variograms = experimental_variogram(
        dataset.coordinates,
        average_mf,
        lag=4.0,
        nlag=30,
    )
    models = fit_variogram_models(lags, variograms)

    assert variograms.shape == (6, 30)
    assert len(models) == 6
    assert all(model.range_ > 0 for model in models)


def test_mapping_restores_benchmark_conditioning_values(
    benchmark_data_path: Path,
) -> None:
    dataset = load_benchmark_dataset(benchmark_data_path)
    generator = np.random.default_rng(20260713)
    paired_mf = paired_morphing_factors(
        empirical_cdf(dataset.values),
        generator,
    )

    mapped = map_to_original_space(
        simulated_mf=paired_mf,
        prediction_coordinates=dataset.coordinates,
        landmarks=dataset,
        paired_mf=paired_mf,
        neighbors=18,
    )

    np.testing.assert_allclose(mapped, dataset.values, rtol=0.0, atol=0.0)


def test_complete_benchmark_pipeline(
    benchmark_data_path: Path,
    benchmark_sgsim_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "benchmark_run"
    config = PipelineConfig(
        dataset_kind="benchmark",
        input_path=benchmark_data_path,
        landmarks=None,
        pairings=1,
        realizations=1,
        neighbors=18,
        lag=4.0,
        nlag=30,
        seed=20260713,
        gslib_dir=benchmark_sgsim_dir,
        output_dir=output_dir,
        diagnostics=True,
        sgsim_timeout=60.0,
    )

    outputs = SmmtPipeline(config).run()

    assert outputs == [output_dir / "realization_000.npz"]
    with np.load(outputs[0]) as realization:
        assert realization["coordinates"].shape == (40_000, 2)
        assert realization["values"].shape == (40_000, 6)
        assert np.isfinite(realization["values"]).all()

        coordinate_lookup = {
            tuple(coordinate): index
            for index, coordinate in enumerate(realization["coordinates"])
        }
        benchmark = load_benchmark_dataset(benchmark_data_path)
        mapped_at_landmarks = np.asarray(
            [
                realization["values"][coordinate_lookup[tuple(coordinate)]]
                for coordinate in benchmark.coordinates
            ]
        )
        np.testing.assert_allclose(
            mapped_at_landmarks,
            benchmark.values,
            rtol=0.0,
            atol=0.0,
        )
        relative_standard_deviation_error = np.abs(
            realization["values"].std(axis=0) - benchmark.values.std(axis=0)
        ) / benchmark.values.std(axis=0)
        assert np.all(relative_standard_deviation_error < 0.25)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["landmark_count"] == 199
    assert metadata["grid"]["nx"] == 200
    assert (output_dir / "diagnostics" / "02_optimal_transport.png").is_file()
    assert (output_dir / "diagnostics" / "04_variograms_01.png").is_file()
    assert (
        output_dir / "diagnostics" / "06_mapped_realization_000_01.png"
    ).is_file()
    assert (
        output_dir / "diagnostics" / "08_correlations_000.png"
    ).is_file()
    assert (
        output_dir
        / "diagnostics"
        / "09_mf_variogram_reproduction_000_01.png"
    ).is_file()
    assert (
        output_dir
        / "diagnostics"
        / "10_mapped_variogram_reproduction_000_01.png"
    ).is_file()
    assert (
        output_dir / "diagnostics" / "quality_metrics_000.json"
    ).is_file()
    assert (output_dir / "gslib" / "sgsim_0_0.log").is_file()
