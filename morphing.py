"""Spatial Multivariate Morphing Transformation (SMMT).

This implementation follows Avalos and Ortiz's three-stage workflow:
optimal-transport pairing, independent conditional SGS of morphing factors,
and local thin-plate-spline mapping back to the original variable space.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import ot
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree, distance
from scipy.stats import norm, rankdata
from tqdm import trange


EPSILON = 1.0e-6
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    x_min: float
    y_min: float
    x_size: float
    y_size: float

    @property
    def coordinates(self) -> np.ndarray:
        x_values = self.x_min + self.x_size * np.arange(self.nx)
        y_values = self.y_min + self.y_size * np.arange(self.ny)
        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
        return np.column_stack((x_grid.ravel(), y_grid.ravel()))


@dataclass(frozen=True)
class Dataset:
    coordinates: np.ndarray
    values: np.ndarray
    variable_names: list[str]
    grid: Grid


@dataclass(frozen=True)
class VariogramModel:
    nugget: float
    partial_sill: float
    range_: float


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable SMMT workflow.")
    parser.add_argument("--dataset", choices=("csv", "benchmark"), default="csv")
    parser.add_argument("--input", type=Path, help="CSV input or benchmark conditioning-data file.")
    parser.add_argument("--variables", help="Comma-separated variable names for CSV input.")
    parser.add_argument("--x-column", help="X-coordinate column name for CSV input; defaults to the first column.")
    parser.add_argument("--y-column", help="Y-coordinate column name for CSV input; defaults to the second column.")
    parser.add_argument("--landmarks", type=int, default=200, help="Number of conditioning landmarks.")
    parser.add_argument("--pairings", type=int, default=20, help="Number of OT pairings used to estimate MF continuity.")
    parser.add_argument("--realizations", type=int, default=1, help="Number of SMMT realizations to generate.")
    parser.add_argument("--neighbors", type=int, help="Spatially closest landmarks used by local TPS.")
    parser.add_argument("--lag", type=float, help="Variogram lag distance; defaults to the largest grid spacing.")
    parser.add_argument("--nlag", type=int, default=30, help="Number of experimental variogram lags.")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--gslib-dir", type=Path, default=Path("gslibexe"))
    parser.add_argument("--output", type=Path, default=Path("result/smmt"))
    parser.add_argument("--skip-mapping", action="store_true", help="Save MF simulations without TPS back-mapping.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def default_benchmark_input() -> Path:
    candidates = (
        Path("benchmark/Conditioning_Data_6dim_Numpy.txt"),
        Path("2022-Morphing-master/Codes/Files/Conditioning_Data_6dim_Numpy.txt"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot locate Conditioning_Data_6dim_Numpy.txt; pass --input explicitly.")


def infer_grid(coordinates: np.ndarray) -> Grid:
    x_values = np.unique(coordinates[:, 0])
    y_values = np.unique(coordinates[:, 1])
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("At least two unique x and y coordinates are required to infer a grid.")
    x_diffs = np.diff(x_values)
    y_diffs = np.diff(y_values)
    x_size = float(np.median(x_diffs))
    y_size = float(np.median(y_diffs))
    if not np.allclose(x_diffs, x_size) or not np.allclose(y_diffs, y_size):
        raise ValueError("The input must define a regular grid.")
    expected_nodes = len(x_values) * len(y_values)
    if len(coordinates) != expected_nodes or len(np.unique(coordinates, axis=0)) != expected_nodes:
        raise ValueError("CSV input must contain each regular-grid node exactly once.")
    return Grid(len(x_values), len(y_values), float(x_values[0]), float(y_values[0]), x_size, y_size)


def load_csv_dataset(
    path: Path,
    variables: Optional[str],
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
) -> Dataset:
    frame = pd.read_csv(path)
    coordinate_columns = [x_column or frame.columns[0], y_column or frame.columns[1]]
    if coordinate_columns[0] == coordinate_columns[1]:
        raise ValueError("X and Y coordinate columns must be different.")
    if variables:
        variable_names = [name.strip() for name in variables.split(",") if name.strip()]
    else:
        excluded = {"dem", "slope", "granites"}
        variable_names = [name for name in frame.columns[2:] if name.lower() not in excluded]
    missing = set(coordinate_columns + variable_names).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    numeric = frame[coordinate_columns + variable_names].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Input contains missing or non-finite values.")
    coordinates = numeric[coordinate_columns].to_numpy(dtype=float)
    values = numeric[variable_names].to_numpy(dtype=float)
    return Dataset(coordinates, values, variable_names, infer_grid(coordinates))


def load_benchmark_dataset(path: Path) -> Dataset:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError("Benchmark input must have x, y, and at least one variable column.")
    coordinates = data[:, :2]
    values = data[:, 2:]
    if not np.isfinite(data).all():
        raise ValueError("Benchmark input contains non-finite values.")
    variable_names = [f"Z{index}" for index in range(1, values.shape[1] + 1)]
    return Dataset(coordinates, values, variable_names, Grid(200, 200, 0.0, 0.0, 1.0, 1.0))


def select_landmarks(dataset: Dataset, count: int, generator: np.random.Generator) -> Dataset:
    if count < dataset.values.shape[1] + 2:
        raise ValueError("Landmark count must exceed the number of variables plus one for TPS.")
    if count > len(dataset.coordinates):
        raise ValueError("Landmark count exceeds the available input samples.")
    indices = generator.choice(len(dataset.coordinates), size=count, replace=False)
    LOGGER.info("Selected %s landmarks from %s input nodes.", count, len(dataset.coordinates))
    return Dataset(dataset.coordinates[indices], dataset.values[indices], dataset.variable_names, dataset.grid)


def empirical_cdf(values: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(values[:, index], method="average") / (len(values) + 1) for index in range(values.shape[1])])


def paired_morphing_factors(target_cdf: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    source = generator.standard_normal(target_cdf.shape)
    source_cdf = norm.cdf(source)
    weights = np.full(len(target_cdf), 1.0 / len(target_cdf))
    cost = ot.dist(target_cdf, source_cdf, metric="euclidean")
    coupling = ot.emd(weights, weights, cost)
    assignment = coupling.argmax(axis=1)
    return source[assignment]


def experimental_variogram(coordinates: np.ndarray, values: np.ndarray, lag: float, nlag: int) -> tuple[np.ndarray, np.ndarray]:
    distances = distance.pdist(coordinates)
    if not np.any(distances > 0):
        raise ValueError("Landmarks must not all occupy the same location.")
    lag_centers = lag * np.arange(1, nlag + 1)
    direct_variograms = np.full((values.shape[1], nlag), np.nan)
    for variable_index in range(values.shape[1]):
        semivariance = 0.5 * distance.pdist(values[:, [variable_index]], metric="sqeuclidean")
        for lag_index, center in enumerate(lag_centers):
            mask = np.abs(distances - center) <= lag / 2
            if np.any(mask):
                direct_variograms[variable_index, lag_index] = np.mean(semivariance[mask])
    return lag_centers, direct_variograms


def exponential_variogram(distance_values: np.ndarray, nugget: float, partial_sill: float, range_: float) -> np.ndarray:
    return nugget + partial_sill * (1.0 - np.exp(-3.0 * distance_values / range_))


def fit_variogram_models(lags: np.ndarray, variograms: np.ndarray) -> list[VariogramModel]:
    models: list[VariogramModel] = []
    for gamma in variograms:
        valid = np.isfinite(gamma)
        x_values, y_values = lags[valid], gamma[valid]
        variance = float(np.nanmax(y_values)) if len(y_values) else 1.0
        if len(x_values) < 3 or variance <= 0:
            models.append(VariogramModel(0.0, 1.0, float(lags[-1])))
            continue
        try:
            parameters, _ = curve_fit(
                exponential_variogram,
                x_values,
                y_values,
                p0=(0.0, variance, float(np.median(x_values))),
                bounds=((0.0, 0.0, max(np.min(x_values) / 10, EPSILON)), (variance * 2, variance * 3, lags[-1] * 3)),
                maxfev=20_000,
            )
            models.append(VariogramModel(*map(float, parameters)))
        except (RuntimeError, ValueError):
            models.append(VariogramModel(0.0, variance, float(np.median(x_values))))
    return models


def write_gslib_data(path: Path, coordinates: np.ndarray, values: np.ndarray) -> None:
    rows = np.column_stack((coordinates, values))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("SMMT morphing factors\n3\nX\nY\nMF\n")
        np.savetxt(handle, rows, fmt="%.10g")


def run_sgsim(
    executable: Path,
    work_dir: Path,
    realization_index: int,
    coordinates: np.ndarray,
    morphing_factors: np.ndarray,
    grid: Grid,
    models: Iterable[VariogramModel],
    generator: np.random.Generator,
) -> np.ndarray:
    output = np.empty((grid.nx * grid.ny, morphing_factors.shape[1]))
    for variable_index, model in enumerate(models):
        data_file = work_dir / f"mf_conditioning_{realization_index}_{variable_index}.dat"
        write_gslib_data(data_file, coordinates, morphing_factors[:, [variable_index]])
        parameter_file = work_dir / f"sgsim_{realization_index}_{variable_index}.par"
        output_file = work_dir / f"sgsim_{realization_index}_{variable_index}.out"
        seed = int(generator.integers(10_001, 999_999)) | 1
        search_radius = max(model.range_ * 2.0, max(grid.x_size, grid.y_size) * 2.0)
        parameter_file.write_text(
            "\n".join(
                (
                    "    Parameters for SGSIM",
                    "    ********************",
                    "",
                    "START OF PARAMETERS:",
                    data_file.name,
                    "1 2 0 3 0 0 - columns for X,Y,Z,vr,wt,sec.var.",
                    "-1.0e21 1.0e21 - trimming limits",
                    "0 - transform the data",
                    "none.trn",
                    "0 - consider reference distribution",
                    "none.dat",
                    "1 0 - reference columns",
                    "-4.1 4.1 - tail limits",
                    "1 -4.1 - lower tail",
                    "1 4.1 - upper tail",
                    "0 - debugging level",
                    "sgsim.dbg",
                    output_file.name,
                    "1 - number of realizations",
                    f"{grid.nx} {grid.x_min} {grid.x_size} - nx,xmn,xsiz",
                    f"{grid.ny} {grid.y_min} {grid.y_size} - ny,ymn,ysiz",
                    "1 0.0 1.0 - nz,zmn,zsiz",
                    f"{seed} - random number seed",
                    "0 24 - min and max original data",
                    "12 - maximum conditioning data",
                    "1 - assign data to grid nodes",
                    "1 3 - multiple grid search",
                    "0 - maximum data per octant",
                    f"{search_radius} {search_radius} 1.0 - search radii",
                    "0.0 0.0 0.0 - search angles",
                    "241 241 1 - covariance lookup table",
                    "0 0.0 1.0 - kriging type",
                    "none.dat",
                    "4 - secondary column",
                    f"1 {model.nugget:.10g} - structures, nugget",
                    f"2 {model.partial_sill:.10g} 0.0 0.0 0.0 - exponential structure",
                    f"{model.range_:.10g} {model.range_:.10g} 1.0 - ranges",
                )
            ),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [str(executable), parameter_file.name],
                cwd=work_dir,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise RuntimeError(f"Unable to launch SGSIM executable {executable}: {error}") from error
        if completed.returncode != 0 or not output_file.exists():
            details = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
            raise RuntimeError(f"SGSIM failed for variable {variable_index}: {details or 'no diagnostic output'}")
        values = np.loadtxt(output_file, skiprows=3, ndmin=1)
        if values.size != output.shape[0]:
            raise RuntimeError(f"Unexpected SGSIM output size for variable {variable_index}: {values.size}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"SGSIM returned non-finite values for variable {variable_index}.")
        output[:, variable_index] = values.reshape(-1)
    return output


class LocalThinPlateSpline:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.source = source
        kernel = distance.cdist(source, source)
        design = np.column_stack((np.ones(len(source)), source))
        system = np.block(((kernel, design), (design.T, np.zeros((source.shape[1] + 1, source.shape[1] + 1)))))
        right_side = np.vstack((target, np.zeros((source.shape[1] + 1, target.shape[1]))))
        self.parameters = np.linalg.lstsq(system, right_side, rcond=None)[0]

    def predict(self, points: np.ndarray) -> np.ndarray:
        kernel = distance.cdist(points, self.source)
        design = np.column_stack((kernel, np.ones(len(points)), points))
        return design @ self.parameters


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def inverse_empirical_cdf(probabilities: np.ndarray, raw_values: np.ndarray) -> np.ndarray:
    output = np.empty_like(probabilities)
    for variable_index in range(raw_values.shape[1]):
        output[:, variable_index] = np.quantile(raw_values[:, variable_index], probabilities[:, variable_index], method="linear")
    return output


def map_to_original_space(
    simulated_mf: np.ndarray,
    prediction_coordinates: np.ndarray,
    landmarks: Dataset,
    paired_mf: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    neighbors = min(neighbors, len(landmarks.coordinates))
    tree = cKDTree(landmarks.coordinates)
    _, neighbor_indices = tree.query(prediction_coordinates, k=neighbors)
    if neighbors == 1:
        neighbor_indices = neighbor_indices[:, None]
    source = logit(norm.cdf(paired_mf))
    target = logit(empirical_cdf(landmarks.values))
    queries = logit(norm.cdf(simulated_mf))
    mapped_cdf = np.empty_like(simulated_mf)
    groups: dict[tuple[int, ...], list[int]] = {}
    for point_index, indices in enumerate(neighbor_indices):
        groups.setdefault(tuple(np.asarray(indices, dtype=int)), []).append(point_index)
    for indices, point_indices in groups.items():
        points = np.asarray(point_indices)
        mapper = LocalThinPlateSpline(source[np.asarray(indices)], target[np.asarray(indices)])
        mapped_cdf[points] = mapper.predict(queries[points])
    return inverse_empirical_cdf(np.clip(mapped_cdf, EPSILON, 1.0 - EPSILON), landmarks.values)


def save_metadata(path: Path, arguments: argparse.Namespace, dataset: Dataset, models: list[VariogramModel]) -> None:
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
        "variables": dataset.variable_names,
        "grid": asdict(dataset.grid),
        "variogram_models": [asdict(model) for model in models],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    if arguments.landmarks < 3 or arguments.pairings < 1 or arguments.realizations < 1 or arguments.nlag < 3:
        raise ValueError("--landmarks, --pairings, --realizations, and --nlag must be positive; --nlag must be at least 3.")
    if arguments.lag is not None and arguments.lag <= 0:
        raise ValueError("--lag must be positive.")
    generator = np.random.default_rng(arguments.seed)
    dataset = (
        load_benchmark_dataset(arguments.input or default_benchmark_input())
        if arguments.dataset == "benchmark"
        else load_csv_dataset(arguments.input or Path("data.csv"), arguments.variables, arguments.x_column, arguments.y_column)
    )
    landmarks = select_landmarks(dataset, arguments.landmarks, generator)
    neighbors = arguments.neighbors or max(18, dataset.values.shape[1] + 2)
    if neighbors < dataset.values.shape[1] + 2:
        raise ValueError("--neighbors must exceed the number of variables plus one for TPS.")
    if neighbors > arguments.landmarks:
        raise ValueError("--neighbors cannot exceed --landmarks.")
    output_dir = arguments.output
    work_dir = output_dir / "gslib"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    executable = (arguments.gslib_dir / "sgsim.exe").resolve()
    if not executable.exists():
        raise FileNotFoundError(f"Cannot find SGSIM executable: {executable}")
    LOGGER.info("Dataset: %s nodes, %s variables, grid %sx%s.", len(dataset.coordinates), len(dataset.variable_names), dataset.grid.nx, dataset.grid.ny)
    lag = arguments.lag or max(dataset.grid.x_size, dataset.grid.y_size)
    target_cdf = empirical_cdf(landmarks.values)
    pairing_mf = np.stack([paired_morphing_factors(target_cdf, generator) for _ in trange(arguments.pairings, desc="OT pairings")], axis=0)
    average_mf = pairing_mf.mean(axis=0)
    lags, variograms = experimental_variogram(landmarks.coordinates, average_mf, lag, arguments.nlag)
    models = fit_variogram_models(lags, variograms)
    LOGGER.info("Fitted %s morphing-factor variogram models.", len(models))
    np.savez_compressed(output_dir / "pairings_and_variograms.npz", pairing_mf=pairing_mf, average_mf=average_mf, lags=lags, variograms=variograms)
    save_metadata(output_dir / "metadata.json", arguments, dataset, models)
    prediction_coordinates = dataset.grid.coordinates
    for realization_index in trange(arguments.realizations, desc="SMMT realizations"):
        paired_mf = pairing_mf[realization_index % arguments.pairings]
        simulated_mf = run_sgsim(executable, work_dir, realization_index, landmarks.coordinates, paired_mf, dataset.grid, models, generator)
        if arguments.skip_mapping:
            np.savez_compressed(output_dir / f"realization_{realization_index:03d}.npz", coordinates=prediction_coordinates, morphing_factors=simulated_mf)
        else:
            simulated_values = map_to_original_space(simulated_mf, prediction_coordinates, landmarks, paired_mf, neighbors)
            np.savez_compressed(output_dir / f"realization_{realization_index:03d}.npz", coordinates=prediction_coordinates, values=simulated_values, variable_names=np.asarray(dataset.variable_names))
        LOGGER.info("Saved realization %03d to %s.", realization_index, output_dir)


if __name__ == "__main__":
    main()
