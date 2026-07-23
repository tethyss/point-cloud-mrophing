"""GSLIB file generation and SGSIM process integration."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .models import Grid, VariogramModel

LOGGER = logging.getLogger(__name__)


class SgsimRunner:
    """Run independent conditional simulations through a local SGSIM binary."""

    def __init__(self, executable: Path, work_dir: Path) -> None:
        self.executable = executable.resolve()
        self.work_dir = work_dir.resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(f"Cannot find SGSIM executable: {self.executable}")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def simulate(
        self,
        realization_index: int,
        coordinates: np.ndarray,
        morphing_factors: np.ndarray,
        grid: Grid,
        models: Sequence[VariogramModel],
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Simulate all morphing-factor columns for one realization."""

        if len(models) != morphing_factors.shape[1]:
            raise ValueError("A variogram model is required for every MF variable.")

        output = np.empty((grid.node_count, morphing_factors.shape[1]))
        for variable_index, model in enumerate(models):
            output[:, variable_index] = self._simulate_variable(
                realization_index=realization_index,
                variable_index=variable_index,
                coordinates=coordinates,
                values=morphing_factors[:, [variable_index]],
                grid=grid,
                model=model,
                generator=generator,
            )
        return output

    def _simulate_variable(
        self,
        realization_index: int,
        variable_index: int,
        coordinates: np.ndarray,
        values: np.ndarray,
        grid: Grid,
        model: VariogramModel,
        generator: np.random.Generator,
    ) -> np.ndarray:
        file_stem = f"{realization_index}_{variable_index}"
        data_file = self.work_dir / f"mf_conditioning_{file_stem}.dat"
        parameter_file = self.work_dir / f"sgsim_{file_stem}.par"
        output_file = self.work_dir / f"sgsim_{file_stem}.out"

        write_gslib_data(data_file, coordinates, values)
        random_seed = int(generator.integers(10_001, 999_999)) | 1
        parameter_text = build_sgsim_parameters(
            data_file=data_file,
            output_file=output_file,
            grid=grid,
            model=model,
            random_seed=random_seed,
        )
        parameter_file.write_text(parameter_text, encoding="utf-8")

        LOGGER.debug(
            "Running SGSIM for realization %s, variable %s.",
            realization_index,
            variable_index,
        )
        completed = self._execute(parameter_file)
        if completed.returncode != 0 or not output_file.exists():
            details = "\n".join(
                message
                for message in (completed.stdout.strip(), completed.stderr.strip())
                if message
            )
            raise RuntimeError(
                f"SGSIM failed for variable {variable_index}: "
                f"{details or 'no diagnostic output'}"
            )

        return read_sgsim_output(
            output_file,
            expected_nodes=grid.node_count,
            variable_index=variable_index,
        )

    def _execute(self, parameter_file: Path) -> subprocess.CompletedProcess[str]:
        """Execute SGSIM with the author's parameter-file invocation pattern."""

        try:
            return subprocess.run(
                [str(self.executable), parameter_file.name],
                cwd=self.work_dir,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise RuntimeError(
                f"Unable to launch SGSIM executable {self.executable}: {error}"
            ) from error


def write_gslib_data(
    path: Path,
    coordinates: np.ndarray,
    values: np.ndarray,
) -> None:
    """Write one conditioning variable in GSLIB ASCII format."""

    rows = np.column_stack((coordinates, values))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("SMMT morphing factors\n3\nX\nY\nMF\n")
        np.savetxt(handle, rows, fmt="%.10g")


def build_sgsim_parameters(
    data_file: Path,
    output_file: Path,
    grid: Grid,
    model: VariogramModel,
    random_seed: int,
) -> str:
    """Build a SGSIM 2.920 parameter file for one MF variable."""

    search_radius = max(
        model.range_ * 2.0,
        max(grid.x_size, grid.y_size) * 2.0,
    )
    lines = (
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
        f"{random_seed} - random number seed",
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
    return "\n".join(lines)


def read_sgsim_output(
    path: Path,
    expected_nodes: int,
    variable_index: int,
) -> np.ndarray:
    """Read and validate one SGSIM output variable."""

    values = np.loadtxt(path, skiprows=3, ndmin=1).reshape(-1)
    if values.size != expected_nodes:
        raise RuntimeError(
            f"Unexpected SGSIM output size for variable {variable_index}: "
            f"expected {expected_nodes}, received {values.size}."
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"SGSIM returned non-finite values for variable {variable_index}."
        )
    return values


def run_sgsim(
    executable: Path,
    work_dir: Path,
    realization_index: int,
    coordinates: np.ndarray,
    morphing_factors: np.ndarray,
    grid: Grid,
    models: Sequence[VariogramModel],
    generator: np.random.Generator,
) -> np.ndarray:
    """Compatibility wrapper around :class:`SgsimRunner`."""

    return SgsimRunner(executable, work_dir).simulate(
        realization_index=realization_index,
        coordinates=coordinates,
        morphing_factors=morphing_factors,
        grid=grid,
        models=models,
        generator=generator,
    )
