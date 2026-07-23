"""Shared access to the optional local benchmark dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DATA = (
    PROJECT_ROOT / "benchmark" / "bmdata" / "Conditioning_Data_6dim_Numpy.txt"
)
SGSIM_EXECUTABLE = PROJECT_ROOT / "benchmark" / "sgsim.exe"


@pytest.fixture(scope="session")
def benchmark_data_path() -> Path:
    """Return the private local benchmark input or skip when unavailable."""

    if not BENCHMARK_DATA.is_file():
        pytest.skip("Local benchmark conditioning data is not available.")
    return BENCHMARK_DATA


@pytest.fixture(scope="session")
def benchmark_sgsim_dir(benchmark_data_path: Path) -> Path:
    """Return the local SGSIM directory or skip when unavailable."""

    if not SGSIM_EXECUTABLE.is_file():
        pytest.skip("Local benchmark SGSIM executable is not available.")
    return SGSIM_EXECUTABLE.parent
