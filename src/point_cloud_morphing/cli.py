"""Command-line interface for the SMMT pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .models import PipelineConfig
from .pipeline import SmmtPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="smmt",
        description="Run the Spatial Multivariate Morphing Transformation.",
    )
    parser.add_argument("--dataset", choices=("csv", "benchmark"), default="csv")
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV input or benchmark conditioning-data file.",
    )
    parser.add_argument(
        "--variables",
        help="Comma-separated variable names for CSV input.",
    )
    parser.add_argument(
        "--x-column",
        help="X-coordinate column name; defaults to the first CSV column.",
    )
    parser.add_argument(
        "--y-column",
        help="Y-coordinate column name; defaults to the second CSV column.",
    )
    parser.add_argument(
        "--landmarks",
        type=int,
        help=(
            "Number of conditioning landmarks. Defaults to all benchmark samples "
            "or 200 CSV nodes."
        ),
    )
    parser.add_argument(
        "--pairings",
        type=int,
        default=20,
        help="OT pairings used to estimate MF continuity.",
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=1,
        help="Number of SMMT realizations.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        help="Spatially closest landmarks used by local TPS.",
    )
    parser.add_argument(
        "--lag",
        type=float,
        help="Variogram lag distance; defaults to the largest cell size.",
    )
    parser.add_argument(
        "--nlag",
        type=int,
        default=30,
        help="Number of experimental variogram lags.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--gslib-dir", type=Path, default=Path("gslibexe"))
    parser.add_argument("--output", type=Path, default=Path("result/smmt"))
    parser.add_argument(
        "--sgsim-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds allowed for each SGSIM process.",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Save MF simulations without TPS back-mapping.",
    )
    parser.add_argument(
        "--no-marginal-correction",
        action="store_true",
        help="Disable rank-based marginal correction after TPS mapping.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable stage-by-stage PNG diagnostic plots.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without starting the pipeline."""

    return build_parser().parse_args(argv)


def parse_config(argv: Sequence[str] | None = None) -> tuple[PipelineConfig, str]:
    """Parse command-line arguments into a validated pipeline configuration."""

    arguments = parse_arguments(argv)
    config = PipelineConfig(
        dataset_kind=arguments.dataset,
        input_path=arguments.input,
        variables=arguments.variables,
        x_column=arguments.x_column,
        y_column=arguments.y_column,
        landmarks=arguments.landmarks,
        pairings=arguments.pairings,
        realizations=arguments.realizations,
        neighbors=arguments.neighbors,
        lag=arguments.lag,
        nlag=arguments.nlag,
        seed=arguments.seed,
        gslib_dir=arguments.gslib_dir,
        output_dir=arguments.output,
        skip_mapping=arguments.skip_mapping,
        marginal_correction=not arguments.no_marginal_correction,
        diagnostics=not arguments.no_diagnostics,
        sgsim_timeout=arguments.sgsim_timeout,
    )
    return config, arguments.log_level


def configure_logging(level: str) -> None:
    """Configure concise process-wide logging for CLI execution."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    config, log_level = parse_config(argv)
    configure_logging(log_level)
    SmmtPipeline(config).run()
    return 0
