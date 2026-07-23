# Point Cloud Morphing

[![CI](https://github.com/tethyss/point-cloud-mrophing/actions/workflows/ci.yml/badge.svg)](https://github.com/tethyss/point-cloud-mrophing/actions/workflows/ci.yml)

Python implementation of the **Spatial Multivariate Morphing Transformation
(SMMT)** for multivariate geostatistical simulation. It follows
Avalos and Ortiz's workflow: pair observed multivariate samples with independent
Gaussian morphing factors using optimal transport, simulate factors independently
with GSLIB `sgsim`, then map them back to the original variable space using local
thin-plate splines in logit space.

> The `2022-Morphing-master/` directory is retained locally as the original
> authors' reference implementation. The production entry point is `morphing.py`.

## Features

- Supports the original **200 × 200, six-variable** benchmark conditioning data.
- Supports regular-grid CSV data, including the **335 × 335** multivariate dataset.
- Validates grid completeness, finite numeric inputs, TPS neighborhood size, and
  SGSIM output shape.
- Persists realization outputs, OT pairings, fitted variograms, and run metadata
  for approved internal workflows.
- Provides structured logs through `--log-level`.
- Includes static code checks, issue forms, and pull-request guidance.

## Requirements

- Python 3.9 or newer
- A local GSLIB `sgsim.exe` executable
- A regular Cartesian grid for CSV input

The GSLIB executable and private/project data are intentionally excluded from
version control. Place `sgsim.exe` at `gslibexe/sgsim.exe`, or provide another
directory with `--gslib-dir`.

## Installation

```powershell
git clone https://github.com/tethyss/point-cloud-mrophing.git
cd point-cloud-mrophing
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable installation exposes the `smmt` command. You can also invoke the
compatible script entry point with `python morphing.py`.

## Data Access

All operational datasets and derived simulation outputs are confidential and are
not distributed with this repository. Consequently, the project does not provide a
public reproduction test, benchmark run, or downloadable sample dataset at this
time. The repository documents the method, software interface, and integration
requirements only.

## CSV Input Contract

CSV input must contain every node of one regular 2D grid exactly once. By default,
the first two columns are used as X and Y coordinates. Select alternatives with
`--x-column` and `--y-column`. By default, all columns after the coordinates are
used except case-insensitive `DEM`, `Slope`, and `granites`; select exact variables
with `--variables`:

```powershell
smmt --dataset csv --input data.csv --x-column x --y-column y --variables Ag,Al,Au
```

All selected values must be finite and numeric. The number of landmarks and TPS
neighbors must exceed the number of variables plus one.

## Method and Outputs

1. **Pairing**: empirical CDFs of landmark observations are matched to sampled
   independent Gaussian vectors using discrete optimal transport.
2. **Simulation**: the mean MF field supplies direct experimental variograms;
   each paired MF realization is conditionally simulated with GSLIB `sgsim`.
3. **Mapping**: simulated MF values are transformed to CDF/logit space and mapped
   locally using spatially closest landmark points and thin-plate splines.

Approved internal runs write to `result/smmt/` by default:

- `metadata.json`: configuration, grid definition, selected variables, variogram models
- `pairings_and_variograms.npz`: OT pairings, average MFs, and experimental variograms
- `realization_*.npz`: coordinates plus simulated values, or MF values with `--skip-mapping`
- `gslib/`: generated conditioning data, SGSIM parameter files, logs, and outputs

## Command Reference

```text
smmt --help
```

Key options: `--dataset`, `--input`, `--variables`, `--landmarks`, `--pairings`,
`--realizations`, `--neighbors`, `--lag`, `--nlag`, `--seed`, `--gslib-dir`,
`--output`, `--skip-mapping`, and `--log-level`.

## Development

```powershell
ruff check .
python -m py_compile morphing.py
```

## Citation and Acknowledgement

Please cite the original SMMT paper when using this implementation in approved
work:

> Avalos, S. and Ortiz, J. M. *Spatial Multivariate Morphing Transformation*.

The original authors' source code is included locally as a reference for scientific
comparison. GSLIB is an external dependency and is not distributed by this project.

## License

This project is released under the [MIT License](LICENSE). Third-party software
and reference implementations retain their respective licenses and notices.
