# Contributing

## Development setup

Use Python 3.9 or newer, then install the editable development environment:

```powershell
python -m pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```powershell
ruff check .
python -m py_compile morphing.py
```

## Scope

Keep changes focused on the SMMT workflow or documentation.
Do not commit confidential exploration data, generated realizations, executable
binaries, output files, or data-derived validation artifacts.

## Pull requests

Explain the method change, list static checks performed, and state any implications
for compatibility with GSLIB.
