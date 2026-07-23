"""Compatibility entry point for the packaged SMMT implementation.

Production code lives in :mod:`point_cloud_morphing`. Existing commands such as
``python morphing.py`` remain supported.
"""

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from point_cloud_morphing import *  # noqa: E402,F403
from point_cloud_morphing.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
