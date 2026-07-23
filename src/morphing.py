"""Installed compatibility module for the historical ``morphing`` import."""

from point_cloud_morphing import *  # noqa: F403
from point_cloud_morphing.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

