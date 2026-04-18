"""Segmented Conformal Transport for predictive CDF recalibration."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _metadata_version
from pathlib import Path
import tomllib


def _read_source_tree_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        version = data["project"]["version"]
    except Exception:
        return "0.0.0+unknown"
    return str(version)


try:
    __version__ = _metadata_version("tsconformal")
except PackageNotFoundError:
    __version__ = _read_source_tree_version()

from tsconformal.api import *  # noqa: E402, F401, F403
