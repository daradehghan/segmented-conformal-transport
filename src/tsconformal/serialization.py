"""Deprecated shim module. Use tsconformal.save_calibrator / load_calibrator directly."""

import warnings

from tsconformal.calibrators import load_calibrator as _load
from tsconformal.calibrators import save_calibrator as _save


def save_calibrator(*args, **kwargs):
    warnings.warn(
        "tsconformal.serialization.save_calibrator is deprecated. "
        "Use tsconformal.save_calibrator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _save(*args, **kwargs)


def load_calibrator(*args, **kwargs):
    warnings.warn(
        "tsconformal.serialization.load_calibrator is deprecated. "
        "Use tsconformal.load_calibrator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load(*args, **kwargs)
