"""Deprecated shim. Use tsconformal.sensitivity_report directly."""

import warnings

from tsconformal.diagnostics import sensitivity_report as _sr


def sensitivity_report(*args, **kwargs):
    warnings.warn(
        "tsconformal.sensitivity.sensitivity_report is deprecated. "
        "Use tsconformal.sensitivity_report instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _sr(*args, **kwargs)
