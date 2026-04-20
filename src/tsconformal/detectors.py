"""Segment detectors for change-point detection in calibration residuals.

This module provides the ``SegmentDetector`` protocol and two reference
implementations: ``CUSUMNormDetector`` and ``PageHinkleyDetector``.

Detectors report threshold exceedance only. Cooldown and confirmation
logic lives in ``SegmentedTransportCalibrator``.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


# -----------------------------------------------------------------------
# SegmentDetector protocol
# -----------------------------------------------------------------------


def _normalized_l2_norm(residual_vector: np.ndarray) -> float:
    """Return the theory-aligned gridwise L2 norm ``||r||_{2,J}``.

    This normalization satisfies

        ||r||_{2,J} = ||r||_2 / sqrt(J),

    where ``J = len(residual_vector)``. Equivalently, it is the RMS norm
    ``sqrt(mean(r_j^2))``. Detector defaults are calibrated on this scale.
    """
    residual_vector = np.asarray(residual_vector, dtype=np.float64)
    return float(np.linalg.norm(residual_vector) / np.sqrt(len(residual_vector)))

@runtime_checkable
class SegmentDetector(Protocol):
    """Protocol for a change-point detector consumed by SCT.

    The ``update`` method accepts a residual vector and returns True
    if the detector's internal statistic exceeds its threshold.
    The ``state`` method exposes internal diagnostics.
    The ``reset`` method clears internal state after a confirmed
    segment boundary.
    """

    def update(self, residual_vector: np.ndarray) -> bool:
        ...

    def state(self) -> Mapping[str, Any]:
        ...

    def reset(self) -> None:
        ...


# -----------------------------------------------------------------------
# CUSUMNormDetector
# -----------------------------------------------------------------------

class CUSUMNormDetector:
    """Scalar CUSUM detector on the normalized L2 norm of residual vectors.

    The statistic tracks:
        S_t = max(0, S_{t-1} + ||r_t||_{2,J} - kappa)

    and signals when S_t > threshold.

    Here ``||r_t||_{2,J}`` is the gridwise L2 norm from the theory,
    defined by ``||r||_{2,J}^2 = (1 / J) * sum_j r_j^2``. So the detector
    uses the RMS norm, equivalently ``||r||_2 / sqrt(J)``, not the raw
    Euclidean norm ``||r||_2``.

    Parameters
    ----------
    kappa : float
        Finite non-negative reference value on the normalized-L2 scale.
        Typical default: 0.02.
    threshold : float
        Finite positive decision threshold on the normalized-L2 scale.
        Typical default: 0.20.
    """

    def __init__(self, kappa: float = 0.02, threshold: float = 0.20):
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and non-negative")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        self.kappa = kappa
        self.threshold = threshold
        self._S: float = 0.0
        self._t: int = 0

    def update(self, residual_vector: np.ndarray) -> bool:
        """Update CUSUM statistic and return True if threshold exceeded."""
        if not np.all(np.isfinite(residual_vector)):
            warnings.warn(
                "CUSUMNormDetector received non-finite residuals; "
                "skipping this update",
                RuntimeWarning,
                stacklevel=2,
            )
            self._t += 1
            return False
        rms_norm = _normalized_l2_norm(residual_vector)
        self._S = max(0.0, self._S + rms_norm - self.kappa)
        self._t += 1
        return self._S > self.threshold

    def state(self) -> Mapping[str, Any]:
        return {
            "detector_type": "CUSUMNorm",
            "kappa": self.kappa,
            "threshold": self.threshold,
            "S": self._S,
            "t": self._t,
        }

    def reset(self) -> None:
        """Reset the CUSUM statistic to zero."""
        self._S = 0.0

    def __repr__(self) -> str:
        return f"CUSUMNormDetector(kappa={self.kappa}, threshold={self.threshold})"


# -----------------------------------------------------------------------
# PageHinkleyDetector
# -----------------------------------------------------------------------

class PageHinkleyDetector:
    """Page-Hinkley detector on the normalized L2 residual norm.

    Tracks:
        m_t = sum_{i=1}^t (||r_i||_{2,J} - delta)
        M_t = min_{i<=t} m_i

    and signals when m_t - M_t > threshold.

    Here ``||r_t||_{2,J}`` is the gridwise L2 norm from the theory,
    defined by ``||r||_{2,J}^2 = (1 / J) * sum_j r_j^2``. So the detector
    uses the RMS norm, equivalently ``||r||_2 / sqrt(J)``, not the raw
    Euclidean norm ``||r||_2``.

    Parameters
    ----------
    delta : float
        Drift allowance on the normalized-L2 scale.
    threshold : float
        Decision threshold on the normalized-L2 scale.
    """

    def __init__(self, delta: float = 0.01, threshold: float = 0.50):
        if not np.isfinite(delta) or delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        self.delta = delta
        self.threshold = threshold
        self._m: float = 0.0
        self._M: float = 0.0
        self._t: int = 0

    def update(self, residual_vector: np.ndarray) -> bool:
        if not np.all(np.isfinite(residual_vector)):
            warnings.warn(
                "PageHinkleyDetector received non-finite residuals; "
                "skipping this update",
                RuntimeWarning,
                stacklevel=2,
            )
            self._t += 1
            return False
        rms_norm = _normalized_l2_norm(residual_vector)
        self._m += rms_norm - self.delta
        self._M = min(self._M, self._m)
        self._t += 1
        return (self._m - self._M) > self.threshold

    def state(self) -> Mapping[str, Any]:
        return {
            "detector_type": "PageHinkley",
            "delta": self.delta,
            "threshold": self.threshold,
            "m": self._m,
            "M": self._M,
            "t": self._t,
        }

    def reset(self) -> None:
        self._m = 0.0
        self._M = 0.0

    def __repr__(self) -> str:
        return f"PageHinkleyDetector(delta={self.delta}, threshold={self.threshold})"
