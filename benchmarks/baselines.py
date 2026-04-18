"""Baseline conformal methods for benchmark comparison.

Implements ACI (Gibbs-Candès) and split-CP for one-step scalar regression.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ACIBaseline:
    """Adaptive Conformal Inference (Gibbs & Candès).

    Online method that adapts the significance level to maintain
    long-run coverage. Uses absolute residual scores.

    Parameters
    ----------
    alpha : float
        Target miscoverage level.
    gamma : float
        Step size for alpha update.
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.01):
        self.alpha_target = alpha
        self.gamma = gamma
        self._alpha_t = alpha
        self._scores = deque(maxlen=500)

    def predict_interval(self, y_hat: float) -> tuple:
        """Return (lower, upper) prediction interval."""
        if len(self._scores) < 2:
            # Not enough history — use wide interval
            return (y_hat - 5.0, y_hat + 5.0)
        scores = np.array(self._scores)
        q = np.quantile(scores, 1.0 - self._alpha_t)
        return (y_hat - q, y_hat + q)

    def update(self, y_t: float, y_hat: float) -> None:
        """Update after observing outcome."""
        # Coverage must be evaluated against the interval that was available
        # before observing y_t; otherwise the update leaks the current outcome.
        lower, upper = self.predict_interval(y_hat)
        covered = float(lower <= y_t <= upper)

        score = abs(y_t - y_hat)
        self._scores.append(score)

        # Adaptive alpha update
        self._alpha_t = self._alpha_t + self.gamma * (self.alpha_target - (1.0 - covered))
        self._alpha_t = np.clip(self._alpha_t, 0.001, 0.999)

    def reset(self):
        self._scores.clear()
        self._alpha_t = self.alpha_target


class SplitCPBaseline:
    """Split conformal prediction for time series.

    Uses a fixed calibration block immediately before the test block.
    No online adaptation.

    Parameters
    ----------
    alpha : float
        Target miscoverage level.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self._calibration_scores: np.ndarray = np.array([])
        self._quantile: float = 5.0

    def calibrate(self, y_cal: np.ndarray, y_hat_cal: np.ndarray) -> None:
        """Fit on a calibration block."""
        self._calibration_scores = np.abs(y_cal - y_hat_cal)
        n = len(self._calibration_scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._quantile = float(np.quantile(self._calibration_scores, level))

    def predict_interval(self, y_hat: float) -> tuple:
        return (y_hat - self._quantile, y_hat + self._quantile)
