"""Forecast CDF interfaces, adapters, and validation.

This module defines the ``ForecastCDF`` protocol, adapter
implementations for quantile grids and predictive samples, the
``TransportedForecastCDF`` wrapper returned by SCT, and forecast
validation helpers.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import interp1d

from tsconformal.utils import clip_cdf, clip_probability, pav_isotonic_unbounded, pav_isotonic_clipped


# -----------------------------------------------------------------------
# Warning / error classes
# -----------------------------------------------------------------------

class InvalidForecastCDFError(Exception):
    """Raised when a ForecastCDF object fails validation."""


class InvalidForecastCDFWarning(UserWarning):
    """Issued when a ForecastCDF object has borderline validity."""


# -----------------------------------------------------------------------
# ForecastCDF protocol
# -----------------------------------------------------------------------

@runtime_checkable
class ForecastCDF(Protocol):
    """Protocol for a predictive CDF object.

    Any object satisfying this protocol can be consumed by
    ``SegmentedTransportCalibrator``. The ``cdf`` method must be
    monotone and clipped to [0, 1]. The ``ppf`` method must implement
    the generalized inverse: ppf(u) = inf{y : cdf(y) >= u}.

    If the forecast is adapter-backed (e.g., from a quantile grid),
    the repr or docstring should state this fact.
    """

    @property
    def is_discrete(self) -> bool:
        ...

    def cdf(self, y: float) -> float:
        ...

    def ppf(self, u: float | np.ndarray) -> float | np.ndarray:
        ...


# -----------------------------------------------------------------------
# TransportedForecastCDF
# -----------------------------------------------------------------------

class TransportedForecastCDF:
    """CDF produced by applying a transport map to a base forecast.

    Parameters
    ----------
    base_cdf : ForecastCDF
        The original base forecast.
    T : callable
        Monotone transport map T: [0,1] -> [0,1].
    T_inv : callable
        Generalized inverse of T.
    """

    def __init__(self, base_cdf: ForecastCDF, T, T_inv):
        self._base = base_cdf
        self._T = T
        self._T_inv = T_inv

    @property
    def is_discrete(self) -> bool:
        return self._base.is_discrete

    def cdf(self, y: float) -> float:
        raw = self._base.cdf(y)
        return float(clip_cdf(self._T(raw)))

    def ppf(self, u: float | np.ndarray) -> float | np.ndarray:
        u_c = np.asarray(clip_probability(u), dtype=np.float64)
        scalar = u_c.ndim == 0
        v = np.asarray(clip_probability(self._T_inv(u_c)), dtype=np.float64)

        try:
            out = np.asarray(self._base.ppf(v), dtype=np.float64)
            if out.shape != u_c.shape:
                raise TypeError("base ppf returned unexpected shape")
        except (TypeError, ValueError):
            if scalar:
                return float(self._base.ppf(float(v)))
            flat = np.array(
                [self._base.ppf(float(val)) for val in v.ravel()],
                dtype=np.float64,
            )
            out = flat.reshape(v.shape)

        return float(out) if scalar else out

    def __repr__(self) -> str:
        return f"TransportedForecastCDF(base={self._base!r})"


# -----------------------------------------------------------------------
# QuantileGridCDFAdapter
# -----------------------------------------------------------------------

class QuantileGridCDFAdapter:
    """Adapter that converts a quantile grid into a ForecastCDF.

    Construction follows the normative specification:
    1. Sort by probability level.
    2. Repair monotonicity via unweighted L2 isotonic regression (PAV).
    3. Piecewise-linear interpolation of repaired quantile function.
    4. Boundary extrapolation using nearest non-zero slope, clipped.
    5. Flat quantile segments are treated as atoms.
    6. CDF is the right-continuous inverse induced by the repaired quantile
       function, with ``cdf_left(y)`` exposing the left limit used by
       randomized PIT.

    Parameters
    ----------
    probabilities : array-like of shape (K,)
        Probability levels in (0, 1).
    quantiles : array-like of shape (K,)
        Raw quantile values (may be non-monotone).
    """

    def __init__(self, probabilities, quantiles):
        probs = np.asarray(probabilities, dtype=np.float64)
        quants = np.asarray(quantiles, dtype=np.float64)

        if len(probs) != len(quants):
            raise ValueError("probabilities and quantiles must have same length")
        if len(probs) < 2:
            raise ValueError("need at least 2 quantile levels")
        if not np.all(np.isfinite(probs)):
            raise InvalidForecastCDFError("probability levels contain non-finite entries")
        if not np.all(np.isfinite(quants)):
            raise InvalidForecastCDFError("quantile values contain non-finite entries")
        if np.any(probs <= 0.0) or np.any(probs >= 1.0):
            raise InvalidForecastCDFError("probability levels must be strictly in (0, 1)")

        # Sort by probability
        order = np.argsort(probs)
        probs = probs[order]
        quants = quants[order]

        # Check strict monotonicity of sorted probabilities
        if np.any(np.diff(probs) <= 0):
            raise InvalidForecastCDFError(
                "probability levels must be strictly increasing (duplicates found)"
            )

        # Repair monotonicity via L2 PAV (unbounded — quantiles are on target scale)
        repaired = pav_isotonic_unbounded(quants)

        # Store repaired knots
        self._probs = probs
        self._quants = repaired
        self.is_discrete = bool(np.any(np.diff(repaired) <= 1e-12))

        # Build extrapolation slopes
        self._slope_lo = self._find_slope(repaired, probs, end="low")
        self._slope_hi = self._find_slope(repaired, probs, end="high")

        # Build interpolator for ppf
        self._ppf_interp = interp1d(
            probs, repaired, kind="linear",
            bounds_error=False, fill_value=(repaired[0], repaired[-1])
        )

    @staticmethod
    def _find_slope(quants, probs, end: str) -> float:
        """Find nearest non-zero local slope for extrapolation."""
        if end == "low":
            for i in range(len(quants) - 1):
                dq = quants[i + 1] - quants[i]
                dp = probs[i + 1] - probs[i]
                if dp > 0 and dq > 1e-15:
                    return dq / dp
            return 0.0
        else:
            for i in range(len(quants) - 1, 0, -1):
                dq = quants[i] - quants[i - 1]
                dp = probs[i] - probs[i - 1]
                if dp > 0 and dq > 1e-15:
                    return dq / dp
            return 0.0

    def ppf(self, u: float | np.ndarray) -> float | np.ndarray:
        """Repaired quantile function with boundary extrapolation."""
        u_c = np.asarray(clip_probability(u), dtype=np.float64)
        scalar = u_c.ndim == 0
        vals = np.asarray(self._ppf_interp(u_c), dtype=np.float64)

        below = u_c < self._probs[0]
        above = u_c > self._probs[-1]
        if np.any(below):
            vals = np.where(
                below,
                self._quants[0] - self._slope_lo * (self._probs[0] - u_c),
                vals,
            )
        if np.any(above):
            vals = np.where(
                above,
                self._quants[-1] + self._slope_hi * (u_c - self._probs[-1]),
                vals,
            )

        return float(vals) if scalar else vals

    def cdf_left(self, y: float) -> float:
        """Left limit F(y-) induced by the repaired quantile function."""
        idx = np.searchsorted(self._quants, y, side="left")
        if idx == 0:
            if y <= self._quants[0]:
                # Extrapolate below
                if self._slope_lo > 1e-15:
                    val = self._probs[0] - (self._quants[0] - y) / self._slope_lo
                    return float(clip_cdf(val))
                return 0.0
            return float(self._probs[0])
        if idx >= len(self._quants):
            if y > self._quants[-1]:
                if self._slope_hi > 1e-15:
                    val = self._probs[-1] + (y - self._quants[-1]) / self._slope_hi
                    return float(clip_cdf(val))
                return 1.0
            return float(self._probs[-1])
        # Linear interpolation between knots
        q_lo, q_hi = self._quants[idx - 1], self._quants[idx]
        p_lo, p_hi = self._probs[idx - 1], self._probs[idx]
        if q_hi - q_lo < 1e-15:
            return float(p_hi)
        frac = (y - q_lo) / (q_hi - q_lo)
        return float(clip_cdf(p_lo + frac * (p_hi - p_lo)))

    def cdf(self, y: float) -> float:
        """Right-continuous CDF induced by the repaired quantile function."""
        idx = np.searchsorted(self._quants, y, side="right")
        if idx == 0:
            if y <= self._quants[0]:
                if self._slope_lo > 1e-15:
                    val = self._probs[0] - (self._quants[0] - y) / self._slope_lo
                    return float(clip_cdf(val))
                return 0.0
            return float(self._probs[0])
        if idx >= len(self._quants):
            if y > self._quants[-1]:
                if self._slope_hi > 1e-15:
                    val = self._probs[-1] + (y - self._quants[-1]) / self._slope_hi
                    return float(clip_cdf(val))
                return 1.0
            if self._slope_hi <= 1e-15:
                return 1.0
            return float(self._probs[-1])

        q_lo, q_hi = self._quants[idx - 1], self._quants[idx]
        p_lo, p_hi = self._probs[idx - 1], self._probs[idx]
        if q_hi - q_lo < 1e-15:
            return float(p_hi)
        frac = (y - q_lo) / (q_hi - q_lo)
        return float(clip_cdf(p_lo + frac * (p_hi - p_lo)))

    def __repr__(self) -> str:
        return (
            f"QuantileGridCDFAdapter(K={len(self._probs)}, "
            f"range=[{self._quants[0]:.4f}, {self._quants[-1]:.4f}], "
            f"is_discrete={self.is_discrete})"
        )


# -----------------------------------------------------------------------
# SampleCDFAdapter
# -----------------------------------------------------------------------

class SampleCDFAdapter:
    """Adapter that converts predictive samples into a ForecastCDF.

    Parameters
    ----------
    samples : array-like of shape (N,)
        Predictive samples. Benchmark default requires N >= 10000.
    smoothing : str
        'none' for raw empirical step CDF (is_discrete=True),
        'gaussian_monotone' for kernel-smoothed monotone-repaired CDF.
        The smoothed path requires a non-degenerate sample range.
    """

    def __init__(self, samples, smoothing: str = "gaussian_monotone"):
        if smoothing not in ("none", "gaussian_monotone"):
            raise ValueError(f"unsupported smoothing: {smoothing!r}")

        samples = np.sort(np.asarray(samples, dtype=np.float64))
        if not np.all(np.isfinite(samples)):
            raise InvalidForecastCDFError("samples contain non-finite values")
        if len(samples) < 100:
            raise InvalidForecastCDFError("need at least 100 samples")

        self._samples = samples
        self._n = len(samples)
        self._smoothing = smoothing
        self.is_discrete = (smoothing == "none")

        if smoothing == "gaussian_monotone":
            sample_range = float(samples[-1] - samples[0])
            if sample_range < 1e-12:
                raise InvalidForecastCDFError(
                    f"degenerate samples: range {sample_range:.2e} too small for gaussian_monotone smoothing"
                )
            self._build_smooth_cdf()
        else:
            # Raw empirical CDF
            self._ecdf_y = samples
            self._ecdf_p = np.arange(1, self._n + 1) / self._n

    def _build_smooth_cdf(self):
        """Build kernel-smoothed, monotone-repaired CDF on a dense grid."""
        from scipy.stats import gaussian_kde

        # Bandwidth via Scott's rule
        try:
            kde = gaussian_kde(self._samples)
        except np.linalg.LinAlgError as exc:
            raise InvalidForecastCDFError(
                "degenerate samples: gaussian_monotone smoothing failed because the sample covariance is singular"
            ) from exc

        # Dense evaluation grid (1000 points spanning data range + margins)
        margin = 3.0 * kde.scotts_factor() * np.std(self._samples)
        lo = self._samples[0] - margin
        hi = self._samples[-1] + margin
        self._dense_y = np.linspace(lo, hi, 1000)

        # Evaluate smoothed CDF by integrating density
        raw_cdf = np.array([kde.integrate_box_1d(-np.inf, y) for y in self._dense_y])

        # Monotone repair via isotonic projection
        raw_cdf = pav_isotonic_clipped(raw_cdf)

        # Ensure boundary conditions
        raw_cdf[0] = max(raw_cdf[0], 0.0)
        raw_cdf[-1] = min(raw_cdf[-1], 1.0)

        self._dense_cdf = raw_cdf

    def cdf(self, y: float) -> float:
        if self._smoothing == "gaussian_monotone":
            return float(clip_cdf(np.interp(y, self._dense_y, self._dense_cdf)))
        else:
            idx = np.searchsorted(self._ecdf_y, y, side="right")
            if idx == 0:
                return 0.0
            if idx >= self._n:
                return 1.0
            return float(self._ecdf_p[idx - 1])

    def cdf_left(self, y: float) -> float:
        if self._smoothing == "gaussian_monotone":
            return self.cdf(y)

        idx = np.searchsorted(self._ecdf_y, y, side="left")
        if idx == 0:
            return 0.0
        if idx >= self._n:
            return 1.0
        return float(self._ecdf_p[idx - 1])

    def ppf(self, u: float | np.ndarray) -> float | np.ndarray:
        u_c = np.asarray(clip_probability(u), dtype=np.float64)
        scalar = u_c.ndim == 0

        if self._smoothing == "gaussian_monotone":
            vals = np.asarray(np.interp(u_c, self._dense_cdf, self._dense_y), dtype=np.float64)
        else:
            idx = np.ceil(u_c * self._n).astype(np.int64) - 1
            idx = np.clip(idx, 0, self._n - 1)
            vals = np.asarray(self._samples[idx], dtype=np.float64)

        return float(vals) if scalar else vals

    def __repr__(self) -> str:
        return (
            f"SampleCDFAdapter(n={self._n}, smoothing='{self._smoothing}', "
            f"range=[{self._samples[0]:.4f}, {self._samples[-1]:.4f}])"
        )


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

_PROBE_PROBS = np.array([1e-6, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0 - 1e-6])


def validate_forecast_cdf(cdf: ForecastCDF) -> None:
    """Validate a ForecastCDF object before it reaches SCT.

    Raises ``InvalidForecastCDFError`` if the object fails hard checks.
    Issues ``InvalidForecastCDFWarning`` for borderline issues.

    Checks
    ------
    1. No NaN/Inf from ppf on probe probabilities.
    2. No NaN/Inf from cdf on probe values.
    3. cdf values in [0, 1].
    4. Non-degenerate quantile range (max - min >= 1e-12), unless the
       forecast is explicitly discrete.
    5. Monotonicity of cdf on probe values.
    """
    # Check ppf on probes
    probe_values = []
    for u in _PROBE_PROBS:
        try:
            val = cdf.ppf(float(u))
        except Exception as exc:
            raise InvalidForecastCDFError(
                f"ppf({u}) raised {type(exc).__name__}: {exc}"
            ) from exc
        if not np.isfinite(val):
            raise InvalidForecastCDFError(f"ppf({u}) returned non-finite: {val}")
        probe_values.append(val)

    probe_values = np.array(probe_values)

    # Check degenerate range
    qrange = probe_values[-1] - probe_values[0]
    if qrange < 1e-12 and not bool(getattr(cdf, "is_discrete", False)):
        raise InvalidForecastCDFError(
            f"degenerate quantile range: {qrange:.2e}"
        )

    # Check cdf on probe values
    cdf_values = []
    for y in probe_values:
        try:
            cv = cdf.cdf(float(y))
        except Exception as exc:
            raise InvalidForecastCDFError(
                f"cdf({y}) raised {type(exc).__name__}: {exc}"
            ) from exc
        if not np.isfinite(cv):
            raise InvalidForecastCDFError(f"cdf({y}) returned non-finite: {cv}")
        if cv < 0.0 or cv > 1.0:
            raise InvalidForecastCDFError(
                f"cdf({y}) outside [0,1]: {cv}"
            )
        cdf_values.append(cv)

    # Monotonicity check
    cdf_arr = np.array(cdf_values)
    if np.any(np.diff(cdf_arr) < -1e-10):
        warnings.warn(
            "cdf is not monotone on probe values",
            InvalidForecastCDFWarning,
            stacklevel=2,
        )
