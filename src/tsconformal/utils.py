"""Internal utility functions for tsconformal.

This module contains helpers for validation, RNG management, numerical
clipping, and windowing. Nothing here is part of the public API.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def clip_probability(p: float | np.ndarray) -> float | np.ndarray:
    """Clip probability values to [1e-12, 1 - 1e-12]."""
    return np.clip(p, 1e-12, 1.0 - 1e-12)


def clip_cdf(v: float | np.ndarray) -> float | np.ndarray:
    """Clip CDF values to [0, 1]."""
    return np.clip(v, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Isotonic projection helpers
# ---------------------------------------------------------------------------

def pav_isotonic_clipped(v: np.ndarray) -> np.ndarray:
    """Isotonic regression clipped to [0,1] — for transport maps only.

    Do NOT use this for quantile repair (quantiles live on the target scale).
    """
    from scipy.optimize import isotonic_regression
    res = isotonic_regression(v, increasing=True)
    out = np.asarray(res.x if hasattr(res, 'x') else res, dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def pav_isotonic_unbounded(v: np.ndarray) -> np.ndarray:
    """Isotonic regression without clipping — for quantile-value repair.

    Quantile values live on the real-valued target scale and must NOT
    be clipped to [0,1].
    """
    from scipy.optimize import isotonic_regression
    res = isotonic_regression(v, increasing=True)
    return np.asarray(res.x if hasattr(res, 'x') else res, dtype=np.float64)


# Keep backward-compatible alias pointing to the clipped version
# (used by calibrators for transport map projection)
pav_isotonic = pav_isotonic_clipped


# ---------------------------------------------------------------------------
# Piecewise-linear interpolation for transport maps
# ---------------------------------------------------------------------------

def build_transport_map(grid: np.ndarray, g: np.ndarray):
    """Return callables (T, T_inv) for piecewise-linear transport.

    Parameters
    ----------
    grid : ndarray of shape (J,)
        Interior grid points u_j in (0, 1).
    g : ndarray of shape (J,)
        Monotone projected grid values g_j.

    Returns
    -------
    T : callable
        Transport map T(u) -> [0, 1], piecewise linear through
        (0, 0), (u_1, g_1), ..., (u_J, g_J), (1, 1).
    T_inv : callable
        Upper generalized inverse T^{-1}_+(u), correct on isotonic
        plateaus.
    """
    knots_u = np.concatenate([[0.0], grid, [1.0]])
    knots_g = np.concatenate([[0.0], g, [1.0]])

    def T(u):
        return np.interp(u, knots_u, knots_g)

    def T_inv(v):
        v = np.asarray(v, dtype=np.float64)
        scalar = v.ndim == 0
        v = np.atleast_1d(v)
        result = np.empty_like(v)

        low = v < knots_g[0]
        high = v >= knots_g[-1]
        result[low] = knots_u[0]
        result[high] = knots_u[-1]

        interior = ~(low | high)
        if np.any(interior):
            idx = np.searchsorted(knots_g, v[interior], side="right") - 1
            g0 = knots_g[idx]
            g1 = knots_g[idx + 1]
            u0 = knots_u[idx]
            u1 = knots_u[idx + 1]

            # side="right" makes exact plateau levels land on the plateau's
            # right endpoint, matching the upper generalized inverse.
            rising = g1 > g0
            vals = np.where(
                rising,
                u0 + (v[interior] - g0) * (u1 - u0) / (g1 - g0),
                u1,
            )
            result[interior] = vals

        return float(result[0]) if scalar else result

    return T, T_inv


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def ensure_rng(rng) -> Generator:
    """Convert seed, None, or Generator to a numpy Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_finite(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if arr contains NaN or Inf."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
