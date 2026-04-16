"""Benchmark-facing evaluation metrics.

Computes the theorem-aligned gridwise L2 calibration error E_r,
marginal coverage, interval width, and adaptation lag.
"""

from __future__ import annotations

import numpy as np


def _ppf_many(cdf, u_grid: np.ndarray) -> np.ndarray:
    """Evaluate ``cdf.ppf`` on a grid, using a batched call when supported."""
    try:
        qs = np.asarray(cdf.ppf(u_grid), dtype=np.float64)
        if qs.shape == u_grid.shape:
            return qs
    except Exception:
        pass

    return np.array([cdf.ppf(float(u)) for u in u_grid], dtype=np.float64)


def gridwise_calibration_error(
    y: np.ndarray,
    calibrated_quantiles: np.ndarray,
    J_eval: int = 99,
) -> float:
    """Compute the gridwise L2 calibration error E_r.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Observed outcomes.
    calibrated_quantiles : ndarray of shape (n, J_eval)
        Calibrated quantiles at each evaluation grid point.
        calibrated_quantiles[t, j] = q_tilde_t(u_j).
    J_eval : int
        Number of evaluation grid points.

    Returns
    -------
    float
        E_r = sqrt((1/J) * sum_j (mean_t 1{y_t <= q_j} - u_j)^2).
    """
    y = np.asarray(y)
    Q = np.asarray(calibrated_quantiles)
    n = len(y)
    grid = np.array([j / (J_eval + 1) for j in range(1, J_eval + 1)])

    # Indicators: (n, J_eval)
    indicators = (y[:, None] <= Q).astype(np.float64)

    # Empirical coverage at each grid point
    emp_cov = indicators.mean(axis=0)  # shape (J_eval,)

    # Squared deviations
    sq_dev = (emp_cov - grid) ** 2

    # E_r
    return float(np.sqrt(np.mean(sq_dev)))


def marginal_coverage(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical marginal coverage.

    Parameters
    ----------
    y : ndarray of shape (n,)
    lower : ndarray of shape (n,)
        Lower interval endpoints.
    upper : ndarray of shape (n,)
        Upper interval endpoints.

    Returns
    -------
    float
        Fraction of y_t in [lower_t, upper_t].
    """
    y = np.asarray(y)
    covered = (y >= np.asarray(lower)) & (y <= np.asarray(upper))
    return float(np.mean(covered))


def mean_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute mean interval width."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def adaptation_lag(
    rolling_cov: np.ndarray,
    target: float,
    eps: float = 0.03,
    min_consecutive: int = 2,
) -> int | None:
    """Compute adaptation lag after a regime boundary.

    Parameters
    ----------
    rolling_cov : ndarray
        Rolling coverage values starting from the regime boundary.
    target : float
        Target coverage level (1 - alpha).
    eps : float
        Tolerance band half-width.
    min_consecutive : int
        Number of consecutive windows that must be within band.

    Returns
    -------
    int or None
        Smallest k such that the next ``min_consecutive`` windows are
        within [target - eps, target + eps]. None if censored.
    """
    rolling_cov = np.asarray(rolling_cov)
    in_band = np.abs(rolling_cov - target) <= eps

    for k in range(len(rolling_cov) - min_consecutive + 1):
        if np.isnan(rolling_cov[k]):
            continue
        if all(in_band[k:k + min_consecutive]):
            return k

    return None  # Censored


def crps(
    y: np.ndarray,
    cdf_list: list,
    n_quadrature: int = 200,
) -> float:
    """Compute mean CRPS for a sequence of CDF forecasts.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Observed outcomes.
    cdf_list : list of ForecastCDF-like objects
        Each must support .cdf(y) and .ppf(u).
    n_quadrature : int
        Number of quadrature points for numerical integration.

    Returns
    -------
    float
        Mean CRPS across all observations.
    """
    y = np.asarray(y)
    n = len(y)
    if n != len(cdf_list):
        raise ValueError("y and cdf_list must have same length")

    u_grid = np.linspace(0, 1, n_quadrature + 2)[1:-1]  # exclude 0, 1
    du = 1.0 / (n_quadrature + 1)

    total = 0.0
    for i in range(n):
        # Use the quantile-score decomposition:
        # CRPS = integral_0^1 QS(u, y) du where QS(u, y) = 2(1{y<=q(u)} - u)(q(u) - y)
        y_i = float(y[i])
        qs = _ppf_many(cdf_list[i], u_grid)
        indicators = (y_i <= qs).astype(np.float64)
        integrand = 2.0 * (indicators - u_grid) * (qs - y_i)
        total += float(np.sum(integrand) * du)

    return float(total / n)
