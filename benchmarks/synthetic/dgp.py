"""Synthetic benchmark DGP generator.

Implements all 7 scenario families from BenchmarkSpec_P1_final §9:
A. Piecewise-stationary PITs
B. Bounded-drift PITs
C. Abrupt regime change
D. Oscillating short regimes
E. Strong within-regime dependence
F. Heavy-tailed noise
G. Gradually drifting forecaster
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Tuple

import numpy as np
from scipy.stats import norm, t as t_dist


@dataclass
class SyntheticStream:
    """A single synthetic benchmark stream."""
    y: np.ndarray
    forecast_mu: np.ndarray
    forecast_sigma: np.ndarray
    regime_ids: np.ndarray
    changepoints: List[int]
    family: str
    params: dict

    def replace_family(self, family: str, params: dict) -> "SyntheticStream":
        return replace(self, family=family, params=params)


def _ar1_innovations(T: int, phi: float, rng, heavy_tail_nu: int = None):
    """Generate AR(1)-copula latent process."""
    Z = np.zeros(T)
    for t in range(1, T):
        xi = rng.standard_t(heavy_tail_nu) if heavy_tail_nu else rng.normal()
        Z[t] = phi * Z[t - 1] + np.sqrt(max(1 - phi ** 2, 1e-10)) * xi
    return Z


# -----------------------------------------------------------------------
# Family A: Piecewise-stationary
# -----------------------------------------------------------------------

def generate_family_A(
    R_reg: int = 3,
    L: int = 500,
    biases: List[float] = None,
    scales: List[float] = None,
    phi: float = 0.5,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    if biases is None:
        biases = [0.0, 0.5, -0.3][:R_reg]
    if scales is None:
        scales = [1.0, 0.7, 1.3][:R_reg]

    T_total = T_burn + R_reg * L
    rng = np.random.default_rng(seed)
    Z = _ar1_innovations(T_total, phi, rng)

    y = Z.copy()  # true mu=0, sigma=1
    mu_hat = np.zeros(T_total)
    sigma_hat = np.ones(T_total)
    regime_ids = np.zeros(T_total, dtype=int)
    changepoints = []

    for r in range(R_reg):
        start = T_burn + r * L
        end = T_burn + (r + 1) * L
        mu_hat[start:end] = biases[r]
        sigma_hat[start:end] = scales[r]
        regime_ids[start:end] = r
        if r > 0:
            changepoints.append(start)

    # Trim burn-in
    y = y[T_burn:]
    mu_hat = mu_hat[T_burn:]
    sigma_hat = sigma_hat[T_burn:]
    regime_ids = regime_ids[T_burn:]
    changepoints = [c - T_burn for c in changepoints]

    return SyntheticStream(
        y=y, forecast_mu=mu_hat, forecast_sigma=sigma_hat,
        regime_ids=regime_ids, changepoints=changepoints,
        family="A", params={"R_reg": R_reg, "L": L, "biases": biases,
                            "scales": scales, "phi": phi},
    )


# -----------------------------------------------------------------------
# Family B: Bounded-drift
# -----------------------------------------------------------------------

def generate_family_B(
    L: int = 1000,
    lambda_r: float = 0.3,
    s_start: float = 1.0,
    phi: float = 0.5,
    n_regimes: int = 3,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    T_total = T_burn + n_regimes * L
    rng = np.random.default_rng(seed)
    Z = _ar1_innovations(T_total, phi, rng)

    y = Z.copy()
    mu_hat = np.zeros(T_total)
    sigma_hat = np.ones(T_total)
    regime_ids = np.zeros(T_total, dtype=int)
    changepoints = []

    for r in range(n_regimes):
        start = T_burn + r * L
        end = T_burn + (r + 1) * L
        for t_idx in range(start, end):
            frac = (t_idx - start) / max(L - 1, 1)
            sigma_hat[t_idx] = s_start + lambda_r * frac
        regime_ids[start:end] = r
        if r > 0:
            changepoints.append(start)

    y = y[T_burn:]
    mu_hat = mu_hat[T_burn:]
    sigma_hat = sigma_hat[T_burn:]
    regime_ids = regime_ids[T_burn:]
    changepoints = [c - T_burn for c in changepoints]

    return SyntheticStream(
        y=y, forecast_mu=mu_hat, forecast_sigma=sigma_hat,
        regime_ids=regime_ids, changepoints=changepoints,
        family="B", params={"L": L, "lambda_r": lambda_r,
                            "s_start": s_start, "phi": phi},
    )


# -----------------------------------------------------------------------
# Family C: Abrupt regime change
# -----------------------------------------------------------------------

def generate_family_C(
    L: int = 500,
    Delta_b: float = 1.0,
    Delta_s: float = 0.5,
    phi: float = 0.5,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    T_total = T_burn + 3 * L
    rng = np.random.default_rng(seed)
    Z = _ar1_innovations(T_total, phi, rng)

    y = Z.copy()
    mu_hat = np.zeros(T_total)
    sigma_hat = np.ones(T_total)
    regime_ids = np.zeros(T_total, dtype=int)

    # Regime 0: well-calibrated
    s0 = T_burn
    # Regime 1: jump
    s1 = T_burn + L
    mu_hat[s1:s1 + L] = Delta_b
    sigma_hat[s1:s1 + L] = 1.0 + Delta_s
    regime_ids[s1:s1 + L] = 1
    # Regime 2: back
    s2 = T_burn + 2 * L
    regime_ids[s2:s2 + L] = 2

    y = y[T_burn:]
    mu_hat = mu_hat[T_burn:]
    sigma_hat = sigma_hat[T_burn:]
    regime_ids = regime_ids[T_burn:]

    return SyntheticStream(
        y=y, forecast_mu=mu_hat, forecast_sigma=sigma_hat,
        regime_ids=regime_ids, changepoints=[L, 2 * L],
        family="C", params={"L": L, "Delta_b": Delta_b,
                            "Delta_s": Delta_s, "phi": phi},
    )


# -----------------------------------------------------------------------
# Family D: Oscillating short regimes
# -----------------------------------------------------------------------

def generate_family_D(
    L_short: int = 50,
    phi: float = 0.5,
    T_scored: int = 5000,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    T_total = T_burn + T_scored
    rng = np.random.default_rng(seed)
    Z = _ar1_innovations(T_total, phi, rng)

    y = Z.copy()
    mu_hat = np.zeros(T_total)
    sigma_hat = np.ones(T_total)
    regime_ids = np.zeros(T_total, dtype=int)
    changepoints = []

    t = T_burn
    r = 0
    while t < T_total:
        end = min(t + L_short, T_total)
        if r % 2 == 0:
            sigma_hat[t:end] = 0.7
            mu_hat[t:end] = 0.5
        else:
            sigma_hat[t:end] = 1.3
            mu_hat[t:end] = -0.5
        regime_ids[t:end] = r
        if r > 0:
            changepoints.append(t)
        t = end
        r += 1

    y = y[T_burn:]
    mu_hat = mu_hat[T_burn:]
    sigma_hat = sigma_hat[T_burn:]
    regime_ids = regime_ids[T_burn:]
    changepoints = [c - T_burn for c in changepoints if c >= T_burn]

    return SyntheticStream(
        y=y, forecast_mu=mu_hat, forecast_sigma=sigma_hat,
        regime_ids=regime_ids, changepoints=changepoints,
        family="D", params={"L_short": L_short, "phi": phi},
    )


# -----------------------------------------------------------------------
# Family E: Strong within-regime dependence
# -----------------------------------------------------------------------

def generate_family_E(
    phi: float = 0.9,
    L: int = 1000,
    s: float = 1.2,
    n_regimes: int = 3,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    return generate_family_A(
        R_reg=n_regimes, L=L,
        biases=[0.0] * n_regimes,
        scales=[s] * n_regimes,
        phi=phi, T_burn=T_burn, seed=seed,
    ).replace_family("E", {"phi": phi, "L": L, "s": s})


# -----------------------------------------------------------------------
# Family F: Heavy-tailed noise
# -----------------------------------------------------------------------

def generate_family_F(
    nu: int = 5,
    s: float = 1.0,
    phi: float = 0.5,
    L: int = 1000,
    n_regimes: int = 3,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    T_total = T_burn + n_regimes * L
    rng = np.random.default_rng(seed)
    Z = _ar1_innovations(T_total, phi, rng, heavy_tail_nu=nu)
    # Standardize t innovations
    Z = Z / np.sqrt(nu / (nu - 2)) if nu > 2 else Z

    y = Z.copy()
    mu_hat = np.zeros(T_total)
    sigma_hat = np.full(T_total, s)
    regime_ids = np.zeros(T_total, dtype=int)

    for r in range(n_regimes):
        start = T_burn + r * L
        end = T_burn + (r + 1) * L
        regime_ids[start:end] = r

    y = y[T_burn:]
    mu_hat = mu_hat[T_burn:]
    sigma_hat = sigma_hat[T_burn:]
    regime_ids = regime_ids[T_burn:]

    return SyntheticStream(
        y=y, forecast_mu=mu_hat, forecast_sigma=sigma_hat,
        regime_ids=regime_ids, changepoints=[r * L for r in range(1, n_regimes)],
        family="F", params={"nu": nu, "s": s, "phi": phi, "L": L},
    )


# -----------------------------------------------------------------------
# Family G: Gradually drifting forecaster
# -----------------------------------------------------------------------

def generate_family_G(
    gamma: float = 0.4,
    phi: float = 0.5,
    T_scored: int = 5000,
    T_burn: int = 500,
    seed: int = 42,
) -> SyntheticStream:
    T_total = T_burn + T_scored
    rng = np.random.default_rng(seed)
    Z = _ar1_innovations(T_total, phi, rng)

    y = Z.copy()  # True law: N(0, 1) + dependence
    mu_hat = np.zeros(T_total)
    sigma_hat = np.ones(T_total)

    # Forecaster scale drifts
    for t in range(T_total):
        sigma_hat[t] = np.exp(gamma * (t / T_total - 0.5))

    y = y[T_burn:]
    mu_hat = mu_hat[T_burn:]
    sigma_hat = sigma_hat[T_burn:]

    return SyntheticStream(
        y=y, forecast_mu=mu_hat, forecast_sigma=sigma_hat,
        regime_ids=np.zeros(len(y), dtype=int),
        changepoints=[],
        family="G", params={"gamma": gamma, "phi": phi},
    )
