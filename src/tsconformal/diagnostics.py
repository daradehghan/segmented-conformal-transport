"""Diagnostics, PIT analysis, and detector sensitivity tooling.

This module provides PIT uniformity tests, rolling coverage,
warm-start occupancy tracking, and detector sensitivity summaries used
by the benchmarks and diagnostics workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
import warnings

import numpy as np
from scipy import stats


# -----------------------------------------------------------------------
# PIT diagnostics
# -----------------------------------------------------------------------

@dataclass
class PITDiagnostics:
    """Results of PIT uniformity tests."""
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ad_pvalue: float
    ljung_box_statistic: Optional[float] = None
    ljung_box_pvalue: Optional[float] = None


def _anderson_normal_pvalue(x: np.ndarray) -> tuple[float, float]:
    """Return Anderson-Darling statistic and p-value for a normal sample.

    SciPy 1.17+ warns unless an explicit p-value method is provided. Prefer the
    built-in interpolated p-value when available, and fall back to the older
    critical-value interpolation on older SciPy versions.
    """
    try:
        ad_result = stats.anderson(x, dist="norm", method="interpolate")
    except TypeError:
        ad_result = stats.anderson(x, dist="norm")
        ad_stat = float(ad_result.statistic)
        crit_vals = ad_result.critical_values  # at 15%, 10%, 5%, 2.5%, 1%
        sig_levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01])
        if ad_stat < crit_vals[0]:
            ad_p = 0.25
        elif ad_stat > crit_vals[-1]:
            ad_p = 0.005
        else:
            ad_p = float(np.interp(ad_stat, crit_vals, sig_levels))
        return ad_stat, ad_p

    return float(ad_result.statistic), float(ad_result.pvalue)


def pit_uniformity_tests(
    pits: np.ndarray,
    lags: int = 10,
) -> PITDiagnostics:
    """Run KS, Anderson-Darling, and Ljung-Box tests on PIT values.

    Parameters
    ----------
    pits : ndarray
        PIT values in [0, 1].
    lags : int
        Lags for Ljung-Box test on centered PITs.
    """
    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]

    if len(pits) < 10:
        return PITDiagnostics(
            ks_statistic=np.nan, ks_pvalue=np.nan,
            ad_statistic=np.nan, ad_pvalue=np.nan,
        )

    # KS test against Uniform(0, 1)
    ks_stat, ks_p = stats.kstest(pits, "uniform")

    # Anderson-Darling against Uniform(0, 1)
    # Transform PITs to standard normal for AD test, then use normal AD
    # This is a valid approach: if PITs ~ Uniform, then Phi^{-1}(PITs) ~ N(0,1)
    transformed = stats.norm.ppf(np.clip(pits, 1e-10, 1 - 1e-10))
    ad_stat, ad_p = _anderson_normal_pvalue(transformed)

    # Ljung-Box on centered PITs
    centered = pits - 0.5
    lb_stat, lb_p = None, None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(centered, lags=[lags], return_df=True)
        lb_stat = float(lb_result["lb_stat"].iloc[0])
        lb_p = float(lb_result["lb_pvalue"].iloc[0])
    except ImportError:
        pass
    except Exception as exc:
        warnings.warn(
            f"Ljung-Box failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    return PITDiagnostics(
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        ad_statistic=ad_stat,
        ad_pvalue=float(ad_p),
        ljung_box_statistic=lb_stat,
        ljung_box_pvalue=lb_p,
    )


# -----------------------------------------------------------------------
# Rolling coverage
# -----------------------------------------------------------------------

def rolling_coverage(
    covered: np.ndarray,
    window: int = 168,
) -> np.ndarray:
    """Compute rolling coverage from a boolean array.

    Parameters
    ----------
    covered : ndarray of bool
        1 if the outcome was covered by the interval, 0 otherwise.
    window : int
        Rolling window size.

    Returns
    -------
    ndarray of float
        Rolling coverage with NaN for initial positions.
    """
    covered = np.asarray(covered, dtype=np.float64)
    n = len(covered)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cumsum = np.cumsum(covered)
    result[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    return result


# -----------------------------------------------------------------------
# Warm-start occupancy
# -----------------------------------------------------------------------

def warm_start_occupancy(
    in_warmup_flags: np.ndarray,
    warm_start_weights: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """Compute warm-start occupancy for a scored regime.

    Parameters
    ----------
    in_warmup_flags : ndarray of bool
        True if SCT was in identity fallback.
    warm_start_weights : ndarray of float
        warm_start_weight at each timestamp.
    threshold : float
        Weight threshold for counting as warm-start dominated.

    Returns
    -------
    float
        Fraction of timestamps in warm-start or with weight > threshold.
    """
    flags = np.asarray(in_warmup_flags, dtype=bool)
    weights = np.asarray(warm_start_weights, dtype=np.float64)
    dominated = flags | (weights > threshold)
    return float(np.mean(dominated))


# -----------------------------------------------------------------------
# Sensitivity report
# -----------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Configuration for a single detector variant."""
    detector_type: str = "CUSUMNorm"
    kappa: float = 0.02
    threshold: float = 0.20
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensitivityResult:
    """Result for one detector configuration."""
    config: DetectorConfig
    n_resets: int = 0
    mean_segment_length: float = 0.0
    warm_start_occupancy: float = 0.0


@dataclass
class SensitivityReport:
    """Full sensitivity report across detector configurations."""
    results: List[SensitivityResult] = field(default_factory=list)
    detector_shopping_risk: bool = False
    variability_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [
                {
                    "config": vars(r.config),
                    "n_resets": r.n_resets,
                    "mean_segment_length": r.mean_segment_length,
                    "warm_start_occupancy": r.warm_start_occupancy,
                }
                for r in self.results
            ],
            "detector_shopping_risk": self.detector_shopping_risk,
            "variability_summary": self.variability_summary,
        }


def sensitivity_report(
    detector_grid: List[DetectorConfig],
    data_stream: Any,
    pit_seed: int = 20260306,
) -> SensitivityReport:
    """Run a standardized detector sensitivity sweep.

    Parameters
    ----------
    detector_grid : list of DetectorConfig
        Detector configurations to evaluate.
    data_stream : iterable of (ForecastCDF, float)
        Stream of (base_cdf, y_t) pairs. Warm-start occupancy is
        computed on the same valid pre-update timestamps used by the
        benchmark runners: fallback steps plus timestamps with
        ``warm_start_weight > 0.10``. Invalid forecasts are skipped and
        do not contribute to occupancy or mean segment length.
    pit_seed : int, default 20260306
        Seed used for randomized PIT draws when a forecast is discrete.

    Returns
    -------
    SensitivityReport
        Summary of metrics across configurations.
    """
    from tsconformal._benchmark_defaults import build_sct_calibrator, make_pit_rng
    from tsconformal.detectors import CUSUMNormDetector, PageHinkleyDetector
    from tsconformal.forecast import InvalidForecastCDFError

    # Materialize the stream once (needed for re-running per config)
    stream_list = list(data_stream)
    if not stream_list:
        return SensitivityReport()

    report = SensitivityReport()

    for config in detector_grid:
        pit_rng = make_pit_rng(pit_seed)
        # Build detector from config
        if config.detector_type == "CUSUMNorm":
            det = CUSUMNormDetector(
                kappa=config.kappa, threshold=config.threshold
            )
        elif config.detector_type == "PageHinkley":
            det = PageHinkleyDetector(
                delta=config.extra.get("delta", 0.01),
                threshold=config.threshold,
            )
        else:
            continue

        # Build calibrator with this detector
        grid_size = config.extra.get("grid_size", 49)
        cal = build_sct_calibrator(
            grid_size=grid_size,
            rho=config.extra.get("rho", 0.99),
            n_eff_min=config.extra.get("n_eff_min", 50.0),
            detector=det,
            cooldown=config.extra.get("cooldown", 168),
            confirm=config.extra.get("confirm", 3),
        )

        # Run online loop
        warmup_flags: List[bool] = []
        warm_start_weights: List[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for base_cdf, y_t in stream_list:
                try:
                    cal.predict_cdf(base_cdf)
                    warmup_flags.append(cal.in_warmup)
                    warm_start_weights.append(cal.warm_start_weight)
                    cal.update(y_t, base_cdf, rng=pit_rng)
                except InvalidForecastCDFError:
                    continue

        valid_steps = len(warmup_flags)
        n_resets = cal.num_resets
        mean_seg_len = (
            valid_steps / max(n_resets + 1, 1) if valid_steps else float("nan")
        )
        if valid_steps:
            ws_occ = warm_start_occupancy(
                np.asarray(warmup_flags, dtype=bool),
                np.asarray(warm_start_weights, dtype=np.float64),
            )
        else:
            ws_occ = float("nan")

        result = SensitivityResult(
            config=config,
            n_resets=n_resets,
            mean_segment_length=mean_seg_len,
            warm_start_occupancy=ws_occ,
        )
        report.results.append(result)

    # Detect shopping risk: high variability in reset counts
    if len(report.results) > 1:
        reset_counts = [r.n_resets for r in report.results]
        rc_range = max(reset_counts) - min(reset_counts)
        rc_cv = float(np.std(reset_counts) / max(np.mean(reset_counts), 1e-8))
        report.detector_shopping_risk = rc_cv > 0.5
        report.variability_summary = {
            "reset_count_range": float(rc_range),
            "reset_count_cv": rc_cv,
            "min_resets": min(reset_counts),
            "max_resets": max(reset_counts),
        }

    return report
