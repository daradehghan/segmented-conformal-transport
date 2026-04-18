"""Example 2: Chronos-2 wrapper with validation and skip-and-log.

Demonstrates wrapping a quantile-only foundation model output
with QuantileGridCDFAdapter, validating it, and applying SCT.

This example uses a synthetic Chronos-like output because the actual
Chronos-2 package is an optional dependency.
"""

import numpy as np

from tsconformal import (
    CUSUMNormDetector,
    QuantileGridCDFAdapter,
    SegmentedTransportCalibrator,
    validate_forecast_cdf,
)
from tsconformal.forecast import InvalidForecastCDFError


def simulate_chronos_output(y_history, rng):
    """Simulate a Chronos-2-like quantile forecast output.

    In a real workflow, this would call:
        model.predict_quantiles(context, quantile_levels=Q)

    Returns probabilities and quantile values.
    """
    Q = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    mu = np.mean(y_history[-50:]) if len(y_history) >= 50 else 0.0
    sigma = np.std(y_history[-50:]) if len(y_history) >= 50 else 1.0
    sigma = max(sigma, 0.01)

    from scipy.stats import norm
    quantiles = norm.ppf(Q, loc=mu, scale=sigma)

    # Simulate occasional non-monotone outputs (FM artifact)
    if rng.random() < 0.05:
        i, j = rng.choice(len(Q), 2, replace=False)
        quantiles[i], quantiles[j] = quantiles[j], quantiles[i]

    return Q, quantiles


def run_example(seed: int = 42, T: int = 500):
    """Run a Chronos-2 wrapper calibration example with skip-and-log."""
    rng = np.random.default_rng(seed)

    # Generate synthetic data with a regime change
    y_data = np.concatenate([
        rng.normal(0, 1, 250),
        rng.normal(2, 0.5, 250),  # mean shift + variance change
    ])

    # Create calibrator
    detector = CUSUMNormDetector(kappa=0.02, threshold=0.20)
    cal = SegmentedTransportCalibrator(
        grid_size=49,
        rho=0.99,
        n_eff_min=30,
        step_schedule=lambda n: min(0.20, 1.0 / max(n ** 0.5, 1e-8)),
        detector=detector,
        cooldown=100,
        confirm=3,
    )

    fm_failures = 0
    calibrated_pits = []
    pit_rng = np.random.default_rng(0)

    for t in range(T):
        # Get "FM output"
        context = y_data[max(0, t - 100):t] if t > 0 else np.array([0.0])
        Q, quantiles = simulate_chronos_output(context, rng)

        # Wrap with adapter. This CDF is adapter-defined rather than
        # model-native analytic.
        try:
            base_cdf = QuantileGridCDFAdapter(probabilities=Q, quantiles=quantiles)
            validate_forecast_cdf(base_cdf)
        except (InvalidForecastCDFError, Exception):
            # Skip-and-log policy: do not replace with fallback
            fm_failures += 1
            continue

        # Predict and update
        cal_cdf = cal.predict_cdf(base_cdf)
        cal.update(y_data[t], base_cdf, rng=pit_rng)
        calibrated_pits.append(cal_cdf.cdf(y_data[t]))

    print(f"Total timestamps: {T}")
    print(f"FM failures (skipped): {fm_failures}")
    print(f"FM failure rate: {fm_failures / T:.3f}")
    print(f"Scored timestamps: {len(calibrated_pits)}")
    print(f"Resets: {cal.num_resets}")
    print(f"Final n_eff: {cal.n_eff:.1f}")
    print(f"Final warm_start_weight: {cal.warm_start_weight:.4f}")

    return cal, np.array(calibrated_pits)


if __name__ == "__main__":
    run_example()
