"""Example 3: Rolling online evaluation with save/load continuation.

Demonstrates:
1. A benchmark loop over a single synthetic series
2. Aligned scoring over time
3. Rolling coverage computation
4. Warning collection
5. Saving and reloading the calibrator state mid-stream
6. Equality check between serialized and non-serialized continuation
"""

import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from tsconformal import (
    CUSUMNormDetector,
    SegmentedTransportCalibrator,
    save_calibrator,
    load_calibrator,
)
from tsconformal.diagnostics import rolling_coverage
from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF


def run_example(seed: int = 42, T: int = 600, save_at: int = 300):
    """Run a rolling evaluation with mid-stream save/load verification."""
    rng = np.random.default_rng(seed)

    # Generate two-regime synthetic stream
    y_data = np.concatenate([
        rng.normal(0, 1, T // 2),
        rng.normal(1, 0.7, T - T // 2),
    ])

    # Miscalibrated forecaster
    def make_forecast(t):
        return GaussianForecastCDF(mu=0.3, sigma=0.8)

    # Create calibrator
    def step_schedule(n):
        return min(0.20, 1.0 / max(n**0.5, 1e-8))

    detector = CUSUMNormDetector(kappa=0.02, threshold=0.20)
    cal = SegmentedTransportCalibrator(
        grid_size=19,
        rho=0.95,
        n_eff_min=10.0,
        step_schedule=step_schedule,
        detector=detector,
        cooldown=50,
        confirm=2,
    )

    # Online loop — first half
    alpha = 0.10
    covered = []
    widths = []
    all_warnings = []

    for t in range(save_at):
        cdf = make_forecast(t)
        cal_cdf = cal.predict_cdf(cdf)

        lower = cal_cdf.ppf(alpha / 2)
        upper = cal_cdf.ppf(1.0 - alpha / 2)
        cov = float(lower <= y_data[t] <= upper)
        covered.append(cov)
        widths.append(upper - lower)

        cal.update(y_data[t], cdf)
        if cal.warnings:
            all_warnings.extend(cal.warnings)

    print(f"=== Phase 1 (t=0..{save_at-1}) ===")
    print(f"  Coverage: {np.mean(covered):.3f} (target: {1-alpha:.2f})")
    print(f"  Mean width: {np.mean(widths):.3f}")
    print(f"  Resets: {cal.num_resets}")
    print(f"  Warnings collected: {len(all_warnings)}")

    # Save state mid-stream
    save_dir = Path(tempfile.mkdtemp()) / "cal_state"
    save_calibrator(cal, save_dir)
    print(f"  Saved at t={save_at}")

    # Continue original for second half
    orig_predictions = []
    for t in range(save_at, T):
        cdf = make_forecast(t)
        cal_cdf = cal.predict_cdf(cdf)
        orig_predictions.append(cal_cdf.cdf(0.5))

        lower = cal_cdf.ppf(alpha / 2)
        upper = cal_cdf.ppf(1.0 - alpha / 2)
        covered.append(float(lower <= y_data[t] <= upper))
        widths.append(upper - lower)

        cal.update(y_data[t], cdf)

    # Reload and continue
    cal_loaded = load_calibrator(save_dir, step_schedule=step_schedule)
    loaded_predictions = []
    for t in range(save_at, T):
        cdf = make_forecast(t)
        cal_cdf = cal_loaded.predict_cdf(cdf)
        loaded_predictions.append(cal_cdf.cdf(0.5))
        cal_loaded.update(y_data[t], cdf)

    # Verify exact continuation
    assert_allclose(orig_predictions, loaded_predictions, atol=1e-10,
                    err_msg="Loaded calibrator diverged!")
    print("\n=== Serialization verification ===")
    print(f"  Predictions match: YES (max diff: {np.max(np.abs(np.array(orig_predictions) - np.array(loaded_predictions))):.2e})")
    print(f"  segment_id match: {cal.segment_id == cal_loaded.segment_id}")
    print(f"  num_resets match: {cal.num_resets == cal_loaded.num_resets}")
    print(f"  n_eff match: {abs(cal.n_eff - cal_loaded.n_eff) < 1e-10}")

    # Rolling coverage
    roll_cov = rolling_coverage(np.array(covered), window=100)
    valid_roll = roll_cov[~np.isnan(roll_cov)]

    print(f"\n=== Phase 2 (t=0..{T-1}) ===")
    print(f"  Overall coverage: {np.mean(covered):.3f}")
    print(f"  Mean width: {np.mean(widths):.3f}")
    print(f"  Rolling coverage MAD: {np.mean(np.abs(valid_roll - (1-alpha))):.4f}")
    print(f"  Total resets: {cal.num_resets}")

    return cal


if __name__ == "__main__":
    run_example()
