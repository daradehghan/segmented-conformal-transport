"""Example 1: Synthetic piecewise-stationary calibration.

Demonstrates SCT on a three-regime synthetic stream with known
miscalibration patterns.
"""

import numpy as np

from tsconformal import (
    CUSUMNormDetector,
    QuantileGridCDFAdapter,
    SegmentedTransportCalibrator,
)


class GaussianForecastCDF:
    """Simple Gaussian forecast CDF for synthetic experiments."""
    is_discrete = False

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = max(sigma, 1e-10)

    def cdf(self, y: float) -> float:
        from scipy.stats import norm
        return float(norm.cdf(y, self.mu, self.sigma))

    def ppf(self, u: float) -> float:
        from scipy.stats import norm
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return float(norm.ppf(u, self.mu, self.sigma))


def run_example(seed: int = 42, T: int = 1500):
    """Run a three-regime synthetic calibration experiment."""
    rng = np.random.default_rng(seed)

    # Define three regimes with different miscalibration
    regimes = [
        {"length": 500, "bias": 0.0, "scale": 1.0},   # well-calibrated
        {"length": 500, "bias": 0.5, "scale": 0.7},    # biased + overconfident
        {"length": 500, "bias": -0.3, "scale": 1.3},   # biased + underconfident
    ]

    # Generate data
    ys = []
    forecasts = []
    oracle_regimes = []
    for r_id, regime in enumerate(regimes):
        n = regime["length"]
        y = rng.normal(0, 1, size=n)
        for yi in y:
            # Miscalibrated forecast
            f = GaussianForecastCDF(
                mu=regime["bias"],
                sigma=regime["scale"],
            )
            ys.append(yi)
            forecasts.append(f)
            oracle_regimes.append(r_id)

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

    # Online loop
    pit_values = []
    warmup_flags = []
    ws_weights = []
    calibrated_quantiles_list = []
    J_eval = 99
    eval_grid = np.array([j / (J_eval + 1) for j in range(1, J_eval + 1)])

    for t in range(len(ys)):
        # Predict
        cal_cdf = cal.predict_cdf(forecasts[t])
        # Record calibrated quantiles for E_r
        cal_q = np.array([cal_cdf.ppf(u) for u in eval_grid])
        calibrated_quantiles_list.append(cal_q)
        # Track diagnostics
        warmup_flags.append(cal.in_warmup)
        ws_weights.append(cal.warm_start_weight)
        # Update
        cal.update(ys[t], forecasts[t])
        # Record calibrated PIT
        pit_values.append(cal_cdf.cdf(ys[t]))

    pit_values = np.array(pit_values)
    calibrated_quantiles = np.array(calibrated_quantiles_list)

    # Summary
    print(f"Total timestamps: {len(ys)}")
    print(f"Resets: {cal.num_resets}")
    print(f"Final segment: {cal.segment_id}")
    print(f"Final n_eff: {cal.n_eff:.1f}")

    # Per-regime reporting
    from tsconformal.metrics import gridwise_calibration_error
    from tsconformal.diagnostics import warm_start_occupancy
    for r_id in range(3):
        mask = np.array(oracle_regimes) == r_id
        regime_pits = pit_values[mask]
        cov = np.mean((regime_pits >= 0.05) & (regime_pits <= 0.95))
        E_r = gridwise_calibration_error(
            np.array(ys)[mask], calibrated_quantiles[mask], J_eval
        )
        ws_occ = warm_start_occupancy(
            np.array(warmup_flags)[mask],
            np.array(ws_weights)[mask],
        )
        print(f"Regime {r_id}: 90% cov={cov:.3f}, E_r={E_r:.4f}, "
              f"warm_start_occ={ws_occ:.3f}")

    return cal, pit_values


if __name__ == "__main__":
    run_example()
