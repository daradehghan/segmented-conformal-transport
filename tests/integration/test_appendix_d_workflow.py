import warnings

import numpy as np
from numpy.testing import assert_allclose


class TriggerOnceDetector:
    def __init__(self, trigger_at=28, calls=0, triggered=False):
        self.trigger_at = trigger_at
        self.calls = calls
        self.triggered = triggered

    def update(self, residual_vector):
        self.calls += 1
        if not self.triggered and self.calls == self.trigger_at:
            self.triggered = True
            return True
        return False

    def state(self):
        return {
            "detector_type": "TriggerOnce",
            "trigger_at": self.trigger_at,
            "calls": self.calls,
            "triggered": self.triggered,
        }

    def reset(self):
        return None


def _make_stream(length: int = 80, seed: int = 20260416):
    from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    mu = 0.6 * np.sin(t / 6.0)
    sigma = 0.45 + 0.05 * ((t.astype(int) % 5))
    y = mu + 0.25 * np.cos(t / 4.0) + 0.15 * rng.normal(size=length)

    return [
        (GaussianForecastCDF(float(mu_t), float(sig_t)), float(y_t))
        for mu_t, sig_t, y_t in zip(mu, sigma, y)
    ]


def _collect_continuation(calibrator, stream):
    from tsconformal.diagnostics import pit_uniformity_tests, rolling_coverage
    from tsconformal.metrics import marginal_coverage, mean_interval_width

    y = []
    lower = []
    upper = []
    pits = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for base_cdf, y_t in stream:
            calibrated = calibrator.predict_cdf(base_cdf)
            lower.append(float(calibrated.ppf(0.05)))
            upper.append(float(calibrated.ppf(0.95)))
            pits.append(float(calibrated.cdf(y_t)))
            y.append(float(y_t))
            calibrator.update(y_t, base_cdf)

    y_arr = np.asarray(y)
    lo_arr = np.asarray(lower)
    hi_arr = np.asarray(upper)
    pits_arr = np.asarray(pits)
    covered = (y_arr >= lo_arr) & (y_arr <= hi_arr)

    return {
        "y": y_arr,
        "lower": lo_arr,
        "upper": hi_arr,
        "pits": pits_arr,
        "coverage": marginal_coverage(y_arr, lo_arr, hi_arr),
        "width": mean_interval_width(lo_arr, hi_arr),
        "rolling_coverage": rolling_coverage(covered, window=10),
        "pit_diag": pit_uniformity_tests(pits_arr),
    }


def test_appendix_d_workflow_save_load_and_continue(tmp_path):
    """The seeded Appendix D workflow must continue exactly after save/load."""
    from tsconformal import SegmentedTransportCalibrator, load_calibrator, save_calibrator

    def step_schedule(n_eff):
        return min(0.20, 1.0 / max(n_eff**0.5, 1.0))

    def detector_factory(state):
        return TriggerOnceDetector(
            trigger_at=state["trigger_at"],
            calls=state["calls"],
            triggered=state["triggered"],
        )

    stream = _make_stream()
    split = 40
    head = stream[:split]
    tail = stream[split:]

    calibrator = SegmentedTransportCalibrator(
        grid_size=21,
        rho=0.97,
        n_eff_min=5.0,
        step_schedule=step_schedule,
        detector=TriggerOnceDetector(trigger_at=28),
        cooldown=1,
        confirm=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for base_cdf, y_t in head:
            calibrator.predict_cdf(base_cdf)
            calibrator.update(y_t, base_cdf)

    assert calibrator.num_resets == 1
    assert calibrator.segment_id == 1

    save_path = tmp_path / "workflow_state"
    save_calibrator(calibrator, save_path)

    loaded = load_calibrator(
        save_path,
        detector_factory=detector_factory,
        step_schedule=step_schedule,
    )

    original_metrics = _collect_continuation(calibrator, tail)
    loaded_metrics = _collect_continuation(loaded, tail)

    assert calibrator.num_resets == loaded.num_resets == 1
    assert calibrator.segment_id == loaded.segment_id == 1
    assert calibrator.last_reset_t == loaded.last_reset_t == 28

    assert_allclose(original_metrics["lower"], loaded_metrics["lower"], atol=1e-12)
    assert_allclose(original_metrics["upper"], loaded_metrics["upper"], atol=1e-12)
    assert_allclose(original_metrics["pits"], loaded_metrics["pits"], atol=1e-12)
    assert_allclose(
        original_metrics["rolling_coverage"],
        loaded_metrics["rolling_coverage"],
        atol=1e-12,
        equal_nan=True,
    )
    assert_allclose(original_metrics["coverage"], loaded_metrics["coverage"], atol=1e-12)
    assert_allclose(original_metrics["width"], loaded_metrics["width"], atol=1e-12)

    original_diag = original_metrics["pit_diag"]
    loaded_diag = loaded_metrics["pit_diag"]
    assert_allclose(original_diag.ks_statistic, loaded_diag.ks_statistic, atol=1e-12)
    assert_allclose(original_diag.ks_pvalue, loaded_diag.ks_pvalue, atol=1e-12)
    assert_allclose(original_diag.ad_statistic, loaded_diag.ad_statistic, atol=1e-12)
    assert_allclose(original_diag.ad_pvalue, loaded_diag.ad_pvalue, atol=1e-12)

    assert np.isfinite(original_metrics["coverage"])
    assert np.isfinite(original_metrics["width"])
    assert 0.0 <= original_metrics["coverage"] <= 1.0
