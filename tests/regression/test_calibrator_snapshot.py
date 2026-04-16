import warnings

import numpy as np
from numpy.testing import assert_allclose


class IdentityCDF:
    is_discrete = False

    def cdf(self, y):
        return float(np.clip(y, 0.0, 1.0))

    def ppf(self, u):
        return float(np.clip(u, 0.0, 1.0))


class TriggerOnceDetector:
    def __init__(self, trigger_step=8):
        self.trigger_step = trigger_step
        self.calls = 0
        self.triggered = False

    def update(self, residual_vector):
        self.calls += 1
        if not self.triggered and self.calls == self.trigger_step:
            self.triggered = True
            return True
        return False

    def state(self):
        return {
            "detector_type": "TriggerOnce",
            "trigger_step": self.trigger_step,
            "calls": self.calls,
            "triggered": self.triggered,
        }

    def reset(self):
        return None


def test_segmented_transport_calibrator_snapshot_with_reset():
    """The public SCT trajectory must remain numerically stable on a fixed stream."""
    from tsconformal import SegmentedTransportCalibrator

    cdf = IdentityCDF()
    ys = [0.15, 0.85, 0.25, 0.75, 0.20, 0.80, 0.30, 0.70, 0.35, 0.65, 0.40, 0.60]
    cal = SegmentedTransportCalibrator(
        grid_size=7,
        rho=0.9,
        n_eff_min=1.0,
        step_schedule=lambda n: 0.2,
        detector=TriggerOnceDetector(trigger_step=8),
        cooldown=1,
        confirm=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for y_t in ys:
            cal.predict_cdf(cdf)
            cal.update(y_t, cdf)

    calibrated = cal.predict_cdf(cdf)

    assert cal.segment_id == 1
    assert cal.num_resets == 1
    assert cal.last_reset_t == 8
    assert cal.in_warmup is False
    assert_allclose(cal.n_eff, 4.892008123282762)
    assert_allclose(cal.warm_start_weight, 0.32768000000000014)
    assert_allclose(cal.pit_history_tail, [0.35, 0.65, 0.4, 0.6])
    assert_allclose(
        cal.calibration_residuals_tail,
        [
            [-0.1, -0.2, 0.7, 0.6, 0.5, 0.2, 0.1],
            [-0.08, -0.16, -0.34526316, -0.42526316, -0.50526316, 0.16, 0.08],
            [-0.064, -0.128, -0.34263119, 0.59336881, 0.52936881, 0.128, 0.064],
            [-0.0512, -0.1024, -0.32121167, -0.43056811, 0.51823189, 0.1024, 0.0512],
        ],
        atol=1e-8,
    )
    assert_allclose(
        [calibrated.cdf(0.2), calibrated.cdf(0.5), calibrated.cdf(0.8)],
        [0.06553600000000004, 0.42401298499015205, 0.9344640000000002],
        atol=1e-12,
    )
    assert_allclose(
        [calibrated.ppf(0.1), calibrated.ppf(0.5), calibrated.ppf(0.9)],
        [0.2607285511394073, 0.6057739182747732, 0.7444096512408502],
        atol=1e-12,
    )
