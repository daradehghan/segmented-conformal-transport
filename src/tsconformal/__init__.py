"""tsconformal — Segmented Conformal Transport for predictive CDF recalibration.

This package implements the SCT algorithm from the paper
"Conformal Calibration under Nonstationarity" with a public API
matching Appendix D of the verified manuscript.

Quick start::

    from tsconformal import (
        SegmentedTransportCalibrator,
        CUSUMNormDetector,
        QuantileGridCDFAdapter,
    )

    detector = CUSUMNormDetector(kappa=0.02, threshold=0.20)
    cal = SegmentedTransportCalibrator(
        grid_size=49,
        rho=0.99,
        n_eff_min=50,
        step_schedule=lambda n: min(0.20, 1.0 / n**0.5),
        detector=detector,
        cooldown=168,
        confirm=3,
    )

    # Online loop
    calibrated_cdf = cal.predict_cdf(base_cdf)
    cal.update(y_t, base_cdf)
"""

__version__ = "0.1.0"

from tsconformal.api import *  # noqa: F401, F403
