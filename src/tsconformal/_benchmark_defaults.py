"""Shared defaults used by the benchmark orchestration layer.

The benchmark scripts and the detector-sensitivity helper reuse the same
SCT detector settings, step schedule, and PIT seed. Centralizing those
values reduces drift between the real-data runner, the synthetic runner,
and the reproducibility-oriented diagnostics tooling.
"""

from __future__ import annotations

import numpy as np

from tsconformal.calibrators import SegmentedTransportCalibrator
from tsconformal.detectors import CUSUMNormDetector, SegmentDetector


DEFAULT_PIT_SEED = 20260306


def default_step_schedule(n_eff: float) -> float:
    """Return the benchmark-standard SCT feedback weight."""
    return min(0.20, 1.0 / max(n_eff**0.5, 1e-8))


def make_pit_rng(seed: int = DEFAULT_PIT_SEED) -> np.random.Generator:
    """Return the benchmark-standard PIT RNG."""
    return np.random.default_rng(seed)


def build_sct_calibrator(
    *,
    rho: float,
    n_eff_min: float,
    cooldown: int,
    confirm: int,
    grid_size: int = 49,
    kappa: float = 0.02,
    threshold: float = 0.20,
    detector: SegmentDetector | None = None,
) -> SegmentedTransportCalibrator:
    """Return an SCT calibrator configured with the benchmark defaults."""
    det = detector or CUSUMNormDetector(kappa=kappa, threshold=threshold)
    return SegmentedTransportCalibrator(
        grid_size=grid_size,
        rho=rho,
        n_eff_min=n_eff_min,
        step_schedule=default_step_schedule,
        detector=det,
        cooldown=cooldown,
        confirm=confirm,
    )
