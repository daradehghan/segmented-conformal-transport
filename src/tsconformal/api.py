"""Public API surface for tsconformal.

This module re-exports exactly the Appendix-D public names.
"""

from tsconformal.forecast import (
    ForecastCDF,
    InvalidForecastCDFError,
    InvalidForecastCDFWarning,
    QuantileGridCDFAdapter,
    SampleCDFAdapter,
    TransportedForecastCDF,
    validate_forecast_cdf,
)
from tsconformal.calibrators import (
    DiscreteForecastWithoutRandomizedPITError,
    ExcessiveResetWarning,
    HighSerialCorrelationWarning,
    LowEffectiveSampleWarning,
    PredictionSequenceError,
    RandomizedPIT,
    SegmentedTransportCalibrator,
    WarmStartDominanceWarning,
    WithinSegmentDriftWarning,
    load_calibrator,
    save_calibrator,
)
from tsconformal.detectors import (
    CUSUMNormDetector,
    PageHinkleyDetector,
    SegmentDetector,
)
from tsconformal.diagnostics import (
    DetectorConfig,
    SensitivityReport,
    sensitivity_report,
)

__all__ = [
    # Forecast
    "ForecastCDF",
    "QuantileGridCDFAdapter",
    "SampleCDFAdapter",
    "TransportedForecastCDF",
    "validate_forecast_cdf",
    "InvalidForecastCDFError",
    "InvalidForecastCDFWarning",
    # PIT
    "RandomizedPIT",
    # Detectors
    "SegmentDetector",
    "CUSUMNormDetector",
    "PageHinkleyDetector",
    # Calibrator
    "SegmentedTransportCalibrator",
    # Serialization
    "save_calibrator",
    "load_calibrator",
    # Sensitivity
    "sensitivity_report",
    "DetectorConfig",
    "SensitivityReport",
    # Warnings
    "LowEffectiveSampleWarning",
    "ExcessiveResetWarning",
    "HighSerialCorrelationWarning",
    "WithinSegmentDriftWarning",
    "WarmStartDominanceWarning",
    "DiscreteForecastWithoutRandomizedPITError",
    "PredictionSequenceError",
]
