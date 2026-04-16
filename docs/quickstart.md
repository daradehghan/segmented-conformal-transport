# Quick Start

## Installation

The current public release installs from source.

```bash
pip install .
# or for development:
pip install -e .
```

## Basic Usage

```python
from tsconformal import (
    SegmentedTransportCalibrator,
    CUSUMNormDetector,
    QuantileGridCDFAdapter,
    validate_forecast_cdf,
)

# 1. Wrap your foundation model output
base_cdf = QuantileGridCDFAdapter(
    probabilities=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99],
    quantiles=model.predict_quantiles(context),
)

# 2. Validate the forecast
validate_forecast_cdf(base_cdf)

# 3. Create the calibrator
step_schedule = lambda n: min(0.20, 1.0 / n**0.5)
detector = CUSUMNormDetector(kappa=0.02, threshold=0.20)
cal = SegmentedTransportCalibrator(
    grid_size=49,
    rho=0.99,
    n_eff_min=50,
    step_schedule=step_schedule,
    detector=detector,
    cooldown=168,
    confirm=3,
)

# 4. Online loop
for t in range(len(data)):
    # Predict (does NOT consume y_t)
    calibrated = cal.predict_cdf(base_cdf)

    # Use the calibrated forecast
    lower = calibrated.ppf(0.05)   # 5th percentile
    upper = calibrated.ppf(0.95)   # 95th percentile

    # Update (reveal y_t)
    cal.update(y_t, base_cdf)

    # Check warnings
    if cal.warnings:
        print(f"Step {t}: {cal.warnings}")
```

## Key Concepts

- **ForecastCDF**: Protocol for any predictive CDF object (must expose `cdf(y)` and `ppf(u)`)
- **QuantileGridCDFAdapter**: Converts quantile grids from foundation models into ForecastCDF
- **SampleCDFAdapter**: Converts predictive samples into ForecastCDF
- **SegmentedTransportCalibrator**: The main SCT calibrator
- **predict_cdf → update**: The two-step online loop (predict first, then reveal outcome)

## Saving and Loading

```python
from tsconformal import save_calibrator, load_calibrator

# Save mid-stream
save_calibrator(cal, "path/to/state")   # directory
save_calibrator(cal, "state.zip")       # zip bundle

# Load and continue
cal_resumed = load_calibrator("path/to/state", step_schedule=step_schedule)
```
