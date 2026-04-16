# Tutorial 1: Synthetic Piecewise-Stationary Calibration

This tutorial demonstrates SCT on a synthetic stream with three known regimes and different miscalibration patterns.

## What you'll learn
- Creating a `GaussianForecastCDF` for synthetic experiments
- Initializing `SegmentedTransportCalibrator` with a CUSUM detector
- Running the online predict/update loop
- Reporting per-regime calibration error (`E_r`) and warm-start occupancy

## Running the example

```bash
python -m tsconformal.examples.synthetic_piecewise_stationary
```

## Code walkthrough

See `src/tsconformal/examples/synthetic_piecewise_stationary.py`.

The example generates three regimes:
1. **Regime 0**: Well-calibrated (bias=0, scale=1)
2. **Regime 1**: Biased + overconfident (bias=0.5, scale=0.7)
3. **Regime 2**: Biased + underconfident (bias=-0.3, scale=1.3)

SCT detects the miscalibration shifts and adjusts the transport map accordingly. The output reports reset count, `E_r`, and warm-start occupancy per regime. `E_r` is lowest in the well-calibrated regime, warm-start occupancy is highest just after regime boundaries, and unusually frequent resets indicate an over-sensitive detector.

## Fixed seeds
The example uses `seed=42` for full reproducibility.
