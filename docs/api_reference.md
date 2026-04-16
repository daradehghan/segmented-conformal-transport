# API Reference

This page records the root-package public surface exposed by `import tsconformal`.

## Forecast interfaces

### `ForecastCDF` (Protocol)
- `cdf(y: float) -> float` — Monotone CDF clipped to [0, 1]
- `ppf(u: float) -> float` — Generalized inverse
- `is_discrete: bool` — Whether the forecast has atoms

### `QuantileGridCDFAdapter(probabilities, quantiles)`
Converts quantile grids into ForecastCDF. Repairs non-monotone quantiles via unbounded L² PAV.

### `SampleCDFAdapter(samples, smoothing="gaussian_monotone")`
Converts predictive samples into ForecastCDF. Modes: `"none"` (empirical step CDF) or `"gaussian_monotone"` (kernel-smoothed, monotone-repaired). Raises `InvalidForecastCDFError` on non-finite, undersized, or degenerate smoothed sample inputs.

### `TransportedForecastCDF`
CDF produced by applying a transport map to a base forecast. Returned by `predict_cdf()`.

### `validate_forecast_cdf(cdf: ForecastCDF) -> None`
Validates a ForecastCDF. Raises `InvalidForecastCDFError` on failure.

## Calibrator

### `SegmentedTransportCalibrator(...)`
Constructor parameters: `grid_size`, `rho`, `n_eff_min`, `step_schedule`, `detector`, `cooldown`, `confirm`, `projection="isotonic_l2"`, `interpolation="piecewise_linear"`, `fallback="identity"`, `warm_start="identity"`.

#### Methods
- `predict_cdf(base_cdf, x_t=None) -> TransportedForecastCDF` — validates the forecast and records lightweight bookkeeping for the update-time heuristic consistency guard
- `update(y_t, base_cdf, rng=None) -> None`

#### Diagnostics (properties)
`segment_id`, `num_resets`, `last_reset_t`, `n_eff`, `rho`, `grid`, `in_warmup`, `warm_start_weight`, `pit_history_tail`, `calibration_residuals_tail`, `warnings`

### `RandomizedPIT`
- `pit(cdf, y, rng) -> float` — Continuous or randomized PIT

## Detectors

### `SegmentDetector` (Protocol)
- `update(residual_vector) -> bool`
- `state() -> Mapping`
- `reset() -> None`

### `CUSUMNormDetector(kappa, threshold)`
### `PageHinkleyDetector(delta, threshold)`

## Serialization

- `save_calibrator(calibrator, path)` — Save to directory or .zip
- `load_calibrator(path, detector_factory=None, step_schedule=None)` — Load and continue; exact continuation with a non-default schedule requires passing the same `step_schedule`

## Sensitivity tooling

- `DetectorConfig` — Detector-configuration record for sensitivity sweeps
- `SensitivityReport` — Aggregate result object returned by `sensitivity_report`
- `sensitivity_report(detector_grid, data_stream, pit_seed=20260306) -> SensitivityReport`

## Module-level benchmark helpers

The benchmark-facing helper functions remain available from their defining modules, not from the root package:

- `tsconformal.diagnostics`: `pit_uniformity_tests`, `rolling_coverage`, `warm_start_occupancy`
- `tsconformal.metrics`: `gridwise_calibration_error`, `marginal_coverage`, `mean_interval_width`, `adaptation_lag`, `crps`

## Warning classes

| Warning | Trigger |
|---|---|
| `LowEffectiveSampleWarning` | `n_eff < n_eff_min` |
| `ExcessiveResetWarning` | ≥3 resets in rolling window |
| `HighSerialCorrelationWarning` | lag-1 autocorrelation > 0.5 in within-segment PITs |
| `WithinSegmentDriftWarning` | KS test on segment halves p < 0.01 |
| `WarmStartDominanceWarning` | `warm_start_weight > 0.5` outside fallback |
| `DiscreteForecastWithoutRandomizedPITError` | Discrete forecast without `rng` |
| `InvalidForecastCDFError` | Forecast fails validation |
| `InvalidForecastCDFWarning` | Borderline forecast validity |
