# Warnings and Failure Modes

## Warning classes

### `LowEffectiveSampleWarning`
**Trigger**: `n_eff < n_eff_min` during `update()`.
**Effect**: Identity fallback activated — the transport map is not applied.
**Action**: Wait for more data to accumulate. If this persists, the regime is too short or `rho` is too aggressive.

### `ExcessiveResetWarning`
**Trigger**: 3 or more detector-confirmed resets within a rolling window of `5 × cooldown` steps.
**Effect**: The calibrator is repeatedly resetting, preventing stable map estimation.
**Action**: Increase `cooldown`, raise detector `threshold`, or reduce detector sensitivity. This may indicate oscillating short regimes (failure mode FM7 in the theory).

### `HighSerialCorrelationWarning`
**Trigger**: Lag-1 autocorrelation of within-segment centered PITs exceeds 0.5.
**Effect**: The β-mixing assumption (A5) may be violated; the theorem's concentration bound may be loose.
**Action**: Consider a larger `rho` (slower forgetting) or acknowledge the dependence limitation.

### `WithinSegmentDriftWarning`
**Trigger**: Two-sample KS test between the first and second half of within-segment PITs has p < 0.01.
**Effect**: Evidence that the PIT law is changing within the current segment, meaning the piecewise-stationarity assumption (A1) is violated.
**Action**: Consider reducing `cooldown` to enable faster detection, or acknowledge that the drift budget V_r is non-negligible.

### `WarmStartDominanceWarning`
**Trigger**: `warm_start_weight > 0.5` while not in identity fallback.
**Effect**: The calibration map is still more than half determined by the identity initialization rather than observed data.
**Action**: Allow more observations before trusting the calibrated output. This is expected immediately after a reset.

### `DiscreteForecastWithoutRandomizedPITError`
**Trigger**: `update()` called with a discrete forecast and `rng=None`.
**Effect**: Exception raised; the update does not proceed.
**Action**: Pass a numpy Generator as `rng` to enable randomized PIT.

### `InvalidForecastCDFError`
**Trigger**: Forecast fails validation or adapter construction (NaN, Inf, impossible monotonicity failure, degenerate smoothed sample cloud, or other structural invalidity).
**Effect**: Exception raised; the forecast is blocked from entering SCT.
**Action**: Check the base forecaster output. Point-mass or other explicitly discrete forecasts are supported, but they must expose randomized PIT semantics.

## Failure modes from the theory

| FM | Description | Package mitigation |
|---|---|---|
| FM1 | Short-regime oscillation | `ExcessiveResetWarning`; `cooldown` parameter |
| FM2 | Tail sparsity | Inherent; gridwise L² averages over tails |
| FM3 | Warm-start bias | `WarmStartDominanceWarning`; exact `warm_start_weight` tracking |
| FM4 | Detector-calibrator feedback | Assumption A4 treats as joint system |
| FM5 | Undetected continuous drift | `WithinSegmentDriftWarning`; acknowledged as non-goal |
| FM6 | False alarms | `sensitivity_report` for detector-shopping risk |
