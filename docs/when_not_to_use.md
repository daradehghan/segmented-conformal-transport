# When Not to Use tsconformal

This package is designed for a specific setting. It is **not appropriate** for:

## Multivariate targets
SCT calibrates scalar predictive CDFs only. Joint or copula calibration of multivariate forecasts is out of scope.

## Multi-horizon joint calibration
The theory and implementation cover one-step-ahead only. Multi-horizon guarantees require a different framework (e.g., simultaneous coverage across horizons).

## Series shorter than `2 × n_eff_min`
If your series has fewer than `2 × n_eff_min` observations, the calibrator will spend the entire series in identity fallback. The output will be the raw base forecast, unchanged.

## Base forecasters that change within a regime
The theory assumes the base predictive CDF is stable within a regime (modulo the bounded drift budget). If your forecaster is retrained or updated within what SCT treats as a single segment, the PIT process is no longer piecewise-stationary and the guarantee does not apply.

## Conditional coverage claims
SCT provides marginal coverage within regimes, not conditional coverage given covariates. If you need coverage conditional on specific features, use a different conformal method.

## Detector guarantees independent of calibrator state
The detector uses residuals computed from the previous transport map (`g_{t-1}`). Detection power depends on the calibrator state. There is no standalone detector guarantee.

## Real-time latency-critical systems
The isotonic projection and piecewise-linear interpolation are fast (sub-millisecond), but the package is designed for research benchmarks, not production latency-sensitive pipelines.

## When the base forecast is already well-calibrated
If the base forecast's PIT process is already close to uniform, SCT will converge toward the identity map. It will not make things worse, but it adds unnecessary complexity. Check raw PIT uniformity first.
