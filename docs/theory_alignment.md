# Theory Alignment

This document maps `tsconformal` v0.2.0 to the verified theory in *Conformal Calibration under Nonstationarity*.

## Theorem scope

The package implements the finite-sample, per-regime, gridwise L² calibration guarantee from Theorem 1. The bound has six terms:

1. **Estimation**: `C_{mix,r} √(log(JT/δ) / n_min^eff)` — controlled by recency-weighted effective sample size
2. **Warm-start transient**: `W_r` — tracked exactly via `warm_start_weight`
3. **Detection delay**: `C_delay · d̄_r / |I_r|`
4. **Drift**: `C_drift · V_r^(J)` — unobservable; diagnostics warn about drift
5. **Discretization**: `C_grid · J^{-1/2}`
6. **Randomization**: `Δ_r^rand` — from discrete PIT randomization

## Algorithm mapping

| Theory (Algorithm 1) | Code | Notes |
|---|---|---|
| Grid `{u_j}` | `SegmentedTransportCalibrator._grid` | `u_j = j/(J+1)` |
| Recency weights | `_W`, `_W2`, `_N` | `W = ρW + 1`, `W² = ρ²W² + 1`, `N = ρN + 1{U≤u}` |
| Smoothing recursion | `_m = (1-η)_m + η Ĝ` | `η = step_schedule(n_eff)` |
| Isotonic projection | `pav_isotonic_clipped(_m)` | L² PAV, clipped to [0,1] for transport maps |
| Transport map | `build_transport_map(grid, g)` | Piecewise linear through (0,0), knots, (1,1) |
| Generalized inverse | Upper generalized inverse on the full knot sequence | Plateau levels map to the right endpoint |
| Detector | `SegmentDetector.update(residuals)` | Threshold exceedance only |
| Cooldown/confirm | `_cooldown`, `_confirm_len` | In calibrator, not detector |
| Reset | `_do_reset()` + continue update | Current observation seeds new archive |
| Identity fallback | `n_eff < n_eff_min` | Exact `n_eff = W²/W²` check |

## Assumptions

| ID | Assumption | Package enforcement |
|---|---|---|
| A1 | Bounded drift budget within regimes | Not enforced; `WithinSegmentDriftWarning` warns |
| A2 | Observable scalar outcomes | Enforced: scalar `y_t` input |
| A3 | Known base predictive CDF | Validated via `validate_forecast_cdf` |
| A4 | Joint detector-calibrator system | Detector uses `g_{t-1}` residuals |
| A5 | Geometric β-mixing within regimes | Not enforced; `HighSerialCorrelationWarning` warns |
| A6 | Bounded grid size | Enforced: `grid_size >= 2` |
| A7 | One-step-ahead | Enforced: single-step predict/update loop |
| A8 | Monotone transport | Enforced: isotonic L² projection |
| A9 | Consequence of A4 | Detector thresholds documented as joint |

## Quantile adapter vs theory

The theory assumes access to a true CDF. When using `QuantileGridCDFAdapter`:
- Quantile repair uses unbounded L² PAV (quantiles live on target scale, not [0,1])
- Flat repaired quantile segments are treated as atoms (`is_discrete=True`)
- `cdf(y)` is the right-continuous inverse, with `cdf_left(y)` used for randomized PIT
- Extrapolation uses nearest non-zero slope, clipped to [0,1]
- This is an adapter-defined pseudo-CDF, not a model-native analytic CDF

## Serialized state

Serialized calibrator state follows a separate compatibility contract. The
package version changes with each public release, while the state schema changes
only when the persisted layout or its semantics change.
