# Tutorial 2: Chronos-2 Wrapper with Validation

This tutorial demonstrates wrapping a quantile-only foundation model (simulated Chronos-2) with `QuantileGridCDFAdapter` and applying SCT with skip-and-log error handling.

## What you'll learn
- Wrapping quantile forecasts with `QuantileGridCDFAdapter`
- Validating forecasts before SCT consumption
- Applying the skip-and-log policy for invalid FM outputs
- Understanding that the CDF is adapter-defined, not model-native analytic

## Running the example

```bash
python -m tsconformal.examples.chronos_wrapper_example
```

## Code walkthrough

See `src/tsconformal/examples/chronos_wrapper_example.py`.

The example simulates a Chronos-2-like quantile forecast output with occasional non-monotone quantiles (a known FM artifact). The adapter repairs monotonicity via L² PAV, `validate_forecast_cdf()` catches any remaining issues, invalid forecasts are skipped and logged, and the resulting CDF remains an adapter-defined pseudo-CDF rather than a model-native analytic object.

## Fixed seeds
The example uses `seed=42`.
