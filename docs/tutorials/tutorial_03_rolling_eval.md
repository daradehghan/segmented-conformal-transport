# Tutorial 3: Rolling Online Evaluation with Save/Load

This tutorial demonstrates a complete benchmark-style evaluation loop with rolling coverage, warning collection, and mid-stream save/load verification.

## What you'll learn
- Running a benchmark loop over a single series
- Computing rolling coverage with `rolling_coverage()`
- Collecting and inspecting warnings
- Saving calibrator state mid-stream and verifying exact continuation

## Running the example

```bash
python -m tsconformal.examples.rolling_evaluation
```

## Code walkthrough

See `src/tsconformal/examples/rolling_evaluation.py`.

The example:
1. Runs 300 steps of online calibration
2. Saves the calibrator state to disk
3. Continues the original for 300 more steps
4. Loads the saved state and continues independently
5. Verifies that predictions match exactly

`save_calibrator` and `load_calibrator` support both directories and `.zip` bundles. Exact continuation requires the same `step_schedule`; when the run uses a non-default schedule, `load_calibrator` must receive that same callable. Rolling coverage MAD quantifies adaptation quality, and accumulated warnings should be inspected after the run.

## Fixed seeds
The example uses `seed=42`.
