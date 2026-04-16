# Benchmark scripts

This directory contains the heavy benchmark orchestration scripts used around the core `tsconformal` package.

## Real-data workflow

1. Stage the raw dataset zip files in `data/raw/`.
2. Run `benchmarks/data_loaders.py` to verify parsing.
3. Cache Chronos-2 forecasts on Colab or another CUDA machine with `benchmarks/cache_fm_forecasts.py`. The public cache CLI currently implements Chronos-2 only.
4. Copy the cached JSON forecast directories into `data/cached_forecasts/`.
5. Run `benchmarks/run_real_data.py` locally. The runner accepts `--pit-seed` for explicit randomized-PIT reproducibility, although the default remains the benchmark-standard seed used for the refreshed paper artifacts.
6. Generate tables and figures from `data/benchmark_results.json`.

The upstream source links and the expected local archive names are recorded in `../data/README.md`.

## Frozen settings used in the recovered benchmark

See:

- `../configs/real_data_actual_settings.json`
- `../configs/synthetic_actual_settings.json`
- `../reproducibility/paper_alignment.md`

## Important note

The benchmark runner defaults and the synchronized cached forecast bundle are the source of truth for refreshed real-data results. The tracked `data/benchmark_results.json` already uses the current `E_series`/`T_eval` schema and remains suitable for local smoke tests and for regenerating the real-data paper tables from the bundled snapshot.

The synthetic runner also accepts `--pit-seed` alongside `--seed-base`. The default values preserve the synchronized benchmark path used for the current paper artifacts.
