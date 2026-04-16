# Data Directory

This directory stores the compact tracked data snapshot used by the public repository.

## Tracked files

The canonical tracked contents are intentionally small.

- `benchmark_results.json`: compact real-data benchmark snapshot on the current `E_series`/`T_eval` schema

## Local working paths

Local benchmark working trees commonly contain additional ignored paths. Those files are staged or generated during reruns and are not part of the public tracked tree.

- `raw/`: extracted dataset zips staged for the real-data benchmark
- `cached_forecasts/`: cached Chronos-2 one-step forecast JSON files
- `results/`: local benchmark outputs such as `benchmark_results.jsonl` and `synthetic_results.json`

## Raw benchmark inputs

The real-data benchmark loaders expect the Monash/Zenodo archives below to be staged locally under `data/raw/`. Those local input files do not need to be tracked for public use of the repository.

- Electricity Hourly
  Local filename: `data/raw/electricity_hourly_dataset.zip`
  Benchmark-ready archive: <https://zenodo.org/records/3898439>
  Original source: <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>
- Traffic Hourly
  Local filename: `data/raw/traffic_hourly_dataset.zip`
  Benchmark-ready archive: <https://zenodo.org/records/4656132>
  Original source: <http://pems.dot.ca.gov/>
- FRED-MD
  Local filename: `data/raw/fred_md_dataset.zip`
  Benchmark-ready archive: <https://zenodo.org/records/4654833>
  Original source: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>

## Reproduction note

A full local real-data rerun stages the three dataset archives under `data/raw/` and the synchronized cached forecasts under `data/cached_forecasts/` before `benchmarks/run_real_data.py` executes. The tracked `benchmark_results.json` remains sufficient for local smoke tests and for regenerating the real-data paper tables from the bundled snapshot.
