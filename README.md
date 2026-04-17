# tsconformal

This repository contains the SCT reference implementation and benchmarks.

`tsconformal` implements Segmented Conformal Transport (SCT) for one-step predictive CDF recalibration under nonstationarity. The repository also contains the benchmark runners, analysis scripts, and reproducibility notes used to evaluate the method.

Companion paper (preprint DOI): https://doi.org/10.13140/RG.2.2.28984.30723

## What is included

The tracked tree is deliberately compact.

- `src/tsconformal/`: core package
- `benchmarks/`: real-data and synthetic benchmark runners
- `analysis/`: scripts for generating tables and figures from saved benchmark results
- `reproducibility/`: workflow notes and frozen benchmark settings
- `data/benchmark_results.json`: compact real-data benchmark snapshot on the current `E_series`/`T_eval` schema

## What is intentionally omitted

The omitted artifacts are the large local ones.

The release does not track the raw benchmark input archives, the cached forecast directories, or the generated benchmark outputs. Those artifacts are large, local, and either externally sourced or mechanically reproducible from the tracked code and settings. The omitted paths are:

- `data/raw/`
- `data/archive/raw_data.zip`
- `data/cached_forecasts/`
- `data/results/benchmark_results_shards/`
- `data/results/*.json`
- `analysis/output/`

The workflow therefore has two operational stages.

1. **Colab / GPU cache stage** for Chronos-2 one-step-ahead forecast caching.
2. **Local CPU/RAM stage** for the benchmark overlay and analysis.

The raw benchmark input sources and expected local filenames are recorded in `data/README.md`.

## Installation

The editable install below matches the development environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[plots,dev]"
pip install ruptures statsmodels "chronos-forecasting>=2.0" torch
```

## Minimal package example

The example below shows the public calibrator workflow.

```python
from tsconformal import (
    CUSUMNormDetector,
    QuantileGridCDFAdapter,
    SegmentedTransportCalibrator,
)

detector = CUSUMNormDetector(kappa=0.02, threshold=0.20)

cal = SegmentedTransportCalibrator(
    grid_size=49,
    rho=0.99,
    n_eff_min=50,
    step_schedule=lambda n: min(0.20, 1.0 / max(n ** 0.5, 1e-8)),
    detector=detector,
    cooldown=168,
    confirm=3,
)

base_cdf = QuantileGridCDFAdapter(
    probabilities=[0.01, 0.10, 0.50, 0.90, 0.99],
    quantiles=[-1.2, -0.4, 0.0, 0.7, 1.4],
)

calibrated_cdf = cal.predict_cdf(base_cdf)
cal.update(y_t=0.35, base_cdf=base_cdf)
```

## Reproducing the benchmark workflow

The workflow is explicit.

### 1. Cache Chronos-2 forecasts on Colab

The repository includes a Colab notebook for the GPU cache stage at `reproducibility/tsconformal_benchmark_colab.ipynb`. The same cache stage is also recorded as plain CLI commands in `docs/cache_step_cli.md`. On a GPU machine, for example Colab, the cache stage runs as follows:

The public cache CLI currently implements Chronos-2 only.

```bash
PYTHONPATH=src:. python benchmarks/data_loaders.py
PYTHONPATH=src:. python benchmarks/cache_fm_forecasts.py --dataset fred_md --model chronos2 --device auto --series-batch-size 128
PYTHONPATH=src:. python benchmarks/cache_fm_forecasts.py --dataset electricity --model chronos2 --device auto --series-batch-size 64
PYTHONPATH=src:. python benchmarks/cache_fm_forecasts.py --dataset traffic --model chronos2 --device auto --series-batch-size 32
```

The cache stage then archives `data/cached_forecasts/...` for transfer back to the local benchmark machine.

### 2. Run the real-data benchmark overlay locally

The local overlay stage consumes cached JSON forecasts at:

```text
data/cached_forecasts/{dataset}/chronos2/*.json
```

The real-data runner then executes as follows:

```bash
PYTHONPATH=src:. python benchmarks/run_real_data.py
```

The runner writes:

```text
data/results/benchmark_results_shards/{dataset}/{model}.jsonl
data/results/benchmark_results.jsonl
data/benchmark_results.json
```

### 3. Run the synthetic benchmark locally

The synthetic benchmark is independent of the cached forecast bundle.

```bash
PYTHONPATH=src:. python benchmarks/run_synthetic.py --n-replicates 10 --seed-base 20260306 --output-json data/results/synthetic_results.json
```

### 4. Generate tables and figures from saved results

The paper-facing artifact stage is separate.

```bash
PYTHONPATH=src:. python analysis/generate_paper_artifacts.py \
  --results data/benchmark_results.json \
  --synthetic-results data/results/synthetic_results.json \
  --output analysis/output/paper
```

The generator accepts `--real-data-log path/to/postcache_reproduction_<run_id>.log`
when the real-data runtime table is required.

The local post-cache reproduction driver is `reproducibility/run_postcache_reproduction.sh`.

The synchronized benchmark settings used by the current workflow are recorded in:

- `configs/real_data_actual_settings.json`
- `configs/synthetic_actual_settings.json`
- `configs/manuscript_claimed_settings.json`

## Settings synchronization

The code and manuscript settings are synchronized. The workflow record is in `reproducibility/paper_alignment.md`, and the machine-readable manuscript mirror is in `configs/manuscript_claimed_settings.json`.

## Bundled results snapshot

The bundled real-data snapshot is current.

The tracked `data/benchmark_results.json` already uses the refreshed `E_series`/`T_eval` schema expected by the current benchmark runner and paper artifact generator. It is sufficient for local smoke tests and for regenerating the real-data paper tables from the tracked snapshot.

The repository does not bundle `data/results/synthetic_results.json`, the sharded benchmark outputs, or `analysis/output/paper/`. The full paper-facing artifact set therefore requires a fresh synthetic run followed by `analysis/generate_paper_artifacts.py`. A full end-to-end real-data rerun additionally requires `benchmarks/run_real_data.py` against the synchronized cached forecast bundle.

## License

MIT
