# GPU cache step (CLI only)

The repository includes the Colab cache-stage notebook at `reproducibility/tsconformal_benchmark_colab.ipynb`. The cache stage remains logically identical when run from the CLI.

The public cache CLI currently implements Chronos-2 only.

The notebook executes the following commands:

```bash
PYTHONPATH=src:. python benchmarks/data_loaders.py

PYTHONPATH=src:. python benchmarks/cache_fm_forecasts.py     --dataset fred_md --model chronos2 --device auto --series-batch-size 128

PYTHONPATH=src:. python benchmarks/cache_fm_forecasts.py     --dataset electricity --model chronos2 --device auto --series-batch-size 64

PYTHONPATH=src:. python benchmarks/cache_fm_forecasts.py     --dataset traffic --model chronos2 --device auto --series-batch-size 32
```

Expected cache layout after transfer back to the local machine:

```text
data/cached_forecasts/fred_md/chronos2/*.json
data/cached_forecasts/electricity/chronos2/*.json
data/cached_forecasts/traffic/chronos2/*.json
```

The benchmark overlay and analysis are then run locally. The corresponding
post-cache reproduction driver is `reproducibility/run_postcache_reproduction.sh`.
