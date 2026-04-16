# Paper/Repo Alignment Notes

This note records the current benchmark workflow and the settings mirrored between the repository and the manuscript.

## Reconstructed workflow

### GPU cache stage

The public repository now includes the Colab cache-stage notebook at `reproducibility/tsconformal_benchmark_colab.ipynb`. The equivalent cache-stage command sequence is also recorded in `docs/cache_step_cli.md` and `reproducibility/colab_workflow.md`. The effective cache commands are:

- `benchmarks/data_loaders.py`
- `benchmarks/cache_fm_forecasts.py --dataset fred_md --model chronos2 --device auto --series-batch-size 128`
- `benchmarks/cache_fm_forecasts.py --dataset electricity --model chronos2 --device auto --series-batch-size 64`
- `benchmarks/cache_fm_forecasts.py --dataset traffic --model chronos2 --device auto --series-batch-size 32`

The cache stage then archives the forecast directories for transfer to the local benchmark machine.

### Local benchmark stage

`benchmarks/run_real_data.py` writes resumable shard outputs and a compact aggregate snapshot with `E_series`, `T_eval`, and `status` fields. The tracked `data/benchmark_results.json` already uses that current schema and remains suitable for local smoke tests and for regenerating the real-data paper tables from the bundled snapshot.

### Analysis stage

`analysis/generate_paper_artifacts.py` generates the paper-facing tables and figures from saved results. The real-data tables can be regenerated directly from the tracked `data/benchmark_results.json`. The full paper-facing artifact set additionally requires a fresh synthetic run, because `data/results/synthetic_results.json` and `analysis/output/paper/` are local derived outputs rather than tracked repository artifacts.

## Settings reflected in the paper and runner defaults

### Real-data runner (`benchmarks/run_real_data.py`)

- evaluation grid for `E_series`: `J_EVAL = 19`
- common scored horizon across methods: `t >= n_cal`
- SCT train grid size: `49`
- SCT hourly settings:
  - `rho = 0.99`
  - `n_eff_min = 50`
  - `step_schedule = min(0.20, 1 / sqrt(n))`
  - detector: `CUSUMNormDetector(kappa=0.02, threshold=0.20)`
  - `cooldown = 168`
  - `confirm = 3`
- SCT monthly settings:
  - `rho = 0.995`
  - `n_eff_min = 24`
  - `step_schedule = min(0.20, 1 / sqrt(n))`
  - detector: `CUSUMNormDetector(kappa=0.02, threshold=0.20)`
  - `cooldown = 12`
  - `confirm = 2`
- ACI: `gamma = 0.005`
- split-CP:
  - hourly `n_cal = 336`
  - monthly `n_cal = 36`

### Synthetic runner (`benchmarks/run_synthetic.py`)

- `n_replicates = 10`
- `seed_base = 20260306`
- SCT:
  - `grid_size = 49`
  - `rho = 0.99`
  - `n_eff_min = 30`
  - `step_schedule = min(0.20, 1 / sqrt(n))`
  - detector: `CUSUMNormDetector(kappa=0.02, threshold=0.20)`
  - `cooldown = 100`
  - `confirm = 3`
- ACI: `gamma = 0.005`
- split-CP: `n_cal = 300`

## Alignment status

- The current runner defaults and the synchronized paper source agree on the hyperparameters above.
- `configs/manuscript_claimed_settings.json` mirrors the paper-facing settings in machine-readable form.
- `reproducibility/run_postcache_reproduction.sh` records the current local post-cache reproduction path for regenerating the omitted local artifacts.

## Additional notes

- The dataset key `electricity_cached` in `data/benchmark_results.json` is a naming artifact of the cache-stage workflow. It corresponds to Electricity Hourly in the paper.
- The remaining reproduction work concerns rebuilding omitted local artifacts such as cached forecasts, synthetic outputs, and generated paper tables. It does not require replacing the tracked real-data snapshot for ordinary repository validation.
