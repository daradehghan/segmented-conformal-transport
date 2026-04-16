# Colab workflow summary

A small Colab notebook was prepared to execute only the GPU-bound cache-stage
steps on Colab, thereby avoiding unnecessary local runtime and reducing
turnaround time. It follows this split execution model:

1. run dataset parsing and Chronos-2 forecast caching on Colab with an A100 GPU
2. export the cached forecast directories
3. run the real-data benchmark overlay locally on a multicore, high-memory machine
4. generate paper tables and figures from the saved JSON results

The local post-cache reproduction driver is
`reproducibility/run_postcache_reproduction.sh`.

The cache-stage notebook uses the following effective batch sizes:

- FRED-MD: `128`
- Electricity Hourly: `64`
- Traffic Hourly: `32`

These values are frozen in `configs/real_data_actual_settings.json`.
