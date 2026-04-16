#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="src:."
export MPLCONFIGDIR="$PWD/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

# Real-data benchmark parallelism.
export TSCONFORMAL_BENCHMARK_WORKERS="${TSCONFORMAL_BENCHMARK_WORKERS:-8}"
export TSCONFORMAL_BENCHMARK_EXECUTOR="${TSCONFORMAL_BENCHMARK_EXECUTOR:-process}"

# Prevent each worker from spawning extra BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/postcache_reproduction_${RUN_ID}.log"
TIMINGS_FILE="$LOG_DIR/postcache_reproduction_${RUN_ID}.timings.tsv"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local rc=$?
  echo
  echo "FAILED"
  echo "timestamp: $(date -Iseconds)"
  echo "line: $1"
  echo "command: $2"
  echo "log: $LOG_FILE"
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

run_step() {
  local name="$1"
  shift
  local start end dur
  start=$(date +%s)
  echo
  echo "=== START ${name} $(date -Iseconds) ==="
  "$@"
  end=$(date +%s)
  dur=$((end - start))
  printf "%s\t%d\n" "$name" "$dur" >> "$TIMINGS_FILE"
  echo "=== END ${name} ${dur}s ==="
}

echo "run_id: $RUN_ID"
echo "started_at: $(date -Iseconds)"
echo "git_commit: $(git rev-parse HEAD)"
echo "python: $(python --version 2>&1)"
echo "macos:"
sw_vers
echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
echo "physical_cpu: $(sysctl -n hw.physicalcpu)"
echo "logical_cpu: $(sysctl -n hw.logicalcpu)"
echo "memory_bytes: $(sysctl -n hw.memsize)"
echo "benchmark_workers: $TSCONFORMAL_BENCHMARK_WORKERS"
echo -e "step\tseconds" > "$TIMINGS_FILE"

run_step "verify_raw_inputs" python benchmarks/data_loaders.py

if [[ "${FRESH_RUN:-1}" == "1" ]]; then
  run_step "cleanup_outputs" bash -lc '
    rm -rf data/results/benchmark_results_shards
    rm -f data/results/benchmark_results.jsonl
    rm -f data/benchmark_results.json
    rm -f data/results/synthetic_results.json
    rm -f data/results/synthetic_results_clean.txt
    rm -rf analysis/output/paper
  '
else
  echo
  echo "=== SKIP cleanup_outputs because FRESH_RUN=${FRESH_RUN} ==="
fi

run_step "real_data_benchmark" python benchmarks/run_real_data.py

run_step "synthetic_benchmark" \
  python benchmarks/run_synthetic.py \
    --n-replicates 10 \
    --seed-base 20260306 \
    --output-json data/results/synthetic_results.json

run_step "paper_artifacts" \
  python analysis/generate_paper_artifacts.py \
    --results data/benchmark_results.json \
    --synthetic-results data/results/synthetic_results.json \
    --real-data-log "$LOG_FILE" \
    --output analysis/output/paper

run_step "regression_tests" \
  pytest tests/regression/test_real_data_outputs.py \
         tests/regression/test_paper_artifact_generation.py

run_step "validate_outputs" python - <<'PY'
from pathlib import Path
import pandas as pd

required = [
    Path("data/benchmark_results.json"),
    Path("data/results/synthetic_results.json"),
    Path("analysis/output/paper/tables/headline_real_data.csv"),
    Path("analysis/output/paper/tables/real_data_runtime_summary.csv"),
    Path("analysis/output/paper/tables/synthetic_representative_summary.csv"),
    Path("analysis/output/paper/figures/eseries_distribution.png"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"Missing outputs: {missing}")

headline = pd.read_csv("analysis/output/paper/tables/headline_real_data.csv")
print(headline.to_string(index=False))
PY

echo
echo "SUCCESS"
echo "finished_at: $(date -Iseconds)"
echo "log: $LOG_FILE"
echo "timings: $TIMINGS_FILE"
