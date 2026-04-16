"""Run the real-data benchmark overlay on cached FM forecasts.

Incremental, resumable, sharded, and compacted:
- writes one JSON line per result into dataset/model shard files
- retries only rows that do not already have status="ok"
- compacts shards to keep only the latest row per key
- writes aggregate outputs from compacted shard data

Requires: cached forecasts in data/cached_forecasts/{dataset}/{model}/*.json
Produces:
    data/results/benchmark_results_shards/{dataset}/{model}.jsonl
    data/results/benchmark_results.jsonl
    data/benchmark_results.json

Usage:
    PYTHONPATH=src:. python benchmarks/run_real_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from tsconformal import (
    QuantileGridCDFAdapter,
)
from tsconformal._benchmark_defaults import (
    DEFAULT_PIT_SEED,
    build_sct_calibrator,
    make_pit_rng,
)
from tsconformal.diagnostics import warm_start_occupancy
from tsconformal.forecast import InvalidForecastCDFError
from tsconformal.metrics import (
    gridwise_calibration_error,
    marginal_coverage,
    mean_interval_width,
)
from benchmarks.baselines import ACIBaseline

J_EVAL = 19
METHODS = ("SCT", "ACI", "split-CP")
RESULTS_DIR = Path("data/results")
SHARD_ROOT = RESULTS_DIR / "benchmark_results_shards"
AGG_JSONL_PATH = RESULTS_DIR / "benchmark_results.jsonl"
COMBINED_JSON_PATH = Path("data/benchmark_results.json")


def run_sct_on_cached(
    forecasts: list,
    alpha: float = 0.10,
    is_monthly: bool = False,
    eval_start: int | None = None,
    pit_seed: int = DEFAULT_PIT_SEED,
):
    """Run SCT on cached forecasts."""
    rho = 0.995 if is_monthly else 0.99
    n_eff_min = 24.0 if is_monthly else 50.0
    cooldown = 12 if is_monthly else 168
    if eval_start is None:
        eval_start = 36 if is_monthly else 336

    cal = build_sct_calibrator(
        grid_size=49,
        rho=rho,
        n_eff_min=n_eff_min,
        cooldown=cooldown,
        confirm=3 if not is_monthly else 2,
    )

    eval_grid = np.array([j / (J_EVAL + 1) for j in range(1, J_EVAL + 1)])
    T = len(forecasts)
    T_eval = max(T - eval_start, 0)

    # Score only the common post-n_cal evaluation horizon shared by all methods.
    valid_y = []
    valid_cal_q = []
    valid_lo = []
    valid_hi = []
    valid_wf = []
    valid_ww = []
    skipped = 0
    pit_rng = make_pit_rng(pit_seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t, fc in enumerate(forecasts):
            try:
                base_cdf = QuantileGridCDFAdapter(
                    probabilities=fc["probabilities"],
                    quantiles=fc["quantiles"],
                )
                cal_cdf = cal.predict_cdf(base_cdf)
            except (InvalidForecastCDFError, ValueError):
                if t >= eval_start:
                    skipped += 1
                continue

            y_t = fc["y_true"]
            if t >= eval_start:
                valid_cal_q.append(np.atleast_1d(cal_cdf.ppf(eval_grid)))
                valid_lo.append(float(cal_cdf.ppf(alpha / 2)))
                valid_hi.append(float(cal_cdf.ppf(1.0 - alpha / 2)))
                valid_y.append(y_t)
                valid_wf.append(cal.in_warmup)
                valid_ww.append(cal.warm_start_weight)
            cal.update(y_t, base_cdf, rng=pit_rng)

    if not valid_y:
        return {
            "method": "SCT",
            "E_series": float("nan"),
            "coverage_90": float("nan"),
            "width_90": float("nan"),
            "resets": 0,
            "ws_occ": float("nan"),
            "skipped": skipped,
            "T": T,
            "T_eval": T_eval,
            "status": "ok",
        }

    y_arr = np.array(valid_y)
    cal_q_arr = np.array(valid_cal_q)
    lo_arr = np.array(valid_lo)
    hi_arr = np.array(valid_hi)
    wf_arr = np.array(valid_wf, dtype=bool)
    ww_arr = np.array(valid_ww, dtype=np.float64)

    return {
        "method": "SCT",
        "E_series": float(gridwise_calibration_error(y_arr, cal_q_arr, J_EVAL)),
        "coverage_90": float(marginal_coverage(y_arr, lo_arr, hi_arr)),
        "width_90": float(mean_interval_width(lo_arr, hi_arr)),
        "resets": int(cal.num_resets),
        "ws_occ": float(warm_start_occupancy(wf_arr, ww_arr)),
        "skipped": int(skipped),
        "T": int(T),
        "T_eval": int(T_eval),
        "status": "ok",
    }


def run_aci_on_cached(
    forecasts: list,
    alpha: float = 0.10,
    eval_start: int = 0,
):
    aci = ACIBaseline(alpha=alpha, gamma=0.005)
    T = len(forecasts)
    T_eval = max(T - eval_start, 0)
    lo = np.zeros(T_eval)
    hi = np.zeros(T_eval)
    y_true = np.zeros(T_eval)
    for t, fc in enumerate(forecasts):
        idx = t - eval_start
        y_hat = np.median(fc["quantiles"])
        l, u = aci.predict_interval(y_hat)
        if idx >= 0:
            y_true[idx] = fc["y_true"]
            lo[idx] = l
            hi[idx] = u
        aci.update(fc["y_true"], y_hat)
    return {
        "method": "ACI",
        "E_series": float("nan"),
        "coverage_90": float(marginal_coverage(y_true, lo, hi)),
        "width_90": float(mean_interval_width(lo, hi)),
        "T": int(T),
        "T_eval": int(T_eval),
        "status": "ok",
    }


def run_split_cp_on_cached(forecasts: list, alpha: float = 0.10, n_cal: int = 336):
    T = len(forecasts)
    T_eval = max(T - n_cal, 0)
    if T_eval <= 0:
        return {
            "method": "split-CP",
            "E_series": float("nan"),
            "coverage_90": float("nan"),
            "width_90": float("nan"),
            "T": int(T),
            "T_eval": int(T_eval),
            "status": "ok",
        }

    y_true = np.array([f["y_true"] for f in forecasts], dtype=np.float64)
    y_hat = np.array([np.median(f["quantiles"]) for f in forecasts], dtype=np.float64)
    scores = np.abs(y_true - y_hat)
    windows = np.lib.stride_tricks.sliding_window_view(scores[:-1], n_cal)
    level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    level = min(level, 1.0)
    q = np.quantile(windows, level, axis=1)

    y_eval = y_true[n_cal:]
    yhat_eval = y_hat[n_cal:]
    lo = yhat_eval - q
    hi = yhat_eval + q
    return {
        "method": "split-CP",
        "E_series": float("nan"),
        "coverage_90": float(marginal_coverage(y_eval, lo, hi)),
        "width_90": float(mean_interval_width(lo, hi)),
        "T": int(T),
        "T_eval": int(T_eval),
        "status": "ok",
    }


def shard_path_for(dataset: str, model: str) -> Path:
    return SHARD_ROOT / dataset / f"{model}.jsonl"


def result_key(result: dict) -> tuple[str, str, str, str]:
    return (
        result.get("dataset", ""),
        result.get("model", ""),
        result.get("series_id", ""),
        result.get("method", ""),
    )


def compact_rows(rows: list[dict]) -> list[dict]:
    latest_by_key: dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = result_key(row)
        if key in latest_by_key:
            del latest_by_key[key]
        latest_by_key[key] = row
    return list(latest_by_key.values())


def read_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    tmp_path.replace(path)


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def compact_shard(path: Path) -> list[dict]:
    rows = read_jsonl_rows(path)
    compacted = compact_rows(rows)
    if rows != compacted:
        write_jsonl_rows(path, compacted)
    return compacted


def load_completed(path: Path) -> set[tuple[str, str, str, str]]:
    done = set()
    for row in compact_shard(path):
        if row.get("status") != "ok":
            continue
        if "T_eval" not in row:
            continue
        if row.get("method") == "SCT" and "E_series" not in row:
            continue
        done.add(result_key(row))
    return done


def maybe_migrate_legacy_results() -> None:
    if any(SHARD_ROOT.glob("*/*.jsonl")):
        return
    legacy_rows = read_jsonl_rows(AGG_JSONL_PATH)
    if not legacy_rows:
        return
    for row in legacy_rows:
        dataset = row.get("dataset")
        model = row.get("model")
        if not dataset or not model:
            continue
        append_result(shard_path_for(dataset, model), row)
    for shard_path in SHARD_ROOT.glob("*/*.jsonl"):
        compact_shard(shard_path)


def process_series_task(
    dataset: str,
    model: str,
    series_path_str: str,
    is_monthly: bool,
    n_cal: int,
    needed_methods: tuple[str, ...],
    pit_seed: int,
) -> list[dict]:
    series_path = Path(series_path_str)
    sid = series_path.stem

    with open(series_path) as f:
        forecasts = json.load(f)
    if len(forecasts) < 50:
        return []

    rows = []
    for method_name in needed_methods:
        try:
            if method_name == "SCT":
                row = run_sct_on_cached(
                    forecasts,
                    is_monthly=is_monthly,
                    pit_seed=pit_seed,
                )
            elif method_name == "ACI":
                row = run_aci_on_cached(forecasts, eval_start=n_cal)
            else:
                row = run_split_cp_on_cached(forecasts, n_cal=n_cal)
        except Exception as exc:
            row = {
                "method": method_name,
                "E_series": float("nan"),
                "coverage_90": float("nan"),
                "width_90": float("nan"),
                "status": "error",
                "error": str(exc),
            }

        row["dataset"] = dataset
        row["model"] = model
        row["series_id"] = sid
        rows.append(row)

    return rows


def resolve_max_workers() -> int:
    raw = os.environ.get("TSCONFORMAL_BENCHMARK_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def create_executor(max_workers: int):
    executor_kind = os.environ.get("TSCONFORMAL_BENCHMARK_EXECUTOR", "process").lower()
    if executor_kind == "thread":
        return ThreadPoolExecutor(max_workers=max_workers), "thread"

    try:
        return ProcessPoolExecutor(max_workers=max_workers), "process"
    except PermissionError:
        return ThreadPoolExecutor(max_workers=max_workers), "thread"


def aggregate_shards() -> list[dict]:
    all_rows = []
    for shard_path in sorted(SHARD_ROOT.glob("*/*.jsonl")):
        all_rows.extend(compact_shard(shard_path))

    compacted_all = compact_rows(all_rows)
    ok_rows = [row for row in compacted_all if row.get("status") == "ok"]

    write_jsonl_rows(AGG_JSONL_PATH, ok_rows)
    COMBINED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_JSON_PATH, "w") as f:
        json.dump(ok_rows, f, indent=2, default=str)

    return ok_rows


def run_dataset_model(
    dataset: str,
    model: str,
    model_dir: Path,
    is_monthly: bool,
    max_workers: int,
    pit_seed: int,
) -> None:
    shard_path = shard_path_for(dataset, model)
    done = load_completed(shard_path)
    if done:
        print(
            f"{dataset}/{model}: resuming with {len(done)} successful results",
            flush=True,
        )

    series_files = sorted(model_dir.glob("*.json"))
    print(f"\n{'=' * 60}", flush=True)
    print(f"{dataset} / {model}: {len(series_files)} series", flush=True)
    print(f"{'=' * 60}", flush=True)

    t_start = time.time()
    n_cal = 36 if is_monthly else 336
    total_ok = len(done)
    pending = []

    for sf in series_files:
        sid = sf.stem
        needed_methods = tuple(
            method for method in METHODS if (dataset, model, sid, method) not in done
        )
        if needed_methods:
            pending.append((sid, sf, needed_methods))

    if not pending:
        print(f"  Nothing to do for {dataset}/{model}", flush=True)
        return

    print(
        f"  Running {len(pending)} pending series with {max_workers} worker(s)",
        flush=True,
    )

    completed_series = 0
    executor, executor_kind = create_executor(max_workers)
    print(f"  Executor: {executor_kind}", flush=True)
    with executor as pool:
        future_to_sid = {
            pool.submit(
                process_series_task,
                dataset,
                model,
                str(sf),
                is_monthly,
                n_cal,
                needed_methods,
                pit_seed,
            ): sid
            for sid, sf, needed_methods in pending
        }

        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            completed_series += 1

            try:
                rows = future.result()
            except Exception as exc:
                rows = [
                    {
                        "dataset": dataset,
                        "model": model,
                        "series_id": sid,
                        "method": "series",
                        "E_series": float("nan"),
                        "coverage_90": float("nan"),
                        "width_90": float("nan"),
                        "status": "error",
                        "error": str(exc),
                    }
                ]

            for row in rows:
                append_result(shard_path, row)
                if row.get("status") == "ok":
                    done.add(result_key(row))
                    total_ok += 1

            if completed_series % 10 == 0 or completed_series == len(pending):
                elapsed = time.time() - t_start
                rate = completed_series / max(elapsed, 1) * 3600
                print(
                    f"  [{completed_series}/{len(pending)}] {rate:.0f} series/hr, "
                    f"{total_ok} total OK results",
                    flush=True,
                )

    shard_rows = compact_shard(shard_path)

    elapsed = time.time() - t_start
    print(f"  Completed {dataset}/{model} in {elapsed:.1f}s", flush=True)

    ok_rows = [row for row in shard_rows if row.get("status") == "ok"]
    if ok_rows:
        df_r = pd.DataFrame(ok_rows)
        summary = (
            df_r.groupby("method")[["E_series", "coverage_90", "width_90"]]
            .median()
            .round(4)
        )
        print(summary.to_string(), flush=True)


def run_benchmark(pit_seed: int = DEFAULT_PIT_SEED) -> None:
    cache_root = Path("data/cached_forecasts")
    if not cache_root.exists():
        print("ERROR: No cached forecasts found at data/cached_forecasts/", flush=True)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    maybe_migrate_legacy_results()
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)

    max_workers = resolve_max_workers()
    monthly_datasets = {"fred_md"}

    for dataset_dir in sorted(cache_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        is_monthly = dataset in monthly_datasets

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            run_dataset_model(
                dataset=dataset,
                model=model_dir.name,
                model_dir=model_dir,
                is_monthly=is_monthly,
                max_workers=max_workers,
                pit_seed=pit_seed,
            )

    all_results = aggregate_shards()

    print(f"\n{'=' * 60}", flush=True)
    print(f"DONE. {len(all_results)} successful results.", flush=True)
    print(f"Shards: {SHARD_ROOT}", flush=True)
    print(f"JSONL:  {AGG_JSONL_PATH}", flush=True)
    print(f"JSON:   {COMBINED_JSON_PATH}", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        summary = (
            df.groupby(["dataset", "method"])[["E_series", "coverage_90", "width_90"]]
            .median()
            .round(4)
        )
        print(f"\n=== HEADLINE TABLE ===", flush=True)
        print(summary.to_string(), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the real-data benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run the real-data SCT benchmark on cached forecasts.",
    )
    parser.add_argument("--pit-seed", type=int, default=DEFAULT_PIT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the benchmark."""
    args = build_parser().parse_args(argv)
    run_benchmark(pit_seed=args.pit_seed)


if __name__ == "__main__":
    main()
