"""Cache one-step-ahead foundation-model forecasts for the real-data benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from benchmarks.data_loaders import (
    load_electricity_hourly,
    load_fred_md,
    load_traffic_hourly,
)

if TYPE_CHECKING:
    import torch

Q_GRID = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]

CONTEXT_LENS = {
    "electricity": 512,
    "traffic": 512,
    "fred_md": 120,
}

DEFAULT_SERIES_BATCH_SIZE = {
    "electricity": 64,
    "traffic": 32,
    "fred_md": 128,
}

DATASETS = {
    "electricity": load_electricity_hourly,
    "traffic": load_traffic_hourly,
    "fred_md": load_fred_md,
}


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to cache Chronos-2 forecasts in this module."
        ) from exc
    return torch


def _resolve_device(device: str) -> str:
    torch = _import_torch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_chronos2_pipeline(model_id: str, device: str):
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        print("ERROR: chronos-forecasting is not installed.")
        print('  pip install "chronos-forecasting>=2.0" torch')
        return None

    resolved_device = _resolve_device(device)
    print(f"Loading Chronos-2 from {model_id} on {resolved_device}...")
    return Chronos2Pipeline.from_pretrained(model_id, device_map=resolved_device)


def _extract_quantiles(prediction: object) -> list[float] | None:
    torch = _import_torch()
    if isinstance(prediction, torch.Tensor):
        arr = prediction.detach().cpu().numpy()
    else:
        arr = np.asarray(prediction)

    if arr.ndim == 0:
        return None

    if arr.shape[-1] == len(Q_GRID):
        q_vals = arr.reshape(-1, len(Q_GRID))[0].astype(float)
    elif arr.ndim >= 2 and arr.shape[0] == len(Q_GRID):
        q_vals = arr[:, 0].astype(float)
    else:
        return None

    if not np.all(np.isfinite(q_vals)):
        return None

    q_vals = np.maximum.accumulate(q_vals)
    return q_vals.tolist()


def _clear_accelerator_cache() -> None:
    torch = _import_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _predict_quantiles_batched(
    pipeline,
    contexts: list["torch.Tensor"],
) -> list[list[float] | None]:
    torch = _import_torch()
    try:
        with torch.inference_mode():
            quantiles, _ = pipeline.predict_quantiles(
                inputs=contexts,
                prediction_length=1,
                quantile_levels=Q_GRID,
            )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _clear_accelerator_cache()
        if len(contexts) == 1:
            return [None]
        mid = len(contexts) // 2
        return _predict_quantiles_batched(
            pipeline, contexts[:mid]
        ) + _predict_quantiles_batched(pipeline, contexts[mid:])

    if isinstance(quantiles, torch.Tensor):
        if quantiles.ndim == 1:
            batch_predictions = [quantiles]
        else:
            batch_predictions = [quantiles[i] for i in range(quantiles.shape[0])]
    else:
        arr = np.asarray(quantiles)
        if arr.ndim == 1:
            batch_predictions = [arr]
        else:
            batch_predictions = [arr[i] for i in range(arr.shape[0])]

    if len(batch_predictions) != len(contexts):
        if len(contexts) == 1:
            return [_extract_quantiles(quantiles)]
        return [None] * len(contexts)

    return [_extract_quantiles(pred) for pred in batch_predictions]


def _iter_series_windows(series, context_len: int):
    torch = _import_torch()
    y = series["y"].to_numpy(dtype=np.float32, copy=False)
    timestamps = series["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    min_context_obs = min(50, context_len)

    for t in range(context_len, len(y)):
        y_true = y[t]
        if not np.isfinite(y_true):
            continue

        context = y[max(0, t - context_len) : t]
        if int(np.isfinite(context).sum()) < min_context_obs:
            continue

        yield torch.tensor(context, dtype=torch.float32), timestamps[t], float(y_true)


def cache_chronos2(
    df,
    output_dir: Path,
    context_len: int = 512,
    device: str = "auto",
    model_id: str = "amazon/chronos-2",
    series_batch_size: int = 64,
    max_series: int | None = None,
    overwrite: bool = False,
):
    """Cache Chronos-2 one-step-ahead quantile forecasts."""
    pipeline = _load_chronos2_pipeline(model_id=model_id, device=device)
    if pipeline is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total_series = int(df["series_id"].nunique())
    planned_total = total_series if max_series is None else min(total_series, max_series)
    model_context_len = getattr(pipeline, "model_context_length", context_len)
    effective_context_len = min(context_len, int(model_context_len))

    t0 = time.time()
    done = 0

    for idx, (sid, series) in enumerate(df.groupby("series_id", sort=False), start=1):
        if max_series is not None and idx > max_series:
            break

        out_path = output_dir / f"{sid}.json"
        if out_path.exists() and not overwrite:
            done += 1
            print(f"  [{done}/{planned_total}] {sid}: cached")
            continue

        series = series.sort_values("timestamp", kind="stable")
        forecasts = []
        failed = 0
        batch_contexts: list[torch.Tensor] = []
        batch_timestamps: list[str] = []
        batch_truths: list[float] = []

        def flush_batch() -> None:
            nonlocal failed
            if not batch_contexts:
                return

            batch_quantiles = _predict_quantiles_batched(pipeline, batch_contexts)
            for ts, y_true, q_vals in zip(batch_timestamps, batch_truths, batch_quantiles):
                if q_vals is None:
                    failed += 1
                    continue
                forecasts.append(
                    {
                        "timestamp": ts,
                        "y_true": y_true,
                        "quantiles": q_vals,
                        "probabilities": Q_GRID,
                    }
                )

            batch_contexts.clear()
            batch_timestamps.clear()
            batch_truths.clear()

        for context, ts, y_true in _iter_series_windows(series, effective_context_len):
            batch_contexts.append(context)
            batch_timestamps.append(ts)
            batch_truths.append(y_true)
            if len(batch_contexts) >= series_batch_size:
                flush_batch()

        flush_batch()

        done += 1
        if not forecasts:
            print(f"  [{done}/{planned_total}] {sid}: skipped")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(forecasts, f)

        elapsed = max(time.time() - t0, 1e-9)
        rate = done / elapsed * 3600.0
        print(
            f"  [{done}/{planned_total}] {sid}: {len(forecasts)} steps, "
            f"{failed} dropped, {rate:.1f} series/hr"
        )

    print(f"Done: cached forecasts to {output_dir}")


def cache_timesfm(
    df,
    output_dir: Path,
    context_len: int = 512,
    device: str = "auto",
    model_id: str | None = None,
    series_batch_size: int = 64,
    max_series: int | None = None,
    overwrite: bool = False,
):
    """Placeholder until the TimesFM cache path is implemented."""
    raise NotImplementedError(
        "TimesFM caching is not implemented in this module. "
        "The public cache CLI currently supports only chronos2."
    )


def cache_moirai(
    df,
    output_dir: Path,
    context_len: int = 512,
    device: str = "auto",
    model_id: str | None = None,
    series_batch_size: int = 64,
    max_series: int | None = None,
    overwrite: bool = False,
):
    """Placeholder until the Moirai cache path is implemented."""
    raise NotImplementedError(
        "Moirai caching is not implemented in this module. "
        "The public cache CLI currently supports only chronos2."
    )


SUPPORTED_MODELS = {
    "chronos2": cache_chronos2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache foundation-model forecasts for the benchmark."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--series-batch-size", type=int, default=None)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = args.dataset
    model = args.model
    batch_size = args.series_batch_size or DEFAULT_SERIES_BATCH_SIZE[dataset]
    model_id = args.model_id
    if model_id is None and model == "chronos2":
        model_id = "amazon/chronos-2"

    print(f"Caching {model} forecasts for {dataset}...")
    df = DATASETS[dataset]()
    out = Path(f"data/cached_forecasts/{dataset}/{model}")

    SUPPORTED_MODELS[model](
        df,
        out,
        context_len=CONTEXT_LENS[dataset],
        device=args.device,
        model_id=model_id,
        series_batch_size=batch_size,
        max_series=args.max_series,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
