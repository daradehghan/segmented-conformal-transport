"""Data loaders for the real-data benchmark.

Parses Monash TSF/TS files from Zenodo zip archives into long tables.
See BenchmarkSpec_P1_final §1.3 for exact preprocessing rules.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_monash_timestamp(timestamp_str: str) -> pd.Timestamp:
    # Monash hourly files sometimes use HH-MM-SS, which pandas may misread
    # as a UTC offset. Convert only the time separators, preserving the date.
    if (
        len(timestamp_str) >= 19
        and timestamp_str[10] == " "
        and timestamp_str[13] == "-"
        and timestamp_str[16] == "-"
    ):
        timestamp_str = (
            timestamp_str[:10]
            + " "
            + timestamp_str[11:13]
            + ":"
            + timestamp_str[14:16]
            + ":"
            + timestamp_str[17:19]
        )
    ts = pd.Timestamp(timestamp_str)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts


def load_monash_tsf(zip_path: str) -> pd.DataFrame:
    """Parse a Monash .tsf/.ts file from a zip archive into a long table."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        ts_names = [n for n in zf.namelist() if n.endswith(".tsf") or n.endswith(".ts")]
        if not ts_names:
            raise ValueError(f"No .tsf/.ts file found in {zip_path}")
        content = _decode_bytes(zf.read(ts_names[0]))

    lines = content.strip().splitlines()
    data_started = False
    frequency = None
    records = []

    for line in lines:
        line = line.strip()
        if line.startswith("@frequency"):
            frequency = line.split()[1]
        if line.startswith("@data"):
            data_started = True
            continue
        if not data_started or not line:
            continue

        parts = line.split(":")
        if len(parts) < 3:
            continue
        series_id = parts[0].strip()
        timestamp_str = parts[1].strip()
        values_str = ":".join(parts[2:]).strip()

        try:
            start_ts = _parse_monash_timestamp(timestamp_str)
        except Exception:
            continue

        values = []
        for v in values_str.split(","):
            v = v.strip()
            if v in {"?", ""}:
                values.append(np.nan)
            else:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(np.nan)

        freq_map = {
            "hourly": "h",
            "monthly": "MS",
            "daily": "D",
            "weekly": "W",
            "yearly": "YS",
        }
        freq = freq_map.get(frequency, "h")
        timestamps = pd.date_range(start=start_ts, periods=len(values), freq=freq)

        for ts, val in zip(timestamps, values):
            records.append(
                {
                    "series_id": series_id,
                    "timestamp": ts,
                    "y": val,
                    "frequency": frequency or "unknown",
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["series_id", "timestamp"], keep="first")
    if len(df) < before:
        print(f"  Dropped {before - len(df)} duplicate timestamps")

    return df


def load_electricity_hourly(
    zip_path: str = "data/raw/electricity_hourly_dataset.zip", min_obs: int = 2000
) -> pd.DataFrame:
    print("Loading Electricity Hourly...")
    df = load_monash_tsf(zip_path)
    counts = df.groupby("series_id")["y"].count()
    valid = counts[counts >= min_obs].index
    df = df[df["series_id"].isin(valid)].copy()
    df["dataset"] = "electricity_hourly"
    print(f"  {df['series_id'].nunique()} series, {len(df)} rows")
    return df


def load_traffic_hourly(
    zip_path: str = "data/raw/traffic_hourly_dataset.zip", min_obs: int = 2000
) -> pd.DataFrame:
    print("Loading Traffic Hourly...")
    df = load_monash_tsf(zip_path)
    counts = df.groupby("series_id")["y"].count()
    valid = counts[counts >= min_obs].index
    df = df[df["series_id"].isin(valid)].copy()
    df["dataset"] = "traffic_hourly"
    print(f"  {df['series_id'].nunique()} series, {len(df)} rows")
    return df


def load_fred_md(
    zip_path: str = "data/raw/fred_md_dataset.zip", min_obs: int = 120
) -> pd.DataFrame:
    print("Loading FRED-MD...")
    df = load_monash_tsf(zip_path).copy()

    # Trim leading NaNs per series without using groupby.apply(),
    # which can behave differently across pandas versions.
    first_valid = (
        df.loc[df["y"].notna()]
        .groupby("series_id", as_index=False)["timestamp"]
        .min()
        .rename(columns={"timestamp": "first_valid_timestamp"})
    )

    df = df.merge(first_valid, on="series_id", how="left")
    df = df[df["first_valid_timestamp"].notna()].copy()
    df = df[df["timestamp"] >= df["first_valid_timestamp"]].copy()
    df = df.drop(columns=["first_valid_timestamp"])

    counts = df.groupby("series_id")["y"].count()
    valid = counts[counts >= min_obs].index
    df = df[df["series_id"].isin(valid)].copy()
    df["dataset"] = "fred_md"
    print(f"  {df['series_id'].nunique()} series, {len(df)} rows")
    return df


if __name__ == "__main__":
    load_electricity_hourly()
    load_traffic_hourly()
    load_fred_md()
