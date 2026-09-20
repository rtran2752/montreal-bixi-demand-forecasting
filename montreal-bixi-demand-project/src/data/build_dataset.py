"""Normalize varying BIXI schemas and build leakage-safe hourly features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.config import PROCESSED, RAW
from src.data.features import make_features

ALIASES = {
    "start_time": ["start_date", "start_time", "started_at", "starttimems", "emplacement_pk_start_date"],
    "start_station_id": ["start_station_code", "start_station_id", "emplacement_pk_start"],
    "start_station_name": ["startstationname", "start_station_name", "start_station"],
    "end_station_id": ["end_station_code", "end_station_id", "emplacement_pk_end"],
    "duration_sec": ["duration_sec", "duration", "duration_seconds"],
    "is_member": ["is_member", "membership", "member"],
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def find_column(columns: list[str], canonical: str, required: bool = True) -> str | None:
    normalized = {normalize_name(c): c for c in columns}
    for alias in ALIASES[canonical]:
        if alias in normalized:
            return normalized[alias]
    if required:
        raise ValueError(f"Missing {canonical}. Available columns: {columns}")
    return None


def load_hourly_trips(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly_parts, station_parts, station_metadata = [], [], []
    csvs = sorted(folder.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")
    for csv in csvs:
        sample = pd.read_csv(csv, nrows=2)
        time_col = find_column(list(sample.columns), "start_time")
        station_col = find_column(list(sample.columns), "start_station_id", required=False)
        station_name_col = find_column(list(sample.columns), "start_station_name", required=False)
        lookup = {normalize_name(c): c for c in sample.columns}
        lat_col = lookup.get("startstationlatitude") or lookup.get("start_station_latitude")
        lon_col = lookup.get("startstationlongitude") or lookup.get("start_station_longitude")
        borough_col = lookup.get("startstationarrondissement") or lookup.get("start_station_arrondissement")
        station_key = station_col or station_name_col
        usecols = list(
            dict.fromkeys(c for c in [time_col, station_key, station_name_col, lat_col, lon_col, borough_col] if c)
        )
        for chunk in pd.read_csv(csv, usecols=usecols, chunksize=500_000):
            if "ms" in normalize_name(time_col):
                dt = (
                    pd.to_datetime(chunk[time_col], unit="ms", errors="coerce", utc=True)
                    .dt.tz_convert("America/Toronto")
                    .dt.tz_localize(None)
                    .dt.floor("h")
                )
            else:
                dt = pd.to_datetime(chunk[time_col], errors="coerce").dt.floor("h")
            valid = pd.DataFrame({"datetime": dt}).dropna()
            hourly_parts.append(valid.groupby("datetime").size().rename("demand"))
            if station_key:
                stations = pd.DataFrame({"datetime": dt, "station_id": chunk[station_key]})
                stations = stations.dropna()
                station_parts.append(stations.groupby(["datetime", "station_id"]).size().rename("departures"))
                if lat_col and lon_col:
                    metadata = pd.DataFrame(
                        {
                            "station_id": chunk[station_key],
                            "name": chunk[station_name_col] if station_name_col else chunk[station_key],
                            "borough": chunk[borough_col] if borough_col else None,
                            "latitude": pd.to_numeric(chunk[lat_col], errors="coerce"),
                            "longitude": pd.to_numeric(chunk[lon_col], errors="coerce"),
                        }
                    ).dropna(subset=["station_id", "latitude", "longitude"])
                    station_metadata.append(metadata.drop_duplicates("station_id"))
    hourly = pd.concat(hourly_parts).groupby(level=0).sum().sort_index().reset_index()
    full_range = pd.date_range(hourly.datetime.min(), hourly.datetime.max(), freq="h")
    hourly = hourly.set_index("datetime").reindex(full_range, fill_value=0)
    hourly.index.name = "datetime"
    station_hourly = (
        pd.concat(station_parts).groupby(level=[0, 1]).sum().reset_index() if station_parts else pd.DataFrame()
    )
    metadata = (
        pd.concat(station_metadata, ignore_index=True).drop_duplicates("station_id", keep="last")
        if station_metadata
        else pd.DataFrame()
    )
    return hourly.reset_index(), station_hourly, metadata


def load_weather(year: int) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RAW.glob(f"weather_{year}_*.csv"))]
    if not frames:
        raise FileNotFoundError("Weather files missing. Run src.data.download first.")
    raw = pd.concat(frames, ignore_index=True)
    columns = {normalize_name(c): c for c in raw.columns}
    date_col = columns.get("date_time_lst") or columns.get("date_time")
    if not date_col:
        raise ValueError("ECCC date/time column not found")
    weather = pd.DataFrame({"datetime": pd.to_datetime(raw[date_col], errors="coerce")})
    mapping = {
        "temp_c": "temp_c",
        "rel_hum": "relative_humidity",
        "wind_spd_km_h": "wind_speed_kmh",
        "precip_amount_mm": "precip_mm",
    }
    for source, target in mapping.items():
        col = columns.get(source)
        weather[target] = pd.to_numeric(raw[col], errors="coerce") if col else 0.0
    return weather.dropna(subset=["datetime"]).drop_duplicates("datetime")


def load_stations() -> pd.DataFrame:
    path = RAW / "stations.json"
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["data"]["stations"])


def build(year: int) -> pd.DataFrame:
    hourly, station_hourly, trip_stations = load_hourly_trips(RAW / f"bixi_{year}")
    weather = load_weather(year)
    merged = hourly.merge(weather, on="datetime", how="left").sort_values("datetime")
    weather_cols = ["temp_c", "relative_humidity", "wind_speed_kmh", "precip_mm"]
    merged[weather_cols] = merged[weather_cols].interpolate(limit_direction="both").fillna(0)
    features = make_features(merged)
    features.to_parquet(PROCESSED / "hourly_features.parquet", index=False)
    hourly.to_parquet(PROCESSED / "hourly_demand.parquet", index=False)
    if not station_hourly.empty:
        station_hourly.to_parquet(PROCESSED / "station_hourly.parquet", index=False)
    stations = trip_stations if not trip_stations.empty else load_stations()
    if not stations.empty:
        stations.to_parquet(PROCESSED / "stations.parquet", index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    result = build(args.year)
    print(f"Built {len(result):,} hourly feature rows")


if __name__ == "__main__":
    main()
