"""Download official BIXI trips, GBFS stations, and ECCC hourly weather."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests

from src.config import BIXI_ARCHIVES, RAW

GBFS_STATIONS = "https://gbfs.velobixi.com/gbfs/2-2/en/station_information.json"
ECCC_HOURLY = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"


def stream_download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def download_bixi(year: int) -> Path:
    if year not in BIXI_ARCHIVES:
        raise ValueError(f"No verified archive URL for {year}. Add it in src/config.py.")
    archive = stream_download(BIXI_ARCHIVES[year], RAW / f"bixi_{year}.zip")
    out = RAW / f"bixi_{year}"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(out)
    return out


def download_stations() -> Path:
    response = requests.get(GBFS_STATIONS, timeout=30)
    response.raise_for_status()
    target = RAW / "stations.json"
    target.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    return target


def download_weather(year: int, station_id: int = 51157) -> list[Path]:
    outputs = []
    for month in range(1, 13):
        params = {
            "format": "csv",
            "stationID": station_id,
            "Year": year,
            "Month": month,
            "Day": 1,
            "timeframe": 1,
            "submit": "Download Data",
        }
        response = requests.get(ECCC_HOURLY, params=params, timeout=60)
        response.raise_for_status()
        target = RAW / f"weather_{year}_{month:02d}.csv"
        target.write_bytes(response.content)
        outputs.append(target)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--station-id", type=int, default=51157)
    parser.add_argument("--skip-trips", action="store_true")
    args = parser.parse_args()
    if not args.skip_trips:
        download_bixi(args.year)
    download_stations()
    download_weather(args.year, args.station_id)
    print(f"Downloaded source data for {args.year} to {RAW}")


if __name__ == "__main__":
    main()
