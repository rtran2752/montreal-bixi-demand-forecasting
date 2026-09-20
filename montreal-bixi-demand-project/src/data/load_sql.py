"""Load processed tables into SQLite (default) or any SQLAlchemy database."""

from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import create_engine

from src.config import PROCESSED


def load(database_url: str) -> None:
    engine = create_engine(database_url)
    tables = {
        "hourly_demand": PROCESSED / "hourly_demand.parquet",
        "stations": PROCESSED / "stations.parquet",
        "station_hourly": PROCESSED / "station_hourly.parquet",
    }
    for name, path in tables.items():
        if path.exists():
            frame = pd.read_parquet(path)
            frame.to_sql(name, engine, if_exists="replace", index=False, chunksize=50_000)
            print(f"Loaded {len(frame):,} rows into {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="sqlite:///data/bixi.db")
    args = parser.parse_args()
    load(args.database_url)
