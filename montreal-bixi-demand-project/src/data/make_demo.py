"""Create deterministic synthetic data so the entire app can be evaluated instantly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED
from src.data.features import make_features


def generate(seed: int = 42, days: int = 240, output_dir=PROCESSED) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2025-04-01", periods=24 * days, freq="h")
    hour, dow = dt.hour.to_numpy(), dt.dayofweek.to_numpy()
    day = np.arange(len(dt)) / 24
    temp = 14 + 10 * np.sin(2 * np.pi * (day - 55) / 365) + rng.normal(0, 3, len(dt))
    precip = np.where(rng.random(len(dt)) < 0.12, rng.gamma(1.3, 1.5, len(dt)), 0)
    wind = np.maximum(0, rng.normal(14, 6, len(dt)))
    commute = 500 * np.exp(-(((hour - 8) / 2.0) ** 2)) + 650 * np.exp(-(((hour - 17) / 2.5) ** 2))
    leisure = 230 * np.exp(-(((hour - 14) / 4) ** 2)) * (dow >= 5)
    base = 100 + commute * (dow < 5) + leisure + 18 * np.maximum(temp, 0)
    demand = np.maximum(0, base * np.exp(-0.16 * precip) * np.exp(-0.012 * wind))
    demand = rng.poisson(demand).astype(int)
    raw = pd.DataFrame(
        {
            "datetime": dt,
            "demand": demand,
            "temp_c": temp.round(1),
            "relative_humidity": np.clip(68 + precip * 5 + rng.normal(0, 10, len(dt)), 25, 100),
            "wind_speed_kmh": wind.round(1),
            "precip_mm": precip.round(1),
        }
    )
    features = make_features(raw)
    raw.to_parquet(output_dir / "hourly_demand.parquet", index=False)
    features.to_parquet(output_dir / "hourly_features.parquet", index=False)
    return features


if __name__ == "__main__":
    frame = generate()
    print(f"Created {len(frame):,} demo rows")
