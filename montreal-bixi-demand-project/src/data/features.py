from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("datetime")
    dt = pd.to_datetime(data["datetime"])
    qc_holidays = holidays.CA(subdiv="QC", years=dt.dt.year.unique())
    data["hour"] = dt.dt.hour
    data["day_of_week"] = dt.dt.dayofweek
    data["month"] = dt.dt.month
    data["day_of_year"] = dt.dt.dayofyear
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_holiday"] = dt.dt.date.map(lambda value: int(value in qc_holidays))
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["year_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
    data["year_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)
    return data


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("datetime")
    demand = data["demand"]
    data["lag_1h"] = demand.shift(1)
    data["lag_24h"] = demand.shift(24)
    data["lag_168h"] = demand.shift(168)
    data["rolling_mean_24h"] = demand.shift(1).rolling(24, min_periods=6).mean()
    data["rolling_mean_168h"] = demand.shift(1).rolling(168, min_periods=24).mean()
    return data


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(add_calendar_features(frame)).dropna().reset_index(drop=True)
