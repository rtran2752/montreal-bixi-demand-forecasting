import numpy as np
import pandas as pd

from src.data.features import make_features


def test_lags_use_only_past_values():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=200, freq="h"),
            "demand": np.arange(200),
            "temp_c": 1,
            "relative_humidity": 50,
            "wind_speed_kmh": 5,
            "precip_mm": 0,
        }
    )
    result = make_features(frame)
    row = result.iloc[0]
    source = frame.loc[frame.datetime == row.datetime].index[0]
    assert row.lag_1h == frame.loc[source - 1, "demand"]
    assert row.lag_168h == frame.loc[source - 168, "demand"]


def test_feature_output_has_no_missing_values():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=200, freq="h"),
            "demand": np.arange(200),
            "temp_c": 1,
            "relative_humidity": 50,
            "wind_speed_kmh": 5,
            "precip_mm": 0,
        }
    )
    assert not make_features(frame).isna().any().any()
