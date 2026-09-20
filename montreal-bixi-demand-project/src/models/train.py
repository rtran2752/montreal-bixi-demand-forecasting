from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import MODELS, PROCESSED

TARGET = "demand"
FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
    "temp_c",
    "relative_humidity",
    "wind_speed_kmh",
    "precip_mm",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
]


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(0, prediction)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
        "WAPE": float(np.abs(y_true - prediction).sum() / max(y_true.sum(), 1)),
    }


def train(
    data_path=PROCESSED / "hourly_features.parquet",
    output_dir=PROCESSED,
    models_dir=MODELS,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(data_path).sort_values("datetime")
    split = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:split], data.iloc[split:]
    x_train, x_test = train_df[FEATURES], test_df[FEATURES]
    y_train, y_test = train_df[TARGET], test_df[TARGET]
    baseline = test_df["lag_168h"].to_numpy()

    models = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.07,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
    }
    results = {"seasonal_naive": metrics(y_test, baseline)}
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])
        pipe.fit(x_train, y_train)
        prediction = np.maximum(0, pipe.predict(x_test))
        results[name] = metrics(y_test, prediction)
        fitted[name] = (pipe, prediction)

    winner = min(fitted, key=lambda name: results[name]["MAE"])
    model, prediction = fitted[winner]
    joblib.dump({"model": model, "features": FEATURES, "name": winner}, models_dir / "model.joblib")
    predictions = test_df[["datetime", TARGET]].copy()
    predictions["prediction"] = prediction
    predictions["baseline"] = baseline
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    payload = {
        "winner": winner,
        "split_datetime": str(test_df.datetime.iloc[0]),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "models": results,
    }
    (models_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROCESSED / "hourly_features.parquet"))
    args = parser.parse_args()
    result = train(args.data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
