"""Evaluate a seasonal baseline and Ridge model; forecast the next 12 months.

Run from the repository root after Project 1 has produced monthly_vehicle_sales.csv.
All forecasts are statistical estimates, not official Statistics Canada projections.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "01-business-intelligence" / "outputs" / "monthly_vehicle_sales.csv"
OUTPUT = REPO / "02-automotive-forecasting" / "outputs"
LAGS = (1, 2, 3, 6, 12, 13, 24)
HOLDOUT_MONTHS = 24
FORECAST_MONTHS = 12


def features(history: list[float], target_date: pd.Timestamp) -> dict[str, float]:
    """Build predictors from months strictly before target_date."""
    if len(history) < max(LAGS):
        raise ValueError("At least 24 earlier months are required.")
    result = {f"lag_{lag}": float(history[-lag]) for lag in LAGS}
    result["recent_3m_mean"] = float(np.mean(history[-3:]))
    result["recent_12m_mean"] = float(np.mean(history[-12:]))
    result["month_sin"] = float(np.sin(2 * np.pi * target_date.month / 12))
    result["month_cos"] = float(np.cos(2 * np.pi * target_date.month / 12))
    result["year"] = float(target_date.year)
    return result


def supervised(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    values = data["vehicles_sold"].to_numpy(dtype=float).tolist()
    rows = [features(values[:i], data.loc[i, "date"]) for i in range(24, len(data))]
    return pd.DataFrame(rows), pd.Series(values[24:], name="vehicles_sold")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "mape_pct": round(float(np.mean(np.abs((actual - predicted) / actual)) * 100), 2),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_file():
        raise FileNotFoundError(f"Run Project 1's data workflow first: {INPUT}")

    data = pd.read_csv(INPUT, parse_dates=["date"])
    data = data[["date", "vehicles_sold"]].sort_values("date").reset_index(drop=True)
    if len(data) < 60:
        raise ValueError("Need at least 60 monthly observations.")
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("Dates must be valid and unique.")
    expected = pd.date_range(data["date"].iloc[0], periods=len(data), freq="MS")
    if not (data["date"].to_numpy() == expected.to_numpy()).all():
        raise ValueError("Data must contain consecutive month-start dates.")
    if data["vehicles_sold"].isna().any() or (data["vehicles_sold"] <= 0).any():
        raise ValueError("Vehicle sales must be positive and nonmissing.")

    X, y = supervised(data)
    split = len(X) - HOLDOUT_MONTHS
    if split <= 0:
        raise ValueError("Not enough observations for the holdout period.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # One-step-ahead backtest: each row uses only already-observed prior months.
    # The seasonal naive comparator forecasts the same month as last year.
    seasonal = X_test["lag_12"].to_numpy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X_train, y_train)
    ridge_pred = np.maximum(0, model.predict(X_test))
    actual = y_test.to_numpy()

    baseline_score = metrics(actual, seasonal)
    ridge_score = metrics(actual, ridge_pred)
    winner = "ridge" if ridge_score["mae"] < baseline_score["mae"] else "seasonal_naive"

    test_dates = data["date"].iloc[-HOLDOUT_MONTHS:].reset_index(drop=True)
       evaluation = pd.DataFrame({
        "date": test_dates,
        "actual": actual,
        "seasonal_naive_prediction": seasonal,
        "ridge_prediction": ridge_pred,
    })

    evaluation.to_csv(
        OUTPUT / "holdout_predictions.csv",
        index=False
    )
