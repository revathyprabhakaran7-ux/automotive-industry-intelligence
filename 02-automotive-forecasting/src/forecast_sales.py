"""Monthly vehicle sales forecasting: seasonal baseline and Ridge regression."""

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
    """Lag and calendar features for one forecast month."""

    if len(history) < max(LAGS):
        raise ValueError("At least 24 earlier months are required.")

    result = {f"lag_{lag}": float(history[-lag]) for lag in LAGS}

    result["recent_3m_mean"] = float(np.mean(history[-3:]))
    result["recent_12m_mean"] = float(np.mean(history[-12:]))

    result["month_sin"] = float(
        np.sin(2 * np.pi * target_date.month / 12)
    )

    result["month_cos"] = float(
        np.cos(2 * np.pi * target_date.month / 12)
    )

    result["year"] = float(target_date.year)

    return result


def supervised(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    values = data["vehicles_sold"].to_numpy(dtype=float).tolist()

    rows = [
        features(values[:i], data.loc[i, "date"])
        for i in range(24, len(data))
    ]

    return (
        pd.DataFrame(rows),
        pd.Series(values[24:], name="vehicles_sold")
    )


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:

    return {
        "mae": round(
            float(mean_absolute_error(actual, predicted)), 2
        ),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 2
        ),
        "mape_pct": round(
            float(np.mean(np.abs((actual - predicted) / actual)) * 100), 2
        ),
    }


def main() -> None:

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Monthly sales data not found: {INPUT}"
        )

    data = pd.read_csv(INPUT, parse_dates=["date"])

    data = (
        data[["date", "vehicles_sold"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(data) < 60:
        raise ValueError("Need at least 60 monthly observations.")

    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("Dates must be valid and unique.")

    expected = pd.date_range(
        data["date"].iloc[0],
        periods=len(data),
        freq="MS"
    )

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

    # Evaluate one-step predictions over the final 24 observed months.

    seasonal = X_test["lag_12"].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    )

    model.fit(X_train, y_train)

    ridge_pred = np.maximum(
        0,
        model.predict(X_test)
    )

    actual = y_test.to_numpy()

    baseline_score = metrics(actual, seasonal)
    ridge_score = metrics(actual, ridge_pred)

    winner = (
        "ridge"
        if ridge_score["mae"] < baseline_score["mae"]
        else "seasonal_naive"
    )

    test_dates = (
        data["date"]
        .iloc[-HOLDOUT_MONTHS:]
        .reset_index(drop=True)
    )

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

    # Refit on full history and generate the forecast.

    model.fit(X, y)

    history = data["vehicles_sold"].astype(float).tolist()
    last_date = data["date"].iloc[-1]

    future = []

    for month in range(1, FORECAST_MONTHS + 1):

        date = last_date + pd.DateOffset(months=month)

        if winner == "ridge":

            row = pd.DataFrame(
                [features(history, date)]
            )[X.columns]

            prediction = max(
                0.0,
                float(model.predict(row)[0])
            )

        else:
            prediction = float(history[-12])

        future.append({
            "date": date,
            "forecast_vehicles_sold": round(prediction)
        })

        history.append(prediction)

    pd.DataFrame(future).to_csv(
        OUTPUT / "next_12_months_forecast.csv",
        index=False
    )

    summary = {
        "dataset_last_month": last_date.strftime("%Y-%m"),
        "holdout_start": test_dates.iloc[0].strftime("%Y-%m"),
        "holdout_end": test_dates.iloc[-1].strftime("%Y-%m"),
        "evaluation_type": "one-step-ahead with observed lag history",
        "baseline": baseline_score,
        "ridge": ridge_score,
        "selected_model": winner,
        "selection_metric": "holdout MAE",
        "forecast_horizon_months": FORECAST_MONTHS,
        "warning": "Forecasts are estimates, not official sales projections.",
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # Record model evaluation results in MLflow.

    mlflow.set_tracking_uri(
        f"sqlite:///{OUTPUT / 'mlflow.db'}"
    )

    experiment_name = "canadian_automotive_sales_forecasting"

    if mlflow.get_experiment_by_name(experiment_name) is None:

        mlflow.create_experiment(
            experiment_name,
            artifact_location=(OUTPUT / "mlruns").as_uri()
        )

    mlflow.set_experiment(experiment_name)

    for model_name, score in (
        ("seasonal_naive", baseline_score),
        ("ridge", ridge_score)
    ):

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("holdout_months", HOLDOUT_MONTHS)
            mlflow.log_param("forecast_horizon_months", FORECAST_MONTHS)
            mlflow.log_param("evaluation_type", "one_step_ahead")
            mlflow.log_param("model", model_name)

            if model_name == "ridge":
                mlflow.log_param("alpha", 10.0)

            mlflow.log_metrics(score)

            mlflow.log_artifact(
                str(OUTPUT / "metrics.json")
            )

            mlflow.log_artifact(
                str(OUTPUT / "holdout_predictions.csv")
            )

    print(json.dumps(summary, indent=2))
    print("Saved forecasts and MLflow runs to:", OUTPUT)


if __name__ == "__main__":
    main()
