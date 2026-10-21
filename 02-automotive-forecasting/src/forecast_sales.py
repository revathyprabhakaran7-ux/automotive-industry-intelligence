
"""
Canadian monthly vehicle-sales forecasting.

Compares:
1. Seasonal naive forecasting
2. Ridge regression with lag and calendar features

Evaluation:
Two non-overlapping 12-month rolling-origin backtests.
Each forecast is generated recursively without using
actual sales from inside its forecast horizon.

Project 01 source files are read-only.
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


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

REPO = Path(__file__).resolve().parents[2]

INPUT = (
    REPO
    / "01-business-intelligence"
    / "outputs"
    / "monthly_vehicle_sales.csv"
)

OUTPUT = (
    REPO
    / "02-automotive-forecasting"
    / "outputs"
)

LAGS = (1, 2, 3, 6, 12, 13, 24)

HOLDOUT_MONTHS = 24
FORECAST_MONTHS = 12
RIDGE_ALPHA = 10.0

MIN_OBSERVATIONS = 60


# ==================================================
# DATA VALIDATION
# ==================================================

def load_and_validate(path: Path) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            f"Monthly sales data not found: {path}"
        )

    data = pd.read_csv(path, parse_dates=["date"])

    required = {"date", "vehicles_sold"}

    if not required.issubset(data.columns):
        raise ValueError(
            "Data must include date and vehicles_sold."
        )

    data = (
        data[["date", "vehicles_sold"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_OBSERVATIONS} monthly "
            "observations are required."
        )

    if (
        data["date"].isna().any()
        or data["date"].duplicated().any()
    ):
        raise ValueError(
            "Dates must be valid and unique."
        )

    expected_dates = pd.date_range(
        start=data["date"].iloc[0],
        periods=len(data),
        freq="MS",
    )

    if not (
        data["date"].to_numpy()
        == expected_dates.to_numpy()
    ).all():
        raise ValueError(
            "Data must contain consecutive "
            "month-start dates."
        )

    data["vehicles_sold"] = pd.to_numeric(
        data["vehicles_sold"],
        errors="coerce",
    )

    if (
        data["vehicles_sold"].isna().any()
        or not np.isfinite(
            data["vehicles_sold"].to_numpy()
        ).all()
        or (data["vehicles_sold"] <= 0).any()
    ):
        raise ValueError(
            "Vehicle sales must be finite, "
            "positive, and nonmissing."
        )

    return data


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def features(
    history: list[float],
    target_date: pd.Timestamp,
) -> dict[str, float]:

    if len(history) < max(LAGS):
        raise ValueError(
            "At least 24 earlier months are required."
        )

    result = {
        f"lag_{lag}": float(history[-lag])
        for lag in LAGS
    }

    result["recent_3m_mean"] = float(
        np.mean(history[-3:])
    )

    result["recent_12m_mean"] = float(
        np.mean(history[-12:])
    )

    result["month_sin"] = float(
        np.sin(
            2 * np.pi * target_date.month / 12
        )
    )

    result["month_cos"] = float(
        np.cos(
            2 * np.pi * target_date.month / 12
        )
    )

    result["year"] = float(
        target_date.year
    )

    return result


def supervised(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    values = (
        data["vehicles_sold"]
        .astype(float)
        .tolist()
    )

    rows = [
        features(
            values[:index],
            data.loc[index, "date"],
        )
        for index in range(24, len(data))
    ]

    X = pd.DataFrame(rows)

    y = pd.Series(
        values[24:],
        name="vehicles_sold",
    )

    return X, y


# ==================================================
# MODELS
# ==================================================

def create_ridge_model():

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def train_ridge(history_data: pd.DataFrame):

    X, y = supervised(history_data)

    model = create_ridge_model()

    model.fit(X, y)

    return model, list(X.columns)


# ==================================================
# RECURSIVE FORECASTING
# ==================================================

def forecast_recursive(
    history_data: pd.DataFrame,
    horizon: int,
    model_name: str,
    model=None,
    feature_columns=None,
) -> pd.DataFrame:
    """
    Generate an entire forecast horizon using only
    observations available at the forecast origin.

    For Ridge, later forecasts use previous predictions.
    For seasonal naive, the prior-year seasonal pattern
    is repeated recursively.
    """

    if model_name not in {
        "ridge",
        "seasonal_naive",
    }:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    history = (
        history_data["vehicles_sold"]
        .astype(float)
        .tolist()
    )

    last_date = history_data["date"].iloc[-1]

    forecasts = []

    for step in range(1, horizon + 1):

        target_date = (
            last_date
            + pd.DateOffset(months=step)
        )

        if model_name == "ridge":

            if model is None or feature_columns is None:
                raise ValueError(
                    "Ridge model and feature columns "
                    "are required."
                )

            row = pd.DataFrame([
                features(history, target_date)
            ])[feature_columns]

            prediction = max(
                0.0,
                float(model.predict(row)[0]),
            )

        else:
            prediction = float(
                history[-12]
            )

        forecasts.append({
            "date": target_date,
            "forecast_vehicles_sold": prediction,
        })

        # Critical: append the prediction, never an
        # unseen future actual value.
        history.append(prediction)

    return pd.DataFrame(forecasts)


# ==================================================
# EVALUATION METRICS
# ==================================================

def metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if (
        len(actual) != len(predicted)
        or len(actual) == 0
    ):
        raise ValueError(
            "Actual and forecast lengths must match."
        )

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    mape = float(
        np.mean(
            np.abs(
                (actual - predicted) / actual
            )
        ) * 100
    )

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
    }


# ==================================================
# TWO ROLLING-ORIGIN BACKTESTS
# ==================================================

def rolling_origin_backtest(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate two consecutive 12-month forecast windows.

    Window 1:
        Train before the final 24 observed months,
        forecast the next 12.

    Window 2:
        Train before the final 12 observed months,
        forecast the next 12.

    No forecast uses actual observations occurring
    after its forecast origin.
    """

    records = []

    total = len(data)

    origins = [
        total - HOLDOUT_MONTHS,
        total - FORECAST_MONTHS,
    ]

    for backtest_number, origin in enumerate(
        origins,
        start=1,
    ):

        training = (
            data.iloc[:origin]
            .reset_index(drop=True)
        )

        holdout = (
            data.iloc[
                origin:origin + FORECAST_MONTHS
            ]
            .reset_index(drop=True)
        )

        if len(training) < 36:
            raise ValueError(
                "Insufficient training history."
            )

        ridge_model, columns = train_ridge(
            training
        )

        ridge_forecast = forecast_recursive(
            history_data=training,
            horizon=FORECAST_MONTHS,
            model_name="ridge",
            model=ridge_model,
            feature_columns=columns,
        )

        seasonal_forecast = forecast_recursive(
            history_data=training,
            horizon=FORECAST_MONTHS,
            model_name="seasonal_naive",
        )

        for index in range(FORECAST_MONTHS):

            records.append({
                "date": holdout.loc[index, "date"],
                "actual": float(
                    holdout.loc[
                        index,
                        "vehicles_sold",
                    ]
                ),
                "seasonal_naive_prediction": float(
                    seasonal_forecast.loc[
                        index,
                        "forecast_vehicles_sold",
                    ]
                ),
                "ridge_prediction": float(
                    ridge_forecast.loc[
                        index,
                        "forecast_vehicles_sold",
                    ]
                ),
                "backtest_window": backtest_number,
                "forecast_horizon": index + 1,
            })

    return pd.DataFrame(records)


# ==================================================
# MLFLOW EXPERIMENT TRACKING
# ==================================================

def log_mlflow(
    summary: dict,
    predictions_path: Path,
    metrics_path: Path,
):

    mlflow.set_tracking_uri(
        f"sqlite:///{(OUTPUT / 'mlflow.db').resolve()}"
    )

    experiment_name = (
        "canadian_automotive_sales_forecasting"
    )

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(
                OUTPUT / "mlruns"
            ).resolve().as_uri(),
        )

    mlflow.set_experiment(
        experiment_name
    )

    for model_name, score in (
        ("seasonal_naive", summary["baseline"]),
        ("ridge", summary["ridge"]),
    ):

        with mlflow.start_run(
            run_name=model_name
        ):

            mlflow.log_param(
                "model",
                model_name,
            )

            mlflow.log_param(
                "evaluation_type",
                "rolling_origin_recursive_12m",
            )

            mlflow.log_param(
                "backtest_windows",
                2,
            )

            mlflow.log_param(
                "holdout_months",
                HOLDOUT_MONTHS,
            )

            mlflow.log_param(
                "forecast_horizon_months",
                FORECAST_MONTHS,
            )

            if model_name == "ridge":
                mlflow.log_param(
                    "alpha",
                    RIDGE_ALPHA,
                )

            mlflow.log_metrics(
                score
            )

            mlflow.log_artifact(
                str(metrics_path)
            )

            mlflow.log_artifact(
                str(predictions_path)
            )


# ==================================================
# MAIN PIPELINE
# ==================================================

def main() -> None:

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_and_validate(
        INPUT
    )

    # Step 1: Evaluate genuine recursive forecasts.
    evaluation = rolling_origin_backtest(
        data
    )

    actual = evaluation["actual"].to_numpy()

    baseline_prediction = evaluation[
        "seasonal_naive_prediction"
    ].to_numpy()

    ridge_prediction = evaluation[
        "ridge_prediction"
    ].to_numpy()

    baseline_score = metrics(
        actual,
        baseline_prediction,
    )

    ridge_score = metrics(
        actual,
        ridge_prediction,
    )

    # Select using unrounded MAE, not rounded metrics.
    baseline_mae = mean_absolute_error(
        actual,
        baseline_prediction,
    )

    ridge_mae = mean_absolute_error(
        actual,
        ridge_prediction,
    )

    winner = (
        "ridge"
        if ridge_mae < baseline_mae
        else "seasonal_naive"
    )

    holdout_path = (
        OUTPUT / "holdout_predictions.csv"
    )

    evaluation.to_csv(
        holdout_path,
        index=False,
    )

    # Step 2: Refit model using the full history.
    full_model, columns = train_ridge(
        data
    )

    # Step 3: Forecast the next 12 months.
    future = forecast_recursive(
        history_data=data,
        horizon=FORECAST_MONTHS,
        model_name=winner,
        model=(
            full_model
            if winner == "ridge"
            else None
        ),
        feature_columns=(
            columns
            if winner == "ridge"
            else None
        ),
    )

    # Preserve existing future forecast columns.
    future["forecast_vehicles_sold"] = (
        future["forecast_vehicles_sold"]
        .round()
        .astype(int)
    )

    future_path = (
        OUTPUT / "next_12_months_forecast.csv"
    )

    future.to_csv(
        future_path,
        index=False,
    )

    test_dates = evaluation["date"]

    summary = {
        "dataset_last_month": (
            data["date"].iloc[-1]
            .strftime("%Y-%m")
        ),
        "holdout_start": (
            test_dates.iloc[0]
            .strftime("%Y-%m")
        ),
        "holdout_end": (
            test_dates.iloc[-1]
            .strftime("%Y-%m")
        ),
        "evaluation_type": (
            "rolling-origin recursive "
            "12-month forecasting"
        ),
        "backtest_windows": 2,
        "holdout_months": HOLDOUT_MONTHS,
        "baseline": baseline_score,
        "ridge": ridge_score,
        "selected_model": winner,
        "selection_metric": "recursive holdout MAE",
        "forecast_horizon_months": FORECAST_MONTHS,
        "warning": (
            "Forecasts are estimates, not official "
            "sales projections. Historical backtests "
            "do not guarantee future performance."
        ),
    }

    metrics_path = OUTPUT / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 4: Track both results in MLflow.
    log_mlflow(
        summary,
        holdout_path,
        metrics_path,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "Saved forecast results to:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
