
"""
Tests for the automotive forecasting pipeline.

Uses synthetic data in temporary directories.
Does not modify Project 01 or Tableau outputs.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# IMPORT FORECAST MODULE
# ==========================================

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "forecast_sales.py"
)

spec = importlib.util.spec_from_file_location(
    "forecast_sales",
    SCRIPT,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# ==========================================
# SYNTHETIC MONTHLY DATA
# ==========================================

def make_data(months=168):

    dates = pd.date_range(
        "2012-01-01",
        periods=months,
        freq="MS",
    )

    values = [
        100_000
        + 500 * index
        + 12_000 * np.sin(
            2 * np.pi * date.month / 12
        )
        for index, date in enumerate(dates)
    ]

    return pd.DataFrame({
        "date": dates,
        "vehicles_sold": values,
    })


# ==========================================
# TESTS
# ==========================================

class ForecastTests(unittest.TestCase):

    def test_feature_lags(self):

        history = list(range(1, 25))

        result = module.features(
            history,
            pd.Timestamp("2025-01-01"),
        )

        self.assertEqual(
            result["lag_1"],
            24,
        )

        self.assertEqual(
            result["lag_12"],
            13,
        )

        self.assertEqual(
            result["lag_24"],
            1,
        )

        self.assertAlmostEqual(
            result["recent_3m_mean"],
            23,
        )

    def test_rolling_origin_backtest(self):

        data = make_data()

        evaluation = module.rolling_origin_backtest(
            data
        )

        self.assertEqual(
            len(evaluation),
            24,
        )

        self.assertEqual(
            evaluation["backtest_window"].tolist(),
            [1] * 12 + [2] * 12,
        )

        self.assertEqual(
            evaluation["forecast_horizon"].tolist(),
            list(range(1, 13)) * 2,
        )

        self.assertTrue(
            (
                evaluation[
                    "seasonal_naive_prediction"
                ] > 0
            ).all()
        )

    def test_backtest_avoids_future_leakage(self):
        """
        Changing actual sales in the last 12 months
        must not change forecasts for those same
        months, because their forecast origin
        occurs before them.
        """

        original = make_data()

        modified = original.copy()

        modified.loc[
            modified.index[-12:],
            "vehicles_sold",
        ] *= 1.5

        original_predictions = (
            module.rolling_origin_backtest(original)
        )

        modified_predictions = (
            module.rolling_origin_backtest(modified)
        )

        for column in [
            "ridge_prediction",
            "seasonal_naive_prediction",
        ]:

            np.testing.assert_allclose(
                original_predictions[
                    column
                ].iloc[-12:].to_numpy(),
                modified_predictions[
                    column
                ].iloc[-12:].to_numpy(),
            )

    def test_missing_month_rejected(self):

        data = make_data(80)

        data = data.drop(
            index=20
        ).reset_index(drop=True)

        with tempfile.TemporaryDirectory() as temp:

            file_path = (
                Path(temp) / "invalid.csv"
            )

            data.to_csv(
                file_path,
                index=False,
            )

            with self.assertRaises(ValueError):
                module.load_and_validate(
                    file_path
                )

    def test_full_pipeline(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            module.INPUT = (
                root / "monthly_vehicle_sales.csv"
            )

            module.OUTPUT = (
                root / "results"
            )

            make_data().to_csv(
                module.INPUT,
                index=False,
            )

            module.main()

            metrics_path = (
                module.OUTPUT / "metrics.json"
            )

            holdout_path = (
                module.OUTPUT
                / "holdout_predictions.csv"
            )

            future_path = (
                module.OUTPUT
                / "next_12_months_forecast.csv"
            )

            self.assertTrue(
                metrics_path.exists()
            )

            self.assertTrue(
                holdout_path.exists()
            )

            self.assertTrue(
                future_path.exists()
            )

            summary = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                summary["forecast_horizon_months"],
                12,
            )

            self.assertEqual(
                summary["backtest_windows"],
                2,
            )

            self.assertIn(
                summary["selected_model"],
                ("ridge", "seasonal_naive"),
            )

            holdout = pd.read_csv(
                holdout_path
            )

            self.assertEqual(
                len(holdout),
                24,
            )

            future = pd.read_csv(
                future_path
            )

            self.assertEqual(
                len(future),
                12,
            )

            self.assertTrue(
                (
                    future["forecast_vehicles_sold"] > 0
                ).all()
            )

            expected_dates = pd.date_range(
                "2026-01-01",
                periods=12,
                freq="MS",
            )

            pd.testing.assert_series_equal(
                pd.to_datetime(
                    future["date"]
                ).reset_index(drop=True),
                pd.Series(
                    expected_dates,
                    name="date",
                ),
            )


if __name__ == "__main__":
    unittest.main()
