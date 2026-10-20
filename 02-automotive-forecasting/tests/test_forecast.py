"""Synthetic-data smoke test; does not alter Project 1 outputs."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "forecast_sales.py"
spec = importlib.util.spec_from_file_location("forecast_sales", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ForecastSmokeTest(unittest.TestCase):
    def test_pipeline(self):
        dates = pd.date_range("2012-01-01", periods=168, freq="MS")
        values = [100_000 + 500 * i + 12_000 * np.sin(2 * np.pi * d.month / 12)
                  for i, d in enumerate(dates)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module.INPUT = root / "monthly_vehicle_sales.csv"
            module.OUTPUT = root / "results"
            pd.DataFrame({"date": dates, "vehicles_sold": values}).to_csv(module.INPUT, index=False)
            module.main()
            summary = json.loads((module.OUTPUT / "metrics.json").read_text())
            self.assertEqual(summary["forecast_horizon_months"], 12)
            self.assertIn(summary["selected_model"], ("ridge", "seasonal_naive"))
            self.assertEqual(len(pd.read_csv(module.OUTPUT / "holdout_predictions.csv")), 24)
            predictions = pd.read_csv(module.OUTPUT / "next_12_months_forecast.csv")
            self.assertEqual(len(predictions), 12)
            self.assertTrue((predictions["forecast_vehicles_sold"] > 0).all())


if __name__ == "__main__":
    unittest.main()
