
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcp_server import (
    SALES_FILE,
    FORECAST_DIR,
    get_annual_sales,
    compare_sales,
    get_latest_sales,
    get_forecast_evaluation,
    get_sales_forecast,
)


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "03-ai-analytics-assistant" / "outputs"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not SALES_FILE.is_file():
        raise FileNotFoundError(f"Missing sales data: {SALES_FILE}")

    metrics_file = FORECAST_DIR / "metrics.json"
    forecast_file = FORECAST_DIR / "next_12_months_forecast.csv"

    if not metrics_file.is_file():
        raise FileNotFoundError(f"Missing evaluation results: {metrics_file}")

    if not forecast_file.is_file():
        raise FileNotFoundError(f"Missing forecasts: {forecast_file}")

    sales = pd.read_csv(SALES_FILE, parse_dates=["date"])

    annual_2024 = get_annual_sales(2024)
    annual_2025 = get_annual_sales(2025)
    comparison = compare_sales(2024, 2025)
    latest = get_latest_sales()
    evaluation = get_forecast_evaluation()
    forecast = get_sales_forecast()

    results = {
        "annual_sales_2024": annual_2024,
        "annual_sales_2025": annual_2025,
        "sales_comparison": comparison,
        "latest_month": latest,
        "model_evaluation": evaluation,
        "forecast": forecast,
    }

    for name, result in results.items():
        if isinstance(result, dict) and "error" in result:
            raise ValueError(f"{name}: {result['error']}")

        if isinstance(result, list):
            if not result or any("error" in row for row in result):
                raise ValueError(f"{name}: Invalid forecast results")

    last_observed = pd.Timestamp(sales["date"].max())
    forecast_dates = pd.to_datetime([row["date"] for row in forecast])

    expected_dates = pd.date_range(
        last_observed + pd.DateOffset(months=1),
        periods=12,
        freq="MS",
    )

    if not forecast_dates.equals(expected_dates):
        raise ValueError("Forecast dates do not follow the observed data.")

    if evaluation["dataset_last_month"] != last_observed.strftime("%Y-%m"):
        raise ValueError("Evaluation and sales datasets are out of sync.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Statistics Canada Table 20-10-0085-01",
        "observations": len(sales),
        "first_month": sales["date"].min().strftime("%Y-%m"),
        "last_month": last_observed.strftime("%Y-%m"),
        "tools_checked": list(results.keys()),
        "results": results,
        "integration_status": "passed",
    }

    report_file = OUTPUT / "integration_report.json"

    report_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Integration passed. Report saved to {report_file}")


if __name__ == "__main__":
    main()
