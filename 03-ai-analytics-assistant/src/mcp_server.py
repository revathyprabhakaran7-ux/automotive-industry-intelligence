import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Automotive Sales Analytics")

REPO = Path(__file__).resolve().parents[2]

SALES_FILE = (
    REPO
    / "01-business-intelligence"
    / "outputs"
    / "monthly_vehicle_sales.csv"
)

FORECAST_DIR = (
    REPO
    / "02-automotive-forecasting"
    / "outputs"
)


def load_sales():
    if not SALES_FILE.exists():
        raise FileNotFoundError(
            "Monthly sales data is missing. Run the Project 1 data workflow."
        )

    data = pd.read_csv(SALES_FILE, parse_dates=["date"])

    return data.sort_values("date").reset_index(drop=True)


@mcp.tool()
def get_annual_sales(year: int) -> dict:
    """Get Canadian vehicle sales for a given year."""

    data = load_sales()
    selected = data[data["date"].dt.year == year]

    if selected.empty:
        return {"error": f"No sales data available for {year}"}

    return {
        "year": year,
        "vehicles_sold": int(selected["vehicles_sold"].sum()),
        "months_available": len(selected),
    }


@mcp.tool()
def compare_sales(year1: int, year2: int) -> dict:
    """Compare vehicle sales between two complete years."""

    first = get_annual_sales(year1)
    second = get_annual_sales(year2)

    if "error" in first:
        return first

    if "error" in second:
        return second

    if first["months_available"] != 12 or second["months_available"] != 12:
        return {
            "error": "Both years must contain 12 months of sales data."
        }

    sales1 = first["vehicles_sold"]
    sales2 = second["vehicles_sold"]

    change_pct = ((sales2 - sales1) / sales1) * 100

    return {
        "year1": year1,
        "year2": year2,
        "sales_year1": sales1,
        "sales_year2": sales2,
        "change_pct": round(change_pct, 2),
    }


@mcp.tool()
def get_latest_sales() -> dict:
    """Get the most recent monthly sales observation."""

    data = load_sales()
    latest = data.iloc[-1]

    return {
        "month": latest["date"].strftime("%Y-%m"),
        "vehicles_sold": int(latest["vehicles_sold"]),
    }


@mcp.tool()
def get_forecast_evaluation() -> dict:
    """Get forecasting model accuracy and comparison results."""

    file = FORECAST_DIR / "evaluation_metrics.json"

    if not file.exists():
        return {
            "error": "Forecast evaluation results are unavailable."
        }

    with file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    return results


@mcp.tool()
def get_sales_forecast() -> list[dict]:
    """Get the next 12 months of vehicle sales forecasts."""

    file = FORECAST_DIR / "forecast_12_months.csv"

    if not file.exists():
        return [
            {"error": "Forecast data is unavailable."}
        ]

    forecast = pd.read_csv(file)

    forecast = forecast.astype(object).where(
        pd.notna(forecast), None
    )

    return forecast.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
