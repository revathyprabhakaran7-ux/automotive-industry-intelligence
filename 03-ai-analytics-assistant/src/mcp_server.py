
import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Automotive Sales Analytics")

REPO = Path(__file__).resolve().parents[2]

SALES_FILE = (
    REPO / "01-business-intelligence"
    / "outputs" / "monthly_vehicle_sales.csv"
)

FORECAST_DIR = REPO / "02-automotive-forecasting" / "outputs"


def load_sales():
    if not SALES_FILE.is_file():
        raise FileNotFoundError(
            "Sales dataset unavailable. Run the Project 1 workflow."
        )

    data = pd.read_csv(SALES_FILE, parse_dates=["date"])

    required = {"date", "vehicles_sold"}
    if not required.issubset(data.columns):
        raise ValueError("The sales dataset has missing columns.")

    if data["date"].isna().any():
        raise ValueError("The sales dataset contains invalid dates.")

    if data["vehicles_sold"].isna().any():
        raise ValueError("The sales dataset contains missing sales values.")

    return data.sort_values("date").reset_index(drop=True)


@mcp.tool()
def get_annual_sales(year: int) -> dict:
    """Get Canadian vehicle sales for the specified year."""

    data = load_sales()
    selected = data[data["date"].dt.year == year]

    if selected.empty:
        return {"error": f"No sales data available for {year}."}

    return {
        "year": year,
        "vehicles_sold": int(selected["vehicles_sold"].sum()),
        "months_available": len(selected),
    }


@mcp.tool()
def compare_sales(year1: int, year2: int) -> dict:
    """Compare vehicle sales between two complete calendar years."""

    first = get_annual_sales(year1)
    second = get_annual_sales(year2)

    if "error" in first:
        return first

    if "error" in second:
        return second

    if first["months_available"] != 12 or second["months_available"] != 12:
        return {"error": "Both years must contain 12 months of data."}

    sales1 = first["vehicles_sold"]
    sales2 = second["vehicles_sold"]

    if sales1 == 0:
        return {"error": "Cannot calculate growth from zero sales."}

    change_pct = (sales2 - sales1) / sales1 * 100

    return {
        "year1": year1,
        "year2": year2,
        "sales_year1": sales1,
        "sales_year2": sales2,
        "change_pct": round(change_pct, 2),
    }


@mcp.tool()
def get_latest_sales() -> dict:
    """Get the latest available monthly vehicle sales figure."""

    data = load_sales()

    if data.empty:
        return {"error": "No sales observations available."}

    latest = data.iloc[-1]

    return {
        "month": latest["date"].strftime("%Y-%m"),
        "vehicles_sold": int(latest["vehicles_sold"]),
    }


@mcp.tool()
def get_forecast_evaluation() -> dict:
    """Get forecasting model performance and the selected model."""

    file = FORECAST_DIR / "metrics.json"

    if not file.is_file():
        return {"error": "Forecast evaluation results unavailable."}

    with file.open("r", encoding="utf-8") as f:
        return json.load(f)


@mcp.tool()
def get_sales_forecast() -> list[dict]:
    """Get the next 12 months of estimated vehicle sales."""

    file = FORECAST_DIR / "next_12_months_forecast.csv"

    if not file.is_file():
        return [{"error": "Forecast results unavailable."}]

    forecast = pd.read_csv(file)

    required = {"date", "forecast_vehicles_sold"}
    if not required.issubset(forecast.columns):
        return [{"error": "Forecast file has missing columns."}]

    if len(forecast) != 12:
        return [{"error": "Expected 12 monthly forecasts."}]

    if forecast[list(required)].isna().any().any():
        return [{"error": "Forecast contains missing values."}]

    return forecast.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
