from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Automotive Sales Analytics")

REPO = Path(__file__).resolve().parents[2]
SALES_FILE = REPO / "01-business-intelligence" / "outputs" / "monthly_vehicle_sales.csv"


def load_sales():
    if not SALES_FILE.exists():
        raise FileNotFoundError(
            "Monthly sales data is missing. Run the Project 1 data workflow."
        )

    data = pd.read_csv(SALES_FILE, parse_dates=["date"])
    return data.sort_values("date")


@mcp.tool()
def get_annual_sales(year: int) -> dict:
    """Return total Canadian new motor vehicle sales for a year."""

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
    """Compare vehicle sales between two years."""

    first = get_annual_sales(year1)
    second = get_annual_sales(year2)

    if "error" in first:
        return first

    if "error" in second:
        return second

    if first["months_available"] != 12 or second["months_available"] != 12:
        return {"error": "Both years must have 12 months of data."}

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
    """Return the most recent monthly sales observation."""

    data = load_sales()
    latest = data.iloc[-1]

    return {
        "month": latest["date"].strftime("%Y-%m"),
        "vehicles_sold": int(latest["vehicles_sold"]),
    }


if __name__ == "__main__":
    mcp.run()
