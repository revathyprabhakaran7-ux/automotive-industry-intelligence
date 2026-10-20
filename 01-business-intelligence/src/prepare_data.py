
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "20100085.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the official Statistics Canada dataset
df = pd.read_csv(RAW_FILE, low_memory=False)

# Select a consistent national monthly sales series
sales = df[
    (df["GEO"] == "Canada")
    & (df["Vehicle type"] == "Total, new motor vehicles")
    & (df["Fuel type"] == "All fuel types")
    & (df["Origin of manufacture"] == "Total, country of manufacture")
    & (df["Sales"] == "Units")
    & (df["Seasonal adjustment"] == "Unadjusted")
].copy()

# Convert dates and sales values
sales["date"] = pd.to_datetime(
    sales["REF_DATE"], format="%b-%y"
)
sales["vehicles_sold"] = pd.to_numeric(
    sales["VALUE"], errors="coerce"
)

# Remove missing observations and sort chronologically
sales = sales.dropna(subset=["date", "vehicles_sold"])
sales = sales.sort_values("date")

# Keep only the fields needed for reporting
sales = sales[["date", "vehicles_sold"]]

# Check that each month appears once
assert not sales["date"].duplicated().any(), "Duplicate months found"

output_file = OUTPUT_DIR / "monthly_vehicle_sales.csv"
sales.to_csv(output_file, index=False)

print(f"Prepared {len(sales)} monthly observations")
print(f"Saved cleaned data to {output_file}")
print(sales.tail())
