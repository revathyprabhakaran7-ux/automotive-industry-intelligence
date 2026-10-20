
from pathlib import Path
import pandas as pd

# Define project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "20100085.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load official Statistics Canada data
df = pd.read_csv(RAW_FILE, low_memory=False)

print("Original rows:", len(df), flush=True)

# Filter for total monthly vehicle sales in Canada
sales = df[
    (df["GEO"] == "Canada")
    & (df["Vehicle type"] == "Total, new motor vehicles")
    & (df["Fuel type"] == "All fuel types")
    & (df["Origin of manufacture"] == "Total, country of manufacture")
    & (df["Sales"] == "Units")
    & (df["Seasonal adjustment"] == "Unadjusted")
].copy()

print("Rows after filtering:", len(sales), flush=True)

# Convert dates such as 1946-01 to proper dates
sales["date"] = pd.to_datetime(
    sales["REF_DATE"],
    format="%Y-%m",
    errors="coerce"
)

# Convert sales values to numbers
sales["vehicles_sold"] = pd.to_numeric(
    sales["VALUE"],
    errors="coerce"
)

# Check valid values
print("Valid dates:", sales["date"].notna().sum(), flush=True)
print("Valid sales values:", sales["vehicles_sold"].notna().sum(), flush=True)

# Remove missing values
sales = sales.dropna(subset=["date", "vehicles_sold"])

# Sort by date
sales = sales.sort_values("date")

# Keep only the columns needed for analysis
sales = sales[["date", "vehicles_sold"]]

# Validate the cleaned data
assert not sales.empty, "No matching sales records found"
assert not sales["date"].duplicated().any(), "Duplicate months found"

# Save the cleaned dataset
output_file = OUTPUT_DIR / "monthly_vehicle_sales.csv"
sales.to_csv(output_file, index=False)

print(f"Prepared {len(sales)} monthly observations")
print(f"Saved cleaned data to {output_file}")

# Preview the first and last five rows
print(sales.head())
print(sales.tail())
