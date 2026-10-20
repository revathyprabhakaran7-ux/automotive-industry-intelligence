from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "20100085.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Statistics Canada data
df = pd.read_csv(RAW_FILE, low_memory=False)

# Select Canadian monthly vehicle sales
sales = df[
    (df["GEO"] == "Canada")
    & (df["Vehicle type"] == "Total, new motor vehicles")
    & (df["Fuel type"] == "All fuel types")
    & (df["Origin of manufacture"] == "Total, country of manufacture")
    & (df["Sales"] == "Units")
    & (df["Seasonal adjustment"] == "Unadjusted")
].copy()

# Convert two-digit years to four-digit years
# Example: Jan-46 becomes January 1946
date_parts = sales["REF_DATE"].astype(str).str.extract(
    r"^([A-Za-z]{3})-(\d{2})$"
)

months = pd.to_datetime(
    date_parts[0], format="%b", errors="coerce"
).dt.month

two_digit_years = pd.to_numeric(
    date_parts[1], errors="coerce"
)

full_years = two_digit_years.where(
    two_digit_years >= 46,
    two_digit_years + 2000
)

full_years = full_years.where(
    full_years < 100,
    full_years
)

full_years = full_years.where(
    two_digit_years < 46,
    two_digit_years + 1900
)

sales["date"] = pd.to_datetime(
    {
        "year": full_years,
        "month": months,
        "day": 1
    },
    errors="coerce"
)

# Convert sales values to numbers
sales["vehicles_sold"] = pd.to_numeric(
    sales["VALUE"], errors="coerce"
)

# Remove missing values and sort
sales = sales.dropna(subset=["date", "vehicles_sold"])
sales = sales.sort_values("date")

# Keep final reporting columns
sales = sales[["date", "vehicles_sold"]]

# Validate the dataset
assert not sales.empty, "No matching sales records found"
assert not sales["date"].duplicated().any(), "Duplicate months found"

# Save cleaned CSV
output_file = OUTPUT_DIR / "monthly_vehicle_sales.csv"
sales.to_csv(output_file, index=False)

print(f"Prepared {len(sales)} monthly observations")
print(f"Saved cleaned data to {output_file}")
print(sales.head())
print(sales.tail())
