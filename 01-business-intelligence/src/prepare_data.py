from pathlib import Path
import pandas as pd

# Define project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "20100085.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the official Statistics Canada dataset
df = pd.read_csv(RAW_FILE, low_memory=False)

# Select Canadian monthly vehicle sales in units
sales = df[
    (df["GEO"] == "Canada")
    & (df["Vehicle type"] == "Total, new motor vehicles")
    & (df["Fuel type"] == "All fuel types")
    & (df["Origin of manufacture"] == "Total, country of manufacture")
    & (df["Sales"] == "Units")
    & (df["Seasonal adjustment"] == "Unadjusted")
].copy()

# Convert dates safely, including historical years
# Example: Jan-46 becomes 1946-01-01
date_parts = sales["REF_DATE"].astype(str).str.extract(
    r"^([A-Za-z]{3})-(\d{2})$"
)

months = pd.to_datetime(
    date_parts[0], format="%b", errors="coerce"
).dt.month

two_digit_years = pd.to_numeric(
    date_parts[1], errors="coerce"
)

# Years 46–99 represent 1946–1999
# Years 00–45 represent 2000–2045
full_years = two_digit_years.where(
    two_digit_years < 46,
    two_digit_years + 1900
)

full_years = full_years.where(
    full_years >= 1900,
    full_years + 2000
)

sales["date"] = pd.to_datetime(
    {
        "year": full_years,
        "month": months,
        "day": 1
    },
    errors="coerce"
)

# Convert vehicle sales to numeric values
sales["vehicles_sold"] = pd.to_numeric(
    sales["VALUE"], errors="coerce"
)

# Remove missing observations
sales = sales.dropna(subset=["date", "vehicles_sold"])

# Sort chronologically
sales = sales.sort_values("date")

# Keep only reporting columns
sales = sales[["date", "vehicles_sold"]]

# Validate the cleaned dataset
assert not sales.empty, "No matching vehicle sales data found"
assert not sales["date"].duplicated().any(), "Duplicate months found"

# Export cleaned CSV
output_file = OUTPUT_DIR / "monthly_vehicle_sales.csv"
sales.to_csv(output_file, index=False)

print(f"Prepared {len(sales)} monthly observations")
print(f"Saved cleaned data to {output_file}")
print(sales.head())
print(sales.tail())
