
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "20100085.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Statistics Canada data
df = pd.read_csv(RAW_FILE, low_memory=False)

print("Original rows:", len(df), flush=True)

# Select Canadian monthly vehicle sales in units
sales = df[
    (df["GEO"] == "Canada")
    & (df["Vehicle type"] == "Total, new motor vehicles")
    & (df["Fuel type"] == "All fuel types")
    & (df["Origin of manufacture"] == "Total, country of manufacture")
    & (df["Sales"] == "Units")
    & (df["Seasonal adjustment"] == "Unadjusted")
].copy()

print("Rows after filtering:", len(sales), flush=True)

# Interpret two-digit years explicitly
# 46–99 = 1946–1999; 00–45 = 2000–2045
parts = sales["REF_DATE"].astype(str).str.extract(
    r"^([A-Za-z]{3})-(\d{2})$"
)

month_numbers = pd.to_datetime(
    parts[0], format="%b", errors="coerce"
).dt.month

year_numbers = pd.to_numeric(parts[1], errors="coerce")

full_years = year_numbers.apply(
    lambda year: (
        year + 1900 if year >= 46
        else year + 2000
    ) if pd.notna(year) else None
)

sales["date"] = pd.to_datetime(
    {
        "year": full_years,
        "month": month_numbers,
        "day": 1
    },
    errors="coerce"
)

sales["vehicles_sold"] = pd.to_numeric(
    sales["VALUE"], errors="coerce"
)

print("Valid dates:", sales["date"].notna().sum(), flush=True)
print("Valid sales values:", sales["vehicles_sold"].notna().sum(), flush=True)

sales = sales.dropna(subset=["date", "vehicles_sold"])
sales = sales.sort_values("date")
sales = sales[["date", "vehicles_sold"]]

assert not sales.empty, "No matching sales records found"
assert not sales["date"].duplicated().any(), "Duplicate months found"

output_file = OUTPUT_DIR / "monthly_vehicle_sales.csv"
sales.to_csv(output_file, index=False)

print(f"Prepared {len(sales)} monthly observations")
print(f"Saved cleaned data to {output_file}")
print(sales.head())
print(sales.tail())
