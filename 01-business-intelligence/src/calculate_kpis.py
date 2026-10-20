
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "outputs" / "monthly_vehicle_sales.csv"
OUTPUT_FILE = PROJECT_DIR / "outputs" / "vehicle_sales_kpis.csv"

# Read cleaned monthly vehicle sales
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Ensure records are in chronological order
df = df.sort_values("date").reset_index(drop=True)

# Month-over-month sales growth (%)
df["mom_growth_pct"] = (
    df["vehicles_sold"].pct_change() * 100
)

# Year-over-year sales growth (%)
df["yoy_growth_pct"] = (
    df["vehicles_sold"].pct_change(periods=12) * 100
)

# Average monthly sales over the last 12 months
df["rolling_12m_avg"] = (
    df["vehicles_sold"].rolling(window=12).mean()
)

# Total vehicles sold over the last 12 months
df["rolling_12m_total"] = (
    df["vehicles_sold"].rolling(window=12).sum()
)

# Save results
df.to_csv(OUTPUT_FILE, index=False)

print("KPI calculation complete.")
print(df.tail(10))
print(f"Saved KPI dataset to: {OUTPUT_FILE}")
