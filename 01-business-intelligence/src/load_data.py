from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# Official Statistics Canada dataset: Table 20-10-0085-01
DATA_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/20100085-eng.zip"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

zip_path = DATA_DIR / "vehicle_sales.zip"

print("Downloading official vehicle sales data...")
urlretrieve(DATA_URL, zip_path)

print("Extracting CSV files...")
with ZipFile(zip_path, "r") as archive:
    archive.extractall(DATA_DIR)

print("Download and extraction complete.")
print(f"Files saved in: {DATA_DIR}")
