
name: Download and Prepare Automotive Data

on:
  workflow_dispatch:

jobs:
  download:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Python dependencies
        run: python -m pip install pandas

      - name: Download Statistics Canada dataset
        run: python 01-business-intelligence/src/load_data.py

      - name: Prepare clean monthly sales data
        run: python 01-business-intelligence/src/prepare_data.py

      - name: Upload cleaned dataset
        uses: actions/upload-artifact@v4
        with:
          name: clean-automotive-sales-data
          path: 01-business-intelligence/outputs/monthly_vehicle_sales.csv
          if-no-files-found: error
          retention-days: 7
