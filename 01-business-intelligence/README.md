# Automotive Business Intelligence

## Overview

Business intelligence analysis of Canadian new motor vehicle sales using Statistics Canada data, Python, and Tableau.

The project automates data preparation, calculates sales performance indicators, and visualizes historical automotive market trends.

## Data Source

[Statistics Canada — Table 20-10-0085-01: New Motor Vehicle Sales](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2010008501)

**Period:** January 1946 – August 2026

**Scope:** Canada-wide, total new motor vehicles, all fuel types, all manufacturing origins, unadjusted sales in units.

## Tech Stack

**Python · pandas · Tableau · GitHub Actions · GitHub**

## Methodology

1. **Data acquisition:** Download and extract the official Statistics Canada dataset using `load_data.py`.
2. **Data preparation:** Filter national vehicle sales, clean dates and values, and validate records using `prepare_data.py`.
3. **KPI calculation:** Calculate month-over-month growth, year-over-year growth, and rolling 12-month indicators using `calculate_kpis.py`.
4. **Visualization:** Use Tableau to analyze monthly vehicle sales trends and annual sales growth.

## Key Performance Indicators

| KPI | Description |
|---|---|
| Monthly Vehicle Sales | Number of new vehicles sold each month |
| Month-over-Month Growth (%) | Percentage change from the preceding month |
| Year-over-Year Growth (%) | Percentage change from the same month in the previous year |
| Rolling 12-Month Average | Average monthly sales over the preceding 12 months |
| Rolling 12-Month Total | Total vehicles sold over the preceding 12 months |

## Tableau Dashboard

Interactive dashboard featuring the latest monthly vehicle sales, historical sales trends, and year-over-year sales growth.

**[View Interactive Dashboard on Tableau Public](https://public.tableau.com/app/profile/revathy.prabhakaran/viz/CanadianAutomotiveSalesIntelligence/CanadianAutomotiveMarketPerformance)**

## Outputs

Generated under `01-business-intelligence/outputs/`:

- `monthly_vehicle_sales.csv` — Cleaned historical vehicle sales dataset.
- `vehicle_sales_kpis.csv` — Monthly vehicle sales with growth rates and rolling indicators.

The cleaned dataset also serves as the input for **Project 02 — Automotive Sales Forecasting**.

## Automation

Python scripts handle data acquisition, preparation, and KPI calculations. GitHub Actions supports the automated data workflow.

## Limitations

The analysis covers aggregate Canadian vehicle sales rather than regional or vehicle-category comparisons. Data is not seasonally adjusted, and historical trends do not establish causal relationships.

---

*Portfolio project using publicly available Statistics Canada data.*
