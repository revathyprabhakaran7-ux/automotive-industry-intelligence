# Canadian Automotive Sales Forecasting

Forecasting monthly Canadian motor vehicle sales using Statistics Canada data, with a comparison of Ridge regression and seasonal naïve forecasting.

## Data Source

[Statistics Canada — Table 20-10-0085-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2010008501)

Monthly vehicle sales: January 1946 – August 2026.

Cleaned data is sourced from Project 01 — Business Intelligence.

## Tech Stack

**Python · pandas · NumPy · scikit-learn · MLflow · GitHub Actions**

## Methodology

1. Validate historical monthly sales data.
2. Engineer lag features, rolling averages, and seasonal indicators.
3. Train Ridge regression (`alpha=10.0`) and compare it against a seasonal naïve baseline.
4. Evaluate both using two rolling-origin, 12-month recursive backtests.
5. Select the model with the lowest backtest MAE.
6. Generate the next 12 months of sales forecasts and track experiments using MLflow.

## Results

**Backtest period:** September 2024 – August 2026

| Metric | Seasonal Naïve | Ridge Regression |
|---|---:|---:|
| MAE | **8,532.83** | 10,197.63 |
| RMSE | **10,073.38** | 12,652.47 |
| MAPE | **5.20%** | 6.35% |

**Selected model: Seasonal Naïve**, achieving approximately 16.3% lower MAE than Ridge.

The selected approach generated monthly forecasts for **September 2026 – August 2027**.

## Outputs

Generated under `02-automotive-forecasting/outputs/`:

- `holdout_predictions.csv` — Recursive backtest predictions.
- `next_12_months_forecast.csv` — Future monthly sales forecasts.
- `metrics.json` — Model evaluation and selection results.
- MLflow tracking database and experiment artifacts.

## Automation

GitHub Actions runs the forecasting pipeline and produces downloadable results. Synthetic-data tests are included to validate forecasting functionality.

## Limitations

Evaluation is based on two historical backtest windows. The models use historical sales and calendar features without external economic predictors, and forecasts do not include prediction intervals.

Historical performance does not guarantee future accuracy.

---

*Portfolio project for analytical purposes. Forecasts are not official Statistics Canada projections.*
