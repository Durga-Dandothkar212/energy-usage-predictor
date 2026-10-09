# Results — Household Energy Consumption Forecasting

- Data: 34,211 hourly records (2006-12-16 to 2010-11-26) from the UCI household power dataset (~2M minute-level smart-meter readings).
- Split: train on all but the last 30 days; test on the final 721 hours.
- Baseline MAE: 0.615 kW · Random Forest MAE: 0.3795 kW → 38.3% lower error than the naive baseline.

| Model | MAE (kW) | RMSE (kW) |
|---|---|---|
| Baseline (same hour yesterday) | 0.615 | 0.9168 |
| Linear Regression | 0.4242 | 0.6074 |
| Random Forest | 0.3795 | 0.5599 |

Figures: `figures/daily_pattern.png`, `figures/forecast_vs_actual.png`, `figures/model_comparison.png`
