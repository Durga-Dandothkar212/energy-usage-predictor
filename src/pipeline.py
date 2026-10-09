"""Household Energy Consumption Forecasting with Machine Learning.

End-to-end pipeline:
  1. Load the UCI Individual Household Electric Power Consumption dataset
     (minute-level smart-meter records).
  2. Clean and resample to hourly average power (kW).
  3. Engineer calendar + lag/rolling features.
  4. Compare three forecasters on a chronological test split (last 30 days):
       - Baseline: "same hour yesterday" (lag-24)
       - Linear Regression
       - Random Forest
  5. Save metrics + figures to results/ and figures/.

Dataset: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
Place household_power_consumption.txt (unzipped) in data/ before running.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/tmp/household_power_consumption.txt")
DATA_OUT = ROOT / "data" / "hourly_power.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
for d in (DATA_OUT.parent, FIG, RES):
    d.mkdir(parents=True, exist_ok=True)

NUM_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(RAW, sep=";", low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # '?' -> NaN
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    hourly = df["Global_active_power"].resample("h").mean().to_frame("kw")
    # Fill short gaps only (<= 3 missing hours); drop anything still missing.
    hourly["kw"] = hourly["kw"].interpolate(limit=3, limit_direction="both")
    hourly = hourly.dropna()
    hourly.to_csv(DATA_OUT)
    return hourly


def make_features(hourly: pd.DataFrame) -> pd.DataFrame:
    f = hourly.copy()
    f["hour"] = f.index.hour
    f["dayofweek"] = f.index.dayofweek
    f["month"] = f.index.month
    f["lag_1h"] = f["kw"].shift(1)
    f["lag_24h"] = f["kw"].shift(24)
    f["lag_168h"] = f["kw"].shift(168)
    f["roll_24h_mean"] = f["kw"].shift(1).rolling(24).mean()
    return f.dropna()


def metrics(name: str, y_true, y_pred) -> dict:
    return {
        "model": name,
        "MAE_kW": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE_kW": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def main() -> None:
    hourly = load_hourly()
    print(f"Hourly series: {len(hourly):,} rows, {hourly.index.min()} -> {hourly.index.max()}")

    feats = make_features(hourly)
    feature_cols = ["hour", "dayofweek", "month", "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean"]

    test_start = feats.index.max() - pd.Timedelta(days=30)
    train = feats[feats.index < test_start]
    test = feats[feats.index >= test_start]
    print(f"Train: {len(train):,} | Test: {len(test):,} (last 30 days)")

    results = [metrics("Baseline (same hour yesterday)", test["kw"], test["lag_24h"])]

    lr = LinearRegression().fit(train[feature_cols], train["kw"])
    results.append(metrics("Linear Regression", test["kw"], lr.predict(test[feature_cols])))

    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(train[feature_cols], train["kw"])
    rf_pred = rf.predict(test[feature_cols])
    results.append(metrics("Random Forest", test["kw"], rf_pred))

    mdf = pd.DataFrame(results)
    mdf.to_csv(RES / "metrics.csv", index=False)
    print(mdf.to_string(index=False))

    # --- Figure 1: average daily consumption pattern ---
    pattern = hourly.groupby(hourly.index.hour)["kw"].mean()
    plt.figure(figsize=(9, 4))
    plt.plot(pattern.index, pattern.values, marker="o")
    plt.xlabel("Hour of day")
    plt.ylabel("Avg power (kW)")
    plt.title("Average household power consumption by hour of day")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "daily_pattern.png", dpi=150)
    plt.close()

    # --- Figure 2: forecast vs actual (first 7 days of test) ---
    week = test.iloc[: 24 * 7]
    plt.figure(figsize=(11, 4))
    plt.plot(week.index, week["kw"], label="Actual", linewidth=1.4)
    plt.plot(week.index, rf.predict(week[feature_cols]), label="Random Forest forecast", linewidth=1.2)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title("Forecast vs actual — first 7 test days")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "forecast_vs_actual.png", dpi=150)
    plt.close()

    # --- Figure 3: model comparison ---
    plt.figure(figsize=(8, 4))
    x = np.arange(len(mdf))
    width = 0.35
    plt.bar(x - width / 2, mdf["MAE_kW"], width, label="MAE (kW)")
    plt.bar(x + width / 2, mdf["RMSE_kW"], width, label="RMSE (kW)")
    plt.xticks(x, mdf["model"], rotation=12, ha="right", fontsize=8)
    plt.ylabel("Error (kW)")
    plt.title("Model comparison on last-30-days test set (lower is better)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "model_comparison.png", dpi=150)
    plt.close()

    baseline_mae = mdf.loc[0, "MAE_kW"]
    rf_mae = mdf.loc[2, "MAE_kW"]
    improvement = 100.0 * (baseline_mae - rf_mae) / baseline_mae
    summary = (
        "# Results — Household Energy Consumption Forecasting\n\n"
        f"- Data: {len(hourly):,} hourly records ({hourly.index.min().date()} to {hourly.index.max().date()}) "
        "from the UCI household power dataset (~2M minute-level smart-meter readings).\n"
        f"- Split: train on all but the last 30 days; test on the final {len(test):,} hours.\n"
        f"- Baseline MAE: {baseline_mae} kW · Random Forest MAE: {rf_mae} kW "
        f"→ {improvement:.1f}% lower error than the naive baseline.\n\n"
        "| Model | MAE (kW) | RMSE (kW) |\n|---|---|---|\n"
        + "\n".join(
            f"| {r.model} | {r.MAE_kW} | {r.RMSE_kW} |" for r in mdf.itertuples()
        )
        + "\n\nFigures: `figures/daily_pattern.png`, `figures/forecast_vs_actual.png`, `figures/model_comparison.png`\n"
    )
    (RES / "summary.md").write_text(summary)
    print(f"\nRandom Forest vs baseline MAE improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
