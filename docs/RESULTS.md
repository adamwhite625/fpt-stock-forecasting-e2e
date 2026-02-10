# Experimental Results

This document summarizes the experimental results obtained so far in the project.
The focus is on evaluating the behavior of linear models under different forecasting
horizons and understanding their limitations.

---

## Dataset

- Asset: FPT stock
- Time span: 2020-08-03 onward
- Features: closing price (primary), volume (auxiliary)

---

## Linear Baseline Results

### Short-Horizon Forecasting (5 days)

- Input length: 30
- MAE: 2.32
- RMSE: 9.91

**Observation**  
Linear models are able to capture short-term trends reasonably well when the
forecasting horizon is limited.

---

### Long-Horizon Forecasting (100 days)

- Input length: 120
- MAE: 37.71
- RMSE: 1812.96

**Observation**  
Performance degrades significantly as the forecasting horizon increases.
Error accumulation and distribution shift lead to severe prediction drift.

---

## Linear Baseline with Bias Correction

- MAE: 36.72
- RMSE: 1519.24

**Observation**  
Bias correction reduces systematic errors and improves overall performance.
However, the improvement is limited, and the model remains unsuitable for
long-horizon forecasting.

---

## Key Takeaways

- Short-horizon performance does not generalize to long-horizon forecasting.
- Linear models, even with post-processing, are fundamentally limited for
  capturing long-term dependencies in stock price movements.
- More expressive models are required for reliable long-horizon forecasting.

## PatchTST Results (100 days)

- Input length: 120
- Horizon: 100
- MAE: 4.70
- RMSE: 28.88

Observation:
PatchTST significantly outperforms linear baselines on long-horizon forecasting.
The model effectively captures long-term dependencies and prevents error
accumulation, resulting in stable and accurate forecasts.

## Forecast Visualization

Figure `forecast_comparison.png` shows a qualitative comparison between
linear baseline and PatchTST forecasts over the first 50 steps of the
100-day horizon.

Linear forecasts exhibit noticeable drift and fail to follow the underlying
price dynamics, while PatchTST produces more stable trajectories that better
align with ground truth trends.
