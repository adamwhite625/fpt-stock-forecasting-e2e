# FPT Stock Price Forecasting

This project is a personal end-to-end implementation of a **long-horizon stock price forecasting system**
using historical data from **FPT**. The project follows the spirit of **Module 6 demo**, but is extended
with deeper analysis, stronger baselines, and a transformer-based forecasting model.

The main objective is to study how different models behave under **long-term forecasting horizons**
and to build a practical forecasting demo.

---

## Problem Description

- Task: Multi-step time series forecasting
- Asset: FPT stock
- Target: Closing price
- Forecast horizon: **100 business days**
- Evaluation metrics: MAE, RMSE

Stock price forecasting is particularly challenging due to:

- High volatility
- Regime shifts
- Error accumulation in long-horizon predictions

---

## Project Structure

```text
.
├── app.py                  # Streamlit demo
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Baseline models
│   ├── train/              # Training & evaluation scripts
│   ├── postproc/           # Bias correction
│   ├── eval/               # Visualization & analysis
│   └── infer/              # Inference scripts
├── data/
│   └── raw/                # Raw stock data
├── docs/
│   └── RESULTS.md          # Experimental results
├── artifacts/              # Generated figures (ignored by git)
└── README.md
```

## Modeling Approach

### 1. Linear Baseline

- Standard linear forecasting model
- Works reasonably well for short horizons
- Fails significantly for long-horizon forecasting due to error accumulation

### 2. Linear + Bias Correction

- Post-processing step to reduce systematic bias
- Improves results slightly but does not solve long-term dependency issues

### 3. PatchTST (Patch-based Transformer)

- Transformer model designed for long time series forecasting
- Captures long-term dependencies effectively
- Significantly outperforms linear baselines on 100-day forecasting

---

## Experimental Results

Quantitative and qualitative results are summarized in:

```
docs/RESULTS.md
```

Key findings:

- Linear models degrade rapidly as forecast horizon increases
- Bias correction provides limited improvement
- PatchTST achieves substantial error reduction and stable long-horizon forecasts

---

## Inference Demo

Generate a **100-day forecast** using the PatchTST model:

```bash
python -m src.infer.predict_latest
```

This script trains the model on the full historical dataset and produces:

- A table of predicted prices
- A visualization of historical prices and future forecasts

---

## Streamlit Demo

Run an interactive forecasting demo locally:

```bash
streamlit run app.py
```

The Streamlit app allows users to:

- Visualize historical FPT stock prices
- Generate a 100-day forecast using PatchTST
- Inspect forecast values and trends interactively
