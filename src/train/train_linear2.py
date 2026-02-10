"""
Train and evaluate linear baseline models with optional bias correction
for long-horizon time series forecasting.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.loader import load_stock_csv
from src.data.features import make_sliding_windows
from src.models.linear_baseline import LinearForecast
from src.postproc.bias_correction import LinearBiasCorrector


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str):
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = mean_squared_error(
        y_true.flatten(), y_pred.flatten()
    )
    print(f"{label}")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    return mae, rmse


def run_linear_with_bias_correction(
    data_path: str,
    input_length: int = 120,
    horizon: int = 100,
    train_ratio: float = 0.6,
    calib_ratio: float = 0.2,
):
    # 1. Load data
    df = load_stock_csv(
        data_path,
        date_col="time",
        price_col="close",
        volume_col="volume",
    )

    series = df["close"]
    X, y = make_sliding_windows(series, input_length, horizon)

    # 2. Split data (time-based)
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_calib = int(n_total * calib_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_calib = X[n_train : n_train + n_calib]
    y_calib = y[n_train : n_train + n_calib]

    X_test = X[n_train + n_calib :]
    y_test = y[n_train + n_calib :]

    print(f"Train samples : {len(X_train)}")
    print(f"Calib samples : {len(X_calib)}")
    print(f"Test samples  : {len(X_test)}")

    # 3. Train linear model
    model = LinearForecast(model_type="linear")
    model.fit(X_train, y_train)

    # 4. Raw predictions
    y_pred_calib = model.predict(X_calib)
    y_pred_test = model.predict(X_test)

    print("\n=== Raw Linear Baseline ===")
    evaluate(y_test, y_pred_test, label="Raw Linear")

    # 5. Bias correction (fit on calibration set)
    corrector = LinearBiasCorrector()
    corrector.fit(y_pred_calib, y_calib)

    y_pred_test_corrected = corrector.apply(y_pred_test)

    print("\n=== Bias-Corrected Linear ===")
    evaluate(y_test, y_pred_test_corrected, label="Bias-Corrected Linear")


if __name__ == "__main__":
    run_linear_with_bias_correction(
        data_path="data/raw/FPT_train.csv",
        input_length=120,
        horizon=100,
    )
