"""
Train and evaluate PatchTST model using NeuralForecast
for long-horizon stock price forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE

from src.data.loader import load_stock_csv


torch.set_float32_matmul_precision("medium")


def prepare_nf_dataframe(csv_path: str) -> pd.DataFrame:
    df = load_stock_csv(
        csv_path,
        date_col="time",
        price_col="close",
        volume_col="volume",
    )

    nf_df = pd.DataFrame({
        "unique_id": "FPT",
        "ds": df["time"],
        "y": df["close"],
    })
    return nf_df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str):
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = mean_squared_error(
        y_true.flatten(), y_pred.flatten()
    )
    print(label)
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    return mae, rmse


def run_patchtst(
    data_path: str,
    input_length: int = 120,
    horizon: int = 100,
    test_ratio: float = 0.2,
):
    df = prepare_nf_dataframe(data_path)

    # split time-based
    n_total = len(df)
    n_test = int(n_total * test_ratio)

    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-(n_test + input_length + horizon):]

    model = PatchTST(
        h=horizon,
        input_size=input_length,
        loss=MAE(),
        max_steps=500,
        batch_size=32,
        learning_rate=1e-3,
        scaler_type="standard",
    )

    nf = NeuralForecast(
        models=[model],
        freq="B",
    )

    # train
    nf.fit(df=train_df)

    # predict on test window
    forecasts = nf.predict(df=test_df)

    # align predictions with ground truth
    y_pred = forecasts["PatchTST"].values
    y_true = test_df["y"].values[-len(y_pred):]

    print("\n=== PatchTST Evaluation (100 days) ===")
    evaluate(y_true, y_pred, label="PatchTST")


if __name__ == "__main__":
    run_patchtst(
        data_path="data/raw/FPT_train.csv",
        input_length=120,
        horizon=100,
    )
