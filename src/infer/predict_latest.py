"""
Inference demo: predict the next 100 business days using PatchTST.
"""

import pandas as pd
import matplotlib.pyplot as plt
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


def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame):
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["ds"],
        history["y"],
        label="Historical",
        linewidth=2,
    )

    plt.plot(
        forecast["ds"],
        forecast["PatchTST"],
        label="Forecast (100 days)",
        linestyle="--",
    )

    plt.title("FPT Stock Price Forecast (PatchTST)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/latest_forecast.png", dpi=150)
    plt.show()
    


def run_inference(
    data_path: str,
    input_length: int = 120,
    horizon: int = 100,
):
    df = prepare_nf_dataframe(data_path)

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

    # Train once on full history
    nf.fit(df=df)

    # Predict future horizon
    forecast = nf.predict().reset_index()

    print("\n=== Forecast (next 100 business days) ===")
    print(forecast.head())

    plot_forecast(
        history=df.tail(300),
        forecast=forecast,
    )


if __name__ == "__main__":
    run_inference(
        data_path="data/raw/FPT_train.csv",
        input_length=120,
        horizon=100,
    )
