"""
Visualization utilities for comparing forecasting models.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data.loader import load_stock_csv
from src.data.features import make_sliding_windows
from src.models.linear_baseline import LinearForecast
from src.postproc.bias_correction import LinearBiasCorrector
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE


def plot_comparison(
    y_true: np.ndarray,
    y_linear: np.ndarray,
    y_patchtst: np.ndarray,
    horizon: int,
    max_plot_steps: int = 50,
    save_path: str | None = "artifacts/forecast_comparison.png",
):
    steps = min(horizon, max_plot_steps)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:steps], label="Ground Truth", linewidth=2)
    plt.plot(y_linear[:steps], label="Linear", linestyle="--")
    plt.plot(y_patchtst[:steps], label="PatchTST")

    plt.title("Forecast Comparison (First Steps of Horizon)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")

    plt.show()


def run_visualization(
    data_path: str,
    input_length: int = 120,
    horizon: int = 100,
):
    # Load raw data
    df = load_stock_csv(
        data_path,
        date_col="time",
        price_col="close",
        volume_col="volume",
    )

    series = df["close"]

    # Create sliding windows
    X, y = make_sliding_windows(series, input_length, horizon)

    # Use last window for visualization
    X_last = X[-1:]
    y_true = y[-1]

    # ---- Linear model ----
    linear = LinearForecast()
    linear.fit(X[:-1], y[:-1])
    y_linear = linear.predict(X_last)[0]

    # ---- PatchTST ----
    nf_df = pd.DataFrame({
        "unique_id": "FPT",
        "ds": df["time"],
        "y": df["close"],
    })

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

    nf.fit(df=nf_df)
    forecast = nf.predict().iloc[-horizon:]

    y_patchtst = forecast["PatchTST"].values

    # ---- Plot ----
    plot_comparison(
        y_true=y_true,
        y_linear=y_linear,
        y_patchtst=y_patchtst,
        horizon=horizon,
    )


if __name__ == "__main__":
    run_visualization(
        data_path="data/raw/FPT_train.csv",
        input_length=120,
        horizon=100,
    )
