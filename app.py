import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE

from src.data.loader import load_stock_csv

torch.set_float32_matmul_precision("medium")


st.set_page_config(
    page_title="FPT Stock Forecast Demo",
    layout="centered",
)

st.title("FPT Stock Price Forecasting")
st.markdown(
    """
This demo predicts the next **100 business days** of FPT stock prices
using a **Patch-based Transformer (PatchTST)** model.

The model is trained on historical data and performs long-horizon
time series forecasting.
"""
)


@st.cache_data
def load_data():
    df = load_stock_csv(
        "data/raw/FPT_train.csv",
        date_col="time",
        price_col="close",
        volume_col="volume",
    )
    return df


@st.cache_resource
def train_model(df, input_length=120, horizon=100):
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
    return nf, nf_df


df = load_data()

st.subheader("Historical Data")
st.line_chart(
    df.set_index("time")["close"]
)

if st.button("Predict next 100 days"):
    with st.spinner("Training model and generating forecast..."):
        nf, nf_df = train_model(df)
        forecast = nf.predict().reset_index()

    st.subheader("Forecast (Next 100 Business Days)")
    st.dataframe(forecast.head())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"].tail(300), df["close"].tail(300), label="Historical")
    ax.plot(
        forecast["ds"],
        forecast["PatchTST"],
        linestyle="--",
        label="Forecast",
    )
    ax.set_title("FPT Stock Price Forecast (PatchTST)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
