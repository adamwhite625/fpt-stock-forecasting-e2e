# src/data/features.py
"""
Feature engineering utilities for time series forecasting.
"""

import numpy as np
import pandas as pd


def make_sliding_windows(
    series: pd.Series,
    input_length: int,
    horizon: int,
):
    """
    Convert a 1D time series into sliding windows.

    X shape: (n_samples, input_length)
    y shape: (n_samples, horizon)
    """
    values = series.values
    X, y = [], []

    for i in range(len(values) - input_length - horizon + 1):
        X.append(values[i : i + input_length])
        y.append(values[i + input_length : i + input_length + horizon])

    return np.array(X), np.array(y)
