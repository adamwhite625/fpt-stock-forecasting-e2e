# src/models/linear_baseline.py
"""
Simple linear baselines for multi-step time series forecasting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet


class LinearForecast:
    def __init__(self, model_type: str = "linear", **kwargs):
        if model_type == "elasticnet":
            self.model = ElasticNet(**kwargs)
        else:
            self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
