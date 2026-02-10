"""
Bias correction utilities for long-horizon forecasts.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearBiasCorrector:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Fit correction model on flattened predictions.
        """
        self.model.fit(
            y_pred.reshape(-1, 1),
            y_true.reshape(-1, 1),
        )

    def apply(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply bias correction.
        """
        y_shape = y_pred.shape
        corrected = self.model.predict(
            y_pred.reshape(-1, 1)
        ).reshape(y_shape)
        return corrected
