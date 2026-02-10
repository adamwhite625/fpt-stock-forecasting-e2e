from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.loader import load_stock_csv
from src.data.features import make_sliding_windows
from src.models.linear_baseline import LinearForecast


def run_linear_baseline(
    data_path: str,
    input_length: int = 30,
    horizon: int = 5,
    test_ratio: float = 0.2,
):
    df = load_stock_csv(
        data_path,
        date_col="time",
        price_col="close",
        volume_col="volume",
    )

    X, y = make_sliding_windows(df["close"], input_length, horizon)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearForecast(model_type="linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    rmse = mean_squared_error(
        y_test.flatten(), y_pred.flatten()
    )

    print("Linear baseline results")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mae, rmse


if __name__ == "__main__":
    run_linear_baseline(
        data_path="data/raw/FPT_train.csv",
        input_length=120,
        horizon=100,
    )
