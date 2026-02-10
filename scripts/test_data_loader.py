from src.data.loader import load_stock_csv
from src.data.features import make_sliding_windows

DATA_PATH = "data/raw/FPT_train.csv"

df = load_stock_csv(
    DATA_PATH,
    date_col="time",
    price_col="close",
    volume_col="volume",
)

print("DataFrame head:")
print(df.head())

X, y = make_sliding_windows(
    df["close"],
    input_length=30,
    horizon=5,
)

print("X shape:", X.shape)
print("y shape:", y.shape)
