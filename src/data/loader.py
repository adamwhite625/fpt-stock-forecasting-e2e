# src/data/loader.py
"""
Data loading utilities for FPT stock time series.
This module is responsible for:
- Reading raw CSV data
- Parsing datetime
- Sorting by time
- Selecting relevant columns
"""

from pathlib import Path
import pandas as pd


def load_stock_csv(
    csv_path: str | Path,
    date_col: str = "date",
    price_col: str = "close",
    volume_col: str | None = None,
) -> pd.DataFrame:
    """
    Load and clean stock price data from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file
    date_col : str
        Name of datetime column
    price_col : str
        Name of closing price column
    volume_col : str or None
        Optional volume column

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe sorted by datetime
    """
    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    cols = [date_col, price_col]
    if volume_col and volume_col in df.columns:
        cols.append(volume_col)

    df = df[cols]
    return df
