"""Data preprocessing utilities."""

import os

import numpy as np
import pandas as pd


def drop_missing(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Drop rows containing missing values.

    Args:
        df: Input DataFrame.
        columns: Optional subset of columns to consider when looking for NaNs.
            If None, all columns are considered.

    Returns:
        A new DataFrame with rows containing NaNs removed and the index reset.
    """
    return df.dropna(subset=columns).reset_index(drop=True)


def normalize(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Min-max normalize the given numeric columns into the [0, 1] range.

    Each column is linearly rescaled so its minimum maps to 0 and its maximum
    to 1. Constant columns (max == min) are mapped to all zeros to avoid
    division by zero.

    Args:
        df: Input DataFrame.
        columns: Iterable of column names to normalize.

    Returns:
        A new DataFrame with the selected columns scaled to [0, 1].
    """
    out = df.copy()
    for col in columns:
        col_min = out[col].min()
        col_max = out[col].max()
        span = col_max - col_min
        if span == 0:
            out[col] = 0.0
        else:
            out[col] = (out[col] - col_min) / span
    return out


def standardize(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Standardize columns to zero mean and unit variance (z-score).

    Constant columns (zero std) are mapped to all zeros.

    Args:
        df: Input DataFrame.
        columns: Iterable of column names to standardize.

    Returns:
        A new DataFrame with the selected columns standardized.
    """
    out = df.copy()
    for col in columns:
        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if std == 0:
            out[col] = 0.0
        else:
            out[col] = (out[col] - mean) / std
    return out


def encode_labels(series: pd.Series):
    """Encode a categorical series into integer labels.

    Args:
        series: Input categorical Series.

    Returns:
        A tuple ``(encoded, classes)`` where ``encoded`` is an int numpy array
        of label indices and ``classes`` is the sorted array of unique values.
    """
    classes = np.sort(series.dropna().unique())
    lookup = {value: idx for idx, value in enumerate(classes)}
    encoded = series.map(lookup).to_numpy()
    return encoded, classes


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk.

    Args:
        path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such dataset: {path}")
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a CSV file without the index.

    Args:
        df: DataFrame to write.
        path: Destination path.
    """
    df.to_csv(path, index=False)


def preprocess_pipeline(path: str, normalize_columns) -> pd.DataFrame:
    """Load a dataset, drop missing rows, and normalize the given columns.

    Args:
        path: Path to the CSV file to load.
        normalize_columns: Columns to min-max normalize.

    Returns:
        The fully preprocessed DataFrame.
    """
    df = load_dataset(path)
    df = drop_missing(df)
    df = normalize(df, normalize_columns)
    return df


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """Split a DataFrame into train and test partitions.

    Args:
        df: Input DataFrame.
        test_size: Fraction of rows (0 < test_size < 1) to allocate to the test set.
        seed: Random seed for reproducible shuffling.

    Returns:
        A tuple ``(train_df, test_df)`` with reset indices.

    Raises:
        ValueError: If ``test_size`` is not strictly between 0 and 1.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    n_test = int(round(len(df) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df
