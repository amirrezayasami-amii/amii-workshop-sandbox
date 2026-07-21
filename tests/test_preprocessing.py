"""Unit tests for the pure (non-I/O) preprocessing functions.

Demonstrates fixtures (see ``conftest.py``) and parametrization.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    drop_missing,
    encode_labels,
    normalize,
    standardize,
    train_test_split,
)


# --- drop_missing ---------------------------------------------------------

def test_drop_missing_removes_nan_rows(df_with_nans):
    result = drop_missing(df_with_nans)
    assert len(result) == 1
    assert result.iloc[0]["a"] == 1.0


def test_drop_missing_resets_index(df_with_nans):
    result = drop_missing(df_with_nans)
    assert result.index.tolist() == [0]


@pytest.mark.parametrize(
    "subset, expected_len",
    [
        (None, 1),       # any NaN drops the row
        (["a"], 2),      # only column "a" considered
        (["b"], 2),      # only column "b" considered
        (["a", "b"], 1), # both considered
    ],
)
def test_drop_missing_subset(df_with_nans, subset, expected_len):
    result = drop_missing(df_with_nans, columns=subset)
    assert len(result) == expected_len


# --- normalize ------------------------------------------------------------

@pytest.mark.parametrize("column", ["a", "b"])
def test_normalize_scales_to_unit_range(sample_df, column):
    result = normalize(sample_df, [column])
    assert result[column].min() == 0.0
    assert result[column].max() == 1.0


def test_normalize_exact_values(sample_df):
    result = normalize(sample_df, ["a"])
    assert result["a"].tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_normalize_constant_column_is_zero():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    result = normalize(df, ["a"])
    assert result["a"].tolist() == [0.0, 0.0, 0.0]


def test_normalize_does_not_mutate_input(sample_df):
    original = sample_df["a"].tolist()
    normalize(sample_df, ["a"])
    assert sample_df["a"].tolist() == original


# --- standardize ----------------------------------------------------------

@pytest.mark.parametrize("column", ["a", "b"])
def test_standardize_zero_mean_unit_variance(sample_df, column):
    result = standardize(sample_df, [column])
    assert result[column].mean() == pytest.approx(0.0, abs=1e-9)
    assert result[column].std(ddof=0) == pytest.approx(1.0)


def test_standardize_constant_column_is_zero():
    df = pd.DataFrame({"a": [7.0, 7.0]})
    result = standardize(df, ["a"])
    assert result["a"].tolist() == [0.0, 0.0]


# --- encode_labels --------------------------------------------------------

def test_encode_labels_returns_sorted_classes(sample_df):
    encoded, classes = encode_labels(sample_df["label"])
    assert classes.tolist() == ["x", "y", "z"]
    assert encoded.tolist() == [0, 1, 0, 2]


@pytest.mark.parametrize(
    "values, expected_classes",
    [
        (["a", "b", "a"], ["a", "b"]),
        (["z", "y", "x"], ["x", "y", "z"]),
        (["a", None, "b", "a"], ["a", "b"]),
        ([1, 3, 2, 1], [1, 2, 3]),
    ],
)
def test_encode_labels_classes(values, expected_classes):
    _, classes = encode_labels(pd.Series(values))
    assert classes.tolist() == expected_classes


# --- train_test_split -----------------------------------------------------

@pytest.mark.parametrize(
    "test_size, expected_test, expected_train",
    [
        (0.25, 1, 3),
        (0.5, 2, 2),
        (0.75, 3, 1),
    ],
)
def test_train_test_split_partition_sizes(
    sample_df, test_size, expected_test, expected_train
):
    train, test = train_test_split(sample_df, test_size=test_size, seed=42)
    assert len(test) == expected_test
    assert len(train) == expected_train


def test_train_test_split_is_a_partition(sample_df):
    train, test = train_test_split(sample_df, test_size=0.5, seed=1)
    combined = pd.concat([train, test])["a"].tolist()
    assert sorted(combined) == sample_df["a"].tolist()


def test_train_test_split_is_reproducible(sample_df):
    t1, _ = train_test_split(sample_df, test_size=0.5, seed=7)
    t2, _ = train_test_split(sample_df, test_size=0.5, seed=7)
    assert t1["a"].tolist() == t2["a"].tolist()


@pytest.mark.parametrize("bad_size", [0, 1, -0.1, 1.5])
def test_train_test_split_rejects_invalid_test_size(sample_df, bad_size):
    with pytest.raises(ValueError):
        train_test_split(sample_df, test_size=bad_size)
