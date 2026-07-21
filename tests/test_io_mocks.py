"""Tests for the I/O functions using mocks.

These demonstrate mocking with ``unittest.mock`` / ``pytest`` so that the
filesystem and pandas' real read/write paths are never touched:

* ``patch`` / ``MagicMock`` to replace ``pd.read_csv`` and ``os.path.exists``
* ``monkeypatch`` (pytest's built-in fixture) as an alternative style
* ``mock_open`` to fake file contents
* asserting on call arguments (``assert_called_once_with``)
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import preprocessing
from preprocessing import load_dataset, preprocess_pipeline, save_dataset


# --- load_dataset: patch decorator style ----------------------------------

@patch("preprocessing.pd.read_csv")
@patch("preprocessing.os.path.exists", return_value=True)
def test_load_dataset_reads_existing_file(mock_exists, mock_read_csv, sample_df):
    mock_read_csv.return_value = sample_df

    result = load_dataset("whatever.csv")

    mock_exists.assert_called_once_with("whatever.csv")
    mock_read_csv.assert_called_once_with("whatever.csv")
    pd.testing.assert_frame_equal(result, sample_df)


@patch("preprocessing.os.path.exists", return_value=False)
def test_load_dataset_missing_file_raises(mock_exists):
    with pytest.raises(FileNotFoundError, match="No such dataset"):
        load_dataset("missing.csv")
    mock_exists.assert_called_once_with("missing.csv")


# --- load_dataset: context-manager + mock_open style ----------------------

def test_load_dataset_with_faked_read_csv(csv_text):
    """Feed fake CSV text in, bypassing the real file read.

    The real ``pd.read_csv`` is captured first so the side_effect can parse
    the in-memory text without recursing into the patched mock.
    """
    real_read_csv = pd.read_csv
    with patch("preprocessing.os.path.exists", return_value=True), patch(
        "preprocessing.pd.read_csv",
        side_effect=lambda p: real_read_csv(io.StringIO(csv_text)),
    ) as mock_read:
        result = load_dataset("data.csv")

    assert mock_read.called
    assert list(result.columns) == ["a", "b", "label"]
    assert len(result) == 4


# --- load_dataset: monkeypatch style --------------------------------------

def test_load_dataset_with_monkeypatch(monkeypatch, sample_df):
    fake_read = MagicMock(return_value=sample_df)
    monkeypatch.setattr(preprocessing.os.path, "exists", lambda p: True)
    monkeypatch.setattr(preprocessing.pd, "read_csv", fake_read)

    result = load_dataset("data.csv")

    fake_read.assert_called_once_with("data.csv")
    pd.testing.assert_frame_equal(result, sample_df)


# --- save_dataset ---------------------------------------------------------

def test_save_dataset_calls_to_csv_without_index():
    mock_df = MagicMock(spec=pd.DataFrame)
    save_dataset(mock_df, "out.csv")
    mock_df.to_csv.assert_called_once_with("out.csv", index=False)


# --- preprocess_pipeline: mock load, exercise real transforms -------------

@patch("preprocessing.load_dataset")
def test_preprocess_pipeline_uses_loaded_data(mock_load):
    mock_load.return_value = pd.DataFrame(
        {"a": [0.0, 5.0, 10.0], "b": [1.0, 2.0, 3.0]}
    )

    result = preprocess_pipeline("any.csv", normalize_columns=["a"])

    mock_load.assert_called_once_with("any.csv")
    # "a" was min-max normalized; "b" left untouched.
    assert result["a"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert result["b"].tolist() == [1.0, 2.0, 3.0]


@patch("preprocessing.load_dataset")
def test_preprocess_pipeline_drops_missing_before_normalizing(mock_load):
    mock_load.return_value = pd.DataFrame(
        {"a": [0.0, None, 10.0], "b": [1.0, 2.0, 3.0]}
    )

    result = preprocess_pipeline("any.csv", normalize_columns=["a"])

    # The NaN row is dropped, leaving two rows normalized to [0, 1].
    assert len(result) == 2
    assert result["a"].tolist() == pytest.approx([0.0, 1.0])


# --- integration sanity check using a real temp file (csv_file fixture) ---

def test_pipeline_end_to_end_with_real_file(csv_file):
    result = preprocess_pipeline(csv_file, normalize_columns=["a", "b"])
    assert result["a"].min() == 0.0
    assert result["a"].max() == 1.0
    assert len(result) == 4
