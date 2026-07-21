"""Shared pytest fixtures for the preprocessing test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """A small, clean DataFrame used across many tests."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "label": ["x", "y", "x", "z"],
        }
    )


@pytest.fixture
def df_with_nans():
    """A DataFrame containing missing values."""
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, np.nan],
        }
    )


@pytest.fixture
def csv_text():
    """Raw CSV content matching ``sample_df`` (used by mock-based tests)."""
    return "a,b,label\n1.0,10.0,x\n2.0,20.0,y\n3.0,30.0,x\n4.0,40.0,z\n"


@pytest.fixture
def csv_file(tmp_path, csv_text):
    """A real CSV file on disk in a temp dir; yields its path."""
    path = tmp_path / "data.csv"
    path.write_text(csv_text)
    return str(path)
