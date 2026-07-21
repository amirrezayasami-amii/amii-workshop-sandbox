# Module 01 — Data Preprocessing

[← Project Setup](00-project-setup.md) · [Index](README.md) · [Next: Testing with pytest →](02-testing-pytest.md)

## Learning objectives

- Write small, **pure**, reusable data-transformation functions with pandas and
  numpy.
- Understand the design principles that make code *testable* (no hidden state,
  no mutation of inputs, predictable outputs).
- Handle edge cases deliberately (missing values, constant columns,
  divide-by-zero).

## Prerequisites

- Module 00 complete (`numpy`, `pandas` installed).

---

## 1. The design philosophy

Every function we write follows three rules that pay off enormously when we
test in Module 02:

1. **Pure where possible** — output depends only on input; no globals, no I/O.
2. **Never mutate the caller's data** — we `.copy()` before changing a
   DataFrame. Surprising a caller by editing their object in place is a classic
   bug source.
3. **Edge cases are explicit** — a constant column has no range to normalize
   into, so we define what happens (map to zeros) rather than letting a
   division by zero produce `NaN`/`inf` silently.

---

## 2. `preprocessing.py`

Create `preprocessing.py`. We'll build it function by function.

### Imports

```python
"""Data preprocessing utilities."""

import os

import numpy as np
import pandas as pd
```

> **Note:** we import only what we use. An earlier draft imported
> `tensorflow`, which was never used — dead imports slow startup and confuse
> readers. We removed it. (`os` returns in Module 05 for file checks.)

### `drop_missing` — remove rows with missing values

```python
def drop_missing(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Drop rows containing missing values.

    Args:
        df: Input DataFrame.
        columns: Optional subset of columns to consider when looking for NaNs.
            If None, all columns are considered.

    Returns:
        A new DataFrame with NaN rows removed and the index reset.
    """
    return df.dropna(subset=columns).reset_index(drop=True)
```

**Why `reset_index(drop=True)`?** After dropping rows, the index has gaps
(`0, 2, 3, ...`). Resetting gives a clean `0..n` index so downstream positional
logic isn't surprised.

### `normalize` — min-max scale to [0, 1]

```python
def normalize(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Min-max normalize the given numeric columns into [0, 1].

    Constant columns (max == min) are mapped to all zeros to avoid division
    by zero.
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
```

Note the two defensive choices: `out = df.copy()` (no mutation) and the
`span == 0` guard (no divide-by-zero).

### `standardize` — zero mean, unit variance (z-score)

```python
def standardize(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Standardize columns to zero mean and unit variance (z-score).

    Constant columns (zero std) are mapped to all zeros.
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
```

> **`ddof=0`** makes this the *population* standard deviation, which is what ML
> scalers (e.g. scikit-learn's `StandardScaler`) use. Being explicit avoids a
> subtle mismatch with pandas' default of `ddof=1`.

### `encode_labels` — categories to integers

```python
def encode_labels(series: pd.Series):
    """Encode a categorical series into integer labels.

    Returns:
        (encoded, classes) — an int array of indices and the sorted unique values.
    """
    classes = np.sort(series.dropna().unique())
    lookup = {value: idx for idx, value in enumerate(classes)}
    encoded = series.map(lookup).to_numpy()
    return encoded, classes
```

Sorting the classes makes the encoding **deterministic** — the same input always
maps to the same integers, which matters for reproducibility and testing.

### `train_test_split` — reproducible partition

```python
def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """Split a DataFrame into train and test partitions.

    Raises:
        ValueError: If test_size is not strictly between 0 and 1.
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
```

Three testable behaviors are baked in on purpose:

- **Validation:** invalid `test_size` raises `ValueError` (not a silent bad
  split).
- **Reproducibility:** a fixed `seed` → identical split every time.
- **Partition integrity:** train + test together equal the original rows, with
  no overlap.

---

## 3. Try it in a REPL

```bash
python
```

```python
>>> import pandas as pd, numpy as np
>>> from preprocessing import normalize, drop_missing
>>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
>>> normalize(df, ["a"])["a"].tolist()
[0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
>>> drop_missing(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))
     a
0  1.0
1  3.0
```

---

## Checkpoint ✅

- [ ] `preprocessing.py` exists with five functions.
- [ ] Each function returns a **new** object (originals unchanged).
- [ ] You can call `normalize` and `drop_missing` from a REPL.

## Exercises

1. Add a `clip_outliers(df, column, lower, upper)` function that caps values to
   a range. Keep it pure (copy first).
2. What should `normalize` do if given a column that doesn't exist? Decide on
   behavior — raise, or skip? (We'll test whichever you choose in Module 02.)

[← Project Setup](00-project-setup.md) · [Index](README.md) · [Next: Testing with pytest →](02-testing-pytest.md)
