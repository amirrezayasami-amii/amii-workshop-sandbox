# Module 02 — Testing with pytest

[← Preprocessing](01-preprocessing.md) · [Index](README.md) · [Next: Continuous Integration →](03-ci-github-actions.md)

## Learning objectives

- Write and run tests with **pytest**.
- Remove duplication with **fixtures** and `conftest.py`.
- Cover many inputs concisely with **parametrization**.
- Isolate code from the filesystem/network using **mocks** (`unittest.mock`,
  `monkeypatch`).
- Understand pytest's **import model** — the single most common "why can't it
  find my module" gotcha.

## Prerequisites

- Module 01 complete (`preprocessing.py` exists).
- `pytest` installed in your venv.

---

## 1. Your first test

pytest discovers any file named `test_*.py`, and inside it any function named
`test_*`. A test "passes" if it runs without an `AssertionError`.

Create `tests/test_preprocessing.py`:

```python
import numpy as np
import pandas as pd

from preprocessing import drop_missing


def test_drop_missing_removes_nan_rows():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result = drop_missing(df)
    assert len(result) == 2
    assert result["a"].tolist() == [1.0, 3.0]
```

Run it:

```bash
pytest -v
```

**Expected output:**

```
tests/test_preprocessing.py::test_drop_missing_removes_nan_rows PASSED    [100%]
============================== 1 passed in 0.05s ==============================
```

> `-v` (verbose) lists each test by name. Without it you get one dot per test.

---

## 2. Fixtures — reusable test setup

Many tests need the same sample data. Copy-pasting a DataFrame into every test
is repetitive and error-prone. A **fixture** is a function whose return value is
injected into any test that names it as an argument.

Fixtures shared across multiple test files go in `tests/conftest.py` — pytest
discovers this file automatically; you never import it.

```python
# tests/conftest.py
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
    return pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
```

Now a test simply *asks* for `sample_df` by name:

```python
def test_normalize_does_not_mutate_input(sample_df):
    original = sample_df["a"].tolist()
    normalize(sample_df, ["a"])
    assert sample_df["a"].tolist() == original   # input untouched
```

pytest sees the `sample_df` parameter, runs the fixture, and passes the result
in. Each test gets a **fresh** copy, so tests never leak state into each other.

### Built-in fixtures worth knowing

- `tmp_path` — a unique temporary directory (a `pathlib.Path`) per test, cleaned
  up automatically. We use it in Module 05/06 to write throwaway files.
- `monkeypatch` — safely patch attributes/env vars for the duration of one test
  (see mocking below).

---

## 3. Parametrization — one test, many inputs

When the *logic* is identical but the *data* varies, don't write ten near-copies.
`@pytest.mark.parametrize` runs the same test body once per case, and reports
each as a separate test.

```python
import pytest

@pytest.mark.parametrize(
    "subset, expected_len",
    [
        (None, 1),        # any NaN drops the row
        (["a"], 2),       # only column "a" considered
        (["b"], 2),       # only column "b" considered
        (["a", "b"], 1),  # both considered
    ],
)
def test_drop_missing_subset(df_with_nans, subset, expected_len):
    result = drop_missing(df_with_nans, columns=subset)
    assert len(result) == expected_len
```

Running this shows **four** tests:

```
test_drop_missing_subset[None-1] PASSED
test_drop_missing_subset[subset1-2] PASSED
test_drop_missing_subset[subset2-2] PASSED
test_drop_missing_subset[subset3-1] PASSED
```

Notice how each case gets an id in `[...]` — when one fails you know exactly
which input broke.

### Testing that an error is raised

Parametrization pairs beautifully with `pytest.raises` for validation testing:

```python
@pytest.mark.parametrize("bad_size", [0, 1, -0.1, 1.5])
def test_train_test_split_rejects_invalid_test_size(sample_df, bad_size):
    with pytest.raises(ValueError):
        train_test_split(sample_df, test_size=bad_size)
```

`with pytest.raises(ValueError):` asserts that the block **must** raise that
exception — the test fails if no error (or the wrong error) occurs.

### Comparing floats

Never assert exact equality on floats. Use `pytest.approx`:

```python
def test_normalize_exact_values(sample_df):
    result = normalize(sample_df, ["a"])
    assert result["a"].tolist() == pytest.approx([0.0, 1/3, 2/3, 1.0])
```

---

## 4. Mocking — isolating code from the outside world

Some code touches things that are slow, non-deterministic, or unavailable in a
test: the filesystem, a network API, a database. **Mocking** replaces those
dependencies with fakes you control, so the test is fast and deterministic and
exercises *your* logic, not pandas' CSV parser or the disk.

We'll mock the I/O functions added in Module 05 (`load_dataset`,
`save_dataset`). For now, understand the techniques.

### Technique A — `@patch` decorator

`unittest.mock.patch` temporarily replaces an object by its import path.

```python
from unittest.mock import patch
import pandas as pd

@patch("preprocessing.pd.read_csv")
@patch("preprocessing.os.path.exists", return_value=True)
def test_load_dataset_reads_existing_file(mock_exists, mock_read_csv, sample_df):
    mock_read_csv.return_value = sample_df

    result = load_dataset("whatever.csv")

    mock_exists.assert_called_once_with("whatever.csv")
    mock_read_csv.assert_called_once_with("whatever.csv")
    pd.testing.assert_frame_equal(result, sample_df)
```

Key points:

- **Patch where it's *used*, not where it's defined.** We patch
  `preprocessing.pd.read_csv` (how `preprocessing.py` refers to it), not
  `pandas.read_csv`. This trips up nearly everyone the first time.
- **Decorators apply bottom-up:** the innermost decorator maps to the *first*
  argument. That's why `mock_exists` comes before `mock_read_csv`.
- `assert_called_once_with(...)` verifies not just the result but *how* the
  dependency was called.

### Technique B — `monkeypatch` (pytest-native)

```python
from unittest.mock import MagicMock
import preprocessing

def test_load_dataset_with_monkeypatch(monkeypatch, sample_df):
    fake_read = MagicMock(return_value=sample_df)
    monkeypatch.setattr(preprocessing.os.path, "exists", lambda p: True)
    monkeypatch.setattr(preprocessing.pd, "read_csv", fake_read)

    result = load_dataset("data.csv")

    fake_read.assert_called_once_with("data.csv")
```

`monkeypatch` auto-reverts after the test — no decorator stacking, no cleanup.

### Technique C — verify a call without a real object

```python
from unittest.mock import MagicMock

def test_save_dataset_calls_to_csv_without_index():
    mock_df = MagicMock(spec=pd.DataFrame)
    save_dataset(mock_df, "out.csv")
    mock_df.to_csv.assert_called_once_with("out.csv", index=False)
```

Here we never write a file — we just assert our function *would* call `to_csv`
with `index=False`. `spec=pd.DataFrame` makes the mock reject calls to methods a
real DataFrame doesn't have, catching typos.

### ⚠️ Pitfall — infinite recursion when a mock calls the real thing

We wrote a test whose fake `read_csv` was defined as
`lambda p: pd.read_csv(io.StringIO(text))` — but `pd.read_csv` was the very
thing being patched, so it called *itself* forever:

```
RecursionError: maximum recursion depth exceeded
```

**Fix:** capture the real function *before* patching, then call that:

```python
real_read_csv = pd.read_csv     # capture original first
with patch("preprocessing.pd.read_csv",
           side_effect=lambda p: real_read_csv(io.StringIO(csv_text))):
    ...
```

> **Teaching moment:** a mock's `side_effect` runs in the patched world. If it
> references the patched symbol, you get recursion. Bind the original first.

---

## 5. ⚠️ The import-path gotcha (critical)

Put your tests in `tests/` and run bare `pytest`, and you may hit:

```
ModuleNotFoundError: No module named 'preprocessing'
```

**Why:** with its default "prepend" import mode, pytest adds the *test file's*
directory (`tests/`) to `sys.path` — **not** the project root. So
`import preprocessing` (which lives in the root) fails.

It often *appears* to work when you run `python -m pytest`, because the `-m`
form adds the current directory to `sys.path`. Relying on that masks the bug —
and it will bite you in Docker and CI, which invoke the bare `pytest`
entrypoint. (We hit exactly this in Module 04.)

**The fix** — a project-root pytest config in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]        # put the project root on sys.path
testpaths = ["tests"]     # where tests live
```

Now bare `pytest` works everywhere — locally, in Docker, and in CI.

---

## 6. Run the whole suite

```bash
pytest -v
```

**Expected output (abridged):**

```
tests/test_io_mocks.py ........                                  [ 22%]
tests/test_preprocessing.py ............................        [100%]
============================== 36 passed in 0.05s ==============================
```

Useful flags:

| Flag | Effect |
| ---- | ------ |
| `-v` | verbose, one line per test |
| `-q` | quiet, just dots |
| `-k "normalize"` | run only tests whose name matches |
| `-x` | stop at the first failure |
| `--lf` | re-run only last-failed tests |

---

## Checkpoint ✅

- [ ] `tests/conftest.py` with shared fixtures.
- [ ] `tests/test_preprocessing.py` using fixtures + parametrization.
- [ ] `tests/test_io_mocks.py` demonstrating three mocking styles.
- [ ] `pyproject.toml` with `pythonpath` set.
- [ ] `pytest -v` is green.

## Exercises

1. Add a parametrized test for `standardize` that checks mean≈0 and std≈1 for
   both columns `a` and `b`.
2. Write a mock test proving `save_dataset` does **not** touch the disk (hint:
   `MagicMock(spec=pd.DataFrame)`).
3. Break the `pythonpath` line in `pyproject.toml` and observe the
   `ModuleNotFoundError`. Restore it. This cements *why* it's there.

[← Preprocessing](01-preprocessing.md) · [Index](README.md) · [Next: Continuous Integration →](03-ci-github-actions.md)
