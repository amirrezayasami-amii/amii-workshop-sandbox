"""Red -> green exercise: make clip_outliers actually clip.

Run `pytest tests/test_clip_outliers.py` — it FAILS until you implement
`clip_outliers` in preprocessing.py. When it passes, open a PR into staging.
"""

import pandas as pd

from preprocessing import clip_outliers


def test_clip_outliers_caps_extremes():
    df = pd.DataFrame({"x": list(range(100)) + [10_000]})  # one huge outlier
    out = clip_outliers(df, ["x"], lower=0.05, upper=0.95)
    upper_q = df["x"].quantile(0.95)
    lower_q = df["x"].quantile(0.05)
    assert out["x"].max() <= upper_q
    assert out["x"].min() >= lower_q


def test_clip_outliers_leaves_inliers_untouched():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 14]})
    out = clip_outliers(df, ["x"], lower=0.0, upper=1.0)
    pd.testing.assert_series_equal(out["x"], df["x"])
