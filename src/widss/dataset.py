from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_sequences(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] = ("voltage_v", "current_a"),
    target_col: str = "soc",
    window_size: int = 20,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    values = frame[list(feature_cols)].to_numpy(dtype=float)
    target = frame[target_col].to_numpy(dtype=float)
    n = len(frame)

    upper = n - window_size - horizon + 1
    if upper <= 0:
        raise ValueError("not enough rows to build sequences for given window_size and horizon")

    x, y = [], []
    for start in range(upper):
        end = start + window_size
        target_idx = end + horizon - 1
        x.append(values[start:end])
        y.append(target[target_idx])

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
