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

    max_start_idx = n - window_size - horizon + 1
    if max_start_idx <= 0:
        raise ValueError("not enough rows to build sequences for given window_size and horizon")

    num_features = len(feature_cols)
    x = np.empty((max_start_idx, window_size, num_features), dtype=float)
    y = np.empty((max_start_idx,), dtype=float)
    for start in range(max_start_idx):
        end = start + window_size
        target_idx = end + horizon - 1
        x[start] = values[start:end]
        y[start] = target[target_idx]

    return x, y
