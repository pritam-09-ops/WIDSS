"""Sliding-window dataset builder for time-series sequence models.

This module converts raw battery simulation timeseries (or any tabular
time-series data) into supervised learning sequences suitable for
recurrent neural networks like LSTMs.

How It Works
------------
Given a DataFrame with columns for features (e.g., ``voltage_v``,
``current_a``) and a target (e.g., ``soc``), the builder creates
overlapping sliding windows:

.. code-block:: text

    Input window (size=3):     Target (horizon=1):
    ┌─────────────────────┐
    │ t=0  t=1  t=2       │ →  SOC at t=3
    └─────────────────────┘
         ┌─────────────────────┐
         │ t=1  t=2  t=3       │ →  SOC at t=4
         └─────────────────────┘
              ┌─────────────────────┐
              │ t=2  t=3  t=4       │ →  SOC at t=5
              └─────────────────────┘

This produces arrays of shape:
- **X**: ``(num_windows, window_size, num_features)``
- **y**: ``(num_windows,)``

For SOH data (cycle-level aggregates), the structure is similar:

.. code-block:: text

    Input window (cycles):     Target (SOH):
    ┌──────────────────────────┐
    │ Cycle 0 to N features    │ →  SOH at cycle N+1
    └──────────────────────────┘

Example
-------
>>> from widss.simulation import build_dataset
>>> from widss.dataset import build_sequences
>>>
>>> frame = build_dataset(duration_s=300, seed=42)
>>> x, y = build_sequences(frame, window_size=30)
>>> print(f"X shape: {x.shape}")  # (269, 30, 2)
>>> print(f"y shape: {y.shape}")  # (269,)

See Also
--------
widss.simulation : Generate the input DataFrame.
widss.degradation : Extract SOH labels and cycle-level features.
widss.model : Build an LSTM that consumes these sequences.
"""

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
    """Build sliding-window sequences from a time-series DataFrame.

    Creates overlapping input windows of ``(window_size, num_features)``
    paired with target values ``horizon`` steps ahead. This is the standard
    data preparation step for training sequence-to-one regression models.

    Args:
        frame: Input DataFrame containing feature and target columns.
            Must have at least ``window_size + horizon`` rows. Typically
            the output of :func:`widss.simulation.build_dataset`.
        feature_cols: Column names to use as input features. Each column
            becomes one feature dimension in the output array.
            Default: ``("voltage_v", "current_a")``.
        target_col: Column name for the prediction target.
            Default: ``"soc"``.
        window_size: Number of past timesteps in each input window.
            Must be positive. Larger values provide more context to the
            model but increase memory usage and training time.
        horizon: Number of steps ahead to predict. The target for each
            window is ``target_col`` at index ``window_end + horizon - 1``.
            Default: ``1`` (predict the very next timestep).

    Returns:
        A tuple ``(x, y)`` of NumPy arrays:

        - **x** — Input features, shape
          ``(num_windows, window_size, len(feature_cols))``
        - **y** — Target values, shape ``(num_windows,)``

        where ``num_windows = len(frame) - window_size - horizon + 1``.

    Raises:
        ValueError: If ``window_size`` or ``horizon`` is not positive.
        ValueError: If the DataFrame has too few rows to create at least
            one window with the given ``window_size`` and ``horizon``.
        KeyError: If any column in ``feature_cols`` or ``target_col``
            is not present in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     "voltage_v": np.linspace(4.2, 3.0, 100),
        ...     "current_a": np.random.uniform(5, 20, 100),
        ...     "soc": np.linspace(1.0, 0.0, 100),
        ... })
        >>> x, y = build_sequences(df, window_size=10, horizon=1)
        >>> print(x.shape)  # (89, 10, 2)
        (89, 10, 2)
    """
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


def build_cycle_sequences(
    cycle_data: pd.DataFrame,
    feature_cols: Sequence[str] = (
        "cycle_num",
        "avg_current_a",
        "max_current_a",
        "avg_voltage_v",
        "soc_delta",
    ),
    target_col: str = "soh",
    window_size: int = 10,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences from cycle-level aggregate data.

    Creates overlapping input windows of cycle-level features paired with
    SOH targets. This is used for training models that predict battery
    degradation (SOH) from cycle-level statistics rather than raw timeseries.

    Args:
        cycle_data: DataFrame containing cycle-level features and targets.
            Each row represents one cycle with aggregate statistics
            (e.g., average current, voltage, SOH). Expected columns include
            cycle number, cycle features (current, voltage, energy, etc.),
            and SOH label.
        feature_cols: Column names to use as input features.
            Default: ``("cycle_num", "avg_current_a", "max_current_a",
            "avg_voltage_v", "soc_delta")``.
        target_col: Column name for the prediction target (SOH).
            Default: ``"soh"``.
        window_size: Number of past cycles in each input window.
            Typical range: 5–30 cycles. Larger values provide more
            historical context but increase computation.
        horizon: Number of steps ahead to predict. The target for each
            window is ``target_col`` at index ``window_end + horizon - 1``.
            Default: ``1`` (predict SOH one cycle ahead).

    Returns:
        A tuple ``(x, y)`` of NumPy arrays:

        - **x** — Cycle features, shape
          ``(num_windows, window_size, len(feature_cols))``
        - **y** — SOH targets, shape ``(num_windows,)``

        where ``num_windows = len(cycle_data) - window_size - horizon + 1``.

    Raises:
        ValueError: If ``window_size`` or ``horizon`` is not positive.
        ValueError: If the DataFrame has too few rows.
        KeyError: If any column in ``feature_cols`` or ``target_col``
            is not present in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> cycles_df = pd.DataFrame({
        ...     "cycle_num": np.arange(100),
        ...     "avg_current_a": np.random.uniform(5, 20, 100),
        ...     "max_current_a": np.random.uniform(20, 80, 100),
        ...     "avg_voltage_v": np.linspace(4.2, 3.5, 100),
        ...     "soc_delta": np.random.uniform(0.5, 1.0, 100),
        ...     "soh": np.linspace(1.0, 0.8, 100),
        ... })
        >>> x, y = build_cycle_sequences(cycles_df, window_size=10)
        >>> print(x.shape, y.shape)  # (89, 10, 5) (89,)
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    values = cycle_data[list(feature_cols)].to_numpy(dtype=float)
    target = cycle_data[target_col].to_numpy(dtype=float)
    n = len(cycle_data)

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
