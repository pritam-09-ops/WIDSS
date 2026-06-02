from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from widss.dataset import build_sequences, build_cycle_sequences
from widss.simulation import build_dataset


def test_build_sequences_shape() -> None:
    frame = build_dataset(duration_s=120, seed=2)
    x, y = build_sequences(frame, window_size=10, horizon=1)

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 10
    assert x.shape[2] == 2


def test_build_sequences_respects_feature_order() -> None:
    frame = build_dataset(duration_s=40, seed=3)
    x, _ = build_sequences(frame, feature_cols=("voltage_v", "current_a"), window_size=5, horizon=1)

    assert x[0, 0, 0] == frame.iloc[0]["voltage_v"]
    assert x[0, 0, 1] == frame.iloc[0]["current_a"]


# ---------------------------------------------------------------------------
# build_cycle_sequences tests
# ---------------------------------------------------------------------------


def test_build_cycle_sequences_shape() -> None:
    """Test that cycle sequences have correct shape."""
    cycles_df = pd.DataFrame(
        {
            "cycle_num": np.arange(100),
            "avg_current_a": np.random.uniform(5, 20, 100),
            "max_current_a": np.random.uniform(20, 80, 100),
            "avg_voltage_v": np.linspace(4.2, 3.5, 100),
            "soc_delta": np.random.uniform(0.5, 1.0, 100),
            "soh": np.linspace(1.0, 0.8, 100),
        }
    )
    x, y = build_cycle_sequences(cycles_df, window_size=10, horizon=1)

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 10
    assert x.shape[2] == 5  # 5 features
    assert x.shape[0] == len(cycles_df) - 10 - 1 + 1


def test_build_cycle_sequences_respects_feature_order() -> None:
    """Test that features are in the correct order."""
    cycles_df = pd.DataFrame(
        {
            "cycle_num": np.arange(50),
            "avg_current_a": np.ones(50) * 10.0,
            "max_current_a": np.ones(50) * 50.0,
            "avg_voltage_v": np.ones(50) * 4.0,
            "soc_delta": np.ones(50) * 0.8,
            "soh": np.linspace(1.0, 0.5, 50),
        }
    )
    x, _ = build_cycle_sequences(cycles_df, window_size=5, horizon=1)

    # First sample, first timestep
    assert x[0, 0, 0] == 0.0  # cycle_num (0-indexed)
    assert x[0, 0, 1] == 10.0  # avg_current_a
    assert x[0, 0, 2] == 50.0  # max_current_a
    assert x[0, 0, 3] == 4.0  # avg_voltage_v
    assert x[0, 0, 4] == 0.8  # soc_delta


def test_build_cycle_sequences_invalid_window_size() -> None:
    """Test that invalid window size raises ValueError."""
    cycles_df = pd.DataFrame(
        {
            "cycle_num": np.arange(100),
            "avg_current_a": np.ones(100) * 10.0,
            "soh": np.ones(100) * 0.9,
        }
    )

    with pytest.raises(ValueError, match="window_size must be > 0"):
        build_cycle_sequences(cycles_df, window_size=0)

    with pytest.raises(ValueError, match="window_size must be > 0"):
        build_cycle_sequences(cycles_df, window_size=-1)


def test_build_cycle_sequences_insufficient_data() -> None:
    """Test that insufficient data raises ValueError."""
    cycles_df = pd.DataFrame(
        {
            "cycle_num": np.arange(5),
            "avg_current_a": np.ones(5) * 10.0,
            "max_current_a": np.ones(5) * 50.0,
            "avg_voltage_v": np.ones(5) * 4.0,
            "soc_delta": np.ones(5) * 0.8,
            "soh": np.ones(5) * 0.9,
        }
    )

    with pytest.raises(ValueError, match="not enough rows"):
        build_cycle_sequences(cycles_df, window_size=10)


def test_build_cycle_sequences_custom_features() -> None:
    """Test cycle sequences with custom feature columns."""
    cycles_df = pd.DataFrame(
        {
            "cycle": np.arange(50),
            "energy": np.random.uniform(1000, 2000, 50),
            "temp": np.random.uniform(20, 50, 50),
            "soh": np.linspace(1.0, 0.8, 50),
        }
    )
    x, y = build_cycle_sequences(
        cycles_df, feature_cols=("cycle", "energy", "temp"), target_col="soh", window_size=5
    )

    assert x.shape[2] == 3
    assert y.shape[0] == x.shape[0]


def test_build_cycle_sequences_with_horizon() -> None:
    """Test that horizon parameter works correctly."""
    cycles_df = pd.DataFrame(
        {
            "cycle_num": np.arange(100),
            "avg_current_a": np.random.uniform(5, 20, 100),
            "max_current_a": np.random.uniform(20, 80, 100),
            "avg_voltage_v": np.linspace(4.2, 3.5, 100),
            "soc_delta": np.random.uniform(0.5, 1.0, 100),
            "soh": np.linspace(1.0, 0.8, 100),
        }
    )

    x1, y1 = build_cycle_sequences(cycles_df, window_size=10, horizon=1)
    x2, y2 = build_cycle_sequences(cycles_df, window_size=10, horizon=2)

    # Horizon=2 should have fewer sequences
    assert len(x2) < len(x1)
    # But same window size
    assert x1.shape[1] == x2.shape[1]
