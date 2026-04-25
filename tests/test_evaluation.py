"""Tests for widss.evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from widss.evaluation import mae, mape, rmse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERFECT = (np.array([0.9, 0.8, 0.7]), np.array([0.9, 0.8, 0.7]))
_TINY_ERROR = (np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------


def test_rmse_perfect_prediction_is_zero() -> None:
    assert rmse(*_PERFECT) == pytest.approx(0.0)


def test_rmse_known_value() -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([1.0, 0.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_returns_float() -> None:
    result = rmse(*_TINY_ERROR)
    assert isinstance(result, float)


def test_rmse_non_negative() -> None:
    assert rmse(*_TINY_ERROR) >= 0.0


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------


def test_mae_perfect_prediction_is_zero() -> None:
    assert mae(*_PERFECT) == pytest.approx(0.0)


def test_mae_known_value() -> None:
    y_true = np.array([0.0, 2.0, 4.0])
    y_pred = np.array([1.0, 1.0, 5.0])
    # |1| + |1| + |1| = 3; mean = 1.0
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_mae_returns_float() -> None:
    assert isinstance(mae(*_TINY_ERROR), float)


# ---------------------------------------------------------------------------
# mape
# ---------------------------------------------------------------------------


def test_mape_perfect_prediction_is_zero() -> None:
    assert mape(*_PERFECT) == pytest.approx(0.0, abs=1e-6)


def test_mape_known_value() -> None:
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.8])
    # |errors|: 0.1/1.0=10%, 0.1/2.0=5%, 0.2/4.0=5%  → mean ≈ 6.667%
    assert mape(y_true, y_pred) == pytest.approx(20.0 / 3.0, rel=1e-3)


def test_mape_returns_float() -> None:
    assert isinstance(mape(*_TINY_ERROR), float)


def test_mape_near_zero_denominator_does_not_raise() -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.1, 0.9])
    result = mape(y_true, y_pred)
    assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Shape / empty validation (shared behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [rmse, mae, mape])
def test_mismatched_shapes_raise(fn) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="shape"):
        fn(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("fn", [rmse, mae, mape])
def test_empty_arrays_raise(fn) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="empty"):
        fn(np.array([]), np.array([]))
