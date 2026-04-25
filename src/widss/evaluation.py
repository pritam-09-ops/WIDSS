"""Evaluation metrics for battery state estimation models.

This module provides standard regression metrics used to assess the
quality of SOC and SOH predictions.

Example:
    >>> import numpy as np
    >>> from widss.evaluation import rmse, mae, mape
    >>> y_true = np.array([0.9, 0.8, 0.7])
    >>> y_pred = np.array([0.88, 0.81, 0.72])
    >>> print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    RMSE: 0.0163
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground-truth target values, shape ``(n,)``.
        y_pred: Model predictions, shape ``(n,)``.

    Returns:
        Scalar RMSE value (non-negative).

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different shapes or
            are empty.

    Example:
        >>> import numpy as np
        >>> rmse(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))
        0.1414...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_shapes(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: Ground-truth target values, shape ``(n,)``.
        y_pred: Model predictions, shape ``(n,)``.

    Returns:
        Scalar MAE value (non-negative).

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different shapes or
            are empty.

    Example:
        >>> import numpy as np
        >>> mae(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))
        0.1333...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        y_true: Ground-truth target values, shape ``(n,)``.  Values very
            close to zero are guarded by ``epsilon`` to avoid division by
            zero.
        y_pred: Model predictions, shape ``(n,)``.
        epsilon: Small value added to the denominator to avoid division by
            zero.  Defaults to ``1e-8``.

    Returns:
        Scalar MAPE value as a *percentage* (e.g. ``2.5`` means 2.5 %).

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different shapes or
            are empty.

    Example:
        >>> import numpy as np
        >>> mape(np.array([1.0, 2.0, 4.0]), np.array([1.1, 1.9, 3.8]))
        5.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate_shapes(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0)


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raise ``ValueError`` if arrays are incompatible for metric computation."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
