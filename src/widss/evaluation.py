"""Evaluation metrics for battery state estimation models.

This module provides standard regression metrics used to assess the
quality of SOC (State-of-Charge) and SOH (State-of-Health) predictions.
All metrics follow a consistent interface and include input validation.

Available Metrics
-----------------

+--------+--------------------------------------+--------+
| Metric | What It Measures                     | Unit   |
+========+======================================+========+
| RMSE   | Root Mean Squared Error               | Same   |
+--------+--------------------------------------+--------+
| MAE    | Mean Absolute Error                   | Same   |
+--------+--------------------------------------+--------+
| MAPE   | Mean Absolute Percentage Error        | %      |
+--------+--------------------------------------+--------+

Interpreting Results (SOC context)
----------------------------------
Since SOC is on a 0–1 scale:

- **RMSE = 0.015** means ±1.5% average SOC error
- **MAE = 0.012** means predictions are off by ~1.2% on average
- **MAPE = 3.0%** means relative error is about 3%

For EV applications, RMSE < 0.02 (2% SOC) is generally considered
acceptable for range estimation.

Example
-------
>>> import numpy as np
>>> from widss.evaluation import rmse, mae, mape
>>>
>>> y_true = np.array([0.9, 0.8, 0.7])
>>> y_pred = np.array([0.88, 0.81, 0.72])
>>>
>>> print(f"RMSE: {rmse(y_true, y_pred):.4f}")
RMSE: 0.0163
>>> print(f"MAE:  {mae(y_true, y_pred):.4f}")
MAE:  0.0133
>>> print(f"MAPE: {mape(y_true, y_pred):.2f}%")
MAPE: 1.59%

See Also
--------
widss.model : Build an LSTM model whose predictions can be evaluated here.
scripts/train_soc_lstm.py : Automatically computes these metrics after training.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    RMSE penalizes large errors more heavily than MAE due to the squaring
    operation, making it sensitive to outlier predictions.

    Args:
        y_true: Ground-truth target values, shape ``(n,)``.
        y_pred: Model predictions, shape ``(n,)``.

    Returns:
        Scalar RMSE value (non-negative). Same units as the input data.

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

    MAE is the average of absolute differences between predictions and
    actual values. It is more robust to outliers than RMSE and provides
    a straightforward interpretation: "on average, predictions are off
    by this much."

    Args:
        y_true: Ground-truth target values, shape ``(n,)``.
        y_pred: Model predictions, shape ``(n,)``.

    Returns:
        Scalar MAE value (non-negative). Same units as the input data.

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

    MAPE expresses prediction error as a percentage of the true value,
    making it scale-independent. Useful for comparing model performance
    across datasets with different value ranges.

    .. warning::
        MAPE is undefined when ``y_true`` contains zeros. The ``epsilon``
        parameter prevents division by zero, but results should be
        interpreted cautiously when true values are near zero.

    Args:
        y_true: Ground-truth target values, shape ``(n,)``. Values very
            close to zero are guarded by ``epsilon`` to avoid division by
            zero.
        y_pred: Model predictions, shape ``(n,)``.
        epsilon: Small value added to the denominator to avoid division by
            zero. Defaults to ``1e-8``.

    Returns:
        Scalar MAPE value as a *percentage* (e.g., ``2.5`` means 2.5%).

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


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raise ``ValueError`` if arrays are incompatible for metric computation.

    Args:
        y_true: Ground-truth array.
        y_pred: Prediction array.

    Raises:
        ValueError: If shapes differ or arrays are empty.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
