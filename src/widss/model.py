"""LSTM model builder for battery State-of-Charge prediction.

This module provides factory functions to construct and compile a
Keras Sequential LSTM model tailored for SOC estimation from
windowed time-series data.

Architecture
------------
The default model architecture is:

.. code-block:: text

    Input (window_size, feature_count)
        ↓
    LSTM (units)            ← learns temporal patterns
        ↓
    Dense (32, ReLU)        ← non-linear feature extraction
        ↓
    Dense (1, sigmoid)      ← SOC output ∈ [0, 1]

The model is compiled with:
- **Optimizer**: Adam (configurable learning rate)
- **Loss**: Mean Squared Error (MSE)
- **Metric**: Root Mean Squared Error (RMSE)

TensorFlow Dependency
---------------------
TensorFlow is a **soft dependency** — it is only imported when you
actually call :func:`build_lstm_soc_model`. The rest of the WIDSS
package (simulation, dataset, evaluation) works without it.

Use :func:`tensorflow_available` to check before building a model:

>>> from widss.model import tensorflow_available
>>> if tensorflow_available():
...     print("TensorFlow is installed — LSTM training is available")

Example
-------
>>> from widss.model import build_lstm_soc_model, tensorflow_available
>>>
>>> if tensorflow_available():
...     model = build_lstm_soc_model(
...         window_size=30,
...         feature_count=2,
...         units=64,
...         learning_rate=1e-3
...     )
...     model.summary()  # prints the architecture

See Also
--------
widss.dataset : Prepare input sequences for the model.
widss.evaluation : Evaluate model predictions with RMSE, MAE, MAPE.
scripts/train_soc_lstm.py : End-to-end training CLI using this module.
"""

from __future__ import annotations

from typing import Any


def tensorflow_available() -> bool:
    """Check whether TensorFlow is importable.

    This is a convenience function for conditional imports. Use it to
    gracefully handle environments where TensorFlow is not installed.

    Returns:
        ``True`` if TensorFlow can be imported, ``False`` otherwise.

    Example:
        >>> from widss.model import tensorflow_available
        >>> available = tensorflow_available()
        >>> print(type(available))
        <class 'bool'>
    """
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def build_lstm_soc_model(
    window_size: int, feature_count: int, units: int = 64, learning_rate: float = 1e-3
) -> Any:
    """Build, compile, and return a Keras LSTM model for SOC prediction.

    Constructs a Sequential model with a single LSTM layer followed by
    two Dense layers. The output uses a sigmoid activation to constrain
    predictions to the [0, 1] SOC range.

    Args:
        window_size: Number of timesteps in each input window. This
            defines the temporal context the LSTM sees. Typical values:
            20–60 for 1 Hz data.
        feature_count: Number of input features per timestep. For the
            default WIDSS pipeline this is 2 (voltage_v, current_a).
        units: Number of LSTM hidden units. Controls model capacity.
            64 is a good starting point; 128 for more complex patterns.
            Higher values increase training time and memory usage.
        learning_rate: Learning rate for the Adam optimizer. Default
            ``1e-3`` works well for most cases. Try ``5e-4`` or ``1e-4``
            if training is unstable.

    Returns:
        A compiled ``tf.keras.Sequential`` model ready for ``.fit()``.
        The return type is ``Any`` to avoid requiring TensorFlow at
        import time.

    Raises:
        RuntimeError: If TensorFlow is not installed. Install it with
            ``pip install tensorflow>=2.13`` or
            ``pip install 'widss[tensorflow]'``.

    Example:
        >>> from widss.model import build_lstm_soc_model, tensorflow_available
        >>> if tensorflow_available():
        ...     model = build_lstm_soc_model(window_size=30, feature_count=2)
        ...     print(f"Model has {model.count_params():,} parameters")
    """
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install it to train the LSTM model."
        ) from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, feature_count)),
            tf.keras.layers.LSTM(units=units),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model
