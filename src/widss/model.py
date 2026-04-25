from __future__ import annotations

from typing import Any


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def build_lstm_soc_model(window_size: int, feature_count: int, units: int = 64, learning_rate: float = 1e-3) -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is not installed. Install it to train the LSTM model.") from exc

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
