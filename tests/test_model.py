"""Tests for widss.model LSTM construction helpers."""

from __future__ import annotations

import pytest

from widss.model import build_lstm_soc_model, tensorflow_available

# ---------------------------------------------------------------------------
# Skip all tests gracefully when TensorFlow is not installed
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not tensorflow_available(),
    reason="TensorFlow is not installed",
)


# ---------------------------------------------------------------------------
# build_lstm_soc_model
# ---------------------------------------------------------------------------


def test_model_builds_without_error() -> None:
    model = build_lstm_soc_model(window_size=20, feature_count=2)
    assert model is not None


def test_model_output_shape() -> None:
    import numpy as np

    model = build_lstm_soc_model(window_size=20, feature_count=2)
    dummy = np.zeros((4, 20, 2), dtype=float)
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (4, 1)


def test_model_output_in_unit_interval() -> None:
    """Sigmoid activation should keep predictions in [0, 1]."""
    import numpy as np

    model = build_lstm_soc_model(window_size=10, feature_count=2)
    dummy = np.random.default_rng(0).standard_normal((16, 10, 2))
    preds = model.predict(dummy, verbose=0)
    assert (preds >= 0.0).all()
    assert (preds <= 1.0).all()


def test_model_custom_units() -> None:
    model = build_lstm_soc_model(window_size=15, feature_count=3, units=32)
    assert model is not None


def test_model_custom_learning_rate() -> None:
    import tensorflow as tf

    model = build_lstm_soc_model(window_size=10, feature_count=2, learning_rate=1e-4)
    opt = model.optimizer
    lr = float(tf.keras.backend.get_value(opt.learning_rate))
    assert lr == pytest.approx(1e-4, rel=1e-4)


def test_tensorflow_available_returns_bool() -> None:
    result = tensorflow_available()
    assert isinstance(result, bool)
