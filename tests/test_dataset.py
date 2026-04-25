from widss.dataset import build_sequences
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
