from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from widss.dataset import build_sequences
from widss.model import build_lstm_soc_model, tensorflow_available
from widss.simulation import BatterySimulationConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM model for SOC prediction on synthetic drive-cycle data.")
    parser.add_argument("--duration-s", type=int, default=7200)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not tensorflow_available():
        print("TensorFlow is not installed. Install TensorFlow first to run LSTM training.")
        return 0

    cfg = BatterySimulationConfig(dt_s=args.dt_s)
    frame = build_dataset(duration_s=args.duration_s, config=cfg, seed=args.seed)
    x, y = build_sequences(frame=frame, window_size=args.window_size)

    split_idx = int(0.8 * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_lstm_soc_model(window_size=x.shape[1], feature_count=x.shape[2])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "soc_lstm.keras")
    np.save(args.output_dir / "history_loss.npy", np.asarray(history.history.get("loss", []), dtype=float))

    print(f"Saved model and history to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
