"""Train an LSTM model for SOC prediction on synthetic drive-cycle data.

This is the main entry-point script for the WIDSS training pipeline.
It orchestrates the full workflow:

1. **Generate** — Create a synthetic EV drive cycle using physics simulation
2. **Build** — Convert the timeseries into sliding-window ML sequences
3. **Train** — Fit a two-layer LSTM model on the data
4. **Save** — Export the trained model, loss history, and run summary

All parameters are configurable via CLI arguments. The script produces
three output artifacts:

- ``soc_lstm.keras`` — Trained Keras model file
- ``history_loss.npy`` — Per-epoch training loss (NumPy array)
- ``training_summary.json`` — Full run config + final metrics (JSON)

Usage Examples
--------------
Quick test run (~2 minutes)::

    python scripts/train_soc_lstm.py --duration-s 300 --epochs 5

Standard training (2 hours of simulated data)::

    python scripts/train_soc_lstm.py --duration-s 7200 --epochs 10 --units 64

Full training with hyperparameter tuning::

    python scripts/train_soc_lstm.py \\
        --duration-s 14400 --epochs 20 --batch-size 32 --window-size 60 \\
        --units 128 --learning-rate 0.0005 --output-dir runs/exp1

Requirements
------------
- TensorFlow >= 2.13 (install with ``pip install 'widss[tensorflow]'``)
- The ``src/`` directory must be on the Python path (handled automatically
  if installed via ``pip install -e .``)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from widss.dataset import build_sequences
from widss.model import build_lstm_soc_model, tensorflow_available
from widss.simulation import BatterySimulationConfig, build_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed argument namespace with all training parameters.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        default=7200,
        metavar="SECONDS",
        help="Length of the simulated drive cycle in seconds (default: 7200 = 2 h).",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=1.0,
        metavar="DT",
        help="Simulation time-step in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        metavar="W",
        help="Number of past time-steps fed to the LSTM (default: 30).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="Training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Mini-batch size (default: 64).",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=64,
        metavar="U",
        help="Number of LSTM units (default: 64).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        metavar="DIR",
        help="Directory where the model and history are saved (default: outputs/).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the complete WIDSS SOC LSTM training pipeline.

    Orchestrates drive-cycle generation, sequence building, model training,
    and artifact saving. Prints progress with emoji-annotated status updates.

    Returns:
        Exit code: 0 on success, 1 if TensorFlow is not available.
    """
    args = parse_args()

    print("⚡ WIDSS – SOC LSTM Training")
    print("=" * 42)

    # ------------------------------------------------------------------
    # Pre-flight check: TensorFlow availability
    # ------------------------------------------------------------------
    if not tensorflow_available():
        print(
            "❌  TensorFlow is not installed.\n"
            "    Install it with:  pip install tensorflow\n"
            "    or:               pip install 'widss[tensorflow]'"
        )
        return 1

    # ------------------------------------------------------------------
    # Stage 1: Generate synthetic drive cycle
    # ------------------------------------------------------------------
    print(f"🔋 Generating {args.duration_s / 3600:.1f} h drive cycle  (seed={args.seed}) …")
    cfg = BatterySimulationConfig(dt_s=args.dt_s)
    frame = build_dataset(duration_s=args.duration_s, config=cfg, seed=args.seed)
    soc_min = frame["soc"].min()
    soc_max = frame["soc"].max()
    print(f"   → {len(frame):,} time-steps  |  SOC range [{soc_min:.3f}, {soc_max:.3f}]")

    # ------------------------------------------------------------------
    # Stage 2: Build sliding-window sequences
    # ------------------------------------------------------------------
    print(f"🪟  Building sequences  (window={args.window_size}) …")
    x, y = build_sequences(frame=frame, window_size=args.window_size)

    split_idx = int(0.8 * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"   → train: {len(x_train):,}  |  val: {len(x_val):,}  |  features: {x.shape[2]}")

    # ------------------------------------------------------------------
    # Stage 3: Build and train LSTM model
    # ------------------------------------------------------------------
    print(f"🧠 Building LSTM model  (units={args.units}, lr={args.learning_rate}) …")
    model = build_lstm_soc_model(
        window_size=x.shape[1],
        feature_count=x.shape[2],
        units=args.units,
        learning_rate=args.learning_rate,
    )

    print(f"🚀 Training for {args.epochs} epoch(s)  (batch={args.batch_size}) …")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Stage 4: Save artifacts
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "soc_lstm.keras"
    history_path = args.output_dir / "history_loss.npy"
    summary_path = args.output_dir / "training_summary.json"

    model.save(model_path)
    np.save(history_path, np.asarray(history.history.get("loss", []), dtype=float))

    # Collect final metrics from training history
    final_loss = history.history.get("loss", [float("nan")])[-1]
    final_val_loss = history.history.get("val_loss", [float("nan")])[-1]
    final_rmse = history.history.get("rmse", [float("nan")])[-1]
    final_val_rmse = history.history.get("val_rmse", [float("nan")])[-1]

    summary = {
        "duration_s": args.duration_s,
        "dt_s": args.dt_s,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "units": args.units,
        "learning_rate": args.learning_rate,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "final_loss": float(final_loss),
        "final_val_loss": float(final_val_loss),
        "final_rmse": float(final_rmse),
        "final_val_rmse": float(final_val_rmse),
    }
    summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------
    print()
    print("✅ Done!")
    print(f"   Model   → {model_path.resolve()}")
    print(f"   History  → {history_path.resolve()}")
    print(f"   Summary  → {summary_path.resolve()}")
    print()
    print("   📊 Final Metrics:")
    print(f"      Train Loss : {final_loss:.6f}")
    print(f"      Val Loss   : {final_val_loss:.6f}")
    print(f"      Train RMSE : {final_rmse:.6f}")
    print(f"      Val RMSE   : {final_val_rmse:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
