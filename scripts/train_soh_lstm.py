"""Train an LSTM model for SOH prediction using cycle-level degradation data.

This script implements the Phase 2 SOH pipeline:

1. **Generate** — Simulate battery aging over multiple charge cycles
2. **Extract** — Compute cycle-level features from the drive profiles
3. **Build** — Convert cycle data into sliding-window ML sequences
4. **Train** — Fit an LSTM model on the cycle-level data
5. **Save** — Export the trained model, loss history, and run summary

All parameters are configurable via CLI arguments. The script produces
three output artifacts:

- ``soh_lstm.keras`` — Trained Keras model file
- ``history_loss.npy`` — Per-epoch training loss (NumPy array)
- ``training_summary.json`` — Full run config + final metrics (JSON)

Usage Examples
--------------
Quick test run (~2 minutes)::

    python scripts/train_soh_lstm.py --cycles 50 --epochs 5

Standard training (500 cycles of degradation)::

    python scripts/train_soh_lstm.py --cycles 500 --epochs 10 --units 32

Full training with hyperparameter tuning::

    python scripts/train_soh_lstm.py \\
        --cycles 1000 --epochs 20 --batch-size 32 --window-size 20 \\
        --units 64 --learning-rate 0.0005 --output-dir runs/soh_exp1

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
import pandas as pd

from widss.dataset import build_cycle_sequences
from widss.degradation import (
    BatteryDegradationConfig,
    build_degradation_profile,
    compute_soh,
    extract_cycle_features,
)
from widss.model import build_lstm_soh_model, tensorflow_available
from widss.simulation import BatterySimulationConfig, build_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SOH training script.

    Returns:
        Parsed argument namespace with all training parameters.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        metavar="N",
        help="Number of charge cycles to simulate (default: 100).",
    )
    parser.add_argument(
        "--duration-per-cycle-s",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Duration of a single cycle in seconds (default: 300 = 5 min).",
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
        default=10,
        metavar="W",
        help="Number of past cycles fed to the LSTM (default: 10).",
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
        default=32,
        metavar="B",
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=32,
        metavar="U",
        help="Number of LSTM units (default: 32).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--capacity-fade-rate",
        type=float,
        default=0.02,
        metavar="RATE",
        help="Logarithmic capacity fade rate (default: 0.02).",
    )
    parser.add_argument(
        "--resistance-growth-rate",
        type=float,
        default=0.01,
        metavar="RATE",
        help="Resistance growth rate (default: 0.01).",
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
    """Run the complete WIDSS SOH LSTM training pipeline.

    Orchestrates cycle-aging simulation, feature extraction, sequence building,
    model training, and artifact saving. Prints progress with emoji-annotated
    status updates.

    Returns:
        Exit code: 0 on success, 1 if TensorFlow is not available.
    """
    args = parse_args()

    print("⚡ WIDSS – Phase 2 SOH LSTM Training")
    print("=" * 50)

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
    # Stage 1: Simulate battery degradation over cycles
    # ------------------------------------------------------------------
    print(f"🔋 Simulating battery aging  ({args.cycles} cycles, seed={args.seed}) …")

    # Build degradation profiles
    deg_cfg = BatteryDegradationConfig(
        capacity_init_ah=60.0,
        capacity_fade_rate=args.capacity_fade_rate,
        resistance_init_ohm=0.02,
        resistance_growth_rate=args.resistance_growth_rate,
    )
    capacity_ah, resistance_ohm = build_degradation_profile(cycles=args.cycles, config=deg_cfg)

    # Compute SOH for each cycle
    soh_values = np.array(
        [compute_soh(capacity_ah[i], capacity_ah[0]) for i in range(len(capacity_ah))]
    )
    print(f"   → SOH range [{soh_values[-1]:.3f}, {soh_values[0]:.3f}]")

    # ------------------------------------------------------------------
    # Stage 2: Extract cycle-level features from synthetic drive cycles
    # ------------------------------------------------------------------
    print("🪟  Extracting cycle features …")

    sim_cfg = BatterySimulationConfig(dt_s=args.dt_s, capacity_ah=capacity_ah[0])

    cycle_data = []
    for cycle_idx in range(args.cycles):
        # Simulate one drive cycle with the degraded battery parameters
        cycle_frame = build_dataset(
            duration_s=args.duration_per_cycle_s,
            config=sim_cfg,
            seed=args.seed + cycle_idx,
        )

        # Extract features
        current_a = cycle_frame["current_a"].values
        voltage_v = cycle_frame["voltage_v"].values
        soc_vals = cycle_frame["soc"].values

        features = extract_cycle_features(current_a, voltage_v, soc_vals, dt_s=args.dt_s)
        features["cycle_num"] = float(cycle_idx)
        features["soh"] = float(soh_values[cycle_idx])

        cycle_data.append(features)

    cycle_df = pd.DataFrame(cycle_data)
    print(f"   → {len(cycle_df)} cycles with {len(cycle_df.columns)} features")

    # ------------------------------------------------------------------
    # Stage 3: Build sliding-window sequences
    # ------------------------------------------------------------------
    print(f"📊 Building cycle sequences  (window={args.window_size}) …")
    x, y = build_cycle_sequences(
        cycle_data=cycle_df,
        feature_cols=(
            "cycle_num",
            "avg_current_a",
            "max_current_a",
            "avg_voltage_v",
            "soc_delta",
        ),
        target_col="soh",
        window_size=args.window_size,
    )

    split_idx = int(0.8 * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"   → train: {len(x_train):,}  |  val: {len(x_val):,}  |  features: {x.shape[2]}")

    # ------------------------------------------------------------------
    # Stage 4: Build and train LSTM model
    # ------------------------------------------------------------------
    print(f"🧠 Building LSTM model  (units={args.units}, lr={args.learning_rate}) …")
    model = build_lstm_soh_model(
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
    # Stage 5: Save artifacts
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "soh_lstm.keras"
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
        "cycles": args.cycles,
        "duration_per_cycle_s": args.duration_per_cycle_s,
        "dt_s": args.dt_s,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "units": args.units,
        "learning_rate": args.learning_rate,
        "capacity_fade_rate": args.capacity_fade_rate,
        "resistance_growth_rate": args.resistance_growth_rate,
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
