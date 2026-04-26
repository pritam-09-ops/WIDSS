"""WIDSS – AI-driven battery state estimation framework.

WIDSS provides a modular pipeline for simulating, training, and evaluating
machine-learning models for battery State-of-Charge (SOC) and
State-of-Health (SOH) estimation.

Modules:
    simulation: Synthetic EV drive-cycle generator and physics-based battery
        state simulator.
    dataset:    Sliding-window time-series builder for supervised learning.
    model:      LSTM model construction helpers (requires TensorFlow).
    evaluation: Regression metrics (RMSE, MAE, MAPE).

Example:
    >>> from widss.simulation import BatterySimulationConfig, build_dataset
    >>> from widss.dataset import build_sequences
    >>> from widss.evaluation import rmse
    >>> cfg = BatterySimulationConfig(capacity_ah=60.0, soc_init=0.95)
    >>> frame = build_dataset(duration_s=600, config=cfg, seed=42)
    >>> x, y = build_sequences(frame, window_size=20)
    >>> print(x.shape, y.shape)
    (579, 20, 2) (579,)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "evaluation",
    "dataset",
    "model",
    "simulation",
]
