"""WIDSS – AI-driven battery state estimation framework.

WIDSS (**W**indowed **I**ntelligent **D**rive-cycle **S**tate e**S**timation)
provides a modular, three-stage pipeline for simulating, training, and
evaluating machine-learning models for battery **State-of-Charge (SOC)** and
**State-of-Health (SOH)** estimation under realistic EV drive cycles.

Architecture
------------
The framework is organized into four independent modules that can be used
individually or composed into a full pipeline:

.. code-block:: text

    Battery Config → simulation → dataset → model → evaluation
                    (physics)   (windows)  (LSTM)   (metrics)

Modules
-------
simulation
    Synthetic EV drive-cycle generator and physics-based battery state
    simulator using an Equivalent Circuit Model (ECM).
dataset
    Sliding-window time-series builder that converts raw simulation output
    into supervised learning sequences ``(X, y)``.
model
    LSTM model construction helpers for SOC prediction. Requires TensorFlow
    (soft dependency — the rest of the package works without it).
evaluation
    Standard regression metrics: RMSE, MAE, MAPE. Designed for comparing
    predicted vs. actual SOC values.

Quick Start
-----------
Generate battery data, build ML sequences, and evaluate — all in a few lines:

>>> from widss.simulation import BatterySimulationConfig, build_dataset
>>> from widss.dataset import build_sequences
>>> from widss.evaluation import rmse
>>>
>>> # Simulate 10 minutes of EV driving
>>> cfg = BatterySimulationConfig(capacity_ah=60.0, soc_init=0.95)
>>> frame = build_dataset(duration_s=600, config=cfg, seed=42)
>>> print(frame.columns.tolist())
['time_s', 'current_a', 'voltage_v', 'soc']
>>>
>>> # Build sliding-window sequences for LSTM training
>>> x, y = build_sequences(frame, window_size=20)
>>> print(x.shape, y.shape)
(579, 20, 2) (579,)

See Also
--------
- README.md : Full documentation with architecture diagrams and examples.
- scripts/train_soc_lstm.py : End-to-end training CLI.
- CONTRIBUTING.md : How to contribute to the project.

License
-------
MIT License. See LICENSE for details.
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
