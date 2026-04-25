<div align="center">

# ⚡ WIDSS

**W**indowed **I**ntelligent **D**rive-cycle **S**tate e**S**timation

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/pritam-09-ops/WIDSS/actions/workflows/tests.yml/badge.svg)](https://github.com/pritam-09-ops/WIDSS/actions/workflows/tests.yml)
[![Lint](https://github.com/pritam-09-ops/WIDSS/actions/workflows/lint.yml/badge.svg)](https://github.com/pritam-09-ops/WIDSS/actions/workflows/lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)

*An open-source, modular framework for AI-driven battery State-of-Charge (SOC) and State-of-Health (SOH) estimation under realistic EV drive cycles.*

[Features](#-features) · [Quick Start](#-quick-start) · [Module Reference](#-module-reference) · [Configuration](#-configuration) · [Roadmap](#-roadmap) · [Contributing](#-contributing)

</div>

---

## ✨ Features

| Feature | Status |
|---|---|
| Synthetic EV drive-cycle generator (idle / cruise / accel / regen) | ✅ |
| Physics-inspired SOC & terminal-voltage simulation | ✅ |
| Sliding-window time-series dataset builder | ✅ |
| LSTM model construction & training (TensorFlow) | ✅ |
| RMSE / MAE / MAPE evaluation metrics | ✅ |
| Multi-version CI (Python 3.10 – 3.12) | ✅ |
| SOH cycle-aging pipeline | 🚧 |
| PyBaMM electrochemical backend | 📋 |
| Transformer / physics-informed baselines | 📋 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip ≥ 21

### Installation

```bash
# Clone the repository
git clone https://github.com/pritam-09-ops/WIDSS.git
cd WIDSS

# Install core dependencies
pip install -e .

# Optional: TensorFlow backend for LSTM training
pip install -e ".[tensorflow]"

# Optional: full development toolchain
pip install -e ".[all]"
```

### Run tests

```bash
python -m pytest
```

### Train your first SOC model

```bash
PYTHONPATH=src python scripts/train_soc_lstm.py \
    --duration-s 7200 \
    --window-size 30 \
    --epochs 10 \
    --output-dir runs/first_run
```

---

## 📖 Module Reference

### `widss.simulation`

Generates synthetic EV current profiles and simulates battery physics.

```python
from widss.simulation import BatterySimulationConfig, build_dataset

cfg = BatterySimulationConfig(
    capacity_ah=60.0,
    soc_init=0.95,
    dt_s=1.0,
    internal_resistance_ohm=0.02,
    ocv_min_v=3.0,
    ocv_max_v=4.2,
)

frame = build_dataset(duration_s=3600, config=cfg, seed=42)
print(frame.head())
#    time_s  current_a  voltage_v       soc
# 0     0.0      -8.43      4.368  0.950039
# 1     1.0     -10.21      4.368  0.950086
# ...
```

### `widss.dataset`

Builds sliding-window sequences for supervised deep learning.

```python
from widss.dataset import build_sequences

# x: (num_windows, window_size, num_features)
# y: (num_windows,)  ← SOC at the next timestep
x, y = build_sequences(
    frame=frame,
    feature_cols=("voltage_v", "current_a"),
    target_col="soc",
    window_size=30,
    horizon=1,
)
print(x.shape, y.shape)  # (3539, 30, 2) (3539,)
```

### `widss.model`

LSTM model factory powered by TensorFlow/Keras.

```python
from widss.model import build_lstm_soc_model, tensorflow_available

if tensorflow_available():
    model = build_lstm_soc_model(
        window_size=30,
        feature_count=2,
        units=64,
        learning_rate=1e-3,
    )
    model.summary()
```

### `widss.evaluation`

Standard regression metrics for model assessment.

```python
import numpy as np
from widss.evaluation import rmse, mae, mape

y_true = np.array([0.9, 0.8, 0.7, 0.6])
y_pred = np.array([0.88, 0.81, 0.69, 0.62])

print(f"RMSE : {rmse(y_true, y_pred):.4f}")
print(f"MAE  : {mae(y_true, y_pred):.4f}")
print(f"MAPE : {mape(y_true, y_pred):.2f} %")
```

---

## ⚙️ Configuration

### Battery Parameters (`BatterySimulationConfig`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `capacity_ah` | `float` | `60.0` | Cell capacity in amp-hours |
| `soc_init` | `float` | `0.95` | Initial state of charge (0–1) |
| `dt_s` | `float` | `1.0` | Simulation time-step (seconds) |
| `internal_resistance_ohm` | `float` | `0.02` | Internal resistance (Ω) |
| `ocv_min_v` | `float` | `3.0` | Minimum open-circuit voltage (V) |
| `ocv_max_v` | `float` | `4.2` | Maximum open-circuit voltage (V) |

### Training Parameters (`train_soc_lstm.py` CLI)

| Flag | Default | Description |
|---|---|---|
| `--duration-s` | `7200` | Drive-cycle length (seconds) |
| `--dt-s` | `1.0` | Simulation time-step |
| `--window-size` | `30` | LSTM input window width |
| `--epochs` | `5` | Training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--seed` | `42` | Random seed |
| `--output-dir` | `outputs/` | Artifact save path |

### LSTM Model (`build_lstm_soc_model`)

| Parameter | Default | Description |
|---|---|---|
| `units` | `64` | LSTM hidden units |
| `learning_rate` | `1e-3` | Adam optimizer LR |

---

## 📁 Project Structure

```
WIDSS/
├── src/widss/
│   ├── __init__.py        ← Package metadata & version
│   ├── simulation.py      ← Drive-cycle generator & battery physics
│   ├── dataset.py         ← Sliding-window sequence builder
│   ├── model.py           ← LSTM factory (TensorFlow)
│   └── evaluation.py      ← RMSE / MAE / MAPE metrics
├── tests/
│   ├── test_simulation.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_evaluation.py
├── scripts/
│   └── train_soc_lstm.py  ← End-to-end training entry point
├── .github/
│   ├── workflows/
│   │   ├── tests.yml      ← CI: pytest + coverage
│   │   └── lint.yml       ← CI: black + flake8 + isort + mypy
│   └── ISSUE_TEMPLATE/
├── pyproject.toml
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

---

## 📊 Benchmarks

Indicative results on a MacBook M1 (TF 2.16, Python 3.11):

| Config | Data | Epochs | Val RMSE | Time |
|---|---|---|---|---|
| LSTM-64 | 2 h (7 200 s) | 10 | ~0.012 | ~45 s |
| LSTM-64 | 4 h (14 400 s) | 20 | ~0.008 | ~90 s |

> **Note**: Performance depends on hardware and drive-cycle randomness.
> Set `--seed` for reproducible runs.

---

## 🗺️ Roadmap

- **v0.1.0** ✅ SOC baseline — synthetic simulator, LSTM, evaluation metrics
- **v0.2.0** 🚧 SOH pipeline — cycle-aging simulation, capacity-fade model
- **v0.3.0** 📋 Production packaging — REST inference API, ONNX export, Docker
- **v0.4.0** 🔮 Advanced ML — Transformer, physics-informed loss, uncertainty quantification

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to open issues, submit pull requests, and follow our code-style standards.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 💬 Support

- Open an [issue](https://github.com/pritam-09-ops/WIDSS/issues) for bug reports or feature requests
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance

