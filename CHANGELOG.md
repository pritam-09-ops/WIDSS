# Changelog

All notable changes to WIDSS are documented here.  
This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed

- Enhanced `scripts/train_soc_lstm.py` with tunable CLI knobs for `--units` and
  `--learning-rate` to simplify model fine-tuning.
- Added `training_summary.json` artifact output with final train/validation loss
  and RMSE, plus run configuration for presentation reporting.
- Updated README training examples and parameter tables to document the fine-tuning
  workflow and new run artifact.
- Polished README wording for clarity and added a dedicated **Results** section
  with benchmark snapshot and reproducible run-metrics example.

---

## [0.1.0] – 2024-01-01

Initial public release of the WIDSS battery state estimation framework.

### Simulation

- Synthetic EV drive-cycle generator with four driving modes: `idle`, `cruise`, `accel`, `regen`
- Configurable drive-mode probability weights (default: 20 / 35 / 30 / 15 %)
- Physics-inspired SOC integration via Coulomb counting
- Linear OCV model with terminal-voltage drop from internal resistance
- `BatterySimulationConfig` dataclass for clean parameterisation
- `build_dataset()` returns a tidy `pandas.DataFrame` with `time_s`, `current_a`, `voltage_v`, `soc`

### Dataset

- `build_sequences()` – sliding-window builder producing `(X, y)` NumPy arrays
- Configurable `feature_cols`, `target_col`, `window_size`, and `horizon`
- Input validation with clear error messages

### Model

- `build_lstm_soc_model()` – two-layer LSTM → Dense(32, ReLU) → Dense(1, sigmoid)
- Compiled with Adam optimizer and MSE loss + RMSE metric
- Soft TensorFlow dependency: import guarded by `tensorflow_available()` helper

### Evaluation

- `rmse(y_true, y_pred)` – Root Mean Squared Error
- `mae(y_true, y_pred)` – Mean Absolute Error
- `mape(y_true, y_pred)` – Mean Absolute Percentage Error (%)
- Guard against division by zero via configurable `epsilon`
- Consistent `ValueError` for shape mismatches and empty arrays

### Scripts

- `scripts/train_soc_lstm.py` – end-to-end CLI for drive-cycle generation, sequence building, LSTM training, and artifact saving
- Saves `soc_lstm.keras` model and `history_loss.npy` to a configurable output directory
- Emoji-annotated progress output for a friendly developer experience

### Infrastructure

- `pyproject.toml` with modern `setuptools` build backend
- Optional dependency groups: `[dev]`, `[tensorflow]`, `[all]`
- Tool configs for Black, isort, mypy, pytest, and coverage
- GitHub Actions CI: multi-version test matrix (Python 3.10 / 3.11 / 3.12) with coverage
- GitHub Actions lint: Black, flake8, isort, mypy
- GitHub issue templates (bug report, feature request)
- GitHub pull request template
- MIT License

---

## Future Releases

### [0.2.0] – SOH Pipeline 🚧

- Cycle-aging simulation (capacity fade, resistance growth)
- SOH label generation from degradation curves
- Extended dataset builder with cycle-level features
- SOH LSTM baseline training script

### [0.3.0] – Production Ready 📋

- REST inference API (FastAPI)
- ONNX model export
- Docker image
- Structured JSON logging

### [0.4.0] – Advanced ML 🔮

- Transformer baseline for SOC
- Physics-informed loss terms
- Uncertainty quantification (Monte-Carlo dropout)
- PyBaMM electrochemical backend integration

---

[Unreleased]: https://github.com/pritam-09-ops/WIDSS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pritam-09-ops/WIDSS/releases/tag/v0.1.0
