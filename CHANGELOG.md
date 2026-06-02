# Changelog

All notable changes to **WIDSS** are documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [Unreleased]

### Added

- (Nothing yet for the next release)

---

## [0.2.0] — Phase 2: SOH Pipeline

> Implements State-of-Health (SOH) prediction via battery cycle-aging simulation
> and degradation modelling.

### Added

#### Degradation Module (`widss.degradation`) — NEW

- `BatteryDegradationConfig` — Configuration for capacity fade and resistance growth
- `build_degradation_profile()` — Simulate battery aging over charge cycles with
  logarithmic capacity fade and square-root resistance growth models
- `compute_soh()` — Calculate State of Health as capacity retention percentage
- `detect_charge_cycles()` — Identify individual cycles from current timeseries
- `extract_cycle_features()` — Extract aggregate statistics from each cycle
  (avg/max current, avg voltage, SOC swing, energy throughput)

#### Dataset Extensions (`widss.dataset`)

- `build_cycle_sequences()` — Build sliding-window sequences from cycle-level
  aggregate data for SOH LSTM training
- Support for cycle-level feature engineering in addition to timestep-level data

#### Model Extensions (`widss.model`)

- `build_lstm_soh_model()` — LSTM model builder for SOH prediction with sigmoid
  activation constrained to [0, 1] SOH range
- Parallel architecture to `build_lstm_soc_model()` with tunable hidden units

#### Training Script (`scripts/train_soh_lstm.py`) — NEW

- End-to-end SOH training pipeline orchestrating:
  - Battery aging simulation over configurable cycles
  - Cycle-level feature extraction from synthetic drive profiles
  - LSTM model training with configurable hyperparameters
  - Artifact saving (model, loss history, summary JSON)
- CLI parameters: `--cycles`, `--duration-per-cycle-s`, `--window-size`,
  `--capacity-fade-rate`, `--resistance-growth-rate`, etc.

#### Tests (`tests/test_degradation.py`) — NEW

- Comprehensive test suite for degradation module (40+ assertions)
- Tests for capacity fade, resistance growth, SOH computation
- Cycle detection and feature extraction validation
- Tests for `build_cycle_sequences()` with various configurations

#### Documentation Updates

- Updated `__init__.py` module docstring with degradation pipeline
- Extended README with SOH section, cycle-level data handling, Phase 2 example
- Package version bumped to 0.2.0

### Changed

- Updated package version from 0.1.0 to 0.2.0
- Enhanced module architecture diagram in docstrings to include degradation stage

---

## [0.1.0] — 2024-01-01

> Initial public release of the WIDSS battery state estimation framework.

### Added

#### Simulation (`widss.simulation`)

- Synthetic EV drive-cycle generator with four driving modes:
  `idle`, `cruise`, `accel`, `regen`
- Configurable drive-mode probability weights
  (default: 20% / 35% / 30% / 15%)
- Physics-inspired SOC integration via Coulomb counting
- Linear OCV model with terminal-voltage drop from internal resistance
- `BatterySimulationConfig` dataclass for clean parameterization
- `build_dataset()` → tidy `pandas.DataFrame` with
  `time_s`, `current_a`, `voltage_v`, `soc`

#### Dataset (`widss.dataset`)

- `build_sequences()` — sliding-window builder producing `(X, y)` NumPy arrays
- Configurable `feature_cols`, `target_col`, `window_size`, and `horizon`
- Input validation with clear error messages

#### Model (`widss.model`)

- `build_lstm_soc_model()` — two-layer LSTM → Dense(32, ReLU) → Dense(1, sigmoid)
- Compiled with Adam optimizer and MSE loss + RMSE metric
- Soft TensorFlow dependency: import guarded by `tensorflow_available()` helper

#### Evaluation (`widss.evaluation`)

- `rmse(y_true, y_pred)` — Root Mean Squared Error
- `mae(y_true, y_pred)` — Mean Absolute Error
- `mape(y_true, y_pred)` — Mean Absolute Percentage Error (%)
- Guard against division by zero via configurable `epsilon`
- Consistent `ValueError` for shape mismatches and empty arrays

#### Scripts

- `scripts/train_soc_lstm.py` — end-to-end CLI for drive-cycle generation,
  sequence building, LSTM training, and artifact saving
- Saves `soc_lstm.keras` model and `history_loss.npy` to configurable output directory
- Emoji-annotated progress output for a friendly developer experience

#### Infrastructure

- `pyproject.toml` with modern `setuptools` build backend
- Optional dependency groups: `[dev]`, `[tensorflow]`, `[all]`
- Tool configs for Black, isort, mypy, pytest, and coverage
- GitHub Actions CI: multi-version test matrix (Python 3.10 / 3.11 / 3.12)
  with coverage upload to Codecov
- GitHub Actions lint: Black, flake8, isort, mypy
- GitHub issue templates (bug report, feature request)
- MIT License

---

## Future Releases

### [0.2.0] — SOH Pipeline 🚧

- Cycle-aging simulation (capacity fade, resistance growth)
- SOH label generation from degradation curves
- Extended dataset builder with cycle-level features
- SOH LSTM baseline training script

### [0.3.0] — Production Ready 📋

- REST inference API (FastAPI)
- ONNX model export for embedded deployment
- Docker image for containerized inference
- Structured JSON logging

### [0.4.0] — Advanced ML 🔮

- Transformer baseline for SOC estimation
- Physics-informed loss terms
- Uncertainty quantification (Monte-Carlo dropout)
- PyBaMM electrochemical backend integration

---

[Unreleased]: https://github.com/pritam-09-ops/WIDSS/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pritam-09-ops/WIDSS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pritam-09-ops/WIDSS/releases/tag/v0.1.0
