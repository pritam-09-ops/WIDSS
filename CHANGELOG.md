# Changelog

All notable changes to **WIDSS** are documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [Unreleased]

### Added

- Tunable CLI parameters `--units` and `--learning-rate` in `scripts/train_soc_lstm.py`
  for simplified model fine-tuning.
- `training_summary.json` artifact output with final train/validation loss,
  RMSE, and complete run configuration for presentation reporting.
- Dedicated **Results** section in README with benchmark snapshot and
  reproducible run-metrics example.
- `SECURITY.md` — vulnerability reporting policy.
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.
- `.github/PULL_REQUEST_TEMPLATE.md` — standardized PR checklist.

### Changed

- README overhauled to presentation-grade with visual pipeline diagram,
  Mermaid roadmap, collapsible sections, and refined formatting.
- CONTRIBUTING.md rewritten with quick-start guide, visual tables, and
  clearer contributor workflow.
- CHANGELOG.md reformatted with proper Added/Changed/Fixed categorization.

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

[Unreleased]: https://github.com/pritam-09-ops/WIDSS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pritam-09-ops/WIDSS/releases/tag/v0.1.0
