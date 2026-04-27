# Contributing to WIDSS 🤝

Thank you for your interest in contributing to **WIDSS**! Whether you're fixing a typo, adding a feature, or reporting a bug — every contribution makes this project better.

This guide walks you through everything you need to know to get started.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Types of Contributions](#-types-of-contributions)
- [Development Setup](#-development-setup)
- [Code Standards](#-code-standards)
- [Testing](#-testing)
- [Commit Convention](#-commit-convention)
- [Pull Request Process](#-pull-request-process)
- [Issue Guidelines](#-issue-guidelines)
- [Code of Conduct](#-code-of-conduct)

---

## 🚀 Quick Start

```bash
# 1. Fork & clone
git clone https://github.com/<your-username>/WIDSS.git
cd WIDSS

# 2. Set up development environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[all]"

# 3. Create a feature branch
git checkout -b feature/my-awesome-feature

# 4. Make changes, run checks
python -m pytest                # tests pass?
black src/ tests/ scripts/      # formatted?
isort src/ tests/ scripts/      # imports sorted?
flake8 src/ tests/ scripts/     # no lint errors?

# 5. Commit and push
git add .
git commit -m "feat(simulation): add temperature-dependent resistance model"
git push origin feature/my-awesome-feature

# 6. Open a Pull Request on GitHub
```

---

## 🎯 Types of Contributions

| Type | What's Involved | Branch Prefix |
|:--|:--|:--|
| 🐛 **Bug Fix** | Fix + regression test that fails before and passes after | `bugfix/` |
| ✨ **New Feature** | Implementation + tests + docstrings. Open an issue first for large features | `feature/` |
| 📖 **Documentation** | README, docstrings, examples. Update TOC if adding new sections | `docs/` |
| ♻️ **Refactoring** | Code improvements without behaviour changes. Include benchmarks if relevant | `refactor/` |
| 🧪 **Test Coverage** | Additional tests for existing functionality | `test/` |

---

## 🛠 Development Setup

### Prerequisites

- **Python** 3.10, 3.11, or 3.12
- **pip** ≥ 21
- **Git**

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install everything (core + TensorFlow + dev tools)
pip install -e ".[all]"
```

This installs:

| Category | Packages |
|:--|:--|
| **Core** | `numpy`, `pandas` |
| **ML Backend** | `tensorflow` |
| **Dev Tools** | `pytest`, `pytest-cov`, `black`, `flake8`, `isort`, `mypy` |

### Add Upstream Remote

```bash
git remote add upstream https://github.com/pritam-09-ops/WIDSS.git
```

---

## 📏 Code Standards

We enforce consistent code quality with automated tools. **All checks run in CI on every pull request.**

### Formatting & Linting

```bash
# Auto-format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/

# Type check
mypy src/
```

### Key Rules

| Rule | Value |
|:--|:--|
| Line length | **100** characters |
| Code style | [PEP 8](https://peps.python.org/pep-0008/) via Black |
| Import order | Enforced by isort (Black-compatible profile) |
| Max complexity | **10** (flake8) |
| Blank lines | Two between top-level definitions, one between methods |

### Type Hints & Docstrings

- **All** public functions must have complete type annotations
- Use **Google-style docstrings** with `Args:`, `Returns:`, and `Raises:` sections
- Include at least one `Example:` block for public API functions

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground-truth values, shape ``(n,)``.
        y_pred: Model predictions, shape ``(n,)``.

    Returns:
        Scalar RMSE (non-negative).

    Raises:
        ValueError: If arrays have different shapes or are empty.

    Example:
        >>> import numpy as np
        >>> rmse(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        0.1
    """
```

---

## 🧪 Testing

### Running Tests

```bash
# Full test suite
python -m pytest

# With coverage report
python -m pytest --cov=widss --cov-report=term-missing

# Specific module
python -m pytest tests/test_evaluation.py -v

# Only run tests matching a pattern
python -m pytest -k "test_rmse"
```

### Test Guidelines

| Guideline | Details |
|:--|:--|
| **Location** | `tests/test_<module>.py` |
| **Determinism** | Use fixed seeds when randomness is involved |
| **Coverage target** | **80%** minimum for new code |
| **Naming** | `test_<function>_<scenario>` (e.g., `test_rmse_perfect_prediction_is_zero`) |
| **Structure** | Arrange → Act → Assert |

### Writing a Test

```python
"""Tests for widss.simulation module."""
import pytest
from widss.simulation import BatterySimulationConfig, build_dataset


def test_build_dataset_soc_stays_within_bounds() -> None:
    """SOC values should always remain in [0, 1]."""
    frame = build_dataset(duration_s=300, seed=42)
    assert (frame["soc"] >= 0.0).all()
    assert (frame["soc"] <= 1.0).all()


def test_build_dataset_negative_duration_raises() -> None:
    """Negative duration should raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        build_dataset(duration_s=-100)
```

---

## 📝 Commit Convention

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types

| Type | When to Use |
|:--|:--|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting (no logic change) |
| `refactor` | Code restructuring (no behaviour change) |
| `test` | Adding or updating tests |
| `chore` | Build, CI, tooling changes |
| `ci` | CI/CD workflow changes |

### Examples

```
feat(evaluation): add MAPE metric function
fix(dataset): handle edge case when window_size equals sequence length
docs(readme): add benchmark results table
test(simulation): increase coverage for regen drive mode
chore(deps): bump tensorflow to >=2.16
```

---

## 🔄 Pull Request Process

### Before Opening a PR

```bash
# 1. Sync with upstream
git fetch upstream
git rebase upstream/main

# 2. Run all checks locally
python -m pytest
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/widss/
```

### PR Checklist

Before submitting, make sure:

- [ ] All tests pass locally (`python -m pytest`)
- [ ] Code formatted with `black` and `isort`
- [ ] `flake8` / `mypy` show no new errors
- [ ] New behaviour is covered by tests
- [ ] Docstrings added/updated for public APIs
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

### PR Workflow

1. Push your branch and open a PR against `main`
2. Fill in the PR template (auto-populated)
3. Ensure all CI checks pass (tests + lint)
4. Request a review from a maintainer
5. Address review comments — push updates to the same branch
6. A maintainer will merge once approved

### Tips for a Smooth Review

- **Keep PRs focused** — one feature or fix per PR
- **Write a clear description** — explain *what* and *why*, not just *how*
- **Add screenshots** for UI or output changes
- **Reference issues** — use `Closes #123` or `Fixes #456` in the PR body

---

## 🐛 Issue Guidelines

### Bug Reports

Use the [Bug Report](https://github.com/pritam-09-ops/WIDSS/issues/new?template=bug_report.md) template. Include:

- Minimal reproducible example
- Expected vs actual behaviour
- Full traceback
- Environment details (Python version, OS, package versions)

### Feature Requests

Use the [Feature Request](https://github.com/pritam-09-ops/WIDSS/issues/new?template=feature_request.md) template. Include:

- Problem / motivation
- Proposed solution
- Alternatives you considered

---

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

**In short:** Be respectful, be constructive, be kind. Harassment or discrimination of any kind is not tolerated.

---

<div align="center">

**Thank you for helping make WIDSS better! 🔋⚡**

</div>
