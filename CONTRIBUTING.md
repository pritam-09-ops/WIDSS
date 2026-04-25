# Contributing to WIDSS

Thank you for considering a contribution to WIDSS! 🎉  
This document explains how to set up your development environment, follow our code standards, and submit a pull request.

---

## Table of Contents

- [Fork & Clone](#fork--clone)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Type Hints & Docstrings](#type-hints--docstrings)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Bug Reports & Feature Requests](#bug-reports--feature-requests)

---

## Fork & Clone

1. Click **Fork** in the top-right corner of the [repository page](https://github.com/pritam-09-ops/WIDSS).
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/WIDSS.git
   cd WIDSS
   ```

3. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/pritam-09-ops/WIDSS.git
   ```

4. Create a feature branch:

   ```bash
   git checkout -b feature/my-awesome-feature
   ```

---

## Development Environment

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package in editable mode with all dev dependencies
pip install -e ".[all]"
```

This installs:

- Core: `numpy`, `pandas`
- ML backend: `tensorflow`
- Dev tools: `pytest`, `black`, `flake8`, `isort`, `mypy`, `coverage`

---

## Code Style

We enforce consistent formatting with **Black** and **flake8**:

```bash
# Auto-format
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/
```

Key rules:

- Line length: **100** characters (Black default as configured in `pyproject.toml`)
- Maximum complexity: **10** (flake8)
- Two blank lines between top-level definitions
- One blank line between methods inside a class

All checks run automatically in CI on every pull request.

---

## Type Hints & Docstrings

- **All** public functions and methods must have complete type annotations.
- Use **Google-style docstrings** with `Args:`, `Returns:`, and `Raises:` sections.
- Include at least one `Example:` block for public API functions.

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

Run mypy for static type checking:

```bash
mypy src/
```

---

## Testing

- Write tests for every new public function or behaviour change.
- Place tests in `tests/` with the filename pattern `test_<module>.py`.
- Tests should be deterministic — use fixed seeds when randomness is involved.

```bash
# Run the full test suite
python -m pytest

# Run with coverage report
python -m pytest --cov=widss --cov-report=term-missing

# Run a specific file
python -m pytest tests/test_evaluation.py -v
```

**Minimum coverage target**: 80 % of new code.

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`

Examples:

```
feat(evaluation): add MAPE metric function
fix(dataset): handle edge case when window_size equals sequence length
docs(readme): add benchmark table
test(simulation): increase coverage for regen drive mode
```

---

## Pull Request Process

1. **Sync** your branch with upstream before opening a PR:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your branch and open a PR against `main`.
3. Fill in the pull request template (checklist auto-populated).
4. Ensure all CI checks pass (tests + lint).
5. Request a review from a maintainer.
6. Address review comments and push updates to the same branch.
7. A maintainer will squash-merge once approved.

### PR Checklist (quick reference)

- [ ] All tests pass locally (`python -m pytest`)
- [ ] Code formatted with Black and isort
- [ ] flake8 / mypy show no new errors
- [ ] New behaviour is covered by tests
- [ ] Docstrings added/updated for public APIs
- [ ] CHANGELOG.md updated under `[Unreleased]`

---

## Bug Reports & Feature Requests

- **Bugs**: Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template. Include a minimal reproducible example, expected vs actual behaviour, and your environment details.
- **Features**: Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template. Describe the use case and alternatives you considered.

Thank you for helping make WIDSS better! 🔋
