# Contributing to WIDSS

Thank you for your interest in contributing to WIDSS! This document provides guidelines and instructions for participating in the project.

## Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/WIDSS.git
cd WIDSS
```

### 2. Set Up Your Development Environment
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e .
pip install pytest pytest-cov tensorflow>=2.13  # Optional, for model training
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
```

## Development Workflow

### Making Changes

1. **Make your changes** in a focused branch. Each PR should address one feature or bug.
2. **Write or update tests** for your changes in the `tests/` directory.
3. **Run tests locally** to ensure nothing breaks:
   ```bash
   pytest
   pytest -v  # Verbose output
   pytest --cov=src/widss --cov-report=html  # Check coverage
   ```
4. **Verify code quality** — keep the codebase readable:
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Keep functions focused and testable

### Commit Messages

Write clear, concise commit messages:
- Use imperative mood: "Add feature" not "Added feature"
- Reference issues if applicable: "Fix #42: Resolve SOC drift in long drives"
- Keep the first line under 50 characters

Example:
```
Add temperature correction to ECM

This implements a simple temperature adjustment to the open-circuit voltage calculation,
improving accuracy in cold/hot environments. Addresses #15.
```

## Types of Contributions

### Bug Reports
If you find a bug, please open an issue and include:
- **Python version** (`python --version`)
- **Operating system** (Windows, macOS, Linux)
- **Steps to reproduce** the issue
- **Expected vs. actual behavior**
- **Error message or stack trace** (if applicable)

### Feature Requests
- Describe the use case clearly
- Explain why this feature would be useful
- Provide examples if possible

### Documentation
- Improvements to README.md, docstrings, or examples are always welcome
- Ensure markdown is properly formatted
- Update the Table of Contents if adding new sections

### Code Improvements
- Refactoring to improve readability
- Performance optimizations (include benchmarks!)
- Additional test coverage
- Code style or consistency improvements

## Testing Requirements

All contributions must include or pass tests:

- **New features** should include unit tests in `tests/`
- **Bug fixes** should include a test that fails before the fix and passes after
- **Target coverage:** >80% of the code you write

Run tests before submitting:
```bash
pytest
pytest --cov=src/widss --cov-report=html
```

## Pull Request Process

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - Use a descriptive title
   - Reference any related issues: "Closes #42"
   - Describe what you changed and why
   - Include screenshots/examples if relevant

3. **Address feedback**:
   - Respond to review comments
   - Make requested changes and push them (no need to squash)
   - Re-request review after updates

4. **Merging**:
   - Maintainers will merge your PR once it's approved and tests pass
   - Thank you for the contribution!

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation
- Aim for lines ≤100 characters (flexible for readability)

### Docstrings
Every public function should have a docstring:
```python
def build_sequences(df, window_size=30):
    """
    Convert a timeseries DataFrame into sliding windows for LSTM training.
    
    Args:
        df (pd.DataFrame): DataFrame with columns [time_s, current_a, voltage_v, soc]
        window_size (int): Number of timesteps per window. Default: 30.
    
    Returns:
        tuple: (x, y) where x is shape (n_windows, window_size, 2) and 
               y is shape (n_windows,)
    """
```

### Type Hints (Encouraged)
```python
from typing import Tuple
import pandas as pd
import numpy as np

def build_sequences(df: pd.DataFrame, window_size: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

## Project Structure

```
WIDSS/
├── src/widss/
│   ├── simulation.py    # Physics simulation & drive cycle
│   ├── dataset.py       # Sliding window builder
│   ├── model.py         # LSTM model definition
│   └── evaluation.py    # Metrics and evaluation
├── scripts/
│   └── train_soc_lstm.py  # End-to-end training CLI
├── tests/               # Unit tests
├── pyproject.toml
├── CONTRIBUTING.md
└── README.md
```

## Licensing

By contributing to WIDSS, you agree that your contributions will be licensed under the MIT License (see LICENSE file).

## Questions or Need Help?

- **For questions:** Open a GitHub discussion or issue
- **For larger changes:** Open an issue first to discuss the approach before implementing
- **Need clarification:** Ask in comments or discussions — we're here to help!

## Code of Conduct

Be respectful and inclusive. We welcome contributions from people of all backgrounds and experience levels. Harassment or discrimination of any kind is not tolerated.

---

Thank you for helping make WIDSS better! 🚗⚡
