---
name: Bug report
about: Report a bug or unexpected behaviour
title: "[Bug] "
labels: bug
assignees: ''
---

## Describe the Bug

A clear and concise description of what the bug is.

## Minimal Reproducible Example

```python
# Paste the smallest code snippet that reproduces the issue
from widss.simulation import build_dataset

frame = build_dataset(duration_s=60)
```

## Expected Behaviour

What you expected to happen.

## Actual Behaviour

What actually happens. Include the full traceback if applicable:

```
Traceback (most recent call last):
  ...
```

## Environment

| | |
|---|---|
| OS | e.g. Ubuntu 22.04 / macOS 14 / Windows 11 |
| Python version | e.g. 3.11.5 |
| WIDSS version | e.g. 0.1.0 |
| TensorFlow version | e.g. 2.16.1 (or N/A) |
| NumPy version | e.g. 1.26.4 |

## Additional Context

Add any other context or screenshots here.
