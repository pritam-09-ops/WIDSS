"""Synthetic EV drive-cycle generator and battery physics simulator.

This module implements a first-order Equivalent Circuit Model (ECM) for
Li-ion batteries and a stochastic drive-cycle generator that produces
realistic EV current profiles.

Physics Model
-------------
The battery is modelled as:

.. code-block:: text

    V_terminal = OCV(SOC) − I × R_internal

where:
- **OCV(SOC)** is the Open-Circuit Voltage, linearly interpolated between
  ``ocv_min_v`` (at SOC=0) and ``ocv_max_v`` (at SOC=1).
- **I** is the instantaneous current (positive = discharge, negative = charge).
- **R_internal** is the lumped internal resistance.

SOC is updated at each timestep via Coulomb counting:

.. code-block:: text

    ΔSOC = −(I × Δt) / (Q × 3600)

where Q is the battery capacity in Ah.

Drive Cycle Modes
-----------------
The generator produces current profiles by randomly sampling from four
driving modes with configurable probabilities:

| Mode    | Current Range     | Default Probability |
|---------|-------------------|---------------------|
| idle    | 0 A               | 20%                 |
| cruise  | 5 – 20 A          | 35%                 |
| accel   | 20 – 80 A         | 30%                 |
| regen   | −40 – −5 A        | 15%                 |

Each mode segment lasts 5–60 timesteps, with Gaussian noise (σ=1 A) added
for realism.

Example
-------
>>> from widss.simulation import BatterySimulationConfig, build_dataset
>>>
>>> # Simulate 1 hour with a 60 Ah battery starting at 95% charge
>>> cfg = BatterySimulationConfig(capacity_ah=60.0, soc_init=0.95)
>>> frame = build_dataset(duration_s=3600, config=cfg, seed=42)
>>> print(frame.head())
   time_s  current_a  voltage_v       soc
0     0.0   5.234291   4.175419  0.950000
1     1.0   4.896742   4.176384  0.949976
...

See Also
--------
widss.dataset : Convert simulation output into ML-ready sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SEGMENT_STEPS = 5
"""Minimum number of timesteps per drive-cycle segment."""

MAX_SEGMENT_STEPS = 60
"""Maximum number of timesteps per drive-cycle segment."""

DRIVE_MODES = ("idle", "cruise", "accel", "regen")
"""Available driving modes for the synthetic drive-cycle generator."""

DRIVE_MODE_PROBABILITIES = (0.2, 0.35, 0.3, 0.15)
"""Default sampling probabilities for each drive mode.

Represents a typical mixed EV usage profile: idle, steady cruise,
acceleration bursts, and regenerative braking.
"""

SECONDS_PER_HOUR = 3600.0
"""Conversion factor used in Coulomb counting: 1 hour = 3600 seconds."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatterySimulationConfig:
    """Configuration for the ECM battery simulator.

    All fields have sensible defaults for a generic Li-ion EV battery pack.
    Adjust these to match your specific cell chemistry and pack topology.

    Attributes:
        capacity_ah: Total usable battery capacity in Amp-hours.
            Typical EV range: 40–100 Ah.
        soc_init: Initial State of Charge (0.0 = empty, 1.0 = full).
            Clamped to [0, 1] during simulation.
        dt_s: Simulation timestep in seconds. Smaller values increase
            accuracy at the cost of computation time.
        internal_resistance_ohm: Lumped internal resistance in Ohms.
            Affects terminal voltage drop under load.
        ocv_min_v: Open-Circuit Voltage at SOC = 0 (fully discharged).
        ocv_max_v: Open-Circuit Voltage at SOC = 1 (fully charged).

    Example:
        >>> cfg = BatterySimulationConfig(
        ...     capacity_ah=80.0,
        ...     soc_init=0.90,
        ...     internal_resistance_ohm=0.03
        ... )
        >>> print(cfg.capacity_ah)
        80.0
    """

    capacity_ah: float = 60.0
    soc_init: float = 0.95
    dt_s: float = 1.0
    internal_resistance_ohm: float = 0.02
    ocv_min_v: float = 3.0
    ocv_max_v: float = 4.2


# ---------------------------------------------------------------------------
# Drive Cycle Generation
# ---------------------------------------------------------------------------


def generate_drive_cycle(
    duration_s: int = 3600, dt_s: float = 1.0, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic EV-like current profile over time.

    Produces a stochastic current waveform by randomly sequencing segments
    of driving modes (idle, cruise, acceleration, regenerative braking).
    Each segment has a random duration and amplitude drawn from mode-specific
    ranges, with Gaussian noise added for realism.

    Args:
        duration_s: Total simulation duration in seconds. Must be positive.
        dt_s: Timestep size in seconds. Must be positive.
        seed: Random seed for reproducibility. Same seed + same parameters
            always produces the same drive cycle.

    Returns:
        A tuple ``(time_s, current_a)`` of 1-D NumPy arrays:

        - **time_s** — timestamps in seconds, shape ``(n_steps,)``
        - **current_a** — current draw in Amperes, shape ``(n_steps,)``
          (positive = discharge, negative = regenerative charging)

    Raises:
        ValueError: If ``duration_s`` or ``dt_s`` is not positive.

    Example:
        >>> time_s, current_a = generate_drive_cycle(duration_s=60, seed=1)
        >>> print(time_s.shape, current_a.shape)
        (60,) (60,)
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / dt_s)
    time_s = np.arange(n_steps) * dt_s

    current_a = np.zeros(n_steps, dtype=float)
    step_idx = 0
    while step_idx < n_steps:
        segment_len = int(rng.integers(MIN_SEGMENT_STEPS, MAX_SEGMENT_STEPS))
        segment_end = min(step_idx + segment_len, n_steps)
        mode = rng.choice(DRIVE_MODES, p=DRIVE_MODE_PROBABILITIES)
        if mode == "idle":
            amp = 0.0
        elif mode == "cruise":
            amp = float(rng.uniform(5.0, 20.0))
        elif mode == "accel":
            amp = float(rng.uniform(20.0, 80.0))
        else:
            amp = float(rng.uniform(-40.0, -5.0))
        current_a[step_idx:segment_end] = amp + rng.normal(0.0, 1.0, segment_end - step_idx)
        step_idx = segment_end

    return time_s, current_a


# ---------------------------------------------------------------------------
# Battery State Simulation
# ---------------------------------------------------------------------------


def simulate_battery_states(
    current_a: np.ndarray, config: BatterySimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate battery SOC and terminal voltage from a current profile.

    Applies Coulomb counting to track SOC and computes terminal voltage
    using the first-order ECM:  ``V = OCV(SOC) − I × R_internal``.

    SOC is clamped to [0, 1] at every timestep to prevent unphysical values.

    Args:
        current_a: 1-D array of current values in Amperes. Positive values
            represent discharge; negative values represent charging
            (regenerative braking).
        config: Battery configuration parameters. See
            :class:`BatterySimulationConfig` for details.

    Returns:
        A tuple ``(soc, voltage_v)`` of 1-D NumPy arrays:

        - **soc** — State of Charge at each timestep, range [0, 1]
        - **voltage_v** — Terminal voltage at each timestep, in Volts

    Raises:
        ValueError: If ``current_a`` is not a 1-D array.

    Example:
        >>> import numpy as np
        >>> cfg = BatterySimulationConfig(capacity_ah=60.0, soc_init=0.95)
        >>> current = np.array([10.0, 10.0, 10.0])  # 10 A discharge
        >>> soc, voltage = simulate_battery_states(current, cfg)
        >>> print(f"SOC dropped from {soc[0]:.4f} to... (Coulomb counting)")
        SOC dropped from 0.9500 to... (Coulomb counting)
    """
    if current_a.ndim != 1:
        raise ValueError("current_a must be a 1D array")

    soc = np.zeros_like(current_a, dtype=float)
    voltage_v = np.zeros_like(current_a, dtype=float)

    soc_prev = float(np.clip(config.soc_init, 0.0, 1.0))
    for idx, cur in enumerate(current_a):
        delta_soc = -(cur * config.dt_s) / (config.capacity_ah * SECONDS_PER_HOUR)
        soc_now = float(np.clip(soc_prev + delta_soc, 0.0, 1.0))
        ocv = config.ocv_min_v + (config.ocv_max_v - config.ocv_min_v) * soc_now
        terminal_v = ocv - cur * config.internal_resistance_ohm

        soc[idx] = soc_now
        voltage_v[idx] = terminal_v
        soc_prev = soc_now

    return soc, voltage_v


# ---------------------------------------------------------------------------
# High-Level Dataset Builder
# ---------------------------------------------------------------------------


def build_dataset(
    duration_s: int = 3600,
    config: BatterySimulationConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a complete battery simulation dataset.

    This is the main entry point for the simulation module. It combines
    drive-cycle generation and battery state simulation into a single call
    that returns a tidy DataFrame ready for analysis or ML.

    Args:
        duration_s: Total simulation duration in seconds.
        config: Battery configuration. Uses default
            :class:`BatterySimulationConfig` if ``None``.
        seed: Random seed for reproducible drive cycles.

    Returns:
        A ``pandas.DataFrame`` with columns:

        - ``time_s`` — timestamp in seconds
        - ``current_a`` — battery current in Amperes
        - ``voltage_v`` — terminal voltage in Volts
        - ``soc`` — State of Charge [0, 1]

    Example:
        >>> frame = build_dataset(duration_s=600, seed=42)
        >>> print(frame.shape)
        (600, 4)
        >>> print(frame.columns.tolist())
        ['time_s', 'current_a', 'voltage_v', 'soc']
    """
    cfg = config or BatterySimulationConfig()
    time_s, current_a = generate_drive_cycle(duration_s=duration_s, dt_s=cfg.dt_s, seed=seed)
    soc, voltage_v = simulate_battery_states(current_a=current_a, config=cfg)

    return pd.DataFrame(
        {
            "time_s": time_s,
            "current_a": current_a,
            "voltage_v": voltage_v,
            "soc": soc,
        }
    )
