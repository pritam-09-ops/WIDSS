"""Cycle-aging simulation and battery degradation modelling.

This module simulates battery capacity fade and resistance growth over
repeated charge-discharge cycles, enabling State-of-Health (SOH) prediction.

Physics Model
-------------
Battery aging is modelled as:

1. **Capacity Fade** — total usable capacity decreases over time
   C(n) = C₀ × (1 - a_c × log(n + 1))
   where n is the cycle number, C₀ is the initial capacity, and a_c
   is the capacity fade rate.

2. **Resistance Growth** — internal resistance increases over time
   R(n) = R₀ × (1 + a_r × √n)
   where R₀ is the initial resistance and a_r is the resistance growth rate.

Both models are empirically motivated and tunable to match specific
battery chemistry and operating conditions.

Example
-------
>>> from widss.degradation import BatteryDegradationConfig, build_degradation_profile
>>>
>>> # Simulate degradation over 1000 cycles
>>> cfg = BatteryDegradationConfig(
...     capacity_init_ah=60.0,
...     capacity_fade_rate=0.02,
...     resistance_growth_rate=0.01
... )
>>> capacity, resistance = build_degradation_profile(cycles=1000, config=cfg)
>>> print(f"Remaining capacity after 1000 cycles: {capacity[-1]:.2f} Ah")
>>> print(f"Final resistance: {resistance[-1]:.4f} Ω")

See Also
--------
widss.simulation : Generate baseline battery characteristics.
widss.dataset : Extract cycle-level features for SOH prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatteryDegradationConfig:
    """Configuration for battery aging simulation.

    This class parameterizes the capacity fade and resistance growth models
    used during cycle-aging simulation. All parameters have sensible defaults
    based on typical Li-ion battery aging under automotive drive cycles.

    Attributes:
        capacity_init_ah: Initial battery capacity in Amp-hours.
            Typical EV range: 40–100 Ah.
        capacity_fade_rate: Logarithmic fade coefficient (0–1).
            Typically 0.01–0.05 for moderate aging. Higher values mean
            faster capacity loss per cycle.
        resistance_init_ohm: Initial internal resistance in Ohms.
            Typical Li-ion: 0.01–0.03 Ω. If not specified, inherited
            from the simulation config.
        resistance_growth_rate: Resistance growth coefficient (0–1).
            Typically 0.001–0.01. Higher values mean faster resistance
            increase per cycle.

    Example:
        >>> cfg = BatteryDegradationConfig(
        ...     capacity_init_ah=60.0,
        ...     capacity_fade_rate=0.02,
        ...     resistance_growth_rate=0.005
        ... )
        >>> print(cfg.capacity_init_ah)
        60.0
    """

    capacity_init_ah: float = 60.0
    capacity_fade_rate: float = 0.02
    resistance_init_ohm: float = 0.02
    resistance_growth_rate: float = 0.01


# ---------------------------------------------------------------------------
# Degradation Profile Generation
# ---------------------------------------------------------------------------


def build_degradation_profile(
    cycles: int,
    config: BatteryDegradationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate capacity fade and resistance growth over charge cycles.

    Generates degradation trajectories using empirical power-law and
    logarithmic models. Useful for creating long-term SOH training data
    where the battery ages over many cycles.

    Args:
        cycles: Total number of charge-discharge cycles to simulate.
            Must be positive. Typical EV: 500–2000 cycles.
        config: Degradation configuration. If ``None``, uses defaults.
            See :class:`BatteryDegradationConfig`.

    Returns:
        A tuple ``(capacity_ah, resistance_ohm)`` of 1-D NumPy arrays:

        - **capacity_ah** — Available capacity at each cycle,
          shape ``(cycles,)``. Starts at ``config.capacity_init_ah``
          and monotonically decreases.
        - **resistance_ohm** — Internal resistance at each cycle,
          shape ``(cycles,)``. Starts at ``config.resistance_init_ohm``
          and monotonically increases.

    Example:
        >>> capacity, resistance = build_degradation_profile(cycles=1000)
        >>> print(capacity.shape, resistance.shape)
        (1000,) (1000,)
        >>> print(f"Capacity fade: {100 * (1 - capacity[-1] / capacity[0]):.1f}%")
    """
    if cycles <= 0:
        raise ValueError("cycles must be positive")

    if config is None:
        config = BatteryDegradationConfig()

    cycle_numbers = np.arange(cycles)

    # Capacity fade: logarithmic model
    # C(n) = C₀ × (1 - a_c × log(n + 1))
    capacity_ah = config.capacity_init_ah * (
        1.0 - config.capacity_fade_rate * np.log(cycle_numbers + 1)
    )
    capacity_ah = np.maximum(capacity_ah, 0.1)  # Floor at 0.1 Ah

    # Resistance growth: square-root model
    # R(n) = R₀ × (1 + a_r × √n)
    resistance_ohm = config.resistance_init_ohm * (
        1.0 + config.resistance_growth_rate * np.sqrt(cycle_numbers)
    )

    return capacity_ah, resistance_ohm


def compute_soh(capacity_current_ah: float, capacity_init_ah: float) -> float:
    """Compute State of Health (SOH) as capacity retention percentage.

    SOH represents the ratio of current usable capacity to the rated
    (initial) capacity, expressed as a percentage. A battery with 80% SOH
    can deliver 80% of its original capacity.

    Args:
        capacity_current_ah: Current usable capacity in Ah.
        capacity_init_ah: Initial (rated) capacity in Ah.

    Returns:
        SOH as a float in [0, 1]. Typically, batteries are end-of-life
        when SOH < 0.80.

    Raises:
        ValueError: If either capacity is negative.

    Example:
        >>> soh = compute_soh(capacity_current_ah=48.0, capacity_init_ah=60.0)
        >>> print(f"SOH: {100 * soh:.1f}%")
        SOH: 80.0%
    """
    if capacity_current_ah < 0:
        raise ValueError("capacity_current_ah must be non-negative")
    if capacity_init_ah <= 0:
        raise ValueError("capacity_init_ah must be positive")

    return capacity_current_ah / capacity_init_ah


# ---------------------------------------------------------------------------
# Cycle Detection and Feature Extraction
# ---------------------------------------------------------------------------


def detect_charge_cycles(current_a: np.ndarray, dt_s: float = 1.0) -> list[tuple[int, int]]:
    """Detect individual charge cycles from current timeseries.

    A cycle is defined as a segment where the cumulative charge changes sign
    (e.g., discharge followed by charge, or vice versa). This is a simplified
    definition suitable for synthetic data.

    Args:
        current_a: Current profile (positive = discharge, negative = charge),
            shape ``(n_steps,)``.
        dt_s: Sampling time-step in seconds. Used for integration.

    Returns:
        A list of cycle tuples ``(start_idx, end_idx)`` where each tuple
        marks the timestep range of one cycle. If fewer than one cycle
        is detected, returns an empty list.

    Example:
        >>> import numpy as np
        >>> current = np.array([1, 2, -1, -2, 1, 2])  # discharge, charge, discharge
        >>> cycles = detect_charge_cycles(current)
        >>> print(len(cycles))  # 2 or 3 depending on threshold
    """
    if len(current_a) == 0:
        return []

    # Integrate current to get charge (Coulombs)
    charge_c = np.cumsum(current_a) * dt_s

    # Find zero crossings (sign changes)
    charge_sign = np.sign(charge_c)
    sign_changes = np.where(np.diff(charge_sign) != 0)[0] + 1

    # Convert sign changes into cycle boundaries
    cycles = []
    if len(sign_changes) > 0:
        cycle_starts = np.concatenate(([0], sign_changes))
        cycle_ends = np.concatenate((sign_changes, [len(current_a)]))
        cycles = list(zip(cycle_starts, cycle_ends))

    return cycles


def extract_cycle_features(
    cycle_current: np.ndarray,
    cycle_voltage: np.ndarray,
    cycle_soc: np.ndarray,
    dt_s: float = 1.0,
) -> dict[str, float]:
    """Extract aggregate features from a single charge cycle.

    Computes statistics that capture the energetic and electrochemical
    characteristics of a cycle, useful for SOH estimation.

    Args:
        cycle_current: Current profile during the cycle (Amperes),
            shape ``(cycle_steps,)``.
        cycle_voltage: Voltage profile during the cycle (Volts),
            shape ``(cycle_steps,)``.
        cycle_soc: SOC profile during the cycle, shape ``(cycle_steps,)``.
        dt_s: Sampling time-step in seconds.

    Returns:
        A dictionary with keys:

        - ``"avg_current_a"`` — Mean absolute current (A)
        - ``"max_current_a"`` — Peak current magnitude (A)
        - ``"avg_voltage_v"`` — Mean voltage (V)
        - ``"soc_delta"`` — SOC swing (ΔSOC) over the cycle
        - ``"energy_wh"`` — Energy throughput (Wh)

    Example:
        >>> current = np.array([10, 15, 20, 15, 10, 0])
        >>> voltage = np.array([4.0, 4.1, 4.2, 4.1, 4.0, 3.9])
        >>> soc = np.array([0.90, 0.85, 0.80, 0.85, 0.90, 0.95])
        >>> features = extract_cycle_features(current, voltage, soc)
        >>> print(f"Avg current: {features['avg_current_a']:.2f} A")
    """
    if len(cycle_current) == 0:
        raise ValueError("Empty cycle")

    features = {
        "avg_current_a": float(np.mean(np.abs(cycle_current))),
        "max_current_a": float(np.max(np.abs(cycle_current))),
        "avg_voltage_v": float(np.mean(cycle_voltage)),
        "soc_delta": float(np.abs(cycle_soc[-1] - cycle_soc[0])),
        "energy_wh": float(np.sum(cycle_voltage * np.abs(cycle_current)) * dt_s / 3600.0),
    }
    return features
