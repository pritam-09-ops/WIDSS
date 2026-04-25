from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_SEGMENT_STEPS = 5
MAX_SEGMENT_STEPS = 60
DRIVE_MODES = ("idle", "cruise", "accel", "regen")
# Typical mixed EV usage profile: idle, steady cruise, acceleration bursts, and regenerative braking.
DRIVE_MODE_PROBABILITIES = (0.2, 0.35, 0.3, 0.15)
SECONDS_PER_HOUR = 3600.0


@dataclass(slots=True)
class BatterySimulationConfig:
    capacity_ah: float = 60.0
    soc_init: float = 0.95
    dt_s: float = 1.0
    internal_resistance_ohm: float = 0.02
    ocv_min_v: float = 3.0
    ocv_max_v: float = 4.2


def generate_drive_cycle(duration_s: int = 3600, dt_s: float = 1.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic EV-like current profile over time."""
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


def simulate_battery_states(current_a: np.ndarray, config: BatterySimulationConfig) -> tuple[np.ndarray, np.ndarray]:
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


def build_dataset(
    duration_s: int = 3600,
    config: BatterySimulationConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
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
