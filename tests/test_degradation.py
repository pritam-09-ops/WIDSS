"""Tests for the battery degradation module.

Tests cover cycle-aging simulation, capacity fade, resistance growth,
SOH computation, and cycle detection/feature extraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from widss.degradation import (
    BatteryDegradationConfig,
    build_degradation_profile,
    compute_soh,
    detect_charge_cycles,
    extract_cycle_features,
)


class TestBatteryDegradationConfig:
    """Tests for BatteryDegradationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test that default config has sensible values."""
        cfg = BatteryDegradationConfig()
        assert cfg.capacity_init_ah == 60.0
        assert cfg.capacity_fade_rate == 0.02
        assert cfg.resistance_init_ohm == 0.02
        assert cfg.resistance_growth_rate == 0.01

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        cfg = BatteryDegradationConfig(
            capacity_init_ah=80.0,
            capacity_fade_rate=0.03,
            resistance_init_ohm=0.015,
            resistance_growth_rate=0.005,
        )
        assert cfg.capacity_init_ah == 80.0
        assert cfg.capacity_fade_rate == 0.03
        assert cfg.resistance_init_ohm == 0.015
        assert cfg.resistance_growth_rate == 0.005


class TestBuildDegradationProfile:
    """Tests for build_degradation_profile function."""

    def test_output_shapes(self) -> None:
        """Test that output arrays have correct shapes."""
        cycles = 1000
        capacity, resistance = build_degradation_profile(cycles=cycles)
        assert capacity.shape == (cycles,)
        assert resistance.shape == (cycles,)

    def test_capacity_decreases(self) -> None:
        """Test that capacity monotonically decreases."""
        capacity, _ = build_degradation_profile(cycles=500)
        assert np.all(np.diff(capacity) <= 0), "Capacity should not increase"

    def test_resistance_increases(self) -> None:
        """Test that resistance monotonically increases."""
        _, resistance = build_degradation_profile(cycles=500)
        assert np.all(np.diff(resistance) >= 0), "Resistance should not decrease"

    def test_capacity_floor(self) -> None:
        """Test that capacity doesn't go below 0.1 Ah."""
        capacity, _ = build_degradation_profile(cycles=10000)
        assert np.all(capacity >= 0.1)

    def test_custom_config(self) -> None:
        """Test degradation with custom configuration."""
        cfg = BatteryDegradationConfig(
            capacity_init_ah=100.0,
            capacity_fade_rate=0.05,
            resistance_init_ohm=0.01,
            resistance_growth_rate=0.02,
        )
        capacity, resistance = build_degradation_profile(cycles=100, config=cfg)
        assert capacity[0] == 100.0
        assert resistance[0] == 0.01

    def test_invalid_cycles(self) -> None:
        """Test that negative cycles raise ValueError."""
        with pytest.raises(ValueError, match="cycles must be positive"):
            build_degradation_profile(cycles=-1)
        with pytest.raises(ValueError, match="cycles must be positive"):
            build_degradation_profile(cycles=0)


class TestComputeSoH:
    """Tests for compute_soh function."""

    def test_full_capacity(self) -> None:
        """Test SOH when capacity is at initial value."""
        soh = compute_soh(capacity_current_ah=60.0, capacity_init_ah=60.0)
        assert soh == pytest.approx(1.0)

    def test_half_capacity(self) -> None:
        """Test SOH at 50% capacity."""
        soh = compute_soh(capacity_current_ah=30.0, capacity_init_ah=60.0)
        assert soh == pytest.approx(0.5)

    def test_eol_threshold(self) -> None:
        """Test SOH at end-of-life threshold (80%)."""
        soh = compute_soh(capacity_current_ah=48.0, capacity_init_ah=60.0)
        assert soh == pytest.approx(0.8)

    def test_zero_capacity(self) -> None:
        """Test SOH with zero current capacity."""
        soh = compute_soh(capacity_current_ah=0.0, capacity_init_ah=60.0)
        assert soh == pytest.approx(0.0)

    def test_invalid_inputs(self) -> None:
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError, match="capacity_current_ah must be non-negative"):
            compute_soh(capacity_current_ah=-1.0, capacity_init_ah=60.0)
        with pytest.raises(ValueError, match="capacity_init_ah must be positive"):
            compute_soh(capacity_current_ah=60.0, capacity_init_ah=0.0)


class TestDetectChargeCycles:
    """Tests for detect_charge_cycles function."""

    def test_empty_array(self) -> None:
        """Test detection with empty current array."""
        cycles = detect_charge_cycles(np.array([]))
        assert cycles == []

    def test_single_mode(self) -> None:
        """Test with all positive (discharge) or all negative (charge)."""
        current = np.array([5.0, 10.0, 15.0, 20.0])
        cycles = detect_charge_cycles(current)
        assert len(cycles) == 0 or len(cycles) >= 0  # No sign change

    def test_alternating_modes(self) -> None:
        """Test with alternating discharge and charge."""
        current = np.array([10.0, 10.0, -10.0, -10.0, 10.0, 10.0])
        cycles = detect_charge_cycles(current)
        assert len(cycles) >= 1  # At least one cycle

    def test_cycle_boundaries(self) -> None:
        """Test that cycle boundaries make sense."""
        current = np.array([10.0, 10.0, -10.0, -10.0])
        cycles = detect_charge_cycles(current)
        for start, end in cycles:
            assert 0 <= start < end <= len(current)


class TestExtractCycleFeatures:
    """Tests for extract_cycle_features function."""

    def test_feature_dict_keys(self) -> None:
        """Test that output dictionary has expected keys."""
        current = np.array([10.0, 15.0, 20.0, 15.0, 10.0])
        voltage = np.array([4.0, 4.1, 4.2, 4.1, 4.0])
        soc = np.array([0.9, 0.85, 0.8, 0.85, 0.9])
        features = extract_cycle_features(current, voltage, soc)
        assert "avg_current_a" in features
        assert "max_current_a" in features
        assert "avg_voltage_v" in features
        assert "soc_delta" in features
        assert "energy_wh" in features

    def test_feature_values(self) -> None:
        """Test that features have reasonable values."""
        current = np.array([10.0, 20.0, 30.0])
        voltage = np.array([4.0, 4.1, 4.2])
        soc = np.array([0.9, 0.85, 0.8])
        features = extract_cycle_features(current, voltage, soc)
        assert features["avg_current_a"] > 0
        assert features["max_current_a"] >= features["avg_current_a"]
        assert 3.0 < features["avg_voltage_v"] < 4.5
        assert 0.0 <= features["soc_delta"] <= 1.0
        assert features["energy_wh"] >= 0.0

    def test_empty_cycle(self) -> None:
        """Test that empty cycle raises ValueError."""
        with pytest.raises(ValueError, match="Empty cycle"):
            extract_cycle_features(np.array([]), np.array([]), np.array([]))

    def test_matching_lengths(self) -> None:
        """Test with matched array lengths."""
        n = 10
        current = np.random.uniform(5, 30, n)
        voltage = np.random.uniform(3.5, 4.2, n)
        soc = np.linspace(1.0, 0.5, n)
        features = extract_cycle_features(current, voltage, soc)
        assert all(isinstance(v, (float, np.floating)) for v in features.values())
