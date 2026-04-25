from widss.simulation import BatterySimulationConfig, build_dataset


def test_build_dataset_has_expected_columns_and_soc_bounds() -> None:
    frame = build_dataset(duration_s=300, config=BatterySimulationConfig(dt_s=1.0), seed=1)

    assert list(frame.columns) == ["time_s", "current_a", "voltage_v", "soc"]
    assert (frame["soc"] >= 0.0).all()
    assert (frame["soc"] <= 1.0).all()
    assert len(frame) == 300
