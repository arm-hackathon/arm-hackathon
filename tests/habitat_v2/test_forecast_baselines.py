from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.baselines import (
    BaselineError,
    RidgeSample,
    fit_direct_ridge,
    flatten_features,
    linear_extrapolation,
    persistence,
)
from aeolus.habitat_v2.forecast.projection import ForecastHistory, ForecastLayout

INPUT = "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
TARGET = "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"


def history(*, rows: int = 4) -> ForecastHistory:
    operational = []
    for source in ("primary_sensor_head", "secondary_sensor_head"):
        for zone in range(8):
            for field in (
                "temperature_k",
                "pressure_pa",
                "co2_ppm",
                "o2_mole_fraction",
                "relative_humidity",
            ):
                operational.append(
                    {
                        "descriptor_id": f"z{zone}/{field}",
                        "unit": "u",
                        "source_kind": source,
                    }
                )
    operational += [
        {
            "descriptor_id": f"branch_airflow_m3_s/z{zone}",
            "unit": "u",
            "source_kind": "operational_feedback_instrument",
        }
        for zone in range(8)
    ]
    operational += [
        {
            "descriptor_id": name,
            "unit": "fraction",
            "source_kind": "operational_feedback_instrument",
        }
        for name in (
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        )
    ]
    operational += [
        {
            "descriptor_id": f"unused/{i}",
            "unit": "u",
            "source_kind": "operational_feedback_instrument",
        }
        for i in range(167 - len(operational))
    ]
    targets = tuple(
        {"descriptor_id": f"z{zone}/{field}", "unit": "u"}
        for zone in range(8)
        for field in (
            "temperature_k",
            "pressure_pa",
            "co2_ppm",
            "o2_mole_fraction",
            "relative_humidity",
            "branch_airflow_m3_s",
        )
    ) + tuple(
        {"descriptor_id": n, "unit": "fraction"}
        for n in (
            "battery_state_of_charge",
            "oxygen_store_fraction",
            "sorbent_remaining_fraction",
        )
    )
    layout = ForecastLayout(tuple(operational), (), (), targets, INPUT, TARGET)
    numeric = np.zeros((rows, 194), dtype=np.float32)
    status = np.zeros((rows, 167, 5), dtype=np.float32)
    status[:, :, 0] = 1
    for r in range(rows):
        for c in range(167):
            numeric[r, c] = 10 + c + 2 * r
    for array in (numeric, status):
        array.setflags(write=False)
    other = []
    for shape in ((rows, 4), (rows, 4), (rows, 287, 4)):
        array = np.zeros(shape, dtype=np.float32)
        array.setflags(write=False)
        other.append(array)
    return ForecastHistory(
        tuple(range(1, rows + 1)),
        tuple(float(60 * (i + 1)) for i in range(rows)),
        numeric,
        status,
        other[0],
        other[1],
        other[2],
        layout,
    )


def test_persistence_averages_heads_falls_back_and_abstains() -> None:
    item = history()
    values = np.array(item.numeric_f32, copy=True)
    # z0 temperature is operational columns 0 and 40; use a known mean.
    values[-1, 0], values[-1, 40] = 10, 14
    statuses = np.array(item.status_f32, copy=True)
    statuses[-1, 41] = (
        0,
        1,
        0,
        0,
        0,
    )  # secondary z0 pressure unavailable -> primary fallback.
    item = replace(item, numeric_f32=values, status_f32=statuses)
    result = persistence(item, horizon_steps=2)
    assert result.status == "PREDICTION"
    assert result.values.shape == (2, 51)
    assert result.values[0, 0] == pytest.approx(12)
    assert result.values[0, 1] == pytest.approx(values[-1, 1])
    statuses[-1, (0, 40)] = (0, 1, 0, 0, 0)
    assert (
        persistence(replace(item, status_f32=statuses), horizon_steps=2).status
        == "ABSTAIN"
    )


def test_linear_ols_persistence_and_no_data_abstention() -> None:
    item = history(rows=4)
    result = linear_extrapolation(item, horizon_steps=2, future_times_s=(300.0, 360.0))
    assert result.status == "PREDICTION"
    # Both environmental heads are averaged: 30 at t=60, +2 per 60 seconds, so t=300 -> 38.
    assert result.values[0, 0] == pytest.approx(38)
    status = np.array(item.status_f32, copy=True)
    status[:2, (0, 40)] = (0, 1, 0, 0, 0)
    one_two = linear_extrapolation(
        replace(item, status_f32=status), horizon_steps=2, future_times_s=(300, 360)
    )
    assert one_two.values[0, 0] == pytest.approx(
        (item.numeric_f32[-1, 0] + item.numeric_f32[-1, 40]) / 2
    )
    status[:, (0, 40)] = (0, 1, 0, 0, 0)
    assert (
        linear_extrapolation(
            replace(item, status_f32=status), horizon_steps=2, future_times_s=(300, 360)
        ).status
        == "ABSTAIN"
    )


def sample(
    sample_id: str, cluster: str, x: float, action: float, y: float
) -> RidgeSample:
    item = history()
    target = np.full((2, 51), y, dtype=np.float32)
    return RidgeSample(
        sample_id,
        cluster,
        "TRAIN",
        item,
        np.full(27, action, dtype=np.float32),
        target,
        INPUT,
        TARGET,
    )


def test_ridge_action_features_blinding_and_determinism() -> None:
    item = history()
    aware = flatten_features(item, np.ones(27, dtype=np.float32), include_action=True)
    blind = flatten_features(item, np.ones(27, dtype=np.float32), include_action=False)
    assert aware.shape[0] == blind.shape[0] + 27
    samples = tuple(
        sample(f"s{i}", f"c{i}", float(i), float(i), float(i)) for i in range(4)
    )
    first = fit_direct_ridge(samples, horizon_steps=2, include_action=True)
    second = fit_direct_ridge(samples, horizon_steps=2, include_action=True)
    assert first.alpha == second.alpha
    assert np.array_equal(first.coef, second.coef)
    assert first.predict(item, np.ones(27, dtype=np.float32)).shape == (2, 51)


def test_ridge_rejects_duplicate_identity_and_bad_cluster_split() -> None:
    items = (
        sample("same", "c1", 0, 0, 0),
        sample("same", "c2", 1, 1, 1),
        sample("s3", "c3", 2, 2, 2),
    )
    with pytest.raises(BaselineError):
        fit_direct_ridge(items, horizon_steps=2)
    with pytest.raises(BaselineError):
        fit_direct_ridge(
            tuple(sample(f"s{i}", "one", i, i, i) for i in range(3)),
            horizon_steps=2,
        )


def test_ridge_rejects_any_nontraining_sample() -> None:
    items = [sample(f"s{i}", f"c{i}", i, i, i) for i in range(4)]
    items[2] = replace(items[2], split_label="VALIDATION")

    with pytest.raises(BaselineError, match="TRAIN"):
        fit_direct_ridge(tuple(items), horizon_steps=2)
