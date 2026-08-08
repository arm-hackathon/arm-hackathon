"""Transient physical-fault contracts for schema-v10 recovery scenarios."""

from __future__ import annotations

import pytest

from aeolus.config import (
    TransientBlockedPath,
    TransientGradualPrimaryFanDegradation,
    parse_scenario,
)


def _reserve_connections():
    pairs = []
    for zone in ("cabin_a", "cabin_b", "lab"):
        pairs.extend(
            [
                {
                    "id": f"reserve_{zone}_to_processing",
                    "from": zone,
                    "to": "processing",
                    "max_airflow": 4.0,
                    "health": 1.0,
                },
                {
                    "id": f"reserve_processing_to_{zone}",
                    "from": "processing",
                    "to": zone,
                    "max_airflow": 4.0,
                    "health": 1.0,
                },
            ]
        )
    return pairs


def _v10(standard_doc, profile):
    standard_doc["version"] = 10
    standard_doc["air_system"]["reserve_airflow_capacity"] = 4.0
    standard_doc["reserve_connections"] = _reserve_connections()
    standard_doc["fault_profiles"] = [profile]
    return standard_doc


def _transient_blocked(**overrides):
    profile = {
        "type": "transient_blocked_path",
        "connection_id": "cabin_a_to_processing",
        "start_tick": 0,
        "end_tick": 3,
        "blocked_effectiveness": 0.65,
    }
    profile.update(overrides)
    return profile


def _transient_gradual(**overrides):
    profile = {
        "type": "transient_gradual_primary_fan_degradation",
        "connection_id": "cabin_a_to_processing",
        "start_tick": 10,
        "end_tick": 13,
        "start_effectiveness": 1.0,
        "end_effectiveness": 0.75,
    }
    profile.update(overrides)
    return profile


def test_transient_blocked_path_uses_half_open_interval(standard_doc):
    config = parse_scenario(_v10(standard_doc, _transient_blocked()))

    profile = config.fault_profiles[0]
    assert profile == TransientBlockedPath(
        connection_id="cabin_a_to_processing",
        start_tick=0,
        end_tick=3,
        blocked_effectiveness=0.65,
    )
    assert [profile.effectiveness_at(tick) for tick in range(5)] == [
        0.65,
        0.65,
        0.65,
        1.0,
        1.0,
    ]


def test_transient_gradual_hits_both_endpoints_then_clears(standard_doc):
    config = parse_scenario(_v10(standard_doc, _transient_gradual()))

    profile = config.fault_profiles[0]
    assert profile == TransientGradualPrimaryFanDegradation(
        connection_id="cabin_a_to_processing",
        start_tick=10,
        end_tick=13,
        start_effectiveness=1.0,
        end_effectiveness=0.75,
    )
    assert profile.effectiveness_at(9) == pytest.approx(1.0)
    assert profile.effectiveness_at(10) == pytest.approx(1.0)
    assert profile.effectiveness_at(11) == pytest.approx(0.875)
    assert profile.effectiveness_at(12) == pytest.approx(0.75)
    assert profile.effectiveness_at(13) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        (_transient_blocked(start_tick=True), "start_tick.*integer"),
        (_transient_blocked(start_tick=-1), "start_tick.*non-negative"),
        (_transient_blocked(end_tick=0), "end_tick must be after start_tick"),
        (_transient_blocked(blocked_effectiveness=1.0), "must be below 1.0"),
        (_transient_blocked(blocked_effectiveness=1.1), "blocked_effectiveness"),
        (_transient_blocked(blocked_effectiveness=float("nan")), "finite"),
        (_transient_blocked(extra=True), "unexpected field"),
        (_transient_gradual(start_tick=True), "start_tick.*integer"),
        (_transient_gradual(start_tick=-1), "start_tick.*non-negative"),
        (_transient_gradual(end_tick=11), "at least two ticks"),
        (_transient_gradual(start_effectiveness=0.5, end_effectiveness=0.6), "degrade"),
        (_transient_gradual(start_effectiveness=1.1), "start_effectiveness"),
        (_transient_gradual(end_effectiveness=-0.1), "end_effectiveness"),
        (_transient_gradual(end_effectiveness=float("inf")), "finite"),
        (_transient_gradual(extra=True), "unexpected field"),
    ],
)
def test_v10_rejects_malformed_transient_faults(standard_doc, profile, match):
    with pytest.raises(ValueError, match=match):
        parse_scenario(_v10(standard_doc, profile))


def test_v9_rejects_transient_fault_type(standard_doc):
    standard_doc["fault_profiles"] = [_transient_blocked(start_tick=1)]

    with pytest.raises(ValueError, match="unsupported fault profile type"):
        parse_scenario(standard_doc)