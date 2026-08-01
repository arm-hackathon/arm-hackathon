"""Schema-v8 fault-profile contracts for the converged simulator."""

from __future__ import annotations

import pytest

from aeolus.config import (
    BlockedPath,
    FrozenSensor,
    GradualPrimaryFanDegradation,
    parse_scenario,
)


FAULT_TYPE = "gradual_primary_fan_degradation"
BLOCKED_TYPE = "blocked_path"
FROZEN_TYPE = "frozen_sensor"


def _profile(**overrides):
    profile = {
        "type": FAULT_TYPE,
        "connection_id": "cabin_a_to_processing",
        "start_tick": 20,
        "end_tick": 80,
        "end_effectiveness": 0.4,
    }
    profile.update(overrides)
    return profile


def _v8(doc: dict, profiles: list[dict] | None = None) -> dict:
    doc["version"] = 9
    doc["fault_profiles"] = [] if profiles is None else profiles
    return doc


def test_v8_parses_gradual_primary_fan_degradation_at_exact_boundaries(standard_doc):
    config = parse_scenario(_v8(standard_doc, [_profile()]))

    assert config.version == 9
    assert config.fault_profiles == (
        GradualPrimaryFanDegradation(
            connection_id="cabin_a_to_processing",
            start_tick=20,
            end_tick=80,
            end_effectiveness=0.4,
        ),
    )
    profile = config.fault_profiles[0]
    assert profile.effectiveness_at(19) == pytest.approx(1.0)
    assert profile.effectiveness_at(20) == pytest.approx(1.0)
    assert profile.effectiveness_at(50) == pytest.approx(0.7)
    assert profile.effectiveness_at(80) == pytest.approx(0.4)
    assert profile.effectiveness_at(120) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("profiles", "match"),
    [
        (None, "fault_profiles"),
        ({}, "must be a list"),
        ([{"type": FAULT_TYPE}], "missing required field"),
        ([_profile(type="instant_failure")], "unsupported fault profile type"),
        ([_profile(connection_id="missing")], "unknown connection"),
        ([_profile(connection_id="processing_to_cabin_a")], "metering path"),
        ([_profile(start_tick=0)], "start_tick.*positive"),
        ([_profile(end_tick=20)], "end_tick must be after start_tick"),
        ([_profile(end_effectiveness=1.0)], "end_effectiveness"),
        ([_profile(end_effectiveness=float("nan"))], "finite"),
        ([_profile(extra_field="not allowed")], "unexpected field"),
        ([_profile(), _profile(end_effectiveness=0.2)], "more than one fault profile"),
    ],
)
def test_v8_rejects_malformed_fault_profiles(standard_doc, profiles, match):
    standard_doc["version"] = 9
    if profiles is None:
        standard_doc.pop("fault_profiles")
    else:
        standard_doc["fault_profiles"] = profiles

    with pytest.raises(ValueError, match=match):
        parse_scenario(standard_doc)


def test_fault_target_must_be_the_outbound_loop_metering_connection(standard_doc):
    config = parse_scenario(_v8(standard_doc, [_profile()]))

    profile = config.fault_profiles[0]
    target = next(connection for connection in config.connections if connection.id == profile.connection_id)
    assert target.from_zone == "cabin_a"
    assert target.to_zone == config.processing_zone().id


def _blocked(**overrides):
    profile = {
        "type": BLOCKED_TYPE,
        "connection_id": "cabin_b_to_processing",
        "start_tick": 30,
        "blocked_effectiveness": 0.05,
    }
    profile.update(overrides)
    return profile


def _frozen(**overrides):
    profile = {
        "type": FROZEN_TYPE,
        "zone_id": "lab",
        "start_tick": 40,
    }
    profile.update(overrides)
    return profile


def test_v8_parses_blocked_path_with_step_semantics(standard_doc):
    config = parse_scenario(_v8(standard_doc, [_blocked()]))

    assert config.fault_profiles == (
        BlockedPath(
            connection_id="cabin_b_to_processing",
            start_tick=30,
            blocked_effectiveness=0.05,
        ),
    )
    profile = config.fault_profiles[0]
    assert profile.effectiveness_at(1) == pytest.approx(1.0)
    assert profile.effectiveness_at(29) == pytest.approx(1.0)
    assert profile.effectiveness_at(30) == pytest.approx(0.05)
    assert profile.effectiveness_at(120) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        (_blocked(blocked_effectiveness=1.0), "blocked_effectiveness"),
        (_blocked(blocked_effectiveness=-0.1), "blocked_effectiveness"),
        (_blocked(blocked_effectiveness=float("nan")), "finite"),
        (_blocked(end_tick=90), "unexpected field"),
        (_blocked(connection_id="processing_to_cabin_b"), "metering path"),
        (_blocked(connection_id="missing"), "unknown connection"),
        (_blocked(start_tick=0), "start_tick.*positive"),
    ],
)
def test_v8_rejects_malformed_blocked_path_profiles(standard_doc, profile, match):
    with pytest.raises(ValueError, match=match):
        parse_scenario(_v8(standard_doc, [profile]))


def test_v8_rejects_blocked_path_missing_required_field(standard_doc):
    profile = _blocked()
    profile.pop("blocked_effectiveness")
    with pytest.raises(ValueError, match="missing required field"):
        parse_scenario(_v8(standard_doc, [profile]))


def test_v8_rejects_two_connection_faults_on_one_connection_across_types(standard_doc):
    profiles = [_profile(), _blocked(connection_id="cabin_a_to_processing")]
    with pytest.raises(ValueError, match="more than one fault profile"):
        parse_scenario(_v8(standard_doc, profiles))


def test_v8_parses_frozen_sensor_with_membership_semantics(standard_doc):
    config = parse_scenario(_v8(standard_doc, [_frozen()]))

    assert config.fault_profiles == (
        FrozenSensor(zone_id="lab", start_tick=40),
    )
    profile = config.fault_profiles[0]
    assert not profile.is_frozen_at(1)
    assert not profile.is_frozen_at(39)
    assert profile.is_frozen_at(40)
    assert profile.is_frozen_at(120)


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        (_frozen(zone_id="missing"), "unknown zone"),
        (_frozen(zone_id="processing"), "non-processing"),
        (_frozen(start_tick=0), "start_tick.*positive"),
        (_frozen(end_tick=90), "unexpected field"),
    ],
)
def test_v8_rejects_malformed_frozen_sensor_profiles(standard_doc, profile, match):
    with pytest.raises(ValueError, match=match):
        parse_scenario(_v8(standard_doc, [profile]))


def test_v8_rejects_frozen_sensor_missing_required_field(standard_doc):
    profile = _frozen()
    profile.pop("zone_id")
    with pytest.raises(ValueError, match="missing required field"):
        parse_scenario(_v8(standard_doc, [profile]))


def test_v8_rejects_two_frozen_sensors_on_one_zone(standard_doc):
    profiles = [_frozen(), _frozen(start_tick=60)]
    with pytest.raises(ValueError, match="more than one fault profile"):
        parse_scenario(_v8(standard_doc, profiles))


def test_config_exposes_connection_and_sensor_fault_views(standard_doc):
    config = parse_scenario(_v8(standard_doc, [_profile(), _frozen()]))

    assert config.connection_faults() == (
        GradualPrimaryFanDegradation(
            connection_id="cabin_a_to_processing",
            start_tick=20,
            end_tick=80,
            end_effectiveness=0.4,
        ),
    )
    assert config.sensor_faults() == (
        FrozenSensor(zone_id="lab", start_tick=40),
    )
