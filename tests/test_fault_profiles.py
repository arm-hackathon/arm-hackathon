"""Schema-v7 fault-profile contracts for the converged simulator."""

from __future__ import annotations

import pytest

from icarus.config import GradualPrimaryFanDegradation, parse_scenario


FAULT_TYPE = "gradual_primary_fan_degradation"


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


def _v7(doc: dict, profiles: list[dict] | None = None) -> dict:
    doc["version"] = 7
    doc["fault_profiles"] = [] if profiles is None else profiles
    return doc


def test_v7_parses_gradual_primary_fan_degradation_at_exact_boundaries(standard_doc):
    config = parse_scenario(_v7(standard_doc, [_profile()]))

    assert config.version == 7
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
def test_v7_rejects_malformed_fault_profiles(standard_doc, profiles, match):
    standard_doc["version"] = 7
    if profiles is None:
        standard_doc.pop("fault_profiles")
    else:
        standard_doc["fault_profiles"] = profiles

    with pytest.raises(ValueError, match=match):
        parse_scenario(standard_doc)


def test_fault_target_must_be_the_outbound_loop_metering_connection(standard_doc):
    config = parse_scenario(_v7(standard_doc, [_profile()]))

    profile = config.fault_profiles[0]
    target = next(connection for connection in config.connections if connection.id == profile.connection_id)
    assert target.from_zone == "cabin_a"
    assert target.to_zone == config.processing_zone().id
