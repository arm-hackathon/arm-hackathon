"""V6 room-physics sweep contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.config import load_scenario
from aeolus.sweep_v6 import generate_v6_sweep, parse_v6_sweep_spec

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "scenarios" / "standard_habitat.json"


def _base_document(*, cabin_a_volume: float, cabin_b_capacity: float) -> dict:
    document = json.loads(STANDARD.read_text())
    zones = {zone["id"]: zone for zone in document["zones"]}
    zones["cabin_a"]["air_volume"] = cabin_a_volume
    for connection in document["connections"]:
        if connection["id"] in {"cabin_b_to_processing", "processing_to_cabin_b"}:
            connection["max_airflow"] = cabin_b_capacity
    return document


def _write_bases(tmp_path: Path) -> None:
    variants = {
        "room-balanced.json": _base_document(cabin_a_volume=100.0, cabin_b_capacity=10.0),
        "room-volume-asymmetric.json": _base_document(cabin_a_volume=70.0, cabin_b_capacity=7.0),
        "room-capacity-constrained.json": _base_document(cabin_a_volume=90.0, cabin_b_capacity=8.0),
        "room-transition-heavy.json": _base_document(cabin_a_volume=130.0, cabin_b_capacity=12.0),
    }
    for name, document in variants.items():
        (tmp_path / name).write_text(json.dumps(document), encoding="utf-8")


def _profile(profile_id: str) -> dict:
    return {
        "id": profile_id,
        "source_multiplier": 1.0,
        "shared_airflow_capacity": 24.0,
        "telemetry": {
            "airflow_noise_fraction": 0.01,
            "airflow_bias_fraction": 0.0,
            "airflow_drift_fraction": 0.0,
            "actuator_position_noise_fraction": 0.0,
            "co2_sensor_noise_fraction": 0.01,
            "co2_sensor_bias_fraction": 0.0,
            "co2_sensor_drift_fraction": 0.0,
        },
    }


def _family(room_family_id: str, role: str, base_scenario: str, seed: int) -> dict:
    return {
        "id": room_family_id,
        "role": role,
        "base_scenario": base_scenario,
        "seeds": [seed],
        "fault_start_ticks": [30],
        "operating_profiles": [_profile(f"{room_family_id}-nominal")],
        "gradual_profiles": [{"duration_ticks": 20, "end_effectiveness": 0.75}],
        "blocked_effectiveness": [0.65],
    }


def _document() -> dict:
    return {
        "schema_version": "aeolus_sweep_v6",
        "suite_role": "development",
        "targets": ["cabin_a"],
        "room_families": [
            _family("room-balanced", "fit", "room-balanced.json", 2100),
            _family("room-volume-asymmetric", "fit", "room-volume-asymmetric.json", 2110),
            _family("room-capacity-constrained", "calibration", "room-capacity-constrained.json", 2120),
            _family("room-transition-heavy", "validation", "room-transition-heavy.json", 2300),
        ],
    }


def test_v6_requires_complete_disjoint_room_family_roles_and_shared_context(tmp_path: Path):
    _write_bases(tmp_path)
    document = _document()

    spec = parse_v6_sweep_spec(document, source_path=tmp_path / "sweep-v6.json")

    assert tuple(item.room_family_id for item in spec.room_families) == (
        "room-balanced",
        "room-capacity-constrained",
        "room-transition-heavy",
        "room-volume-asymmetric",
    )
    assert {item.role for item in spec.room_families} == {"fit", "calibration", "validation"}
    assert len({item.context_metadata["selector_sha256"] for item in spec.room_families}) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda document: document["room_families"][1].__setitem__("id", "room-balanced"),
            "duplicate room family",
        ),
        (
            lambda document: document["room_families"][2].__setitem__("seeds", [2100]),
            "seed cluster",
        ),
        (
            lambda document: document["room_families"][3].__setitem__("role", "fit"),
            "validation",
        ),
    ),
)
def test_v6_rejects_room_family_or_role_leakage(tmp_path: Path, mutate, message: str):
    _write_bases(tmp_path)
    document = _document()
    mutate(document)

    with pytest.raises(ValueError, match=message):
        parse_v6_sweep_spec(document, source_path=tmp_path / "sweep-v6.json")


def test_v6_accepts_room_base_scenarios_under_a_relative_child_directory(tmp_path: Path):
    scenario_dir = tmp_path / "v6"
    scenario_dir.mkdir()
    _write_bases(scenario_dir)
    document = _document()
    for family in document["room_families"]:
        family["base_scenario"] = f"v6/{family['base_scenario']}"

    parsed = parse_v6_sweep_spec(document, source_path=tmp_path / "sweep-v6.json")

    assert all(item.base_scenario_path.parent == scenario_dir for item in parsed.room_families)


def test_v6_generation_preserves_base_physics_in_every_paired_fault(tmp_path: Path):
    _write_bases(tmp_path)
    spec_path = tmp_path / "sweep-v6.json"
    spec_path.write_text(json.dumps(_document()), encoding="utf-8")

    receipt = generate_v6_sweep(spec_path, tmp_path / "generated")

    assert receipt["families_by_role"] == {"calibration": 3, "fit": 6, "validation": 3}
    manifest = json.loads((tmp_path / "generated" / "families-v6.json").read_text())
    assert {entry["room_family_id"] for entry in manifest["families"]} == {
        "room-balanced",
        "room-volume-asymmetric",
        "room-capacity-constrained",
        "room-transition-heavy",
    }
    for entry in manifest["families"]:
        reference = json.loads((tmp_path / "generated" / entry["reference_scenario"]).read_text())
        fault = json.loads((tmp_path / "generated" / entry["fault_scenario"]).read_text())
        assert fault["fault_profiles"]
        fault["fault_profiles"] = []
        assert fault == reference
        load_scenario(tmp_path / "generated" / entry["reference_scenario"])
        load_scenario(tmp_path / "generated" / entry["fault_scenario"])
