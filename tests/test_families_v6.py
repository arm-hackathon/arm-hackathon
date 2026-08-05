"""V6 room-family manifest identity contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.families_v6 import load_v6_family_manifest
from aeolus.sweep_v6 import generate_v6_sweep, load_v6_sweep_spec

ROOT = Path(__file__).resolve().parents[1]
V6_SPEC = ROOT / "scenarios" / "sweep-v6-development.json"


def test_v6_manifest_binds_generated_scenarios_to_room_family_and_base_physics(tmp_path):
    generated = tmp_path / "generated"
    generate_v6_sweep(V6_SPEC, generated)
    spec = load_v6_sweep_spec(V6_SPEC)

    manifest = load_v6_family_manifest(
        generated / "families-v6.json", expected_sweep=spec
    )

    assert manifest.sweep_spec_sha256 == spec.sha256
    assert manifest.family_count == 144
    assert set(manifest.families_by_role) == {"fit", "calibration", "validation"}
    first = manifest.families[0]
    assert len(first.reference_scenario_sha256) == 64
    assert len(first.fault_scenario_sha256) == 64
    assert first.reference_scenario_sha256 != first.fault_scenario_sha256


def test_v6_manifest_rejects_a_changed_generated_scenario_even_when_names_match(tmp_path):
    generated = tmp_path / "generated"
    generate_v6_sweep(V6_SPEC, generated)
    spec = load_v6_sweep_spec(V6_SPEC)
    first = json.loads((generated / "families-v6.json").read_text())["families"][0]
    reference_path = generated / first["reference_scenario"]
    document = json.loads(reference_path.read_text())
    document["air_system"]["shared_airflow_capacity"] += 0.25
    reference_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="reference scenario digest"):
        load_v6_family_manifest(generated / "families-v6.json", expected_sweep=spec)
