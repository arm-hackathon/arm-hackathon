from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.physics_provenance import (
    PHYSICS_PROVENANCE_FILENAME,
    PHYSICS_PROVENANCE_MANIFEST_ID,
    PROVENANCE_CLASSIFICATIONS,
    PhysicsProvenanceError,
    generator_variable_parameters,
    load_physics_provenance_manifest,
    parameter_by_id,
    parameters_for_system,
    sample_band,
    validate_physics_provenance_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def manifest() -> dict:
    loaded, digest = load_physics_provenance_manifest(REPO_ROOT)
    assert len(digest) == 64
    return loaded


def test_manifest_identity_and_coverage(manifest: dict) -> None:
    assert manifest["manifest_id"] == PHYSICS_PROVENANCE_MANIFEST_ID
    assert manifest["status"] == "ACCEPTED_PROVENANCE_MANIFEST"
    assert (REPO_ROOT / "contracts" / PHYSICS_PROVENANCE_FILENAME).is_file()
    assert manifest["authorization"]["authorized_via_issue"] == 71
    assert tuple(manifest["classifications"]) == PROVENANCE_CLASSIFICATIONS
    assert len(manifest["parameters"]) >= 90


def test_every_record_is_complete(manifest: dict) -> None:
    ids = set()
    for record in manifest["parameters"]:
        assert set(record) == {
            "parameter_id",
            "value",
            "unit",
            "valid_range",
            "classification",
            "citation",
            "source_path",
            "equation",
            "generator_variable",
            "uncertainty_distribution",
            "affected_systems",
            "affected_metrics",
        }
        assert record["parameter_id"] not in ids
        ids.add(record["parameter_id"])
        assert record["unit"]
        assert record["source_path"]
        if record["value"] is None:
            assert record["equation"]
        if record["classification"] in ("physical_constant", "public_requirement"):
            assert record["citation"]
        if record["generator_variable"]:
            assert record["uncertainty_distribution"] is not None
        else:
            assert record["uncertainty_distribution"] is None


def test_key_parameters_are_present_and_classified(manifest: dict) -> None:
    gas = parameter_by_id(manifest, "gas_constant_j_per_mol_k")
    assert gas["classification"] == "physical_constant"
    assert gas["value"] == pytest.approx(8.31446261815324, rel=1e-12)
    o2_bound = parameter_by_id(manifest, "oxygen_upper_mole_fraction_bound")
    assert o2_bound["classification"] == "public_requirement"
    assert o2_bound["value"] == 0.30
    assert "NASA-STD-3001" in o2_bound["citation"]
    fan_curve = parameter_by_id(manifest, "fan_pressure_curve")
    assert fan_curve["classification"] == "physics_derived"
    assert fan_curve["equation"]
    exchange = parameter_by_id(manifest, "well_mixed_exchange_fraction")
    assert "1 - exp(" in exchange["equation"]
    initial_o2 = parameter_by_id(manifest, "initial_o2_mole_fraction")
    assert "fixture revision 2" in initial_o2["citation"]


def test_helper_queries(manifest: dict) -> None:
    thermal = parameters_for_system(manifest, "thermal")
    assert thermal
    assert all("thermal" in record["affected_systems"] for record in thermal)
    gen = generator_variable_parameters(manifest)
    assert gen
    assert all(record["generator_variable"] for record in gen)
    assert all(record["uncertainty_distribution"] for record in gen)
    with pytest.raises(PhysicsProvenanceError, match="undeclared affected system"):
        parameters_for_system(manifest, "warp_core")
    with pytest.raises(PhysicsProvenanceError, match="unknown provenance parameter"):
        parameter_by_id(manifest, "not_a_parameter")


def test_sample_band_edges(manifest: dict) -> None:
    battery = parameter_by_id(manifest, "battery_capacity_wh")
    assert sample_band(battery, -1.0) == pytest.approx(14000.0)
    assert sample_band(battery, 0.0) == pytest.approx(20000.0)
    assert sample_band(battery, 1.0) == pytest.approx(26000.0)
    load_scale = parameter_by_id(manifest, "operating_load_scale_band")
    assert sample_band(load_scale, -1.0) == pytest.approx(0.75)
    assert sample_band(load_scale, 1.0) == pytest.approx(1.3)
    with pytest.raises(PhysicsProvenanceError, match=r"\[-1, 1\]"):
        sample_band(battery, 1.5)
    constant = parameter_by_id(manifest, "gas_constant_j_per_mol_k")
    with pytest.raises(PhysicsProvenanceError, match="no declared band"):
        sample_band(constant, 0.5)


def _mutate(manifest: dict) -> dict:
    return copy.deepcopy(manifest)


def test_validator_rejects_drift(manifest: dict) -> None:
    drifted = _mutate(manifest)
    drifted["manifest_id"] = "something_else"
    with pytest.raises(PhysicsProvenanceError, match="identity"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["parameters"][0]["citation"] = None
    with pytest.raises(PhysicsProvenanceError, match="requires a citation"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["parameters"][25]["value"] = 99.0
    with pytest.raises(PhysicsProvenanceError, match="outside its valid range"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["parameters"][30]["classification"] = "measured_flight_value"
    with pytest.raises(PhysicsProvenanceError, match="unknown classification"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["parameters"].append(copy.deepcopy(drifted["parameters"][0]))
    with pytest.raises(PhysicsProvenanceError, match="duplicated"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["parameters"][24]["generator_variable"] = True
    with pytest.raises(PhysicsProvenanceError):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    for record in drifted["parameters"]:
        if record["generator_variable"]:
            record["generator_variable"] = False
            break
    with pytest.raises(PhysicsProvenanceError, match="without generator variability"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    for record in drifted["parameters"]:
        if (
            record["generator_variable"]
            and record["uncertainty_distribution"]["kind"] == "uniform_relative"
        ):
            record["uncertainty_distribution"]["low"] = 0.2
            break
    with pytest.raises(PhysicsProvenanceError, match="straddle zero"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    for record in drifted["parameters"]:
        if (
            record["generator_variable"]
            and record["uncertainty_distribution"]["kind"] == "uniform_absolute"
            and record["valid_range"] is not None
        ):
            record["uncertainty_distribution"]["high"] = record["valid_range"][1] * 10.0
            break
    with pytest.raises(PhysicsProvenanceError, match="exceeds its valid range"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    for record in drifted["parameters"]:
        if (
            record["generator_variable"]
            and record["uncertainty_distribution"]["kind"] == "uniform_relative"
            and record["valid_range"] is not None
        ):
            record["uncertainty_distribution"]["high"] = 50.0
            break
    with pytest.raises(PhysicsProvenanceError, match="relative band exceeds"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["parameters"][40]["affected_systems"] = ["warp_core"]
    with pytest.raises(PhysicsProvenanceError, match="undeclared entries"):
        validate_physics_provenance_manifest(drifted)

    drifted = _mutate(manifest)
    drifted["conservation_tolerances"]["species_closure_mol"] = -1.0
    with pytest.raises(PhysicsProvenanceError, match="must be positive"):
        validate_physics_provenance_manifest(drifted)


def test_loader_rejects_duplicate_keys_and_missing_file(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    raw = (REPO_ROOT / "contracts" / PHYSICS_PROVENANCE_FILENAME).read_text(
        encoding="utf-8"
    )
    injected = raw.replace(
        '"manifest_id": "habitat_v2_physics_provenance_v1",',
        '"manifest_id": "habitat_v2_physics_provenance_v1",\n  "manifest_id": "other",',
        1,
    )
    (contracts_dir / PHYSICS_PROVENANCE_FILENAME).write_text(injected, encoding="utf-8")
    with pytest.raises(PhysicsProvenanceError, match="duplicate manifest key"):
        load_physics_provenance_manifest(tmp_path)

    empty = tmp_path / "empty"
    (empty / "contracts").mkdir(parents=True)
    with pytest.raises(PhysicsProvenanceError, match="unreadable"):
        load_physics_provenance_manifest(empty)


def test_manifest_json_is_canonical_text(manifest: dict) -> None:
    raw = (REPO_ROOT / "contracts" / PHYSICS_PROVENANCE_FILENAME).read_bytes()
    assert json.loads(raw) == json.loads(json.dumps(manifest))
