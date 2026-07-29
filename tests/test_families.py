"""Gate-2 scenario-family manifest contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icarus.config import load_scenario
from icarus.families import load_family_manifest, parse_family_manifest
from icarus.model_input import build_model_input_contract, model_artifact_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "scenarios"
FAMILY_MANIFEST_PATH = SCENARIOS / "families.json"
HIGH_DEMAND_PATH = SCENARIOS / "high_demand_healthy.json"


def _manifest_document() -> dict:
    return json.loads(FAMILY_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_load_family_manifest_is_canonical_and_bound_to_gate_one_contract():
    manifest = load_family_manifest(FAMILY_MANIFEST_PATH)

    assert tuple(family.family_id for family in manifest.families) == (
        "blocked-path-v1",
        "degradation-primary-fan-v1",
        "frozen-sensor-v1",
    )
    assert manifest.contract_metadata == model_artifact_metadata(
        build_model_input_contract(load_scenario(HIGH_DEMAND_PATH))
    )
    assert len(manifest.manifest_sha256) == 64
    assert manifest.canonical_json == json.dumps(
        json.loads(manifest.canonical_json),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def test_family_manifest_hash_is_independent_of_source_family_order():
    document = _manifest_document()
    reversed_document = {**document, "families": list(reversed(document["families"]))}

    original = parse_family_manifest(document, base_dir=SCENARIOS)
    reordered = parse_family_manifest(reversed_document, base_dir=SCENARIOS)

    assert reordered.canonical_json == original.canonical_json
    assert reordered.manifest_sha256 == original.manifest_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda document: document.__setitem__("unexpected", "value"),
            "unknown fields",
        ),
        (
            lambda document: document.__setitem__("selector_sha256", "0" * 64),
            "does not match",
        ),
        (
            lambda document: document["families"][0].__setitem__("split", "random"),
            "split",
        ),
    ),
)
def test_family_manifest_rejects_malformed_top_level_contract(mutation, message):
    document = _manifest_document()
    mutation(document)

    with pytest.raises(ValueError, match=message):
        parse_family_manifest(document, base_dir=SCENARIOS)


def test_family_manifest_rejects_duplicate_family_identity():
    document = _manifest_document()
    document["families"][1]["family_id"] = document["families"][0]["family_id"]

    with pytest.raises(ValueError, match="duplicate family_id"):
        parse_family_manifest(document, base_dir=SCENARIOS)


def test_family_manifest_rejects_faulted_reference_scenario():
    document = _manifest_document()
    document["families"][0]["reference_scenario"] = "blocked_path.json"

    with pytest.raises(ValueError, match="reference scenario must declare no fault profiles"):
        parse_family_manifest(document, base_dir=SCENARIOS)


def test_family_manifest_rejects_fault_class_that_disagrees_with_pair():
    document = _manifest_document()
    document["families"][0]["fault_class"] = "frozen_sensor"

    with pytest.raises(ValueError, match="fault_class does not match"):
        parse_family_manifest(document, base_dir=SCENARIOS)
