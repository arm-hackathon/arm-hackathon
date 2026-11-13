from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.bdm_v1_benchmark_contract import validate_group_disjointness
from aeolus.habitat_v2.bdm_v1_families import (
    BDM_V1_PARTITIONS_ORDERED,
    DECISION_STEPS,
    EPISODE_STEPS,
    GENERATOR_FAULT_DURATION_RANGE,
    GENERATOR_FAULT_MULTIPLIER_BANDS,
    GENERATOR_FAULT_ONSET_CLASSES,
    GENERATOR_INITIAL_O2_BANDS,
    GENERATOR_O2_EXCESS_THRESHOLD,
    GENERATOR_RAMP_SHAPES,
    GENERATOR_REGISTRY_SEED,
    GENERATOR_SENSOR_BIAS_BANDS,
    GENERATOR_SENSOR_BUNDLE_KINDS,
    GENERATOR_VERSION,
    TEMPLATES,
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    raw = REGISTRY_PATH.read_bytes()
    loaded = json.loads(raw)
    assert canonical_json_bytes(loaded) == raw
    return loaded


def test_registry_identity_and_bindings(registry: dict) -> None:
    assert registry["registry_id"] == "habitat_v2_bdm_v1_family_custody_v1"
    assert registry["status"] == "FROZEN_FAMILY_CUSTODY_REGISTRY"
    assert registry["authorization"]["authorized_via_issue"] == 72
    generator = registry["generator"]
    assert generator["generator_version"] == GENERATOR_VERSION
    assert generator["registry_seed"] == GENERATOR_REGISTRY_SEED
    assert generator["episode_steps"] == EPISODE_STEPS
    assert generator["decision_steps"] == list(DECISION_STEPS)
    base_data, base_sha = load_base_scenario_data(REPO_ROOT)
    assert generator["base_scenario_sha256"] == base_sha
    _manifest, manifest_sha = load_physics_provenance_manifest(REPO_ROOT)
    assert generator["provenance_manifest_sha256"] == manifest_sha


def test_registry_sizing_is_consistent(registry: dict) -> None:
    sizing = registry["sizing"]
    assert sizing["groups_per_partition"] == {
        "TRAIN": 40,
        "DEV": 16,
        "CALIBRATION": 12,
        "BLIND_FINAL": 12,
    }
    assert sizing["total_groups"] == 80
    assert sizing["total_families"] == 160
    groups = registry["groups"]
    assert len(groups) == 80
    counts = {partition: 0 for partition in BDM_V1_PARTITIONS_ORDERED}
    family_ids = set()
    for row in groups:
        counts[row["partition"]] += 1
        assert len(row["family_ids"]) == 2
        assert set(row["scenario_sha256"]) == {"a", "b"}
        for digest in row["scenario_sha256"].values():
            assert len(digest) == 64
        for family_id in row["family_ids"]:
            assert family_id not in family_ids
            family_ids.add(family_id)
    assert counts == {
        "TRAIN": 40,
        "DEV": 16,
        "CALIBRATION": 12,
        "BLIND_FINAL": 12,
    }
    assert len(family_ids) == 160


def test_registry_partitions_are_group_disjoint(registry: dict) -> None:
    partition_groups = {partition: [] for partition in BDM_V1_PARTITIONS_ORDERED}
    for row in registry["groups"]:
        partition_groups[row["partition"]].append(row["group_id"])
    validate_group_disjointness(partition_groups)


def test_registry_strata_and_template_coverage(registry: dict) -> None:
    strata = registry["strata_counts_by_partition"]
    expected_strata = {template.stratum for template in TEMPLATES}
    for partition in BDM_V1_PARTITIONS_ORDERED:
        assert set(strata[partition]) == expected_strata, partition
        assert all(count >= 1 for count in strata[partition].values())
    template_counts = registry["template_counts"]
    assert set(template_counts) == {template.template_id for template in TEMPLATES}
    assert sum(template_counts.values()) == 80


def test_registry_blind_seal(registry: dict) -> None:
    seal = registry["blind_seal"]
    assert seal["group_count"] == 12
    assert seal["family_count"] == 24
    assert len(seal["definitions_digest"]) == 64
    assert seal["outcome_status"] == "NOT_COMPUTED"
    assert seal["size_justification"] == "TBD_FROM_PILOT" or seal[
        "size_justification"
    ].startswith("pilot_receipt:")


def test_registry_declared_vocabularies_mirror_module(registry: dict) -> None:
    vocab = registry["declared_generator_vocabularies"]
    assert vocab["initial_oxygen_bands"] == {
        key: list(band) for key, band in GENERATOR_INITIAL_O2_BANDS.items()
    }
    assert vocab["o2_excess_threshold"] == GENERATOR_O2_EXCESS_THRESHOLD
    assert vocab["fault_multiplier_bands"] == {
        key: list(band) for key, band in GENERATOR_FAULT_MULTIPLIER_BANDS.items()
    }
    assert vocab["fault_onset_classes"] == {
        key: list(band) for key, band in GENERATOR_FAULT_ONSET_CLASSES.items()
    }
    assert vocab["fault_duration_range"] == list(GENERATOR_FAULT_DURATION_RANGE)
    assert vocab["ramp_shapes"] == list(GENERATOR_RAMP_SHAPES)
    assert vocab["sensor_bias_bands"] == {
        key: list(band) for key, band in GENERATOR_SENSOR_BIAS_BANDS.items()
    }
    assert vocab["sensor_bundle_kinds"] == list(GENERATOR_SENSOR_BUNDLE_KINDS)
    assert "sensor_noise_amplitudes" in vocab["excluded_from_drawing"]


def test_registry_digests_survive_regeneration(registry: dict) -> None:
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    config = GeneratorConfig()
    plans = {plan.group_index: plan for plan in assign_partitions(config)}
    by_partition: dict[str, dict] = {}
    for row in registry["groups"]:
        by_partition.setdefault(row["partition"], row)
    for partition in BDM_V1_PARTITIONS_ORDERED:
        row = by_partition[partition]
        plan = plans[row["group_index"]]
        for variant_index, variant in enumerate(("a", "b")):
            definition, _scenario = build_family(
                config, plan, variant_index, int(row["attempts"][variant]), base_data, manifest
            )
            assert definition.family_id == row["family_ids"][variant_index]
            assert definition.scenario_sha256 == row["scenario_sha256"][variant]
            assert definition.group_key == row["group_key"]


def test_registry_checks_are_all_passed(registry: dict) -> None:
    assert registry["checks"] == {
        "schema_validation": "passed",
        "feasibility_smoke_replay": "passed",
        "group_disjointness": "passed",
        "regeneration_digests": "passed",
    }


def test_generation_script_rejects_foreign_seed(tmp_path: Path) -> None:
    import importlib.util
    from aeolus.habitat_v2.bdm_v1_families import GeneratorConfig

    spec = importlib.util.spec_from_file_location(
        "generate_bdm_v1_families",
        REPO_ROOT / "scripts" / "generate_bdm_v1_families.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(module.GenerationError, match="declared registry seed"):
        module.generate_roster(tmp_path, config=GeneratorConfig(seed="wrong-seed"))
