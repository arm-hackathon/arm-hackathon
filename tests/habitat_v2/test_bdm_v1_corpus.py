from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.bdm_v1_benchmark_contract import load_bdm_v1_benchmark_contract
from aeolus.habitat_v2.bdm_v1_corpus import (
    BDM_V1_CORPUS_SCHEMA_VERSION,
    FEATURE_FIELD_NAMES,
    LABEL_FIELD_NAMES,
    WINDOW_STEPS,
    BdmV1CorpusError,
    CorpusFamilyMeta,
    collect_family_samples,
    corpus_feature_manifest,
    validate_sample_against_contract,
)
from aeolus.habitat_v2.bdm_v1_families import (
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_corpus_protocol_v1.json"
REGISTRY_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json"


@pytest.fixture(scope="module")
def bundle():
    return load_forecast_contracts(REPO_ROOT)


@pytest.fixture(scope="module")
def contract():
    contract_data, _ = load_bdm_v1_benchmark_contract(REPO_ROOT)
    return contract_data


@pytest.fixture(scope="module")
def base_and_manifest():
    base_data, _ = load_base_scenario_data(REPO_ROOT)
    manifest, _ = load_physics_provenance_manifest(REPO_ROOT)
    return base_data, manifest


def _meta(definition) -> CorpusFamilyMeta:
    return CorpusFamilyMeta(
        family_id=definition.family_id,
        group_id=definition.group_id,
        group_index=definition.group_index,
        partition=definition.partition,
        template_id=definition.template_id,
        stratum=definition.stratum,
        sensor_variant=definition.sensor_variant,
        scenario_sha256=definition.scenario_sha256,
    )


@pytest.fixture(scope="module")
def healthy_samples(bundle, contract, base_and_manifest):
    base_data, manifest = base_and_manifest
    config = GeneratorConfig(
        groups_per_partition={"TRAIN": 1, "DEV": 0, "CALIBRATION": 0, "BLIND_FINAL": 0}
    )
    plans = assign_partitions(config)
    definition, scenario = build_family(config, plans[0], 0, 0, base_data, manifest)
    samples = collect_family_samples(bundle, scenario, _meta(definition), decision_steps=(16,))
    for sample in samples:
        validate_sample_against_contract(contract, sample)
    return samples


def test_feature_names_match_contract_exactly(healthy_samples, contract) -> None:
    from aeolus.habitat_v2.bdm_v1_benchmark_contract import declared_input_field_names

    sample = healthy_samples[0]
    assert set(sample["features"].keys()) == set(FEATURE_FIELD_NAMES)
    assert set(FEATURE_FIELD_NAMES) == set(declared_input_field_names(contract))


def test_window_is_causal_and_contiguous(healthy_samples) -> None:
    for sample in healthy_samples:
        window = sample["window_completed_steps"]
        assert window == list(range(1, WINDOW_STEPS + 1))
        assert window[-1] == sample["decision_step"] == 16


def test_healthy_family_is_fully_observed(healthy_samples) -> None:
    sample = healthy_samples[0]
    mask = sample["features"]["observed_value_mask"]
    stale = sample["features"]["steps_since_last_valid_observation"]
    assert all(flag for step in mask for zone in step for flag in zone)
    assert all(value == 0 for step in stale for zone in step for value in zone)


def test_stuck_family_freezes_observed_channel(bundle, contract, base_and_manifest) -> None:
    base_data, manifest = base_and_manifest
    config = GeneratorConfig()
    candidates = [
        plan for plan in assign_partitions(config)
        if plan.template.template_id
        in ("t08_sensor_defect_bundle", "t10_compound_physical_sensor")
    ]
    chosen = None
    for plan in candidates:
        for variant_index in (0, 1):
            definition, scenario = build_family(
                config, plan, variant_index, 0, base_data, manifest
            )
            profiles = definition.generator_draws["sensor_defect_profiles"]
            stuck = [
                profile
                for profile in profiles
                if profile["type"] == "sensor_stuck"
                and profile.get("sensor_head") == "primary"
                and 16 < int(profile["start_step"]) <= 49
            ]
            if stuck:
                chosen = (definition, scenario, stuck[0])
                break
        if chosen is not None:
            break
    assert chosen is not None, "no sensor-defect family draws a stuck window inside the corpus range"
    definition, scenario, profile = chosen
    onset = int(profile["start_step"])
    decision = onset + 14
    samples = collect_family_samples(
        bundle, scenario, _meta(definition), decision_steps=(decision,)
    )
    for sample in samples:
        validate_sample_against_contract(contract, sample)
    zone_ids = sample["features"]["topology_configuration_descriptor"]["zone_order"]
    zone_index = zone_ids.index(str(profile["zone_id"]))
    channel_index = (
        "temperature_k",
        "pressure_pa",
        "co2_ppm",
        "o2_mole_fraction",
        "relative_humidity",
    ).index(str(profile["channel"]))
    onset_row = onset - (decision - 15)
    env_field = (
        "zone_temperature_k",
        "zone_pressure_pa",
        "zone_co2_ppm",
        "zone_o2_mole_fraction",
        "zone_relative_humidity",
    )[channel_index]
    column = [step_row[zone_index] for step_row in samples[0]["features"][env_field]]
    frozen = column[onset_row]
    assert all(value == frozen for value in column[onset_row:])
    assert onset_row >= 1
    mask = samples[0]["features"]["observed_value_mask"]
    assert all(mask[row][zone_index][channel_index] for row in range(16))


def test_action_minus_hold_is_consistent(healthy_samples) -> None:
    for sample in healthy_samples:
        for name in LABEL_FIELD_NAMES:
            delta = float(sample["action_minus_hold"][name])
            direct = float(sample["labels"]["decision_targets"][name]) - float(
                sample["hold_reference"]["decision_targets"][name]
            )
            assert abs(delta - direct) < 1e-3


def test_collection_is_deterministic(bundle, base_and_manifest) -> None:
    base_data, manifest = base_and_manifest
    config = GeneratorConfig(
        groups_per_partition={"TRAIN": 1, "DEV": 0, "CALIBRATION": 0, "BLIND_FINAL": 0}
    )
    plans = assign_partitions(config)
    digests = []
    for _ in range(2):
        definition, scenario = build_family(config, plans[0], 0, 0, base_data, manifest)
        samples = collect_family_samples(bundle, scenario, _meta(definition), decision_steps=(16,))
        digests.append(tuple(sample["sample_sha256"] for sample in samples))
    assert digests[0] == digests[1]


def test_blind_partition_is_rejected() -> None:
    with pytest.raises(BdmV1CorpusError, match="sealed fail-closed"):
        CorpusFamilyMeta(
            family_id="bdm-v1-f0000-a",
            group_id="bdmv1g0000",
            group_index=0,
            partition="BLIND_FINAL",
            template_id="t01_healthy_steady",
            stratum="no_action",
            sensor_variant="a",
            scenario_sha256="0" * 64,
        )


def test_decision_step_bounds_are_enforced(bundle, base_and_manifest) -> None:
    base_data, manifest = base_and_manifest
    config = GeneratorConfig(
        groups_per_partition={"TRAIN": 1, "DEV": 0, "CALIBRATION": 0, "BLIND_FINAL": 0}
    )
    plans = assign_partitions(config)
    definition, scenario = build_family(config, plans[0], 0, 0, base_data, manifest)
    with pytest.raises(BdmV1CorpusError, match="complete causal window"):
        collect_family_samples(bundle, scenario, _meta(definition), decision_steps=(8,))
    with pytest.raises(BdmV1CorpusError, match="full rollout"):
        collect_family_samples(bundle, scenario, _meta(definition), decision_steps=(70,))


def test_validate_rejects_tampered_samples(healthy_samples, contract) -> None:
    import copy

    tampered = copy.deepcopy(healthy_samples[0])
    tampered["window_completed_steps"] = list(range(2, WINDOW_STEPS + 2))
    with pytest.raises(BdmV1CorpusError, match="past its decision step"):
        validate_sample_against_contract(contract, tampered)
    tampered = copy.deepcopy(healthy_samples[0])
    tampered["features"]["steps_since_last_valid_observation"][0][0][0] = 3
    with pytest.raises(BdmV1CorpusError, match="zero staleness"):
        validate_sample_against_contract(contract, tampered)
    tampered = copy.deepcopy(healthy_samples[0])
    del tampered["labels"]["decision_targets"]["safety_exposure"]
    with pytest.raises(BdmV1CorpusError, match="missing label"):
        validate_sample_against_contract(contract, tampered)


def test_feature_manifest_binds_catalogue(bundle) -> None:
    manifest = corpus_feature_manifest(bundle)
    assert manifest["schema_version"] == BDM_V1_CORPUS_SCHEMA_VERSION
    assert manifest["catalogue_binding"]["ordering"] == [
        action.action_id for action in bundle.actions
    ]
    assert "counterfactual_outcomes_as_features" in manifest["prohibited_inputs"]


def test_protocol_contract_binds_live_identities() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_bytes())
    registry_raw = REGISTRY_PATH.read_bytes()
    base_data, base_sha = load_base_scenario_data(REPO_ROOT)
    manifest, manifest_sha = load_physics_provenance_manifest(REPO_ROOT)
    bindings = protocol["bindings"]
    assert bindings["custody_registry_sha256"] == hashlib.sha256(registry_raw).hexdigest()
    assert bindings["base_scenario_sha256"] == base_sha
    assert bindings["provenance_manifest_sha256"] == manifest_sha
    assert protocol["corpus_schema_version"] == BDM_V1_CORPUS_SCHEMA_VERSION
    assert protocol["partitions"] == ["TRAIN", "DEV", "CALIBRATION"]


def test_builder_rejects_output_outside_out(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_bdm_v1_corpus", REPO_ROOT / "scripts" / "build_bdm_v1_corpus.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(module.CorpusBuildError, match="ignored out/"):
        module._resolve_output(tmp_path / "corpus")


def test_load_partition_samples_deterministic_and_complete() -> None:
    from aeolus.habitat_v2.bdm_v1_corpus import load_partition_samples

    corpus = REPO_ROOT / "out" / "bdm-v1-corpus-v1"
    if not corpus.is_dir():
        pytest.skip("local corpus artifact not present")
    train = load_partition_samples(corpus, "TRAIN")
    again = load_partition_samples(corpus, "TRAIN")
    assert [sample["sample_sha256"] for sample in train] == [
        sample["sample_sha256"] for sample in again
    ]
    assert len(train) == 4160
    keys = [
        (sample["family_id"], sample["decision_step"], sample["features"]["candidate_action_index"])
        for sample in train
    ]
    assert keys == sorted(keys)
    dev = load_partition_samples(corpus, "DEV")
    cal = load_partition_samples(corpus, "CALIBRATION")
    assert len(dev) == 1664
    assert len(cal) == 1248


def test_load_partition_samples_fail_closed(tmp_path: Path) -> None:
    from aeolus.habitat_v2.bdm_v1_corpus import load_partition_samples

    with pytest.raises(BdmV1CorpusError, match="shards missing"):
        load_partition_samples(tmp_path, "TRAIN")
    corpus = REPO_ROOT / "out" / "bdm-v1-corpus-v1"
    if not corpus.is_dir():
        pytest.skip("local corpus artifact not present")
    with pytest.raises(BdmV1CorpusError, match="no BLIND_FINAL samples"):
        load_partition_samples(corpus, "BLIND_FINAL")
