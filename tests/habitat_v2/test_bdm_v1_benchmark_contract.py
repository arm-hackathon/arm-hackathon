from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aeolus.habitat_v2.bdm_v1_benchmark_contract import (
    BDM_V1_CATALOGUE_ACTIONS,
    BDM_V1_CONTRACT_FILENAME,
    BDM_V1_CONTRACT_ID,
    BDM_V1_DECISION_TARGETS,
    BDM_V1_DECISION_THRESHOLD_NAMES,
    BDM_V1_GROUPING_KEYS,
    BDM_V1_PARTITIONS,
    BDM_V1_PROHIBITED_INPUT_ITEMS,
    BDM_V1_ROSTER_ARM_IDS,
    BDM_V1_STOP_CRITERIA,
    BDM_V1_TBD_MARKER,
    BdmV1ContractError,
    declared_input_field_names,
    load_bdm_v1_benchmark_contract,
    threshold_is_frozen,
    validate_bdm_v1_benchmark_contract,
    validate_causal_window,
    validate_group_disjointness,
    validate_metrics_declared,
    validate_model_input_fields,
    validate_targets_declared,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract() -> dict:
    loaded, digest = load_bdm_v1_benchmark_contract(REPO_ROOT)
    assert len(digest) == 64
    return loaded


def test_contract_identity_and_file(contract: dict) -> None:
    assert contract["contract_id"] == BDM_V1_CONTRACT_ID
    assert contract["status"] == "ACCEPTED_RESEARCH_CONTRACT"
    assert (REPO_ROOT / "contracts" / BDM_V1_CONTRACT_FILENAME).is_file()
    assert contract["authorization"]["authorized_via_issue"] == 70


def test_evidence_status_matrix_covers_all_categories(contract: dict) -> None:
    matrix = contract["evidence_status_matrix"]
    for category in ("implemented", "historical", "proposed"):
        assert matrix[category], category
        for entry in matrix[category]:
            assert entry["id"] and entry["summary"] and entry["evidence"]
    assert matrix["not_claimed"]
    implemented_ids = {entry["id"] for entry in matrix["implemented"]}
    assert "issue_56_v4_line_concluded" in implemented_ids
    assert "issue_53_dropout_forecast_lane" in implemented_ids
    historical_ids = {entry["id"] for entry in matrix["historical"]}
    assert "closed_loop_advisory_campaign" in historical_ids


def test_every_field_record_carries_all_rules(contract: dict) -> None:
    names = set()
    for field_class in contract["input_schema"]["field_classes"]:
        assert field_class["class_id"] and field_class["description"]
        for record in field_class["fields"]:
            assert set(record) == {
                "name",
                "dtype",
                "shape",
                "unit",
                "timing",
                "missingness",
                "observability",
                "provenance",
            }
            assert record["name"] not in names
            names.add(record["name"])
    assert declared_input_field_names(contract) == frozenset(names)
    assert "zone_o2_mole_fraction" in names
    assert "steps_since_last_valid_observation" in names


def test_frozen_lists_are_exact(contract: dict) -> None:
    assert tuple(contract["prohibited_fields"]["items"]) == BDM_V1_PROHIBITED_INPUT_ITEMS
    assert tuple(contract["action_catalogue"]["actions"]) == BDM_V1_CATALOGUE_ACTIONS
    assert contract["action_catalogue"]["mutable_in_study"] is False
    assert contract["horizons"]["step_keys"] == [4, 16, 32]
    assert contract["horizons"]["episode_remaining_key"] == 0
    assert tuple(contract["labels"]["decision_targets"]) == BDM_V1_DECISION_TARGETS
    assert contract["labels"]["counterfactual_rollouts"] == "labels_only_never_runtime_inputs"
    arm_ids = tuple(arm["arm_id"] for arm in contract["comparison_roster"]["arms"])
    assert arm_ids == BDM_V1_ROSTER_ARM_IDS
    assert tuple(contract["split_custody"]["grouping_keys"]) == BDM_V1_GROUPING_KEYS
    assert set(contract["split_custody"]["partitions"]) == set(BDM_V1_PARTITIONS)
    assert tuple(contract["stop_criteria"]) == BDM_V1_STOP_CRITERIA


def test_statistical_unit_is_causal_group(contract: dict) -> None:
    metrics = contract["metrics"]
    assert metrics["independent_statistical_unit"] == "causal_group"
    assert metrics["bootstrap"]["unit"] == "causal_group"
    assert set(metrics["non_independent_units"]) == {
        "paired_sensor_variants",
        "counterfactual_action_branches",
        "decision_steps_within_one_family",
    }
    assert metrics["aggregate_forecast_error_status"] == "secondary_diagnostic_only"


def test_thresholds_are_preregistered_or_tbd(contract: dict) -> None:
    thresholds = contract["thresholds"]
    assert thresholds["model_path_latency_p99_ms_maximum"] == 250.0
    decision = thresholds["decision_thresholds"]
    assert set(decision) == set(BDM_V1_DECISION_THRESHOLD_NAMES)
    for name, value in decision.items():
        assert value == BDM_V1_TBD_MARKER or isinstance(value, (int, float)), name
        assert not isinstance(value, bool)
    assert thresholds["pilot_definition"]["data"] == "DEV_partition_only"
    assert threshold_is_frozen(contract, "safety_non_inferiority_margin") is (
        decision["safety_non_inferiority_margin"] != BDM_V1_TBD_MARKER
    )
    with pytest.raises(BdmV1ContractError):
        threshold_is_frozen(contract, "not_a_threshold")


def test_validate_model_input_fields_enforcement(contract: dict) -> None:
    validate_model_input_fields(
        ["zone_temperature_k", "candidate_action_index"], contract
    )
    for prohibited in BDM_V1_PROHIBITED_INPUT_ITEMS:
        with pytest.raises(BdmV1ContractError, match="prohibited"):
            validate_model_input_fields([prohibited], contract)
    with pytest.raises(BdmV1ContractError, match="undeclared"):
        validate_model_input_fields(["future_o2_reading"], contract)
    with pytest.raises(BdmV1ContractError, match="undeclared"):
        validate_model_input_fields(["fault_effectiveness_multiplier"], contract)


def test_validate_causal_window_enforcement(contract: dict) -> None:
    validate_causal_window(list(range(1, 17)), 16, contract)
    validate_causal_window([12, 16], 16, contract)
    with pytest.raises(BdmV1ContractError, match="future step"):
        validate_causal_window([16, 17], 16, contract)
    with pytest.raises(BdmV1ContractError, match="exceeds"):
        validate_causal_window(list(range(0, 18)), 20, contract)
    with pytest.raises(BdmV1ContractError, match="history bound"):
        validate_causal_window([0, 16], 16, contract)
    with pytest.raises(BdmV1ContractError, match="at least one"):
        validate_causal_window([], 16, contract)
    with pytest.raises(BdmV1ContractError):
        validate_causal_window([4], -1, contract)


def test_validate_group_disjointness_enforcement() -> None:
    validate_group_disjointness(
        {
            "TRAIN": ["g0001", "g0002"],
            "DEV": ["g0003"],
            "CALIBRATION": ["g0004"],
            "BLIND_FINAL": ["g0005"],
        }
    )
    with pytest.raises(BdmV1ContractError, match="appears in both"):
        validate_group_disjointness({"TRAIN": ["g0001"], "DEV": ["g0001"]})
    with pytest.raises(BdmV1ContractError, match="appears in both"):
        validate_group_disjointness({"TRAIN": ["g0001", "g0001"]})
    with pytest.raises(BdmV1ContractError, match="unknown custody partitions"):
        validate_group_disjointness({"HOLDOUT": ["g0001"]})
    with pytest.raises(BdmV1ContractError, match="non-empty strings"):
        validate_group_disjointness({"TRAIN": [""]})


def test_validate_targets_and_metrics_enforcement(contract: dict) -> None:
    validate_targets_declared(list(BDM_V1_DECISION_TARGETS) + ["trajectory_targets"], contract)
    with pytest.raises(BdmV1ContractError, match="undeclared label target"):
        validate_targets_declared(["future_exposure"], contract)
    validate_metrics_declared(
        ["safety_exposure", "action_ranking_quality"], contract
    )
    with pytest.raises(BdmV1ContractError, match="undeclared metric"):
        validate_metrics_declared(["vibes"], contract)


def _mutate(contract: dict) -> dict:
    return copy.deepcopy(contract)


def test_validator_rejects_identity_and_section_drift(contract: dict) -> None:
    drifted = _mutate(contract)
    drifted["schema_version"] = "something_else"
    with pytest.raises(BdmV1ContractError, match="identity"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["horizons"]["step_keys"] = [4, 8, 16]
    with pytest.raises(BdmV1ContractError, match="horizons"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["metrics"]["polarity"]["higher_is_better"].append("safety_exposure")
    drifted["metrics"]["polarity"]["lower_is_better"].remove("safety_exposure")
    with pytest.raises(BdmV1ContractError, match="metrics"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["split_custody"]["group_disjoint"] = False
    with pytest.raises(BdmV1ContractError, match="custody"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["split_custody"]["partitions"]["BLIND_FINAL"] = ""
    with pytest.raises(BdmV1ContractError, match="custody"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["prohibited_fields"]["items"] = drifted["prohibited_fields"]["items"][:-1]
    with pytest.raises(BdmV1ContractError, match="prohibited"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["comparison_roster"]["arms"] = drifted["comparison_roster"]["arms"][:5]
    with pytest.raises(BdmV1ContractError, match="roster"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["action_catalogue"]["actions"][3] = "normal-contingency-v2"
    with pytest.raises(BdmV1ContractError, match="action catalogue"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["stop_criteria"] = drifted["stop_criteria"][:4]
    with pytest.raises(BdmV1ContractError, match="stop criteria"):
        validate_bdm_v1_benchmark_contract(drifted)


def test_validator_rejects_threshold_and_field_record_drift(contract: dict) -> None:
    drifted = _mutate(contract)
    drifted["thresholds"]["model_path_latency_p99_ms_maximum"] = 500.0
    with pytest.raises(BdmV1ContractError, match="latency"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["thresholds"]["decision_thresholds"]["interval_coverage_minimum"] = "roughly_high"
    with pytest.raises(BdmV1ContractError, match="TBD_FROM_PILOT"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["thresholds"]["decision_thresholds"]["harmful_admission_rate_maximum"] = True
    with pytest.raises(BdmV1ContractError, match="numeric"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["input_schema"]["field_classes"][0]["fields"][0]["timing"] = "any_future_step"
    with pytest.raises(BdmV1ContractError, match="invalid timing"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    del drifted["input_schema"]["field_classes"][0]["fields"][0]["unit"]
    with pytest.raises(BdmV1ContractError, match="field set drifted"):
        validate_bdm_v1_benchmark_contract(drifted)

    drifted = _mutate(contract)
    drifted["input_schema"]["field_classes"][1]["fields"][0]["name"] = "zone_temperature_k"
    with pytest.raises(BdmV1ContractError, match="duplicated"):
        validate_bdm_v1_benchmark_contract(drifted)


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    raw = (REPO_ROOT / "contracts" / BDM_V1_CONTRACT_FILENAME).read_text(encoding="utf-8")
    injected = raw.replace(
        '"contract_id": "habitat_v2_bdm_v1_benchmark_contract_v1",',
        '"contract_id": "habitat_v2_bdm_v1_benchmark_contract_v1",\n  "contract_id": "other",',
        1,
    )
    (contracts_dir / BDM_V1_CONTRACT_FILENAME).write_text(injected, encoding="utf-8")
    with pytest.raises(BdmV1ContractError, match="duplicate contract key"):
        load_bdm_v1_benchmark_contract(tmp_path)


def test_loader_rejects_missing_contract(tmp_path: Path) -> None:
    (tmp_path / "contracts").mkdir()
    with pytest.raises(BdmV1ContractError, match="unreadable"):
        load_bdm_v1_benchmark_contract(tmp_path)


def test_claim_boundaries_keep_advisory_language(contract: dict) -> None:
    boundaries = json.dumps(contract["claim_boundaries"])
    assert "advisory-only" in boundaries
    assert "development evidence" in boundaries
    assert "published unchanged" in boundaries
