from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from aeolus.habitat_v2.hmc_contract import (
    HMCContract,
    HMCContractError,
    load_hmc_contract,
)


def test_checked_in_hmc_contract_has_canonical_identity() -> None:
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"

    contract = load_hmc_contract(path)

    assert type(contract) is HMCContract
    assert contract.schema_version == "aeolus_habitat_v2_hmc_contract_v1"
    assert (
        contract.snapshot_schema_version == "aeolus_habitat_v2_operational_snapshot_v1"
    )
    assert contract.control_trace_schema_version == "aeolus_habitat_v2_control_trace_v1"
    assert (
        contract.hmc_contract_sha256
        == hashlib.sha256(contract.canonical_bytes).hexdigest()
    )
    assert contract.canonical_bytes == path.read_bytes().rstrip(b"\n")

    identities = (
        contract.snapshot_schema_sha256,
        contract.snapshot_verification_contract_sha256,
        contract.external_command_contract_sha256,
        contract.preflight_contract_sha256,
        contract.health_policy_sha256,
        contract.safety_policy_sha256,
        contract.safe_action_catalogue_sha256,
        contract.proposal_receipt_schema_sha256,
        contract.arbitration_receipt_schema_sha256,
        contract.step_receipt_schema_sha256,
        contract.terminal_receipt_schema_sha256,
        contract.control_trace_schema_sha256,
    )
    assert all(len(identity) == 64 for identity in identities)
    assert all(identity == identity.lower() for identity in identities)


def _checked_mapping() -> dict[str, object]:
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_hmc_contract_rejects_unknown_nested_policy_fields() -> None:
    forged = deepcopy(_checked_mapping())
    forged["health_policy"]["unexpected"] = True  # type: ignore[index]

    with pytest.raises(HMCContractError, match="health_policy fields"):
        HMCContract.from_mapping(forged)


def test_hmc_contract_binds_reviewed_v5_noise_and_excludes_unsafe_tracking() -> None:
    contract = HMCContract.from_mapping(_checked_mapping())

    assert (
        contract.reviewed_noise_configuration["feedback_sensor_noise_amplitude"]
        == 0.001
    )
    assert (
        contract.reviewed_noise_configuration["primary_noise_amplitude"]["co2_ppm"]
        == 1.5
    )
    assert (
        contract.reviewed_noise_configuration["secondary_noise_amplitude"]["co2_ppm"]
        == 2.25
    )
    assert contract.tracked_actuator_channels == (
        "fan_speed_fraction",
        "damper_position_by_id",
        "cooling_delivery_w",
    )
    assert "oxygen_delivery_mol_s" not in contract.tracked_actuator_channels
    assert "branch_airflow_m3_s" not in contract.tracked_actuator_channels


def test_hmc_contract_rejects_tracking_threshold_inside_noise_only_envelope() -> None:
    forged = deepcopy(_checked_mapping())
    forged["health_policy"]["tracking"]["fan_speed_fraction"]["warning_enter"] = 0.001  # type: ignore[index]

    with pytest.raises(HMCContractError, match="noise-only envelope"):
        HMCContract.from_mapping(forged)
