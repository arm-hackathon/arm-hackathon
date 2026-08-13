from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from aeolus.habitat_v2.hmc import HabitatManagementComputer, HMCResetValidationError
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.scenario import Scenario, derive_run_id


def _checked_v5_scenario() -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    return Scenario.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _checked_hmc_contract():
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def _self_consistent_forged_scenario() -> Scenario:
    source = _checked_v5_scenario()
    data = deepcopy(dict(source.data))
    data["undeclared_reset_probe"] = {"looks_consistent": True}
    canonical = json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    scenario_sha256 = hashlib.sha256(canonical).hexdigest()
    run_id = derive_run_id(
        scenario_sha256=scenario_sha256,
        scenario_schema_version=source.scenario_schema_version,
        trace_schema_version=source.trace_schema_version,
        equation_contract_revision=source.equation_contract_revision,
        actuator_feedback_contract_revision=source.actuator_feedback_contract_revision,
    )
    forged = Scenario(
        data=data,
        canonical_bytes=canonical,
        scenario_sha256=scenario_sha256,
        scenario_schema_version=source.scenario_schema_version,
        trace_schema_version=source.trace_schema_version,
        equation_contract_revision=source.equation_contract_revision,
        run_id=run_id,
        actuator_feedback_contract_revision=source.actuator_feedback_contract_revision,
    )
    forged.validate_contract_identities()
    return forged


def test_reset_rejects_self_consistent_forged_scenario_before_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aeolus.habitat_v2.hmc as hmc_module

    def physics_must_not_run(_scenario: Scenario):
        raise AssertionError("initial_state reached before closed-schema rejection")

    monkeypatch.setattr(hmc_module, "initial_state", physics_must_not_run)

    with pytest.raises(HMCResetValidationError, match="closed scenario schema"):
        HabitatManagementComputer.reset(
            _self_consistent_forged_scenario(),
            _checked_hmc_contract(),
            b"r" * 32,
        )


@pytest.mark.parametrize(
    "reset_nonce",
    [None, "00" * 32, bytearray(32), b"", b"x" * 31, b"x" * 33],
)
def test_reset_rejects_every_non_exact_32_byte_nonce_before_physics(
    reset_nonce: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aeolus.habitat_v2.hmc as hmc_module

    def physics_must_not_run(_scenario: Scenario):
        raise AssertionError("initial_state reached before nonce rejection")

    monkeypatch.setattr(hmc_module, "initial_state", physics_must_not_run)

    with pytest.raises(HMCResetValidationError, match="exact bytes of length 32"):
        HabitatManagementComputer.reset(
            _checked_v5_scenario(),
            _checked_hmc_contract(),
            reset_nonce,  # type: ignore[arg-type]
        )


def test_reset_uses_a_newly_reparsed_scenario_for_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aeolus.habitat_v2.hmc as hmc_module

    supplied = _checked_v5_scenario()
    parsed_scenarios: list[Scenario] = []
    real_initial_state = hmc_module.initial_state

    def capture_initial_state(scenario: Scenario):
        parsed_scenarios.append(scenario)
        return real_initial_state(scenario)

    monkeypatch.setattr(hmc_module, "initial_state", capture_initial_state)

    hmc = HabitatManagementComputer.reset(
        supplied,
        _checked_hmc_contract(),
        b"r" * 32,
    )

    assert len(parsed_scenarios) == 1
    assert parsed_scenarios[0] is not supplied
    assert hmc._scenario is parsed_scenarios[0]
    assert hmc._scenario.canonical_bytes == supplied.canonical_bytes


def _domain_hash(label: str, *parts: str | bytes) -> str:
    payload = bytearray(label.encode("utf-8"))
    for part in parts:
        payload.extend(part if isinstance(part, bytes) else bytes.fromhex(part))
    return hashlib.sha256(payload).hexdigest()


def test_reset_derives_deterministic_run_epoch_and_topology_identities_without_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aeolus.habitat_v2.physics as physics_module

    scenario = _checked_v5_scenario()
    contract = _checked_hmc_contract()

    def timeline_must_not_be_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reset identity reached the scenario timeline")

    monkeypatch.setattr(physics_module, "_segment_for_step", timeline_must_not_be_read)

    first = HabitatManagementComputer.reset(scenario, contract, b"a" * 32)
    replay = HabitatManagementComputer.reset(scenario, contract, b"a" * 32)
    distinct = HabitatManagementComputer.reset(scenario, contract, b"b" * 32)

    expected_run_id = _domain_hash(
        "aeolus-habitat-v2-hmc-run-v1",
        scenario.scenario_sha256,
        contract.hmc_contract_sha256,
        first.snapshot_schema_sha256,
        first.observable_topology_sha256,
        b"a" * 32,
    )
    expected_epoch = _domain_hash(
        "aeolus-habitat-v2-hmc-epoch-v1",
        expected_run_id,
        b"a" * 32,
    )

    assert first.control_run_id == replay.control_run_id == expected_run_id
    assert first.authority_epoch == replay.authority_epoch == expected_epoch
    assert first.observable_topology_sha256 == replay.observable_topology_sha256
    assert first.control_run_id != distinct.control_run_id
    assert first.authority_epoch != distinct.authority_epoch
    assert first.lifecycle_phase == "RESET"


def test_reset_rejects_valid_v5_scenario_with_unreviewed_feedback_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aeolus.habitat_v2.hmc as hmc_module

    mapping = json.loads(
        (
            Path(__file__).parents[2]
            / "scenarios"
            / "habitat_v2_actuator_feedback.json"
        ).read_text(encoding="utf-8")
    )
    mapping["actuator_feedback"]["feedback_sensor_noise_amplitude"] = 0.0005
    scenario = Scenario.from_mapping(mapping)

    def physics_must_not_run(_scenario: Scenario) -> object:
        raise AssertionError("unreviewed noise reached plant construction")

    monkeypatch.setattr(hmc_module, "initial_state", physics_must_not_run)

    with pytest.raises(HMCResetValidationError, match="feedback sensor noise"):
        HabitatManagementComputer.reset(
            scenario,
            _checked_hmc_contract(),
            b"n" * 32,
        )
