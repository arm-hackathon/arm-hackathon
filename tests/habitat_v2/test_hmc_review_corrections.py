from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

import aeolus.habitat_v2.control_trace as control_trace_module
import aeolus.habitat_v2.hmc as hmc_module
import aeolus.habitat_v2.proposal as proposal_module
import aeolus.habitat_v2.safety as safety_module
import aeolus.habitat_v2.snapshot as snapshot_module
from aeolus.habitat_v2.health import (
    HealthReduction,
    HealthTracker,
    OperationalAlarm,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.hmc_contract import load_hmc_contract
from aeolus.habitat_v2.physics import (
    PreflightResult,
    StepResult,
    command_from_achieved_state,
    validate_external_command,
)
from aeolus.habitat_v2.scenario import Scenario


def _scenario(*, low_battery: bool = False) -> Scenario:
    path = Path(__file__).parents[2] / "scenarios" / "habitat_v2_actuator_feedback.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if low_battery:
        mapping["initial_utility"]["battery_energy_wh"] = 3800.0
    return Scenario.from_mapping(mapping)


def _contract():
    path = Path(__file__).parents[2] / "contracts" / "habitat_v2_hmc_v1.json"
    return load_hmc_contract(path)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _install_health(
    monkeypatch: pytest.MonkeyPatch, *, state: str, alarms: tuple[OperationalAlarm, ...]
) -> None:
    def fake_reduce_health(*, measurement, **_kwargs):
        return HealthReduction(
            health_state=state,
            alarms=alarms,
            tracker=HealthTracker(
                completed_step=measurement.completed_step,
                tracks=MappingProxyType({}),
            ),
        )

    monkeypatch.setattr(hmc_module, "reduce_health", fake_reduce_health)


def _valid_proposal(
    hmc: HabitatManagementComputer, snapshot, command: dict[str, object]
) -> dict[str, object]:
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": "external-adviser-v1",
        "source_type": "external_adviser",
        "completed_observation_step": snapshot.to_mapping()["completed_step"],
        "observation_snapshot_sha256": snapshot.snapshot_sha256,
        "requested_application_step": snapshot.to_mapping()["completed_step"],
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": command,
        "confidence": 0.8,
    }
    return {
        **body,
        "proposal_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


def _prepare_arbitrated_no_proposal(hmc: HabitatManagementComputer):
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    proposal = hmc.propose(None)
    arbitration = hmc.arbitrate()
    return snapshot, verification, proposal, arbitration


def _closed_preflight(
    *,
    classification: str,
    application_step: int,
    command_sha256: str,
    preflight_contract_sha256: str,
) -> PreflightResult:
    body = {
        "classification": classification,
        "application_step": application_step,
        "command_sha256": command_sha256,
        "preflight_contract_sha256": preflight_contract_sha256,
    }
    return PreflightResult(
        **body,
        preflight_result_sha256=hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    )


def test_unknown_health_overrides_a_valid_proposal_with_safe_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_health(
        monkeypatch,
        state="UNKNOWN",
        alarms=(
            OperationalAlarm(
                alarm_id="telemetry_unknown/fan_speed_fraction/critical",
                family="telemetry_unknown",
                target="fan_speed_fraction",
                severity="CRITICAL",
                lifecycle="ACTIVE",
            ),
        ),
    )
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"u" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    requested = command_from_achieved_state(scenario, hmc._state).command.to_mapping()
    requested["fan_speed_fraction"] = 1.0
    hmc.propose(_valid_proposal(hmc, snapshot, requested))
    hold = command_from_achieved_state(scenario, hmc._state).command

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert mapping["accepted_proposal_sha256"] is None
    assert mapping["requested_command"] == requested
    assert mapping["final_command"] == hold.to_mapping()
    assert mapping["final_command_sha256"] == hold.sha256
    assert mapping["disposition"] == "REJECTED"
    assert mapping["reason_codes"] == ["safe_hold_telemetry_unknown"]
    assert mapping["command_owner"] == "baseline_hold"
    assert mapping["emergency_override"] is False


def test_conflicting_critical_environmental_alarms_select_deterministic_emergency_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_health(
        monkeypatch,
        state="CRITICAL",
        alarms=(
            OperationalAlarm(
                alarm_id="high_co2/laboratory/critical",
                family="high_co2",
                target="laboratory",
                severity="CRITICAL",
                lifecycle="ACTIVE",
            ),
            OperationalAlarm(
                alarm_id="high_temperature/common_galley/critical",
                family="high_temperature",
                target="common_galley",
                severity="CRITICAL",
                lifecycle="ACTIVE",
            ),
        ),
    )
    scenario = _scenario()
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"e" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    requested = command_from_achieved_state(scenario, hmc._state).command.to_mapping()
    hmc.propose(_valid_proposal(hmc, snapshot, requested))

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    final = mapping["final_command"]
    assert mapping["accepted_proposal_sha256"] is None
    assert mapping["requested_command"] == requested
    assert mapping["disposition"] == "MODIFIED"
    assert mapping["reason_codes"] == ["emergency_override"]
    assert mapping["command_owner"] == "emergency_safe_action"
    assert mapping["emergency_override"] is True
    assert mapping["emergency_reserve_use"] is False
    assert final["fan_speed_fraction"] == 1.0
    assert final["scrubber_duty"] == 1.0
    assert final["damper_position_by_id"]["laboratory_supply_damper"] == 1.0
    assert final["damper_position_by_id"]["common_galley_supply_damper"] >= 0.7
    assert final["cooling_removed_w"]["common_galley"] == 1000.0


def test_emergency_override_records_normal_reserve_floor_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_health(
        monkeypatch,
        state="CRITICAL",
        alarms=(
            OperationalAlarm(
                alarm_id="high_co2/laboratory/critical",
                family="high_co2",
                target="laboratory",
                severity="CRITICAL",
                lifecycle="ACTIVE",
            ),
        ),
    )
    scenario = _scenario(low_battery=True)
    hmc = HabitatManagementComputer.reset(scenario, _contract(), b"r" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None)

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert mapping["reason_codes"] == ["emergency_override"]
    assert mapping["emergency_override"] is True
    assert mapping["emergency_reserve_use"] is True


def test_physics_failure_after_authority_consumption_emits_one_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"f" * 32)
    snapshot, verification, proposal, arbitration = _prepare_arbitrated_no_proposal(hmc)
    committed_state = hmc._state
    committed_sensor_memory = hmc._sensor_memory
    committed_health_tracker = hmc._health_tracker
    committed_sequence = hmc._sequence
    prior_events = hmc.control_events

    def fail_physics(*_args, **_kwargs):
        raise RuntimeError("injected hidden physics detail")

    monkeypatch.setattr(hmc_module, "advance_one_step_with_command", fail_physics)

    receipt = hmc.step()

    assert type(receipt).__name__ == "TerminalFailureReceipt"
    assert hmc.lifecycle_phase == "TERMINAL"
    assert hmc._step_capability is None
    assert hmc._state is committed_state
    assert hmc._sensor_memory is committed_sensor_memory
    assert hmc._health_tracker is committed_health_tracker
    assert hmc._sequence == committed_sequence
    assert len(hmc.control_events) == len(prior_events) + 1
    terminal_event = hmc.control_events[-1]
    assert terminal_event.event_kind == "TERMINAL"
    mapping = receipt.to_mapping()
    assert set(mapping) == set(contract.data["receipt_schemas"]["terminal"]["fields"])
    assert mapping["receipt_schema_sha256"] == contract.terminal_receipt_schema_sha256
    assert (
        mapping["terminal_contract_sha256"] == contract.terminal_receipt_schema_sha256
    )
    assert mapping["hmc_contract_sha256"] == contract.hmc_contract_sha256
    assert mapping["observable_topology_sha256"] == hmc.observable_topology_sha256
    assert mapping["control_run_id"] == hmc.control_run_id
    assert mapping["authority_epoch"] == hmc.authority_epoch
    assert mapping["sequence"] == 0
    assert mapping["application_step"] == 0
    assert mapping["lifecycle_phase"] == "ARBITRATED"
    assert mapping["last_good_snapshot_sha256"] == snapshot.snapshot_sha256
    assert mapping["last_good_verification_receipt_sha256"] == (
        verification.snapshot_verification_receipt_sha256
    )
    assert (
        mapping["last_good_step_receipt_sha256"]
        == contract.data["null_roots"]["step_receipt"]["sha256"]
    )
    assert mapping["proposal_receipt_sha256"] == proposal.proposal_receipt_sha256
    assert (
        mapping["arbitration_receipt_sha256"] == arbitration.arbitration_receipt_sha256
    )
    assert mapping["final_command_sha256"] == arbitration.final_command_sha256
    assert mapping["candidate_plant_receipt_digest"] is None
    assert mapping["plant_state_committed"] is False
    assert mapping["reason_code"] == "PHYSICS_EXECUTION_FAILED"
    rendered = receipt.canonical_bytes.decode("utf-8")
    assert "injected hidden physics detail" not in rendered
    assert terminal_event.receipt == mapping

    for causal_call in (
        hmc.observe,
        lambda: hmc.verify_snapshot(snapshot, verification),
        lambda: hmc.propose(None),
        hmc.arbitrate,
        hmc.step,
    ):
        with pytest.raises(RuntimeError, match="TERMINAL"):
            causal_call()


@pytest.mark.parametrize(
    ("failure_stage", "reason_code", "candidate_digest_expected"),
    (
        ("physics", "PHYSICS_EXECUTION_FAILED", False),
        ("command_digest", "COMMAND_DIGEST_MISMATCH", True),
        ("plant_receipt", "PLANT_RECEIPT_INVALID", True),
        ("measurement", "OPERATIONAL_MEASUREMENT_INVALID", True),
        ("health", "HEALTH_REDUCTION_FAILED", True),
        ("snapshot", "SNAPSHOT_ISSUANCE_FAILED", True),
    ),
)
def test_each_post_authority_failure_is_one_atomic_terminal_transition(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    reason_code: str,
    candidate_digest_expected: bool,
) -> None:
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"t" * 32)
    snapshot, verification, _proposal, _arbitration = _prepare_arbitrated_no_proposal(
        hmc
    )
    committed = {
        "state": hmc._state,
        "sensor_memory": hmc._sensor_memory,
        "health_tracker": hmc._health_tracker,
        "measurement": hmc._last_operational_measurement,
        "sequence": hmc._sequence,
        "snapshot": hmc._cached_snapshot,
        "verification": hmc._cached_verification_receipt,
        "events": hmc.control_events,
    }
    original_advance = hmc_module.advance_one_step_with_command

    if failure_stage == "physics":

        def fail_physics(*_args, **_kwargs):
            raise RuntimeError("hidden-physics-detail")

        monkeypatch.setattr(
            hmc_module,
            "advance_one_step_with_command",
            fail_physics,
        )
    elif failure_stage == "command_digest":

        def wrong_digest(*args, **kwargs):
            result = original_advance(*args, **kwargs)
            return StepResult(
                state=result.state,
                receipt={**result.receipt, "external_command_digest": "00" * 32},
            )

        monkeypatch.setattr(
            hmc_module,
            "advance_one_step_with_command",
            wrong_digest,
        )
    elif failure_stage == "plant_receipt":

        def fail_plant_validation(*_args, **_kwargs):
            raise RuntimeError("hidden-plant-detail")

        monkeypatch.setattr(
            hmc_module,
            "validate_external_step_result",
            fail_plant_validation,
        )
    elif failure_stage == "measurement":

        def fail_measurement(*_args, **_kwargs):
            raise RuntimeError("hidden-measurement-detail")

        monkeypatch.setattr(
            hmc_module,
            "instrument_v5_operational_measurement",
            fail_measurement,
        )
    elif failure_stage == "health":

        def fail_health(*_args, **_kwargs):
            raise RuntimeError("hidden-health-detail")

        monkeypatch.setattr(hmc_module, "reduce_health", fail_health)
    elif failure_stage == "snapshot":

        def fail_snapshot_stage(*_args, **_kwargs):
            raise RuntimeError("hidden-snapshot-detail")

        monkeypatch.setattr(
            HabitatManagementComputer,
            "_stage_completed_cycle",
            fail_snapshot_stage,
        )
    else:  # pragma: no cover - the parameter set is closed above
        raise AssertionError(f"unsupported failure stage {failure_stage}")

    receipt = hmc.step()

    assert type(receipt).__name__ == "TerminalFailureReceipt"
    mapping = receipt.to_mapping()
    assert mapping["reason_code"] == reason_code
    assert (mapping["candidate_plant_receipt_digest"] is not None) is (
        candidate_digest_expected
    )
    assert mapping["plant_state_committed"] is False
    assert hmc.lifecycle_phase == "TERMINAL"
    assert hmc._step_capability is None
    assert hmc._state is committed["state"]
    assert hmc._sensor_memory is committed["sensor_memory"]
    assert hmc._health_tracker is committed["health_tracker"]
    assert hmc._last_operational_measurement is committed["measurement"]
    assert hmc._sequence == committed["sequence"]
    assert hmc._cached_snapshot is committed["snapshot"]
    assert hmc._cached_verification_receipt is committed["verification"]
    assert len(hmc.control_events) == len(committed["events"]) + 1
    assert hmc.control_events[-1].event_kind == "TERMINAL"
    assert hmc.control_events[-1].receipt == mapping
    rendered = receipt.canonical_bytes.decode("utf-8")
    assert "hidden-" not in rendered
    assert mapping["last_good_snapshot_sha256"] == snapshot.snapshot_sha256
    assert mapping["last_good_verification_receipt_sha256"] == (
        verification.snapshot_verification_receipt_sha256
    )


def test_infeasible_safe_hold_enters_terminal_before_step_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    hmc = HabitatManagementComputer.reset(_scenario(), contract, b"i" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None)
    committed_state = hmc._state
    events_before = hmc.control_events

    def infeasible_preflight(_scenario, _state, command, application_step):
        command_sha256 = validate_external_command(_scenario, command).sha256
        return _closed_preflight(
            classification="INFEASIBLE",
            application_step=application_step,
            command_sha256=command_sha256,
            preflight_contract_sha256=contract.preflight_contract_sha256,
        )

    monkeypatch.setattr(
        hmc_module,
        "preflight_external_command",
        infeasible_preflight,
    )

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert type(receipt).__name__ == "TerminalFailureReceipt"
    assert mapping["reason_code"] == "SAFE_HOLD_INFEASIBLE"
    assert mapping["lifecycle_phase"] == "PROPOSED"
    assert mapping["arbitration_receipt_sha256"] is None
    assert mapping["plant_state_committed"] is False
    assert hmc._state is committed_state
    assert hmc.lifecycle_phase == "TERMINAL"
    assert len(hmc.control_events) == len(events_before) + 1
    assert hmc.control_events[-1].event_kind == "TERMINAL"


def test_infeasible_emergency_preflight_falls_back_to_feasible_safe_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_health(
        monkeypatch,
        state="CRITICAL",
        alarms=(
            OperationalAlarm(
                alarm_id="high_co2/laboratory/critical",
                family="high_co2",
                target="laboratory",
                severity="CRITICAL",
                lifecycle="ACTIVE",
            ),
        ),
    )
    scenario = _scenario()
    contract = _contract()
    hmc = HabitatManagementComputer.reset(scenario, contract, b"g" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    requested = command_from_achieved_state(scenario, hmc._state).command.to_mapping()
    requested["fan_speed_fraction"] = 0.4
    hmc.propose(_valid_proposal(hmc, snapshot, requested))
    safe_hold = command_from_achieved_state(scenario, hmc._state).command
    original_preflight = hmc_module.preflight_external_command
    calls: list[str] = []

    def selective_preflight(_scenario, state, command, application_step):
        canonical = validate_external_command(_scenario, command)
        calls.append(canonical.sha256)
        if canonical.sha256 != safe_hold.sha256:
            return _closed_preflight(
                classification="INFEASIBLE",
                application_step=application_step,
                command_sha256=canonical.sha256,
                preflight_contract_sha256=contract.preflight_contract_sha256,
            )
        return original_preflight(_scenario, state, command, application_step)

    monkeypatch.setattr(
        hmc_module,
        "preflight_external_command",
        selective_preflight,
    )

    receipt = hmc.arbitrate()

    mapping = receipt.to_mapping()
    assert len(calls) == 2
    assert calls[0] != safe_hold.sha256
    assert calls[1] == safe_hold.sha256
    assert mapping["requested_command"] == requested
    assert mapping["accepted_proposal_sha256"] is None
    assert mapping["final_command"] == safe_hold.to_mapping()
    assert mapping["disposition"] == "REJECTED"
    assert mapping["reason_codes"] == ["emergency_override"]
    assert mapping["command_owner"] == "baseline_hold"
    assert mapping["emergency_override"] is True
    assert mapping["emergency_reserve_use"] is False
    assert mapping["preflight_result"]["classification"] == "FEASIBLE"
    assert hmc.lifecycle_phase == "ARBITRATED"


def test_old_public_receipt_minting_names_are_not_exposed() -> None:
    assert not hasattr(control_trace_module, "issue_step_receipt")
    assert not hasattr(proposal_module, "issue_proposal_receipt")
    assert not hasattr(safety_module, "issue_arbitration_receipt")
    for name in (
        "issue_operational_snapshot",
        "issue_snapshot_verification_receipt",
        "issue_control_event",
        "issue_receipt_control_event",
    ):
        assert not hasattr(snapshot_module, name)


def test_second_cycle_pre_arbitration_terminal_does_not_link_prior_arbitration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"l" * 32)
    _prepare_arbitrated_no_proposal(hmc)
    first_step = hmc.step()
    assert type(first_step).__name__ == "StepReceipt"
    snapshot_one, verification_one = hmc.observe()
    hmc.verify_snapshot(snapshot_one, verification_one)
    second_proposal = hmc.propose(None)
    assert hmc._cached_arbitration_receipt is None

    def infeasible_preflight(_scenario, _state, command, application_step):
        canonical = validate_external_command(_scenario, command)
        return _closed_preflight(
            classification="INFEASIBLE",
            application_step=application_step,
            command_sha256=canonical.sha256,
            preflight_contract_sha256=hmc._contract.preflight_contract_sha256,
        )

    monkeypatch.setattr(
        hmc_module,
        "preflight_external_command",
        infeasible_preflight,
    )

    terminal = hmc.arbitrate()

    mapping = terminal.to_mapping()
    assert mapping["reason_code"] == "SAFE_HOLD_INFEASIBLE"
    assert mapping["sequence"] == 1
    assert mapping["application_step"] == 1
    assert mapping["proposal_receipt_sha256"] == (
        second_proposal.proposal_receipt_sha256
    )
    assert mapping["arbitration_receipt_sha256"] is None
    assert mapping["last_good_step_receipt_sha256"] == first_step.step_receipt_sha256
    assert hmc.lifecycle_phase == "TERMINAL"


@pytest.mark.parametrize(
    ("failure_stage", "reason_code"),
    (
        ("measurement", "OPERATIONAL_MEASUREMENT_INVALID"),
        ("health", "HEALTH_REDUCTION_FAILED"),
        ("snapshot", "SNAPSHOT_ISSUANCE_FAILED"),
    ),
)
def test_initial_observation_failure_enters_closed_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    reason_code: str,
) -> None:
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"z" * 32)

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"hidden-initial-{failure_stage}-detail")

    if failure_stage == "measurement":
        monkeypatch.setattr(
            hmc_module,
            "instrument_v5_operational_measurement",
            fail,
        )
    elif failure_stage == "health":
        monkeypatch.setattr(hmc_module, "reduce_health", fail)
    elif failure_stage == "snapshot":
        monkeypatch.setattr(snapshot_module, "_issue_operational_snapshot", fail)
        monkeypatch.setattr(hmc_module, "_issue_operational_snapshot", fail)

    receipt = hmc.observe()

    assert type(receipt).__name__ == "TerminalFailureReceipt"
    mapping = receipt.to_mapping()
    assert mapping["reason_code"] == reason_code
    assert mapping["application_step"] is None
    assert mapping["lifecycle_phase"] == "RESET"
    assert mapping["last_good_snapshot_sha256"] == hmc._contract.data["null_roots"][
        "snapshot"
    ]["sha256"]
    assert mapping["proposal_receipt_sha256"] is None
    assert mapping["arbitration_receipt_sha256"] is None
    assert mapping["final_command_sha256"] is None
    assert mapping["candidate_plant_receipt_digest"] is None
    assert mapping["plant_state_committed"] is False
    assert hmc.lifecycle_phase == "TERMINAL"
    assert len(hmc.control_events) == 1
    assert hmc.control_events[0].event_kind == "TERMINAL"
    assert "hidden-initial" not in receipt.canonical_bytes.decode("utf-8")


def test_unexpected_emergency_preflight_failure_falls_back_then_terminal_if_hold_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_health(
        monkeypatch,
        state="CRITICAL",
        alarms=(
            OperationalAlarm(
                alarm_id="high_co2/laboratory/critical",
                family="high_co2",
                target="laboratory",
                severity="CRITICAL",
                lifecycle="ACTIVE",
            ),
        ),
    )
    hmc = HabitatManagementComputer.reset(_scenario(), _contract(), b"x" * 32)
    snapshot, verification = hmc.observe()
    hmc.verify_snapshot(snapshot, verification)
    hmc.propose(None)
    calls = 0

    def fail_preflight(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("hidden-preflight-detail")

    monkeypatch.setattr(hmc_module, "preflight_external_command", fail_preflight)

    receipt = hmc.arbitrate()

    assert calls == 2
    assert type(receipt).__name__ == "TerminalFailureReceipt"
    assert receipt.to_mapping()["reason_code"] == "SAFE_HOLD_INFEASIBLE"
    assert "hidden-preflight" not in receipt.canonical_bytes.decode("utf-8")
    assert hmc.lifecycle_phase == "TERMINAL"
