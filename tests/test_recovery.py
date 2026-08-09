"""Deterministic reserve-authority state-machine contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus.config import parse_scenario
from aeolus.model_input import build_model_input_contract
from aeolus.recovery import (
    AdvisoryAcceptanceSettings,
    AuthorityState,
    DeterministicRecoverySupervisor,
    RecoveryAdvisory,
    RecoveryObservation,
    RecoverySettings,
    ReserveCommandOwner,
    validate_recovery_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCENARIO = REPO_ROOT / "scenarios" / "recovery_habitat.json"


@pytest.fixture
def config():
    import json

    return parse_scenario(json.loads(RECOVERY_SCENARIO.read_text(encoding="utf-8")))


@pytest.fixture
def supervisor(config):
    return DeterministicRecoverySupervisor(
        config,
        run_id="run-1",
        contract=build_model_input_contract(config),
    )


def _observation(
    supervisor: DeterministicRecoverySupervisor,
    *,
    target: str | None = None,
    ambiguous: bool = False,
    clear: bool = False,
    reserve_position: float = 0.0,
    reserve_delivery: float | None = None,
    co2: float = 0.2,
    target_delivered: float | None = None,
) -> RecoveryObservation:
    decision = supervisor.last_decision
    assert decision is not None
    zone_ids = supervisor.zone_ids
    requested = {zone_id: 10.0 for zone_id in zone_ids}
    delivered = dict(requested)
    if target is not None and not clear:
        delivered[target] = 6.5 if target_delivered is None else target_delivered
    if ambiguous:
        delivered[zone_ids[0]] = 6.5
        delivered[zone_ids[1]] = 6.5
    co2_by_zone = {zone_id: co2 for zone_id in zone_ids}
    reserve_position_by_zone = {zone_id: 0.0 for zone_id in zone_ids}
    reserve_position_by_zone[target or zone_ids[0]] = reserve_position
    reserve_delivery_by_zone = {zone_id: 0.0 for zone_id in zone_ids}
    if reserve_delivery is None:
        reserve_delivery = reserve_position * supervisor.reserve_capacity
    reserve_delivery_by_zone[target or zone_ids[0]] = reserve_delivery

    vector = [0.0] * len(supervisor.contract.fields)
    for index, field in enumerate(supervisor.contract.fields):
        if field.group == "zones":
            vector[index] = co2_by_zone.get(field.entity_id, 0.0)
        elif field.group == "connections":
            zone_id = supervisor.zone_for_primary_outbound(field.entity_id)
            if field.field == "requested_airflow":
                vector[index] = requested[zone_id]
            elif field.field == "delivered_airflow":
                vector[index] = delivered[zone_id]
            else:
                vector[index] = requested[zone_id] - delivered[zone_id]

    return RecoveryObservation(
        run_id=decision.run_id,
        authority_epoch=decision.authority_epoch,
        completed_tick=decision.decision_tick,
        sequence=decision.sequence,
        model_input_v1=tuple(vector),
        selector_sha256=supervisor.contract.selector_hash,
        topology_sha256=supervisor.contract.topology_hash,
        zone_ids=zone_ids,
        primary_outbound_ids={
            zone_id: supervisor.primary_outbound_id(zone_id) for zone_id in zone_ids
        },
        reserve_outbound_ids={
            zone_id: supervisor.reserve_outbound_id(zone_id) for zone_id in zone_ids
        },
        reserve_return_ids={
            zone_id: supervisor.reserve_return_id(zone_id) for zone_id in zone_ids
        },
        co2_concentration=co2_by_zone,
        primary_requested_airflow=requested,
        primary_delivered_airflow=delivered,
        reserve_actual_position=reserve_position_by_zone,
        reserve_delivered_airflow=reserve_delivery_by_zone,
        applied_command_digest=decision.command_digest,
    )


def _start(supervisor):
    decision = supervisor.cold_start(decision_tick=1)
    assert decision.state is AuthorityState.NOMINAL
    assert decision.reserve_command_owner is ReserveCommandOwner.RESERVE_OFF
    assert set(decision.reserve_commands) == set(supervisor.zone_ids)
    assert set(decision.reserve_commands.values()) == {0.0}
    return decision


def _advance(supervisor, **kwargs):
    return supervisor.decide(_observation(supervisor, **kwargs))


ADVISORY_ARTIFACT_SHA256 = "a" * 64


def _advisory_supervisor(config):
    return DeterministicRecoverySupervisor(
        config,
        run_id="advisory-run",
        contract=build_model_input_contract(config),
        advisory_settings=AdvisoryAcceptanceSettings(
            artifact_sha256=ADVISORY_ARTIFACT_SHA256,
            minimum_probability=0.57,
            minimum_margin=0.17,
            minimum_residual_ratio=0.04,
        ),
    )


def _advisory(
    supervisor: DeterministicRecoverySupervisor,
    observation: RecoveryObservation,
    *,
    target: str = "cabin_a",
    probability: float = 0.80,
    margin: float = 0.30,
) -> RecoveryAdvisory:
    return RecoveryAdvisory(
        run_id=supervisor.run_id,
        authority_epoch=supervisor.authority_epoch,
        completed_tick=observation.completed_tick,
        sequence=observation.sequence,
        target_zone_id=target,
        probability=probability,
        margin=margin,
        selector_sha256=supervisor.contract.selector_hash,
        topology_sha256=supervisor.contract.topology_hash,
        artifact_sha256=ADVISORY_ARTIFACT_SHA256,
    )


def _arm(supervisor, target="cabin_a"):
    _start(supervisor)
    first = _advance(supervisor, target=target)
    second = _advance(supervisor, target=target)
    assert first.state is AuthorityState.DEGRADED
    if second.state is AuthorityState.PROTECT:
        assert supervisor.settings.entry_persistence_ticks == 2
        return second
    assert second.state is AuthorityState.DEGRADED
    third = _advance(supervisor, target=target)
    assert third.state is AuthorityState.PROTECT
    return third


class TestRecoverySettings:
    def test_defaults_match_frozen_development_candidate(self):
        settings = RecoverySettings()
        assert (
            settings.entry_residual_ratio,
            settings.entry_isolation_margin,
            settings.entry_persistence_ticks,
            settings.exit_residual_ratio,
            settings.handback_abort_residual_ratio,
            settings.handback_abort_persistence_ticks,
        ) == (0.10, 0.05, 2, 0.06, 0.08, 2)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("entry_residual_ratio", math.nan),
            ("entry_persistence_ticks", True),
            ("entry_persistence_ticks", 1),
            ("reserve_command_delta", 0.0),
            ("maximum_reserve_command", 1.1),
            ("minimum_reserve_delivery_ratio", -0.1),
            ("maximum_handback_ticks", 35),
            ("handback_abort_persistence_ticks", 0),
        ],
    )
    def test_rejects_malformed_settings(self, field, value):
        with pytest.raises(ValueError, match="recovery"):
            RecoverySettings(**{field: value})

    def test_requires_hysteretic_threshold_ordering(self):
        with pytest.raises(ValueError, match="hysteresis"):
            RecoverySettings(exit_residual_ratio=0.2)
        with pytest.raises(ValueError, match="hysteresis"):
            RecoverySettings(handback_abort_residual_ratio=0.18)

    def test_rejects_reserve_path_capacity_mismatch(self, config):
        first = replace(config.reserve_connections[0], max_airflow=3.0)
        unsafe = replace(
            config,
            reserve_connections=(first, *config.reserve_connections[1:]),
        )
        with pytest.raises(ValueError, match="reserve path max airflow"):
            DeterministicRecoverySupervisor(
                unsafe,
                run_id="unsafe-reserve",
                contract=build_model_input_contract(unsafe),
            )

    def test_rejects_primary_command_below_evidence_floor(self, config):
        document = __import__("json").loads(RECOVERY_SCENARIO.read_text(encoding="utf-8"))
        document["control"]["minimum_command"] = 0.04
        unsafe = parse_scenario(document)
        with pytest.raises(ValueError, match="minimum primary command"):
            DeterministicRecoverySupervisor(
                unsafe,
                run_id="unsafe",
                contract=build_model_input_contract(unsafe),
            )


class TestObservationAndApplicationGates:
    def test_rejects_wrong_identity_sequence_hash_topology_and_non_finite(
        self, supervisor
    ):
        _start(supervisor)
        valid = _observation(supervisor, target="cabin_a")
        mutations = (
            replace(valid, run_id="other"),
            replace(valid, authority_epoch=99),
            replace(valid, completed_tick=2),
            replace(valid, sequence=2),
            replace(valid, selector_sha256="0" * 64),
            replace(valid, topology_sha256="0" * 64),
            replace(valid, zone_ids=valid.zone_ids[:-1]),
            replace(
                valid,
                primary_requested_airflow={
                    **valid.primary_requested_airflow,
                    "cabin_a": math.nan,
                },
            ),
            replace(valid, applied_command_digest="f" * 64),
        )
        for malformed in mutations:
            fresh = DeterministicRecoverySupervisor(
                supervisor.config,
                run_id="run-1",
                contract=supervisor.contract,
            )
            fresh.cold_start(decision_tick=1)
            with pytest.raises(ValueError):
                fresh.decide(malformed)

    def test_application_gate_rejects_wrong_owner_tick_keys_bounds_and_digest(
        self, supervisor
    ):
        decision = _start(supervisor)
        expected = {
            "config": supervisor.config,
            "expected_run_id": "run-1",
            "expected_authority_epoch": 0,
            "expected_decision_tick": 1,
            "expected_state": AuthorityState.NOMINAL,
        }
        assert validate_recovery_decision(decision, **expected) == {
            zone_id: 0.0 for zone_id in supervisor.zone_ids
        }
        bad = (
            replace(decision, run_id="other"),
            replace(decision, decision_tick=2),
            replace(
                decision,
                reserve_command_owner=ReserveCommandOwner.DETERMINISTIC_RECOVERY_SUPERVISOR,
            ),
            replace(decision, reserve_commands={"cabin_a": 0.0}),
            replace(
                decision,
                reserve_commands={zone_id: 2.0 for zone_id in supervisor.zone_ids},
            ),
            replace(decision, command_digest="0" * 64),
        )
        for malformed in bad:
            with pytest.raises(ValueError):
                validate_recovery_decision(malformed, **expected)
        with pytest.raises(ValueError, match="reason"):
            validate_recovery_decision(
                replace(decision, reason="invented_reason"), **expected
            )


class TestAdvisoryAcceptanceBoundary:
    def test_no_advisory_preserves_existing_below_threshold_behavior(self, config):
        supervisor = _advisory_supervisor(config)
        _start(supervisor)
        decision = supervisor.decide(
            _observation(supervisor, target="cabin_a", target_delivered=9.4)
        )

        assert decision.state is AuthorityState.NOMINAL
        assert decision.reason == "no_concern"
        assert set(decision.reserve_commands.values()) == {0.0}

    def test_two_persistent_supported_advisories_can_enter_protect(self, config):
        supervisor = _advisory_supervisor(config)
        _start(supervisor)
        first_observation = _observation(
            supervisor, target="cabin_a", target_delivered=9.4
        )
        first = supervisor.decide(
            first_observation, _advisory(supervisor, first_observation)
        )
        second_observation = _observation(
            supervisor, target="cabin_a", target_delivered=9.4
        )
        second = supervisor.decide(
            second_observation, _advisory(supervisor, second_observation)
        )

        assert first.state is AuthorityState.DEGRADED
        assert first.reason == "advisory_unique_concern"
        assert set(first.reserve_commands.values()) == {0.0}
        assert second.state is AuthorityState.PROTECT
        assert second.reason == "advisory_entry_persistence_met"
        assert second.target_zone_id == "cabin_a"
        assert second.reserve_commands["cabin_a"] > 0.0
        assert all(
            value == 0.0
            for zone_id, value in second.reserve_commands.items()
            if zone_id != "cabin_a"
        )

    @pytest.mark.parametrize(
        "mutation",
        (
            {"artifact_sha256": "b" * 64},
            {"completed_tick": 999},
            {"target_zone_id": "lab"},
            {"probability": 0.56},
            {"margin": 0.16},
        ),
    )
    def test_malformed_stale_low_confidence_and_out_of_scope_advisories_are_refused(
        self, config, mutation
    ):
        supervisor = _advisory_supervisor(config)
        _start(supervisor)
        observation = _observation(
            supervisor, target="cabin_a", target_delivered=9.4
        )
        advisory = replace(_advisory(supervisor, observation), **mutation)
        decision = supervisor.decide(observation, advisory)

        assert decision.state is AuthorityState.NOMINAL
        assert decision.reason == "no_concern"
        assert set(decision.reserve_commands.values()) == {0.0}

    def test_advisory_target_must_match_unique_physical_shortfall(self, config):
        supervisor = _advisory_supervisor(config)
        _start(supervisor)
        observation = _observation(
            supervisor, target="cabin_a", target_delivered=9.4
        )
        wrong_target = _advisory(supervisor, observation, target="cabin_b")
        decision = supervisor.decide(observation, wrong_target)

        assert decision.state is AuthorityState.NOMINAL
        assert set(decision.reserve_commands.values()) == {0.0}

    def test_advisory_is_refused_when_current_shortfall_is_ambiguous(self, config):
        supervisor = _advisory_supervisor(config)
        _start(supervisor)
        observation = _observation(supervisor, ambiguous=True)
        decision = supervisor.decide(
            observation, _advisory(supervisor, observation)
        )

        assert decision.state is AuthorityState.DEGRADED
        assert decision.reason == "ambiguous_concern"
        assert set(decision.reserve_commands.values()) == {0.0}


class TestAuthorityStateTable:
    def test_two_same_target_observations_arm_with_one_tick_causality(
        self, supervisor
    ):
        decision = _arm(supervisor)
        assert decision.decision_tick == 3
        assert decision.observation_tick == 2
        assert decision.target_zone_id == "cabin_a"
        assert decision.reserve_command_owner is (
            ReserveCommandOwner.DETERMINISTIC_RECOVERY_SUPERVISOR
        )
        assert decision.reserve_commands["cabin_a"] == pytest.approx(0.1)
        assert all(
            value == 0.0
            for zone, value in decision.reserve_commands.items()
            if zone != "cabin_a"
        )

    def test_target_flapping_and_ambiguity_never_grant_authority(self, supervisor):
        _start(supervisor)
        for target in ("cabin_a", "cabin_b", "cabin_a", "cabin_b"):
            assert _advance(supervisor, target=target).state is AuthorityState.DEGRADED
        for _ in range(4):
            assert _advance(supervisor, ambiguous=True).state is AuthorityState.DEGRADED
        assert all(
            event.to_state is not AuthorityState.PROTECT
            for event in supervisor.event_history
        )

    def test_three_clear_ticks_return_degraded_to_nominal(self, supervisor):
        _start(supervisor)
        assert _advance(supervisor, target="cabin_a").state is AuthorityState.DEGRADED
        assert _advance(supervisor, clear=True).state is AuthorityState.DEGRADED
        assert _advance(supervisor, clear=True).state is AuthorityState.DEGRADED
        assert _advance(supervisor, clear=True).state is AuthorityState.NOMINAL

    def test_dropout_is_fail_closed_in_nominal_degraded_and_protect(self, supervisor):
        _start(supervisor)
        degraded = supervisor.decide_unavailable(
            completed_tick=1,
            sequence=1,
            applied_command_digest=supervisor.last_decision.command_digest,
        )
        assert degraded.state is AuthorityState.DEGRADED
        assert set(degraded.reserve_commands.values()) == {0.0}
        held = supervisor.decide_unavailable(
            completed_tick=2,
            sequence=2,
            applied_command_digest=degraded.command_digest,
        )
        assert held.state is AuthorityState.DEGRADED

        supervisor.reset(run_id="protect-run")
        protect = _arm(supervisor)
        unavailable = supervisor.decide_unavailable(
            completed_tick=protect.decision_tick,
            sequence=protect.sequence,
            applied_command_digest=protect.command_digest,
        )
        assert unavailable.state is AuthorityState.HANDBACK
        assert set(unavailable.reserve_commands.values()) == {0.0}
        held = supervisor.decide_unavailable(
            completed_tick=unavailable.decision_tick,
            sequence=unavailable.sequence,
            applied_command_digest=unavailable.command_digest,
        )
        assert held.state is AuthorityState.HANDBACK
        assert set(held.reserve_commands.values()) == {0.0}

    def test_protect_command_only_increases_and_is_slew_bounded(self, supervisor):
        protect = _arm(supervisor)
        previous = protect.reserve_commands["cabin_a"]
        for _ in range(5):
            decision = _advance(supervisor, target="cabin_a")
            current = decision.reserve_commands["cabin_a"]
            assert previous <= current <= previous + 0.1 + 1e-12
            previous = current

    def test_persistent_fault_never_begins_recovery_clear_handback(self, supervisor):
        _arm(supervisor)
        for _ in range(30):
            assert _advance(supervisor, target="cabin_a").state is AuthorityState.PROTECT

    def test_clear_guard_enters_handback_then_recurrence_restores_protect(
        self, supervisor
    ):
        protect = _arm(supervisor)
        protect_command = protect.reserve_commands["cabin_a"]
        for _ in range(9):
            assert _advance(
                supervisor,
                clear=True,
                reserve_position=protect_command,
            ).state is AuthorityState.PROTECT
        handback = _advance(
            supervisor,
            clear=True,
            reserve_position=protect_command,
        )
        assert handback.state is AuthorityState.HANDBACK
        assert handback.reserve_commands["cabin_a"] == protect_command
        recurrence = _advance(
            supervisor,
            target="cabin_a",
            reserve_position=protect_command,
        )
        assert recurrence.state is AuthorityState.PROTECT
        assert recurrence.reserve_commands["cabin_a"] == protect_command

    def test_soft_handback_recurrence_requires_configured_persistence(self, config):
        supervisor = DeterministicRecoverySupervisor(
            config,
            run_id="soft-recurrence",
            contract=build_model_input_contract(config),
            settings=RecoverySettings(
                entry_residual_ratio=0.10,
                entry_isolation_margin=0.05,
                entry_persistence_ticks=2,
                exit_residual_ratio=0.06,
                handback_abort_residual_ratio=0.09,
                handback_abort_persistence_ticks=2,
            ),
        )
        protect = _arm(supervisor)
        command = protect.reserve_commands["cabin_a"]
        for _ in range(10):
            handback = _advance(
                supervisor,
                clear=True,
                reserve_position=command,
            )
        assert handback.state is AuthorityState.HANDBACK

        first_soft = _advance(
            supervisor,
            target="cabin_a",
            target_delivered=9.05,
            reserve_position=command,
        )
        assert first_soft.state is AuthorityState.HANDBACK
        cleared = _advance(
            supervisor,
            clear=True,
            reserve_position=first_soft.reserve_commands["cabin_a"],
        )
        assert cleared.state is AuthorityState.HANDBACK

        second_first_soft = _advance(
            supervisor,
            target="cabin_a",
            target_delivered=9.05,
            reserve_position=cleared.reserve_commands["cabin_a"],
        )
        assert second_first_soft.state is AuthorityState.HANDBACK
        second_soft = _advance(
            supervisor,
            target="cabin_a",
            target_delivered=9.05,
            reserve_position=second_first_soft.reserve_commands["cabin_a"],
        )
        assert second_soft.state is AuthorityState.PROTECT

    def test_entry_level_handback_recurrence_aborts_immediately(self, config):
        supervisor = DeterministicRecoverySupervisor(
            config,
            run_id="strong-recurrence",
            contract=build_model_input_contract(config),
            settings=RecoverySettings(
                entry_residual_ratio=0.10,
                entry_isolation_margin=0.05,
                entry_persistence_ticks=2,
                exit_residual_ratio=0.06,
                handback_abort_residual_ratio=0.09,
                handback_abort_persistence_ticks=2,
            ),
        )
        protect = _arm(supervisor)
        command = protect.reserve_commands["cabin_a"]
        for _ in range(10):
            handback = _advance(
                supervisor,
                clear=True,
                reserve_position=command,
            )
        assert handback.state is AuthorityState.HANDBACK
        recurrence = _advance(
            supervisor,
            target="cabin_a",
            reserve_position=command,
        )
        assert recurrence.state is AuthorityState.PROTECT

    def test_physical_zero_acknowledgement_requires_five_fresh_ticks(self, supervisor):
        protect = _arm(supervisor)
        command = protect.reserve_commands["cabin_a"]
        for _ in range(10):
            decision = _advance(
                supervisor,
                clear=True,
                reserve_position=command,
            )
        assert decision.state is AuthorityState.HANDBACK
        while decision.reserve_commands["cabin_a"] > 0.0:
            decision = _advance(
                supervisor,
                clear=True,
                reserve_position=decision.reserve_commands["cabin_a"],
            )
        for count in range(1, 5):
            decision = _advance(
                supervisor,
                clear=True,
                reserve_position=0.0,
                reserve_delivery=0.0,
            )
            assert decision.state is AuthorityState.HANDBACK, count
        decision = _advance(
            supervisor,
            clear=True,
            reserve_position=0.0,
            reserve_delivery=0.0,
        )
        assert decision.state is AuthorityState.NOMINAL
        assert decision.reserve_command_owner is ReserveCommandOwner.RESERVE_OFF

    def test_reserve_delivery_failure_latches_shuts_down_and_cannot_rearm(
        self, supervisor
    ):
        _arm(supervisor)
        first = _advance(
            supervisor,
            target="cabin_a",
            reserve_position=0.1,
            reserve_delivery=0.0,
        )
        assert first.state is AuthorityState.PROTECT
        failed = _advance(
            supervisor,
            target="cabin_a",
            reserve_position=0.2,
            reserve_delivery=0.0,
        )
        assert failed.state is AuthorityState.HANDBACK
        assert supervisor.reserve_failed
        handback_tick = failed.decision_tick

        decision = failed
        while decision.state is AuthorityState.HANDBACK:
            position = 0.0 if decision.reserve_commands["cabin_a"] == 0.0 else 0.1
            decision = _advance(
                supervisor,
                target="cabin_a",
                reserve_position=position,
                reserve_delivery=0.0,
            )
        assert decision.state is AuthorityState.DEGRADED
        assert decision.decision_tick - handback_tick <= 36
        for _ in range(6):
            decision = _advance(supervisor, target="cabin_a")
            assert decision.state is AuthorityState.DEGRADED

    def test_reset_clears_failure_latch_and_event_log_is_deterministic(self, config):
        def replay():
            instance = DeterministicRecoverySupervisor(
                config,
                run_id="deterministic",
                contract=build_model_input_contract(config),
            )
            _arm(instance)
            for _ in range(4):
                _advance(instance, target="cabin_a")
            return tuple(instance.event_history)

        assert replay() == replay()
        supervisor = DeterministicRecoverySupervisor(
            config,
            run_id="first",
            contract=build_model_input_contract(config),
        )
        supervisor._reserve_failed = True
        supervisor.reset(run_id="second")
        assert not supervisor.reserve_failed
        assert supervisor.last_decision is None
        assert supervisor.event_history == []

    def test_delivery_failure_during_handback_latches_the_epoch(self, supervisor):
        protect = _arm(supervisor)
        for _ in range(7):
            protect = _advance(supervisor, target="cabin_a")
        assert protect.reserve_commands["cabin_a"] == pytest.approx(0.8)

        handback = protect
        for _ in range(10):
            handback = _advance(
                supervisor,
                clear=True,
                reserve_position=protect.reserve_commands["cabin_a"],
            )
        assert handback.state is AuthorityState.HANDBACK

        _advance(
            supervisor,
            clear=True,
            reserve_position=0.8,
            reserve_delivery=0.0,
        )
        failed = _advance(
            supervisor,
            clear=True,
            reserve_position=0.7,
            reserve_delivery=0.0,
        )

        assert failed.state is AuthorityState.HANDBACK
        assert failed.reason == "reserve_delivery_failure"
        assert supervisor.reserve_failed

    def test_handback_times_out_to_a_latched_reserve_off_state(self, supervisor):
        protect = _arm(supervisor)
        for _ in range(7):
            protect = _advance(supervisor, target="cabin_a")
        assert protect.reserve_commands["cabin_a"] == pytest.approx(0.8)

        handback = protect
        for _ in range(10):
            handback = _advance(
                supervisor,
                clear=True,
                reserve_position=protect.reserve_commands["cabin_a"],
            )
        assert handback.state is AuthorityState.HANDBACK
        handback_entry_tick = handback.decision_tick

        decision = handback
        for _ in range(36):
            decision = _advance(
                supervisor,
                clear=True,
                reserve_position=0.8,
                reserve_delivery=0.8 * supervisor.reserve_capacity,
            )

        assert decision.decision_tick - handback_entry_tick == 36
        assert decision.state is AuthorityState.DEGRADED
        assert decision.reason == "handback_timeout"
        assert set(decision.reserve_commands.values()) == {0.0}
        assert supervisor.reserve_failed
        for _ in range(3):
            assert _advance(supervisor, target="cabin_a").state is AuthorityState.DEGRADED
