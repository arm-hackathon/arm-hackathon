from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _first_control_continuation():
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )

    design = load_approved_pilot_design(ROOT)
    continuation = next(iter_pilot_continuations(design))
    assert continuation.variant == "MATCHED_CONTROL"
    return design, continuation


def test_control_continuation_executes_full_lifecycle_with_strict_replay() -> None:
    from aeolus.habitat_v2.forecast.pilot import materialize_pilot_scenario
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_control_continuation,
    )

    design, continuation = _first_control_continuation()
    bundle = run_pilot_control_continuation(ROOT, design, continuation)

    assert bundle.continuation_id == continuation.continuation_id
    assert bundle.pair_id == continuation.pair_id
    assert bundle.matched_control_id == continuation.continuation_id
    assert bundle.noise_seed == continuation.noise_seed
    assert bundle.hmc_reset_nonce_hex == continuation.hmc_reset_nonce_hex
    expected_scenario = materialize_pilot_scenario(
        ROOT,
        design,
        cluster_id=continuation.cluster_id,
        member_id=continuation.member_id,
        repetition_id=continuation.repetition_id,
    )
    assert bundle.scenario_sha256 == expected_scenario.scenario_sha256
    assert bundle.control_run_id
    assert bundle.authority_epoch
    assert len(bundle.witnesses) == 72
    assert all(
        witness["hmc_plant_receipt_digest"] == witness["shadow_plant_receipt_digest"]
        for witness in bundle.witnesses
    )
    assert tuple(witness["application_step"] for witness in bundle.witnesses) == tuple(
        range(72)
    )
    assert set(bundle.states) == set(range(73))
    assert set(bundle.snapshots) == set(range(1, 72))
    assert bundle.committed_step_count == 72
    assert bundle.trace_sha256
    assert bundle.replay_final_state_sha256 == bundle.trace_final_state_sha256



def test_control_runner_refuses_action_proposal_variant() -> None:
    from dataclasses import replace

    import pytest

    from aeolus.habitat_v2.forecast.pilot_execution import (
        PilotExecutionError,
        run_pilot_control_continuation,
    )

    design, continuation = _first_control_continuation()
    action_variant = replace(
        continuation, variant="ACTION_PROPOSAL", action_id="normal-occupied-v1"
    )
    with pytest.raises(PilotExecutionError, match="MATCHED_CONTROL"):
        run_pilot_control_continuation(ROOT, design, action_variant)


def test_control_runner_refuses_tampered_reset_nonce() -> None:
    import hashlib
    from dataclasses import replace

    import pytest

    from aeolus.habitat_v2.forecast.pilot_execution import (
        PilotExecutionError,
        run_pilot_control_continuation,
    )

    design, continuation = _first_control_continuation()
    tampered = replace(
        continuation,
        hmc_reset_nonce_hex=hashlib.sha256(b"tampered-nonce").hexdigest(),
    )
    with pytest.raises(PilotExecutionError, match="nonce"):
        run_pilot_control_continuation(ROOT, design, tampered)


def test_control_runner_refuses_broken_control_self_reference() -> None:
    from dataclasses import replace

    import pytest

    from aeolus.habitat_v2.forecast.pilot_execution import (
        PilotExecutionError,
        run_pilot_control_continuation,
    )

    design, continuation = _first_control_continuation()
    tampered = replace(continuation, matched_control_id="not-the-control-id")
    with pytest.raises(PilotExecutionError, match="matched control"):
        run_pilot_control_continuation(ROOT, design, tampered)


def test_control_execution_is_byte_deterministic_across_runs() -> None:
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_control_continuation,
    )

    design, continuation = _first_control_continuation()
    first = run_pilot_control_continuation(ROOT, design, continuation)
    second = run_pilot_control_continuation(ROOT, design, continuation)
    assert first.trace_canonical_bytes == second.trace_canonical_bytes
    assert first.control_run_id == second.control_run_id
    assert first.replay_final_state_sha256 == second.replay_final_state_sha256



def _first_action_continuation():
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )

    design = load_approved_pilot_design(ROOT)
    continuation = next(
        item
        for item in iter_pilot_continuations(design)
        if item.variant == "ACTION_PROPOSAL"
    )
    return design, continuation


def test_action_continuation_injects_catalogue_proposal_only_at_anchor() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_action_continuation,
    )

    design, continuation = _first_action_continuation()
    bundle = run_pilot_action_continuation(ROOT, design, continuation)

    catalogue = {
        action.action_id: action
        for action in load_forecast_contracts(ROOT).actions
    }
    expected = catalogue[continuation.action_id]
    assert bundle.action_id == continuation.action_id
    assert bundle.continuation_id == continuation.continuation_id
    assert bundle.matched_control_id != continuation.continuation_id
    assert bundle.requested_command_sha256 == expected.command_sha256
    assert len(bundle.witnesses) == 72
    assert all(
        witness["hmc_plant_receipt_digest"] == witness["shadow_plant_receipt_digest"]
        for witness in bundle.witnesses
    )
    assert set(bundle.states) == set(range(73))
    assert set(bundle.snapshots) == set(range(1, 72))
    anchor = bundle.anchor
    assert anchor["application_step"] == continuation.anchor_completed_step
    proposal = anchor["proposal_receipt"]
    assert proposal["attempt_class"] == "CANONICAL_PROPOSAL"
    assert proposal["validation_outcome"] == "VALID"
    assert anchor["final_command_sha256"]
    non_anchor = [
        witness["application_step"]
        for witness in bundle.witnesses
        if witness["application_step"] != continuation.anchor_completed_step
    ]
    assert len(non_anchor) == 71
    assert bundle.committed_step_count == 72
    assert bundle.replay_final_state_sha256 == bundle.trace_final_state_sha256



def test_action_runner_refuses_matched_control_variant() -> None:
    import pytest

    from aeolus.habitat_v2.forecast.pilot_execution import (
        PilotExecutionError,
        run_pilot_action_continuation,
    )

    design, continuation = _first_control_continuation()
    with pytest.raises(PilotExecutionError):
        run_pilot_action_continuation(ROOT, design, continuation)


def test_action_runner_refuses_unplanned_action() -> None:
    from dataclasses import replace

    import pytest

    from aeolus.habitat_v2.forecast.pilot_execution import (
        PilotExecutionError,
        run_pilot_action_continuation,
    )

    design, continuation = _first_action_continuation()
    tampered = replace(continuation, action_id="normal-bogus-v1")
    with pytest.raises(PilotExecutionError, match="outside the frozen plan"):
        run_pilot_action_continuation(ROOT, design, tampered)


def test_action_runner_refuses_tampered_reset_nonce() -> None:
    import hashlib
    from dataclasses import replace

    import pytest

    from aeolus.habitat_v2.forecast.pilot_execution import (
        PilotExecutionError,
        run_pilot_action_continuation,
    )

    design, continuation = _first_action_continuation()
    tampered = replace(
        continuation,
        hmc_reset_nonce_hex=hashlib.sha256(b"tampered-action-nonce").hexdigest(),
    )
    with pytest.raises(PilotExecutionError, match="nonce"):
        run_pilot_action_continuation(ROOT, design, tampered)


def test_action_and_control_pair_share_scenario_but_diverge_in_trace() -> None:
    from aeolus.habitat_v2.forecast.pilot import iter_pilot_continuations
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_action_continuation,
        run_pilot_control_continuation,
    )

    design, action_continuation = _first_action_continuation()
    control_continuation = next(
        item
        for item in iter_pilot_continuations(design)
        if item.pair_id == action_continuation.pair_id
        and item.variant == "MATCHED_CONTROL"
    )
    control = run_pilot_control_continuation(ROOT, design, control_continuation)
    action = run_pilot_action_continuation(ROOT, design, action_continuation)

    assert action.matched_control_id == control.continuation_id
    assert action.scenario_sha256 == control.scenario_sha256
    assert action.noise_seed == control.noise_seed
    assert action.hmc_reset_nonce_hex == control.hmc_reset_nonce_hex
    assert action.control_run_id == control.control_run_id
    assert action.authority_epoch == control.authority_epoch
    assert action.continuation_id != control.continuation_id
    assert action.trace_canonical_bytes != control.trace_canonical_bytes
