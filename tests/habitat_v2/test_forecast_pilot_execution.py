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
