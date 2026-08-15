"""D2 pilot HMC execution boundary.

This module executes frozen ``PilotContinuation`` plans through the real
``reset -> observe -> verify -> propose -> arbitrate -> step`` lifecycle with
independent shadow physics and strict trace replay.  It performs no scenario
generation campaign, no persistence and no model training: it returns one
in-memory run bundle per continuation.  Batch custody, the evidence-record
contract and the resource preflight are separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.control_trace import parse_control_trace, replay_control_trace
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics import (
    advance_one_step_with_command,
    initial_state,
    validate_external_command,
)

from .contracts import canonical_json_bytes, load_forecast_contracts
from .pilot import (
    _HMC_CONTRACT_SHA256,
    _HMC_IMPLEMENTATION_GIT_SHA,
    _identity,
    _noise_seed,
    _reset_nonce,
    PilotContinuation,
    PilotDesign,
    materialize_pilot_scenario,
)
from .pipeline import _proposal, _state_sha256


class PilotExecutionError(ValueError):
    """A pilot continuation cannot be executed or its evidence is invalid."""


_PAIR_IDENTITY_DOMAIN = b"aeolus-habitat-v2-forecast-pilot-pair-v1"
_CONTINUATION_IDENTITY_DOMAIN = (
    b"aeolus-habitat-v2-forecast-pilot-continuation-v1"
)


@dataclass(frozen=True, slots=True)
class PilotControlRunBundle:
    """In-memory replay-verified evidence from one MATCHED_CONTROL run."""

    continuation_id: str
    pair_id: str
    matched_control_id: str
    cluster_id: str
    repetition_id: str
    member_id: str
    anchor_completed_step: int
    noise_seed: int
    hmc_reset_nonce_hex: str
    scenario_sha256: str
    control_run_id: str
    authority_epoch: str
    trace_canonical_bytes: bytes
    trace_sha256: str
    trace_final_state_sha256: str
    replay_final_state_sha256: str
    committed_step_count: int
    witnesses: tuple[dict[str, Any], ...]
    snapshots: dict[int, tuple[Any, Any]]
    states: dict[int, Any]


@dataclass(frozen=True, slots=True)
class PilotActionRunBundle:
    """In-memory replay-verified evidence from one ACTION_PROPOSAL run."""

    continuation_id: str
    pair_id: str
    matched_control_id: str
    cluster_id: str
    repetition_id: str
    member_id: str
    anchor_completed_step: int
    noise_seed: int
    hmc_reset_nonce_hex: str
    scenario_sha256: str
    control_run_id: str
    authority_epoch: str
    trace_canonical_bytes: bytes
    trace_sha256: str
    trace_final_state_sha256: str
    replay_final_state_sha256: str
    committed_step_count: int
    witnesses: tuple[dict[str, Any], ...]
    snapshots: dict[int, tuple[Any, Any]]
    states: dict[int, Any]
    action_id: str
    requested_command_sha256: str
    anchor: dict[str, Any]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_plan_identity(
    design: PilotDesign, continuation: PilotContinuation
) -> None:
    pair_id = _identity(
        _PAIR_IDENTITY_DOMAIN,
        {
            "cluster_id": continuation.cluster_id,
            "repetition_id": continuation.repetition_id,
            "member_id": continuation.member_id,
            "anchor_completed_step": continuation.anchor_completed_step,
        },
    )
    control_id = _identity(
        _CONTINUATION_IDENTITY_DOMAIN,
        {"pair_id": pair_id, "action_id": "NO_PROPOSAL"},
    )
    continuation_id = _identity(
        _CONTINUATION_IDENTITY_DOMAIN,
        {"pair_id": pair_id, "action_id": continuation.action_id},
    )
    if (
        continuation.pair_id != pair_id
        or continuation.matched_control_id != control_id
        or continuation.continuation_id != continuation_id
    ):
        raise PilotExecutionError(
            "matched control or continuation identity drifts from frozen plan"
        )


def _validate_noise_and_nonce(
    design: PilotDesign, continuation: PilotContinuation
) -> None:
    if (
        continuation.noise_seed
        != _noise_seed(design, continuation.cluster_id, continuation.repetition_id)
        or continuation.hmc_reset_nonce_hex
        != _reset_nonce(
            design,
            continuation.cluster_id,
            continuation.repetition_id,
            continuation.anchor_completed_step,
        ).hex()
    ):
        raise PilotExecutionError("noise seed or reset nonce drifts from frozen plan")


def _load_contracts(root: Path) -> Any:
    contracts = load_forecast_contracts(root)
    if contracts.hmc_contract.hmc_contract_sha256 != _HMC_CONTRACT_SHA256:
        raise PilotExecutionError("pilot HMC contract identity drifts")
    return contracts


def run_pilot_control_continuation(
    repo_root: str | Path,
    design: PilotDesign,
    continuation: PilotContinuation,
) -> PilotControlRunBundle:
    """Execute one matched NO_PROPOSAL control through strict replay closure."""
    if type(design) is not PilotDesign or type(continuation) is not PilotContinuation:
        raise PilotExecutionError("control execution requires exact pilot plan types")
    if (
        continuation.variant != "MATCHED_CONTROL"
        or continuation.action_id != "NO_PROPOSAL"
    ):
        raise PilotExecutionError(
            "control execution requires a MATCHED_CONTROL continuation"
        )
    if continuation.anchor_completed_step not in design.anchor_completed_steps:
        raise PilotExecutionError("continuation anchor is outside the frozen plan")
    _validate_plan_identity(design, continuation)
    _validate_noise_and_nonce(design, continuation)
    root = Path(repo_root).resolve()
    contracts = _load_contracts(root)
    scenario = materialize_pilot_scenario(
        root,
        design,
        cluster_id=continuation.cluster_id,
        member_id=continuation.member_id,
        repetition_id=continuation.repetition_id,
    )
    executed = _execute_lifecycle(
        scenario,
        contracts,
        continuation,
        action=None,
    )
    executed.pop("anchor")
    return PilotControlRunBundle(
        continuation_id=continuation.continuation_id,
        pair_id=continuation.pair_id,
        matched_control_id=continuation.matched_control_id,
        cluster_id=continuation.cluster_id,
        repetition_id=continuation.repetition_id,
        member_id=continuation.member_id,
        anchor_completed_step=continuation.anchor_completed_step,
        noise_seed=continuation.noise_seed,
        hmc_reset_nonce_hex=continuation.hmc_reset_nonce_hex,
        scenario_sha256=scenario.scenario_sha256,
        **executed,
    )


def run_pilot_action_continuation(
    repo_root: str | Path,
    design: PilotDesign,
    continuation: PilotContinuation,
) -> PilotActionRunBundle:
    """Execute one catalogue action proposal through strict replay closure."""
    if type(design) is not PilotDesign or type(continuation) is not PilotContinuation:
        raise PilotExecutionError("action execution requires exact pilot plan types")
    if continuation.variant != "ACTION_PROPOSAL":
        raise PilotExecutionError(
            "action execution requires an ACTION_PROPOSAL continuation"
        )
    if continuation.action_id not in design.action_ids:
        raise PilotExecutionError("continuation action is outside the frozen plan")
    if continuation.anchor_completed_step not in design.anchor_completed_steps:
        raise PilotExecutionError("continuation anchor is outside the frozen plan")
    _validate_plan_identity(design, continuation)
    _validate_noise_and_nonce(design, continuation)
    root = Path(repo_root).resolve()
    contracts = _load_contracts(root)
    actions = {action.action_id: action for action in contracts.actions}
    action = actions.get(continuation.action_id)
    if action is None:
        raise PilotExecutionError("catalogue does not bind the planned action")
    scenario = materialize_pilot_scenario(
        root,
        design,
        cluster_id=continuation.cluster_id,
        member_id=continuation.member_id,
        repetition_id=continuation.repetition_id,
    )
    canonical = validate_external_command(scenario, action.command.to_mapping())
    if canonical.sha256 != action.command_sha256:
        raise PilotExecutionError("catalogue command drifts against pilot scenario")
    executed = _execute_lifecycle(
        scenario,
        contracts,
        continuation,
        action=action,
    )
    anchor = executed.pop("anchor")
    if anchor is None:
        raise PilotExecutionError("action continuation never reached its anchor")
    return PilotActionRunBundle(
        continuation_id=continuation.continuation_id,
        pair_id=continuation.pair_id,
        matched_control_id=continuation.matched_control_id,
        cluster_id=continuation.cluster_id,
        repetition_id=continuation.repetition_id,
        member_id=continuation.member_id,
        anchor_completed_step=continuation.anchor_completed_step,
        noise_seed=continuation.noise_seed,
        hmc_reset_nonce_hex=continuation.hmc_reset_nonce_hex,
        scenario_sha256=scenario.scenario_sha256,
        action_id=continuation.action_id,
        requested_command_sha256=action.command_sha256,
        anchor=anchor,
        **executed,
    )


def _execute_lifecycle(
    scenario: Any,
    contracts: Any,
    continuation: PilotContinuation,
    *,
    action: Any | None,
) -> dict[str, Any]:
    """Run the HMC lifecycle; shadow physics and strict replay close the run."""
    total_steps = int(scenario.data["steps"])
    nonce = bytes.fromhex(continuation.hmc_reset_nonce_hex)
    hmc = HabitatManagementComputer.reset(scenario, contracts.hmc_contract, nonce)
    shadow = initial_state(scenario)
    snapshots: dict[int, tuple[Any, Any]] = {}
    states: dict[int, Any] = {0: shadow}
    witnesses: list[dict[str, Any]] = []
    anchor: dict[str, Any] | None = None
    for application_step in range(total_steps):
        observed = hmc.observe()
        if type(observed) is not tuple or len(observed) != 2:
            raise PilotExecutionError("HMC enters terminal state during observation")
        snapshot, verification = observed
        if (
            verification.completed_step != application_step
            or snapshot.snapshot_sha256 != verification.snapshot_sha256
        ):
            raise PilotExecutionError("issued snapshot and verification drift")
        handle = hmc.verify_snapshot(snapshot, verification)
        if application_step:
            snapshots[application_step] = (snapshot, verification)
        at_anchor = (
            action is not None
            and application_step == continuation.anchor_completed_step
        )
        proposal_value = None
        if at_anchor:
            proposal_value = _proposal(
                hmc,
                snapshot.snapshot_sha256,
                application_step,
                action.command.to_mapping(),
                action.action_id,
            )
        proposal_receipt = hmc.propose(proposal_value, handle)
        proposal_mapping = proposal_receipt.to_mapping()
        if at_anchor:
            if (
                proposal_mapping["attempt_class"],
                proposal_mapping["validation_outcome"],
            ) != ("CANONICAL_PROPOSAL", "VALID"):
                raise PilotExecutionError("anchor canonical proposal was not admitted")
        elif proposal_mapping["validation_outcome"] != "NO_PROPOSAL":
            raise PilotExecutionError(
                "pilot continuation admits a proposal outside the anchor"
            )
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise PilotExecutionError("HMC enters terminal state during arbitration")
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise PilotExecutionError("HMC enters terminal state during step")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        shadow_digest = _sha256(canonical_json_bytes(shadow_result.receipt))
        if shadow_digest != step_receipt.plant_receipt_digest:
            raise PilotExecutionError("shadow plant receipt diverges from HMC receipt")
        shadow = shadow_result.state
        states[shadow.step] = shadow
        witnesses.append(
            {
                "application_step": application_step,
                "final_command_sha256": arbitration.final_command_sha256,
                "hmc_plant_receipt_digest": step_receipt.plant_receipt_digest,
                "shadow_plant_receipt_digest": shadow_digest,
                "shadow_state_sha256": _state_sha256(shadow),
            }
        )
        if at_anchor:
            anchor = {
                "application_step": application_step,
                "proposal_receipt": proposal_mapping,
                "proposed_command": action.command.to_mapping(),
                "final_command_sha256": arbitration.final_command_sha256,
                "plant_receipt_digest": step_receipt.plant_receipt_digest,
                "requested_command_sha256": action.command_sha256,
                "shadow_plant_receipt_digest": shadow_digest,
            }
    if action is not None and anchor is None:
        raise PilotExecutionError("action continuation never reached its anchor")
    trace = hmc.export_control_trace(_HMC_IMPLEMENTATION_GIT_SHA)
    parsed = parse_control_trace(
        trace.canonical_bytes,
        scenario=scenario,
        contract=contracts.hmc_contract,
    )
    replay = replay_control_trace(
        trace.canonical_bytes,
        scenario=scenario,
        contract=contracts.hmc_contract,
    )
    if (
        parsed.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != total_steps
        or replay.final_state_sha256 != parsed.footer["final_state_sha256"]
    ):
        raise PilotExecutionError("pilot trace fails strict completed replay")
    return {
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "trace_canonical_bytes": trace.canonical_bytes,
        "trace_sha256": _sha256(trace.canonical_bytes),
        "trace_final_state_sha256": str(parsed.footer["final_state_sha256"]),
        "replay_final_state_sha256": replay.final_state_sha256,
        "committed_step_count": replay.committed_step_count,
        "witnesses": tuple(witnesses),
        "snapshots": snapshots,
        "states": states,
        "anchor": anchor,
    }
