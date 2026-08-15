"""D2 pilot pair custody: canonical run records, matched-pair rules, staging.

Turns replay-verified run bundles into canonical, self-hashed run records,
enforces the frozen matched-pair shape (exactly one NO_PROPOSAL control plus
the four frozen action proposals sharing scenario, noise and reset lineage),
and stages pair packets atomically.  This layer performs no corpus campaign,
no projection and no training: it persists execution evidence only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_json_bytes
from .pilot import (
    APPROVED_PROFILE_ACTION_SHA256,
    APPROVED_ROSTER_SHA256,
    PilotContinuation,
    PilotDesign,
)
from .pilot_execution import (
    PilotActionRunBundle,
    PilotControlRunBundle,
    PilotExecutionError,
    _validate_noise_and_nonce,
    _validate_plan_identity,
)

RUN_RECORD_SCHEMA: str = "aeolus_habitat_v2_forecast_pilot_run_record_v1"
PAIR_MANIFEST_SCHEMA: str = "aeolus_habitat_v2_forecast_pilot_pair_manifest_v1"

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_kind",
        "continuation_id",
        "pair_id",
        "matched_control_id",
        "cluster_id",
        "operating_mode",
        "load_regime",
        "semantic_profile_role",
        "repetition_id",
        "member_id",
        "treatment_duration",
        "treatment_interval",
        "anchor_completed_step",
        "variant",
        "action_id",
        "requested_command_sha256",
        "noise_seed",
        "hmc_reset_nonce_hex",
        "scenario_sha256",
        "control_run_id",
        "authority_epoch",
        "trace_sha256",
        "trace_final_state_sha256",
        "replay_final_state_sha256",
        "committed_step_count",
        "witnesses",
        "anchor",
        "record_sha256",
    }
)

_SHARED_PLAN_FIELDS = (
    "continuation_id",
    "pair_id",
    "matched_control_id",
    "cluster_id",
    "repetition_id",
    "member_id",
    "anchor_completed_step",
    "noise_seed",
    "hmc_reset_nonce_hex",
)

_PAIR_MATCHED_FIELDS = (
    "pair_id",
    "cluster_id",
    "operating_mode",
    "load_regime",
    "semantic_profile_role",
    "repetition_id",
    "member_id",
    "treatment_duration",
    "treatment_interval",
    "anchor_completed_step",
    "noise_seed",
    "hmc_reset_nonce_hex",
    "scenario_sha256",
    "control_run_id",
    "authority_epoch",
)


class PilotCustodyError(ValueError):
    """Pilot run evidence or pair custody is outside its frozen contract."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_run_record(
    design: PilotDesign,
    continuation: PilotContinuation,
    bundle: PilotControlRunBundle | PilotActionRunBundle,
) -> dict[str, Any]:
    """Bind one executed run bundle to the frozen plan as a canonical record."""
    if type(design) is not PilotDesign or type(continuation) is not PilotContinuation:
        raise PilotCustodyError("run record requires exact pilot plan types")
    expected_bundle = (
        PilotControlRunBundle
        if continuation.variant == "MATCHED_CONTROL"
        else PilotActionRunBundle
    )
    if type(bundle) is not expected_bundle:
        raise PilotCustodyError("run bundle variant does not match continuation")
    try:
        _validate_plan_identity(design, continuation)
        _validate_noise_and_nonce(design, continuation)
    except PilotExecutionError as error:
        raise PilotCustodyError(
            "continuation identity drifts from frozen plan"
        ) from error
    for field in _SHARED_PLAN_FIELDS:
        if getattr(bundle, field) != getattr(continuation, field):
            raise PilotCustodyError("run bundle does not match its continuation")
    if continuation.variant == "ACTION_PROPOSAL" and (
        bundle.action_id != continuation.action_id  # type: ignore[union-attr]
    ):
        raise PilotCustodyError("run bundle does not match its continuation")

    anchor = getattr(bundle, "anchor", None)
    requested = getattr(bundle, "requested_command_sha256", None)
    record: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA,
        "record_kind": "PILOT_RUN",
        "continuation_id": continuation.continuation_id,
        "pair_id": continuation.pair_id,
        "matched_control_id": continuation.matched_control_id,
        "cluster_id": continuation.cluster_id,
        "operating_mode": continuation.operating_mode,
        "load_regime": continuation.load_regime,
        "semantic_profile_role": continuation.semantic_profile_role,
        "repetition_id": continuation.repetition_id,
        "member_id": continuation.member_id,
        "treatment_duration": continuation.treatment_duration,
        "treatment_interval": (
            list(continuation.treatment_interval)
            if continuation.treatment_interval is not None
            else None
        ),
        "anchor_completed_step": continuation.anchor_completed_step,
        "variant": continuation.variant,
        "action_id": continuation.action_id,
        "requested_command_sha256": requested,
        "noise_seed": continuation.noise_seed,
        "hmc_reset_nonce_hex": continuation.hmc_reset_nonce_hex,
        "scenario_sha256": bundle.scenario_sha256,
        "control_run_id": bundle.control_run_id,
        "authority_epoch": bundle.authority_epoch,
        "trace_sha256": bundle.trace_sha256,
        "trace_final_state_sha256": bundle.trace_final_state_sha256,
        "replay_final_state_sha256": bundle.replay_final_state_sha256,
        "committed_step_count": bundle.committed_step_count,
        "witnesses": [dict(witness) for witness in bundle.witnesses],
        "anchor": dict(anchor) if anchor is not None else None,
    }
    record["record_sha256"] = _sha256(canonical_json_bytes(record))
    return record


def validate_run_record(record: Mapping[str, Any]) -> None:
    """Reparse one canonical run record; any drift or tampering fails closed."""
    if not isinstance(record, Mapping):
        raise PilotCustodyError("run record must be a mapping")
    actual = set(record)
    if actual != _RECORD_FIELDS:
        raise PilotCustodyError(
            f"run record has unknown={sorted(actual - _RECORD_FIELDS)}, "
            f"missing={sorted(_RECORD_FIELDS - actual)}"
        )
    body = dict(record)
    declared = body.pop("record_sha256")
    if declared != _sha256(canonical_json_bytes(body)):
        raise PilotCustodyError("run record self-hash is invalid")
    if (
        record["schema_version"] != RUN_RECORD_SCHEMA
        or record["record_kind"] != "PILOT_RUN"
    ):
        raise PilotCustodyError("run record schema or kind drifts")
    if record["variant"] == "MATCHED_CONTROL":
        if (
            record["action_id"] != "NO_PROPOSAL"
            or record["matched_control_id"] != record["continuation_id"]
            or record["anchor"] is not None
            or record["requested_command_sha256"] is not None
        ):
            raise PilotCustodyError("control run record is malformed")
    elif record["variant"] == "ACTION_PROPOSAL":
        if (
            record["action_id"] == "NO_PROPOSAL"
            or record["matched_control_id"] == record["continuation_id"]
            or not isinstance(record["anchor"], Mapping)
            or type(record["requested_command_sha256"]) is not str
        ):
            raise PilotCustodyError("action run record is malformed")
    else:
        raise PilotCustodyError("run record variant is unsupported")
    if (
        type(record["committed_step_count"]) is not int
        or record["committed_step_count"] != 72
        or not isinstance(record["witnesses"], list)
        or len(record["witnesses"]) != record["committed_step_count"]
    ):
        raise PilotCustodyError("run record execution closure drifts")
    if record["replay_final_state_sha256"] != record["trace_final_state_sha256"]:
        raise PilotCustodyError("run record replay identity drifts")


def validate_pilot_pair(
    design: PilotDesign, records: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
) -> None:
    """Enforce one shared control plus the four frozen action records."""
    if type(design) is not PilotDesign:
        raise PilotCustodyError("pair validation requires the exact pilot design")
    items = list(records)
    for record in items:
        validate_run_record(record)
    if len({record["continuation_id"] for record in items}) != len(items):
        raise PilotCustodyError("pair contains duplicate run identities")
    controls = [
        record for record in items if record["variant"] == "MATCHED_CONTROL"
    ]
    if len(controls) != 1:
        raise PilotCustodyError("pair requires exactly one control record")
    control = controls[0]
    actions = [record for record in items if record["variant"] == "ACTION_PROPOSAL"]
    if sorted(record["action_id"] for record in actions) != sorted(design.action_ids):
        raise PilotCustodyError("pair must contain the four frozen action records")
    for field in _PAIR_MATCHED_FIELDS:
        if len({_freeze_json(record[field]) for record in items}) != 1:
            raise PilotCustodyError("matched pair fields diverge")
    for record in actions:
        if record["matched_control_id"] != control["continuation_id"]:
            raise PilotCustodyError("action record references the wrong matched control")


def _freeze_json(value: Any) -> bytes:
    return canonical_json_bytes(value)


def stage_pair_packet(
    destination: str | Path,
    design: PilotDesign,
    records: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Write one validated pair packet; existing destinations are refused."""
    validate_pilot_pair(design, records)
    target = Path(destination)
    if target.exists():
        raise PilotCustodyError("pair packet destination already exists")
    jsonl = b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)
    manifest: dict[str, Any] = {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "pair_id": records[0]["pair_id"],
        "record_count": len(records),
        "records_sha256": _sha256(jsonl),
        "roster_sha256": APPROVED_ROSTER_SHA256,
        "profile_action_sha256": APPROVED_PROFILE_ACTION_SHA256,
    }
    manifest["manifest_sha256"] = _sha256(canonical_json_bytes(manifest))
    try:
        target.mkdir(parents=True)
        (target / "records.jsonl").write_bytes(jsonl)
        (target / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    except OSError as error:
        raise PilotCustodyError("pair packet cannot be staged") from error
    return manifest
