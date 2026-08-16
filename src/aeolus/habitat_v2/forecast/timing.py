"""D1's deliberately underpowered timing and baseline stop receipts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final

RELEASE_TIER: Final = "DEVELOPMENT_FIXTURE_ONLY"
INPUT_MANIFEST_SHA256: Final = (
    "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
)
TARGET_MANIFEST_SHA256: Final = (
    "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"
)
OUTCOME: Final = "STOP_UNDERPOWERED"
WINDOW_CANDIDATES: Final = frozenset((4, 8, 16))
HORIZON_CANDIDATES: Final = frozenset((2, 4, 8))


class TimingError(ValueError):
    """D1 timing/baseline gate evidence or outcome is outside the frozen scope."""


@dataclass(frozen=True, slots=True)
class StopReceipt:
    receipt_kind: str
    release_tier: str
    outcome: str
    window_steps: int | None
    horizon_steps: int | None
    input_manifest_sha256: str
    target_manifest_sha256: str
    evidence_sha256: str
    receipt_sha256: str


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TimingError("receipt evidence is not canonical finite JSON") from error


def _validate_manifests(
    input_manifest_sha256: str, target_manifest_sha256: str
) -> None:
    if (input_manifest_sha256, target_manifest_sha256) != (
        INPUT_MANIFEST_SHA256,
        TARGET_MANIFEST_SHA256,
    ):
        raise TimingError("receipt manifest identity drift")


def validate_candidate_timing(window_steps: int, horizon_steps: int) -> None:
    if (
        type(window_steps) is not int
        or type(horizon_steps) is not int
        or window_steps not in WINDOW_CANDIDATES
        or horizon_steps not in HORIZON_CANDIDATES
    ):
        raise TimingError("only frozen D1 W/H candidates are supported")


def _receipt(
    kind: str,
    evidence: dict[str, Any],
    input_manifest_sha256: str,
    target_manifest_sha256: str,
    *,
    window_steps: int | None,
    horizon_steps: int | None,
    outcome: str,
) -> StopReceipt:
    if outcome != OUTCOME:
        raise TimingError("D1 emits only STOP_UNDERPOWERED")
    if type(evidence) is not dict:
        raise TimingError("receipt evidence must be one canonical object")
    _validate_manifests(input_manifest_sha256, target_manifest_sha256)
    evidence_sha = hashlib.sha256(_canonical(evidence)).hexdigest()
    body = {
        "receipt_kind": kind,
        "release_tier": RELEASE_TIER,
        "outcome": OUTCOME,
        "window_steps": window_steps,
        "horizon_steps": horizon_steps,
        "input_manifest_sha256": input_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "evidence_sha256": evidence_sha,
    }
    return StopReceipt(
        **body, receipt_sha256=hashlib.sha256(_canonical(body)).hexdigest()
    )


def emit_timing_receipt(
    window_steps: int,
    horizon_steps: int,
    *,
    timing_evidence: dict[str, Any],
    input_manifest_sha256: str,
    target_manifest_sha256: str,
    outcome: str = OUTCOME,
) -> StopReceipt:
    """Bind every supplied timing-evidence byte while refusing timing selection."""
    validate_candidate_timing(window_steps, horizon_steps)
    return _receipt(
        "D1_TIMING_GATE",
        timing_evidence,
        input_manifest_sha256,
        target_manifest_sha256,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        outcome=outcome,
    )


def emit_baseline_gate_receipt(
    *,
    baseline_evidence: dict[str, Any],
    input_manifest_sha256: str,
    target_manifest_sha256: str,
    outcome: str = OUTCOME,
) -> StopReceipt:
    """Bind baseline evidence without fabricating action-information support."""
    return _receipt(
        "D1_BASELINE_GATE",
        baseline_evidence,
        input_manifest_sha256,
        target_manifest_sha256,
        window_steps=None,
        horizon_steps=None,
        outcome=outcome,
    )
