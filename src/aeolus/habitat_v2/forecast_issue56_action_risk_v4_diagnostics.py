"""Pre-model diagnostics for the next Issue #56 action-risk study.

This module deliberately does not train or run a learned model.  It defines the
measurement and provenance boundary needed before a V4 experiment is authorized.
V3 artifacts remain readable through their existing contracts, and HMC remains
the only source of final-command and plant-step authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .forecast.contracts import canonical_json_bytes


ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION = "aeolus_habitat_v2_risk_issue_56_v4_diagnostics_v1"
V4_BOOTSTRAP_SEED = 560057
V4_BOOTSTRAP_RESAMPLES = 10_000
V4_PROPOSAL_OUTCOMES = (
    "PROPOSED_ACCEPTED",
    "PROPOSED_MODIFIED",
    "PROPOSED_REJECTED_TO_HOLD",
    "EMERGENCY_OVERRIDDEN",
)
V4_DISPOSITION_TYPES = V4_PROPOSAL_OUTCOMES + ("ABSTAINED_TO_HOLD",)
V4_PROVENANCE_FIELDS = (
    "source_identity_sha256",
    "hmc_binding_sha256",
    "hmc_contract_sha256",
    "scenario_sha256",
    "action_catalogue_sha256",
    "alarm_manifest_sha256",
    "feature_manifest_sha256",
    "label_manifest_sha256",
    "point_artifact_sha256",
)


class Issue56V4DiagnosticsError(ValueError):
    """Raised when a V4 diagnostic record or metric is malformed."""


def _sha(value: object) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError) as error:
        raise Issue56V4DiagnosticsError("V4 digest input is not canonical JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Issue56V4DiagnosticsError(f"{label} must be lowercase SHA-256")
    return value


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise Issue56V4DiagnosticsError(f"{label} must be a non-empty string")
    return value


def _require_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Issue56V4DiagnosticsError("V4 decision step must be a positive integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise Issue56V4DiagnosticsError(f"V4 {label} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class V4CandidateObservation:
    """One action-conditioned screening result.

    Candidate observations are not executed-command observations.  The latter
    require a separate record because HMC can modify or reject a proposal.
    """

    condition_group_id: str
    family_id: str
    decision_step: int
    action_id: str
    model_rejected: bool
    dangerous: bool
    observation_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.condition_group_id, "V4 condition group")
        _require_identifier(self.family_id, "V4 family")
        _require_step(self.decision_step)
        _require_identifier(self.action_id, "V4 action")
        _require_bool(self.model_rejected, "candidate rejection")
        _require_bool(self.dangerous, "candidate dangerous label")
        _require_sha(self.observation_sha256, "V4 candidate observation")
        if self.observation_sha256 != _sha(self._body()):
            raise Issue56V4DiagnosticsError("V4 candidate observation digest is inconsistent")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.candidate",
            "condition_group_id": self.condition_group_id,
            "family_id": self.family_id,
            "decision_step": self.decision_step,
            "action_id": self.action_id,
            "model_rejected": self.model_rejected,
            "dangerous": self.dangerous,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "observation_sha256": self.observation_sha256}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V4CandidateObservation":
        expected = set(cls._field_names())
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V4DiagnosticsError("V4 candidate observation fields drift")
        if mapping["schema_version"] != f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.candidate":
            raise Issue56V4DiagnosticsError("V4 candidate observation schema drift")
        return cls(
            mapping["condition_group_id"],
            mapping["family_id"],
            mapping["decision_step"],
            mapping["action_id"],
            mapping["model_rejected"],
            mapping["dangerous"],
            mapping["observation_sha256"],
        )

    @staticmethod
    def _field_names() -> tuple[str, ...]:
        return (
            "schema_version",
            "condition_group_id",
            "family_id",
            "decision_step",
            "action_id",
            "model_rejected",
            "dangerous",
            "observation_sha256",
        )


@dataclass(frozen=True, slots=True)
class V4ExecutedObservation:
    """One decision's selected/requested/final/executed HMC outcome."""

    condition_group_id: str
    family_id: str
    arm: str
    decision_step: int
    selected_action_id: str | None
    actual_dangerous: bool
    requested_command_sha256: str | None
    final_command_sha256: str
    executed_command_sha256: str
    disposition: str
    observation_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.condition_group_id, "V4 condition group")
        _require_identifier(self.family_id, "V4 family")
        _require_identifier(self.arm, "V4 arm")
        _require_step(self.decision_step)
        _require_bool(self.actual_dangerous, "executed dangerous label")
        _require_sha(self.final_command_sha256, "V4 final command")
        _require_sha(self.executed_command_sha256, "V4 executed command")
        if self.final_command_sha256 != self.executed_command_sha256:
            raise Issue56V4DiagnosticsError("V4 final and executed commands differ")
        if self.disposition not in V4_DISPOSITION_TYPES:
            raise Issue56V4DiagnosticsError("V4 disposition is invalid")
        if self.selected_action_id is None:
            if self.requested_command_sha256 is not None:
                raise Issue56V4DiagnosticsError("V4 abstention has a requested command")
            if self.disposition != "ABSTAINED_TO_HOLD":
                raise Issue56V4DiagnosticsError("V4 no-proposal outcome is invalid")
        else:
            _require_identifier(self.selected_action_id, "V4 selected action")
            if self.requested_command_sha256 is None:
                raise Issue56V4DiagnosticsError("V4 selected action lacks a requested command")
            _require_sha(self.requested_command_sha256, "V4 requested command")
            if self.disposition == "ABSTAINED_TO_HOLD":
                raise Issue56V4DiagnosticsError("V4 proposal cannot be an abstention")
        _require_sha(self.observation_sha256, "V4 executed observation")
        if self.observation_sha256 != _sha(self._body()):
            raise Issue56V4DiagnosticsError("V4 executed observation digest is inconsistent")

    @property
    def proposal_submitted(self) -> bool:
        return self.selected_action_id is not None

    @property
    def hmc_mismatch(self) -> bool:
        return self.proposal_submitted and self.requested_command_sha256 != self.final_command_sha256

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.executed",
            "condition_group_id": self.condition_group_id,
            "family_id": self.family_id,
            "arm": self.arm,
            "decision_step": self.decision_step,
            "selected_action_id": self.selected_action_id,
            "actual_dangerous": self.actual_dangerous,
            "requested_command_sha256": self.requested_command_sha256,
            "final_command_sha256": self.final_command_sha256,
            "executed_command_sha256": self.executed_command_sha256,
            "disposition": self.disposition,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._body(), "observation_sha256": self.observation_sha256}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "V4ExecutedObservation":
        expected = set(cls._field_names())
        if type(mapping) is not dict or set(mapping) != expected:
            raise Issue56V4DiagnosticsError("V4 executed observation fields drift")
        if mapping["schema_version"] != f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.executed":
            raise Issue56V4DiagnosticsError("V4 executed observation schema drift")
        return cls(
            mapping["condition_group_id"],
            mapping["family_id"],
            mapping["arm"],
            mapping["decision_step"],
            mapping["selected_action_id"],
            mapping["actual_dangerous"],
            mapping["requested_command_sha256"],
            mapping["final_command_sha256"],
            mapping["executed_command_sha256"],
            mapping["disposition"],
            mapping["observation_sha256"],
        )

    @staticmethod
    def _field_names() -> tuple[str, ...]:
        return (
            "schema_version",
            "condition_group_id",
            "family_id",
            "arm",
            "decision_step",
            "selected_action_id",
            "actual_dangerous",
            "requested_command_sha256",
            "final_command_sha256",
            "executed_command_sha256",
            "disposition",
            "observation_sha256",
        )


def _validate_unique_keys(records: Sequence[Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    items = tuple(records)
    if not items:
        raise Issue56V4DiagnosticsError("V4 diagnostics require observations")
    keys = [tuple(getattr(item, field) for field in fields) for item in items]
    if len(set(keys)) != len(keys):
        raise Issue56V4DiagnosticsError("V4 observations contain duplicate decision/action rows")
    return items


def validate_condition_groups(
    records: Sequence[V4CandidateObservation | V4ExecutedObservation],
    *,
    expected_families_per_group: int = 2,
) -> dict[str, tuple[str, ...]]:
    """Validate paired-family grouping before equal-weight aggregation."""

    if isinstance(expected_families_per_group, bool) or expected_families_per_group < 1:
        raise Issue56V4DiagnosticsError("V4 group family count is invalid")
    items = tuple(records)
    if not items:
        raise Issue56V4DiagnosticsError("V4 diagnostics require observations")
    families: dict[str, set[str]] = defaultdict(set)
    for item in items:
        _require_identifier(item.condition_group_id, "V4 condition group")
        _require_identifier(item.family_id, "V4 family")
        families[item.condition_group_id].add(item.family_id)
    result = {
        group_id: tuple(sorted(group_families))
        for group_id, group_families in sorted(families.items())
    }
    if any(len(group) != expected_families_per_group for group in result.values()):
        raise Issue56V4DiagnosticsError("V4 condition groups do not contain paired families")
    return result


def candidate_screening_metrics(
    observations: Sequence[V4CandidateObservation],
) -> dict[str, Any]:
    """Measure screening behavior over all candidate actions only."""

    items = _validate_unique_keys(
        observations,
        ("condition_group_id", "family_id", "decision_step", "action_id"),
    )
    dangerous = sum(item.dangerous for item in items)
    rejected_dangerous = sum(item.dangerous and item.model_rejected for item in items)
    retained = sum(not item.model_rejected for item in items)
    retained_dangerous = sum(item.dangerous and not item.model_rejected for item in items)
    return {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.candidate-metrics",
        "sample_count": len(items),
        "dangerous_sample_count": dangerous,
        "retained_sample_count": retained,
        "rejected_dangerous_count": rejected_dangerous,
        "retained_dangerous_count": retained_dangerous,
        "dangerous_event_recall": rejected_dangerous / dangerous if dangerous else None,
        "false_safe_rate": retained_dangerous / retained if retained else None,
        "unfiltered_reference_false_safe_rate": dangerous / len(items),
        "support": {
            "dangerous_labels": dangerous > 0,
            "retained_candidates": retained > 0,
        },
    }


def executed_action_metrics(
    observations: Sequence[V4ExecutedObservation],
) -> dict[str, Any]:
    """Measure selected, executed, abstention, and HMC arbitration outcomes."""

    items = _validate_unique_keys(
        observations,
        ("condition_group_id", "family_id", "arm", "decision_step"),
    )
    selected = [item for item in items if item.proposal_submitted]
    dangerous = sum(item.actual_dangerous for item in items)
    selected_dangerous = sum(item.actual_dangerous for item in selected)
    mismatches = sum(item.hmc_mismatch for item in items)
    dispositions = Counter(item.disposition for item in items)
    return {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.executed-metrics",
        "decision_count": len(items),
        "proposal_count": len(selected),
        "abstention_count": len(items) - len(selected),
        "dangerous_executed_count": dangerous,
        "selected_dangerous_count": selected_dangerous,
        "proposal_rate": len(selected) / len(items),
        "abstention_rate": (len(items) - len(selected)) / len(items),
        "executed_command_dangerous_event_rate": dangerous / len(items),
        "selected_action_false_safe_rate": (
            selected_dangerous / len(selected) if selected else None
        ),
        "hmc_mismatch_count": mismatches,
        "hmc_mismatch_rate": mismatches / len(selected) if selected else None,
        "disposition_counts": {
            disposition: dispositions.get(disposition, 0)
            for disposition in V4_DISPOSITION_TYPES
        },
        "support": {
            "selected_actions": bool(selected),
            "dangerous_executed_labels": dangerous > 0,
        },
    }


def equal_weight_group_mean(group_values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Aggregate within groups, then give every group equal weight."""

    if not group_values:
        raise Issue56V4DiagnosticsError("V4 group aggregation requires values")
    means: dict[str, float] = {}
    for group_id, values in sorted(group_values.items()):
        _require_identifier(group_id, "V4 condition group")
        array = np.asarray(tuple(values), dtype=np.float64)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise Issue56V4DiagnosticsError("V4 group values must be finite and non-empty")
        means[group_id] = float(np.mean(array))
    return {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.group-mean",
        "group_count": len(means),
        "group_means": means,
        "equal_weight_mean": float(np.mean(np.asarray(tuple(means.values()), dtype=np.float64))),
    }


def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    result = np.empty((repetitions, count), dtype=np.int64)
    for replicate in range(repetitions):
        for draw in range(count):
            digest = hashlib.sha256(
                f"issue56-v4-bootstrap-v1|{seed}|{replicate}|{draw}".encode("utf-8")
            ).digest()
            result[replicate, draw] = int.from_bytes(digest[:8], "big") % count
    return result


def bootstrap_equal_weight_group_mean(
    group_means: Mapping[str, float],
    *,
    seed: int = V4_BOOTSTRAP_SEED,
    resamples: int = V4_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap independent group means with a content-independent sampler."""

    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise Issue56V4DiagnosticsError("V4 bootstrap resample count is invalid")
    if len(group_means) < 2:
        raise Issue56V4DiagnosticsError("V4 bootstrap requires at least two groups")
    identifiers = tuple(sorted(group_means))
    values = np.asarray([group_means[key] for key in identifiers], dtype=np.float64)
    if not np.isfinite(values).all():
        raise Issue56V4DiagnosticsError("V4 bootstrap group means must be finite")
    draws = np.mean(values[_bootstrap_indices(seed, resamples, len(values))], axis=1)
    return {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.bootstrap",
        "group_ids": list(identifiers),
        "group_count": len(identifiers),
        "point_estimate": float(np.mean(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def observation_manifest_sha256(
    observations: Sequence[V4CandidateObservation | V4ExecutedObservation],
) -> str:
    """Hash canonical observation rows in a stable order."""

    items = tuple(observations)
    if not items:
        raise Issue56V4DiagnosticsError("V4 observation manifest requires observations")
    rows = sorted(canonical_json_bytes(item.to_mapping()) for item in items)
    return _sha_bytes(
        b"issue56-v4-observation-manifest-v1\n" + b"\n".join(rows) + b"\n"
    )


def provenance_manifest_sha256(manifest: Mapping[str, str]) -> str:
    """Validate and hash the complete V4 provenance identity set."""

    if type(manifest) is not dict or set(manifest) != set(V4_PROVENANCE_FIELDS):
        raise Issue56V4DiagnosticsError("V4 provenance fields drift")
    for field in V4_PROVENANCE_FIELDS:
        _require_sha(manifest[field], f"V4 {field}")
    return _sha(
        {
            "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.provenance",
            "identities": {field: manifest[field] for field in V4_PROVENANCE_FIELDS},
        }
    )
