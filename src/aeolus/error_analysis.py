"""Strict, role-isolated reports for historical model-error forensics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPORT_FORMAT = "aeolus_forensic_error_report_v1"
FORENSIC_EVIDENCE_ROLE = "historical_forensic_only"
EXCLUDED_TRANSITION_LABEL = "excluded_transition"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPORT_KEYS = frozenset(
    {
        "format",
        "evidence_role",
        "source_manifest_sha256",
        "source_model_sha256",
        "input_row_count",
        "scored_row_count",
        "excluded_transition_row_count",
        "error_count",
        "groups",
    }
)
_GROUP_KEYS = frozenset(
    {
        "true_class",
        "predicted_class",
        "family_id",
        "scenario_role",
        "operating_profile_id",
        "count",
    }
)


def reject_forensic_report_input(value: object) -> None:
    """Fail closed when a v4 API receives historical-forensic evidence."""
    if isinstance(value, Mapping) and (
        value.get("evidence_role") == FORENSIC_EVIDENCE_ROLE
        or value.get("format") == REPORT_FORMAT
    ):
        raise ValueError(
            "historical forensic reports are not valid v4 development input"
        )


def write_error_report(path: str | Path, report: object) -> None:
    """Validate and write one deterministic finite forensic JSON document."""
    validate_error_report(report)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )


def validate_error_report(report: object) -> None:
    """Validate the exact forensic schema, provenance, and count invariants."""
    _require_finite_json(report, "forensic report")
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise ValueError("forensic report schema mismatch")
    if report["format"] != REPORT_FORMAT:
        raise ValueError("forensic report format is unsupported")
    if report["evidence_role"] != FORENSIC_EVIDENCE_ROLE:
        raise ValueError("forensic report evidence_role is unsupported")
    _require_sha256(report["source_manifest_sha256"], "source manifest hash")
    _require_sha256(report["source_model_sha256"], "source model hash")

    count_names = (
        "input_row_count",
        "scored_row_count",
        "excluded_transition_row_count",
        "error_count",
    )
    for name in count_names:
        if not _is_non_negative_int(report[name]):
            raise ValueError(f"forensic report {name} must be a non-negative integer")
    if report["input_row_count"] != (
        report["scored_row_count"] + report["excluded_transition_row_count"]
    ):
        raise ValueError("forensic report row counts are inconsistent")
    if report["error_count"] > report["scored_row_count"]:
        raise ValueError("forensic report error count exceeds scored rows")

    groups = report["groups"]
    if not isinstance(groups, list):
        raise ValueError("forensic report groups must be a list")
    identities: list[tuple[str, str, str, str, str]] = []
    group_error_count = 0
    for group in groups:
        if not isinstance(group, dict) or set(group) != _GROUP_KEYS:
            raise ValueError("forensic report group schema mismatch")
        identity_values = tuple(
            group[name]
            for name in (
                "true_class",
                "predicted_class",
                "family_id",
                "scenario_role",
                "operating_profile_id",
            )
        )
        if any(not isinstance(value, str) or not value for value in identity_values):
            raise ValueError(
                "forensic report group identities must be non-empty strings"
            )
        identity = identity_values
        if identity[0] == identity[1]:
            raise ValueError("forensic report groups must contain only errors")
        count = group["count"]
        if not _is_non_negative_int(count) or count == 0:
            raise ValueError("forensic report group count must be a positive integer")
        identities.append(identity)
        group_error_count += count
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("forensic report groups must be unique and sorted")
    if group_error_count != report["error_count"]:
        raise ValueError("forensic report grouped error count is inconsistent")


def build_error_report(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    family_evidence: Mapping[str, object],
    *,
    source_manifest_sha256: str,
    source_model_sha256: str,
) -> dict[str, object]:
    """Build deterministic grouped diagnostics from explicitly supplied evidence."""
    _require_sha256(source_manifest_sha256, "source manifest hash")
    _require_sha256(source_model_sha256, "source model hash")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("forensic rows must be a sequence")
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise ValueError("forensic predictions must be a sequence")
    if len(rows) != len(predictions):
        raise ValueError("forensic rows and predictions must have equal length")
    if not rows:
        raise ValueError("forensic error analysis requires at least one row")
    if not isinstance(family_evidence, Mapping) or not family_evidence:
        raise ValueError("forensic family evidence must be a non-empty mapping")

    groups: Counter[tuple[str, str, str, str, str]] = Counter()
    scored_row_count = 0
    excluded_transition_row_count = 0
    for index, (row, predicted_class) in enumerate(
        zip(rows, predictions, strict=True), start=1
    ):
        if not isinstance(row, Mapping):
            raise ValueError(f"forensic row {index} must be an object")
        true_class = _required_string(row, "label", f"forensic row {index}")
        family_id = _required_string(row, "family_id", f"forensic row {index}")
        scenario_role = _required_string(row, "scenario_role", f"forensic row {index}")
        if not isinstance(predicted_class, str) or not predicted_class:
            raise ValueError(f"forensic prediction {index} must be a non-empty string")
        evidence = family_evidence.get(family_id)
        if evidence is None:
            raise ValueError(
                f"forensic row {index} family_id is absent from family evidence"
            )
        evidence_family_id = _optional_field(evidence, "family_id")
        if evidence_family_id is not None and evidence_family_id != family_id:
            raise ValueError("forensic family evidence identity is inconsistent")
        split = _resolve_split(row, evidence, index)
        operating_profile_id = _resolve_operating_profile(
            row,
            evidence,
            family_id=family_id,
            split=split,
            row_number=index,
        )

        if true_class == EXCLUDED_TRANSITION_LABEL:
            excluded_transition_row_count += 1
            continue
        scored_row_count += 1
        if predicted_class != true_class:
            groups[
                (
                    true_class,
                    predicted_class,
                    family_id,
                    scenario_role,
                    operating_profile_id,
                )
            ] += 1

    grouped_errors = [
        {
            "true_class": key[0],
            "predicted_class": key[1],
            "family_id": key[2],
            "scenario_role": key[3],
            "operating_profile_id": key[4],
            "count": count,
        }
        for key, count in sorted(groups.items())
    ]
    error_count = sum(group["count"] for group in grouped_errors)
    report: dict[str, object] = {
        "format": REPORT_FORMAT,
        "evidence_role": FORENSIC_EVIDENCE_ROLE,
        "source_manifest_sha256": source_manifest_sha256,
        "source_model_sha256": source_model_sha256,
        "input_row_count": len(rows),
        "scored_row_count": scored_row_count,
        "excluded_transition_row_count": excluded_transition_row_count,
        "error_count": error_count,
        "groups": grouped_errors,
    }
    validate_error_report(report)
    return report


def _resolve_split(
    row: Mapping[str, Any], evidence: object, row_number: int
) -> str | None:
    row_split = row.get("split")
    evidence_split = _optional_field(evidence, "split")
    for value in (row_split, evidence_split):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"forensic row {row_number} split is malformed")
    if (
        row_split is not None
        and evidence_split is not None
        and row_split != evidence_split
    ):
        raise ValueError(
            f"forensic row {row_number} split disagrees with family evidence"
        )
    if isinstance(row_split, str):
        return row_split
    if isinstance(evidence_split, str):
        return evidence_split
    return None


def _resolve_operating_profile(
    row: Mapping[str, Any],
    evidence: object,
    *,
    family_id: str,
    split: str | None,
    row_number: int,
) -> str:
    row_profile = row.get("operating_profile_id")
    evidence_profile = _optional_field(evidence, "operating_profile_id")
    for value in (row_profile, evidence_profile):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"forensic row {row_number} operating_profile_id is malformed"
            )
    if (
        row_profile is not None
        and evidence_profile is not None
        and row_profile != evidence_profile
    ):
        raise ValueError(
            f"forensic row {row_number} operating profile disagrees with family evidence"
        )
    explicit = row_profile if isinstance(row_profile, str) else evidence_profile
    if isinstance(explicit, str):
        return explicit
    if split is None:
        raise ValueError(
            f"forensic row {row_number} cannot derive operating profile without split"
        )
    pattern = re.compile(
        rf"^{re.escape(split)}-s[0-9]+-"
        r"(?P<profile>[a-z0-9]+(?:-[a-z0-9]+)*)-t[0-9]+-.+$"
    )
    match = pattern.fullmatch(family_id)
    if match is None:
        raise ValueError(
            f"forensic row {row_number} cannot derive operating profile from family_id"
        )
    return match.group("profile")


def _required_string(value: Mapping[str, Any], field: str, description: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{description} {field} must be a non-empty string")
    return item


def _optional_field(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _require_sha256(value: object, description: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")


def _require_finite_json(value: object, description: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite number")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _require_finite_json(nested, f"{description}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json(nested, f"{description}[{index}]")


def _is_non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0
