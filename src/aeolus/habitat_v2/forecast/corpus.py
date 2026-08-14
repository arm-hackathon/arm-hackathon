"""Strict canonical records and leakage guards for the D1 development fixture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import hmac
import json
import math
from pathlib import PurePosixPath
from typing import Any


RELEASE_TIER = "DEVELOPMENT_FIXTURE_ONLY"
PATH_PREFIX = "development-fixture-only/"
FORBIDDEN_FIELDS = frozenset(
    {
        "custodian_id",
        "custody_receipt",
        "final_release_manifest",
        "final_score_receipt",
        "final_prediction_sha256",
    }
)
_NUMERIC_FIELDS = frozenset(
    {
        "anchor_completed_step",
        "application_step",
        "byte_length",
        "completed_observation_step",
        "final_sequence",
        "horizon_steps",
        "row_count",
        "transition_count",
        "window_steps",
    }
)


class CorpusValidationError(ValueError):
    """A D1 corpus byte, record, split, or lineage boundary is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CorpusValidationError("value is not canonical JSON") from error


def canonical_jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)


def _reject_constant(value: str) -> None:
    raise CorpusValidationError(f"non-finite JSON constant {value!r} is forbidden")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise CorpusValidationError(f"{label} must be lowercase SHA-256")
    return value


def _walk_json(value: Any, *, label: str = "record") -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is not str:
                raise CorpusValidationError(f"{label} has a non-string key")
            if key in FORBIDDEN_FIELDS:
                raise CorpusValidationError(f"{label} contains forbidden final-custody field {key}")
            _walk_json(nested, label=f"{label}.{key}")
    elif type(value) is list:
        for index, nested in enumerate(value):
            _walk_json(nested, label=f"{label}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise CorpusValidationError(f"{label} contains a non-finite number")


def _exact_fields(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual, closed = set(value), set(expected)
    if actual != closed:
        raise CorpusValidationError(
            f"{label} has unknown={sorted(actual - closed)}, missing={sorted(closed - actual)}"
        )


def _check_numeric_positions(value: Mapping[str, Any], label: str) -> None:
    for field in _NUMERIC_FIELDS & set(value):
        number = value[field]
        if type(number) is not int or number < 0:
            raise CorpusValidationError(f"{label}.{field} must be a non-boolean non-negative integer")


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise CorpusValidationError(f"{label} must be bytes")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CorpusValidationError) as error:
        if isinstance(error, CorpusValidationError):
            raise
        raise CorpusValidationError(f"{label} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise CorpusValidationError(f"{label} must be one JSON object")
    _walk_json(value, label=label)
    if canonical_json_bytes(value) != raw:
        raise CorpusValidationError(f"{label} bytes are not canonical")
    return value


def record_identity(record: Mapping[str, Any], specification: Mapping[str, Any]) -> str:
    """Return the contract-framed stable ID for one closed record body."""
    fields = specification.get("identity_fields")
    domain = specification.get("identity_domain")
    if not isinstance(fields, (list, tuple)) or type(domain) is not str or not domain:
        raise CorpusValidationError("record specification has no closed identity contract")
    if len(fields) != len(set(fields)) or any(type(field) is not str for field in fields):
        raise CorpusValidationError("record identity fields are malformed")
    if any(field not in record for field in fields):
        raise CorpusValidationError("record is missing an identity field")
    body = {field: record[field] for field in fields}
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + canonical_json_bytes(body)).hexdigest()


def record_self_hash(record: Mapping[str, Any], field: str = "record_sha256") -> str:
    if field not in record:
        raise CorpusValidationError(f"record lacks self-hash field {field}")
    body = dict(record)
    body.pop(field)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def parse_canonical_jsonl(data: bytes) -> list[dict[str, Any]]:
    """Decode only canonical UTF-8 JSONL with one LF after every row."""
    if type(data) is not bytes or not data or not data.endswith(b"\n"):
        raise CorpusValidationError("JSONL requires at least one row and one final LF")
    lines = data.splitlines(keepends=True)
    if any(not line.endswith(b"\n") or line.endswith(b"\r\n") for line in lines):
        raise CorpusValidationError("JSONL requires exactly LF line endings")
    records = [_strict_json_object(line[:-1], label=f"JSONL row {index}") for index, line in enumerate(lines)]
    if canonical_jsonl_bytes(records) != data:
        raise CorpusValidationError("JSONL bytes are not canonical")
    return records


def validate_record(
    record: Mapping[str, Any], specification: Mapping[str, Any], *, self_hash_field: str = "record_sha256"
) -> dict[str, Any]:
    if type(record) is not dict:
        raise CorpusValidationError("record must be an object")
    required = specification.get("required_fields")
    schema_version = specification.get("schema_version")
    if not isinstance(required, (list, tuple)) or type(schema_version) is not str:
        raise CorpusValidationError("record specification is malformed")
    _exact_fields(record, required, "record")
    _walk_json(record)
    _check_numeric_positions(record, "record")
    if record.get("release_tier") != RELEASE_TIER:
        raise CorpusValidationError("record release tier is not development-fixture-only")
    if record.get("schema_version") != schema_version:
        raise CorpusValidationError("record schema version is not closed")
    if self_hash_field == "record_sha256":
        id_field = specification.get("id_field")
        if type(id_field) is not str or record.get(id_field) != record_identity(record, specification):
            raise CorpusValidationError("record stable identity drifts from identity body")
    if not _is_sha256(record.get(self_hash_field)) or record[self_hash_field] != record_self_hash(record, self_hash_field):
        raise CorpusValidationError("record self hash is invalid")
    return dict(record)


def validate_jsonl_records(data: bytes, specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [validate_record(row, specification) for row in parse_canonical_jsonl(data)]


def deterministic_rank_hmac(*, split_key: bytes, stratum: str, family_cluster_id: str) -> str:
    if type(split_key) is not bytes or not split_key:
        raise CorpusValidationError("split key must be non-empty bytes")
    if type(stratum) is not str or not stratum or not _is_sha256(family_cluster_id):
        raise CorpusValidationError("split rank identity is malformed")
    payload = b"aeolus-forecast-d1-split-rank-v1\0" + stratum.encode("utf-8") + b"\0" + family_cluster_id.encode("ascii")
    return hmac.new(split_key, payload, hashlib.sha256).hexdigest()


def assign_cluster_splits(
    clusters: Sequence[Mapping[str, Any]],
    *,
    split_key: bytes,
    split_policy_sha256: str,
    split_key_id: str,
    development_only: bool = False,
) -> list[dict[str, str]]:
    """Assign one deterministic split per cluster; rankings are stratum-local."""
    _require_sha256(split_policy_sha256, "split policy")
    if type(split_key_id) is not str or not split_key_id:
        raise CorpusValidationError("split key ID is required")
    seen: set[str] = set()
    ranked: list[tuple[str, str]] = []
    for cluster in clusters:
        cluster_id, stratum = cluster.get("family_cluster_id"), cluster.get("stratum")
        if not _is_sha256(cluster_id) or type(stratum) is not str or not stratum or cluster_id in seen:
            raise CorpusValidationError("cluster split unit is malformed or duplicated")
        seen.add(cluster_id)
        ranked.append((stratum, cluster_id))
    result: list[dict[str, str]] = []
    by_stratum: dict[str, list[str]] = {}
    for stratum, cluster_id in ranked:
        by_stratum.setdefault(stratum, []).append(cluster_id)
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda cluster_id: deterministic_rank_hmac(
                split_key=split_key, stratum=stratum, family_cluster_id=cluster_id
            ),
        )
        for index, cluster_id in enumerate(ordered):
            rank = deterministic_rank_hmac(split_key=split_key, stratum=stratum, family_cluster_id=cluster_id)
            label = "DEVELOPMENT" if development_only else ("TRAIN" if index * 5 < len(ordered) * 3 else "VALIDATION" if index * 5 < len(ordered) * 4 else "FINAL")
            body = {
                "family_cluster_id": cluster_id,
                "split_policy_sha256": split_policy_sha256,
                "split_label": label,
                "rank_hmac_sha256": rank,
            }
            result.append({
                "family_cluster_id": cluster_id,
                "stratum": stratum,
                "split_label": label,
                "split_policy_sha256": split_policy_sha256,
                "split_key_id": split_key_id,
                "rank_hmac_sha256": rank,
                "split_assignment_id": hashlib.sha256(
                    b"aeolus-forecast-d1-split-assignment-v1\0" + canonical_json_bytes(body)
                ).hexdigest(),
            })
    return result


def validate_lineage(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Reject cluster/family/run/witness/sample overlap or semantic aliases."""
    required = {"family_clusters", "families", "scenario_members", "split_assignments", "control_runs", "control_traces", "replay_witnesses", "samples"}
    if set(tables) != required:
        raise CorpusValidationError("lineage requires exactly the eight JSONL tables")
    cluster_rows = tables["family_clusters"]
    clusters = {row["family_cluster_id"]: row for row in cluster_rows}
    if len(clusters) != len(cluster_rows):
        raise CorpusValidationError("duplicate family cluster identity")
    assignments = {row["family_cluster_id"]: row for row in tables["split_assignments"]}
    if len(assignments) != len(tables["split_assignments"]) or set(assignments) != set(clusters):
        raise CorpusValidationError("split assignment is not one-to-one with family clusters")
    families = {row["family_id"]: row for row in tables["families"]}
    if len(families) != len(tables["families"]):
        raise CorpusValidationError("duplicate family identity")
    for family in families.values():
        if family["family_cluster_id"] not in clusters:
            raise CorpusValidationError("family references unknown cluster")
    scenario_members = {row["scenario_member_id"]: row for row in tables["scenario_members"]}
    if len(scenario_members) != len(tables["scenario_members"]):
        raise CorpusValidationError("duplicate scenario member identity")
    plant_runs: set[str] = set()
    for member in scenario_members.values():
        if member["family_id"] not in families or member["plant_run_id"] in plant_runs:
            raise CorpusValidationError("family/plant-run semantic alias")
        plant_runs.add(member["plant_run_id"])
    control_runs = {row["control_run_record_id"]: row for row in tables["control_runs"]}
    hmc_runs: set[str] = set()
    for run in control_runs.values():
        if run["scenario_member_id"] not in scenario_members or run["control_run_id"] in hmc_runs:
            raise CorpusValidationError("scenario/HMC-run semantic alias")
        hmc_runs.add(run["control_run_id"])
    traces = {row["control_trace_record_id"]: row for row in tables["control_traces"]}
    witnesses = {row["replay_witness_id"]: row for row in tables["replay_witnesses"]}
    samples = {row["sample_id"]: row for row in tables["samples"]}
    if len(traces) != len(tables["control_traces"]) or len(witnesses) != len(tables["replay_witnesses"]) or len(samples) != len(tables["samples"]):
        raise CorpusValidationError("duplicate trace, witness, or sample identity")
    trace_runs: set[str] = set()
    for trace in traces.values():
        if trace["control_run_id"] not in hmc_runs or trace["control_run_id"] in trace_runs:
            raise CorpusValidationError("control trace semantic alias")
        trace_runs.add(trace["control_run_id"])
    witness_runs: set[str] = set()
    for witness in witnesses.values():
        if witness["control_run_id"] not in hmc_runs or witness["control_trace_record_id"] not in traces or witness["control_run_id"] in witness_runs:
            raise CorpusValidationError("replay witness semantic alias")
        witness_runs.add(witness["control_run_id"])
    for sample in samples.values():
        family = families.get(sample["family_id"])
        member = scenario_members.get(sample["scenario_member_id"])
        run = control_runs.get(sample["control_run_record_id"])
        assignment = assignments.get(sample["family_cluster_id"])
        if not family or not member or not run or not assignment:
            raise CorpusValidationError("sample lineage is incomplete")
        if family["family_cluster_id"] != sample["family_cluster_id"] or member["family_id"] != sample["family_id"] or run["scenario_member_id"] != sample["scenario_member_id"]:
            raise CorpusValidationError("sample lineage aliases a different family/run")
        if sample["split_assignment_id"] != assignment["split_assignment_id"] or sample["split_label"] != assignment["split_label"]:
            raise CorpusValidationError("sample split does not inherit cluster split")
        if sample["replay_witness_id"] not in witnesses or run["replay_witness_id"] != sample["replay_witness_id"]:
            raise CorpusValidationError("sample replay witness lineage drifts")


def iter_training_samples(samples: Sequence[Mapping[str, Any]], split_assignments: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    assignment = {row["split_assignment_id"]: row for row in split_assignments}
    result: list[Mapping[str, Any]] = []
    for sample in samples:
        split = assignment.get(sample.get("split_assignment_id"))
        if split is None or split.get("family_cluster_id") != sample.get("family_cluster_id"):
            raise CorpusValidationError("fit sample has unbound cluster split")
        if sample.get("split_label") != split.get("split_label"):
            raise CorpusValidationError("fit sample split label drift")
        if split["split_label"] == "TRAIN":
            result.append(sample)
    return tuple(result)


def validate_relative_packet_path(value: Any) -> str:
    if type(value) is not str or not value.startswith(PATH_PREFIX):
        raise CorpusValidationError("packet path must begin development-fixture-only/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value == PATH_PREFIX:
        raise CorpusValidationError("packet path is unsafe")
    return value
