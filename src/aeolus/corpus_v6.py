"""V6 observable-context corpus generation and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from aeolus.config import load_scenario
from aeolus.families_v6 import V6FamilyManifest, V6ScenarioFamily
from aeolus.observable_context import (
    OBSERVABLE_CONTEXT_VERSION,
    ObservableContextContract,
    build_observable_context_contract,
    observable_context_metadata,
    observable_context_v1,
)
from aeolus.scenario import run_scenario
from aeolus.trace import TickRecord

V6_CORPUS_VERSION = "aeolus_corpus_v6"
_EXCLUDED_TRANSITION = "excluded_transition"
_ROW_KEYS = frozenset(
    {
        "family_id",
        "room_family_id",
        "split",
        "scenario_role",
        "window_index",
        "start_tick",
        "end_tick",
        "observable_onset_tick",
        "label",
        "observable_context_version",
        "selector_sha256",
        "topology_sha256",
        "features",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "sweep_spec_sha256",
        "family_manifest_sha256",
        "observable_context",
        "window_ticks",
        "stride_ticks",
        "family_count",
        "total_windows",
        "label_counts",
        "corpus_jsonl_sha256",
        "manifest_sha256",
    }
)


def generate_v6_corpus(
    families: V6FamilyManifest,
    output_dir: str | Path,
    *,
    window_ticks: int,
    stride_ticks: int,
) -> dict[str, object]:
    """Generate V6 rows from exact room-family scenario bytes and context contract."""
    _validate_generation_arguments(families, output_dir, window_ticks, stride_ticks)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    label_counts: dict[str, int] = {}

    for family in families.families:
        reference_config = load_scenario(family.reference_path)
        fault_config = load_scenario(family.fault_path)
        reference_contract = build_observable_context_contract(reference_config)
        fault_contract = build_observable_context_contract(fault_config)
        metadata = observable_context_metadata(reference_contract)
        if metadata != families.observable_context or observable_context_metadata(fault_contract) != metadata:
            raise ValueError("V6 corpus family context does not match manifest")
        reference_records = run_scenario(reference_config)
        fault_records = run_scenario(fault_config)
        onset = _observable_onset(
            reference_records, fault_records, reference_contract, fault_contract
        )
        for scenario_role, records in (
            ("reference", reference_records),
            ("fault", fault_records),
        ):
            for index, start_index in enumerate(
                range(0, len(records) - window_ticks + 1, stride_ticks)
            ):
                window = records[start_index : start_index + window_ticks]
                start_tick = window[0].tick
                end_tick = window[-1].tick
                label = _window_label(
                    scenario_role=scenario_role,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    onset=onset,
                    fault_class=family.fault_class,
                )
                rows.append(
                    {
                        "family_id": family.family_id,
                        "room_family_id": family.room_family_id,
                        "split": family.role,
                        "scenario_role": scenario_role,
                        "window_index": index,
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "observable_onset_tick": onset,
                        "label": label,
                        **metadata,
                        "features": [
                            observable_context_v1(record, reference_contract).tolist()
                            for record in window
                        ],
                    }
                )
                label_counts[label] = label_counts.get(label, 0) + 1

    rows.sort(key=lambda row: (str(row["family_id"]), str(row["scenario_role"]), int(row["end_tick"])))
    corpus_path = destination / "corpus-v6.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    manifest: dict[str, object] = {
        "schema_version": V6_CORPUS_VERSION,
        "sweep_spec_sha256": families.sweep_spec_sha256,
        "family_manifest_sha256": families.manifest_sha256,
        "observable_context": dict(families.observable_context),
        "window_ticks": window_ticks,
        "stride_ticks": stride_ticks,
        "family_count": families.family_count,
        "total_windows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "corpus_jsonl_sha256": _file_sha256(corpus_path),
    }
    manifest["manifest_sha256"] = _sha256(_canonical_json(manifest))
    (destination / "manifest-v6.json").write_text(
        _canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def validate_v6_corpus(
    corpus_dir: str | Path, *, expected_families: V6FamilyManifest
) -> list[dict[str, object]]:
    """Validate every persisted V6 row against exact family/context identities."""
    if not isinstance(expected_families, V6FamilyManifest):
        raise ValueError("V6 corpus validation requires a V6 family manifest")
    source = Path(corpus_dir)
    try:
        manifest = json.loads((source / "manifest-v6.json").read_text(encoding="utf-8"))
        lines = (source / "corpus-v6.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read V6 corpus: {exc}") from None
    _validate_manifest(manifest, expected_families, source / "corpus-v6.jsonl")
    if not lines:
        raise ValueError("V6 corpus has no rows")
    rows: list[dict[str, object]] = []
    families = {family.family_id: family for family in expected_families.families}
    stream_indices: dict[tuple[str, str], list[int]] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"V6 corpus row {line_number} is not JSON: {exc}") from None
        _validate_row(row, families, expected_families.observable_context)
        family_id = row["family_id"]
        scenario_role = row["scenario_role"]
        assert isinstance(family_id, str) and isinstance(scenario_role, str)
        stream_indices.setdefault((family_id, scenario_role), []).append(row["window_index"])
        rows.append(row)
    for family in expected_families.families:
        for scenario_role in ("reference", "fault"):
            indices = stream_indices.get((family.family_id, scenario_role))
            if not indices or sorted(indices) != list(range(len(indices))):
                raise ValueError("V6 corpus contains an incomplete family stream")
    if len(rows) != manifest["total_windows"]:
        raise ValueError("V6 corpus total window count does not match manifest")
    return rows


def _validate_generation_arguments(
    families: object, output_dir: str | Path, window_ticks: int, stride_ticks: int
) -> None:
    if not isinstance(families, V6FamilyManifest):
        raise ValueError("V6 corpus generation requires a V6 family manifest")
    for value, name in ((window_ticks, "window_ticks"), (stride_ticks, "stride_ticks")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"V6 corpus {name} must be a positive integer")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("V6 corpus output directory is not empty")


def _observable_onset(
    reference_records: Sequence[TickRecord],
    fault_records: Sequence[TickRecord],
    reference_contract: ObservableContextContract,
    fault_contract: ObservableContextContract,
) -> int:
    if len(reference_records) != len(fault_records):
        raise ValueError("V6 paired traces have unequal lengths")
    for reference, fault in zip(reference_records, fault_records, strict=True):
        reference_context = observable_context_v1(reference, reference_contract)
        fault_context = observable_context_v1(fault, fault_contract)
        if not np.array_equal(reference_context, fault_context):
            return reference.tick
    raise ValueError("V6 fault family never becomes observable in V6 context")


def _window_label(
    *, scenario_role: str, start_tick: int, end_tick: int, onset: int, fault_class: str
) -> str:
    if scenario_role == "reference" or end_tick < onset:
        return "nominal"
    if start_tick < onset <= end_tick:
        return _EXCLUDED_TRANSITION
    return fault_class


def _validate_manifest(
    value: object, expected_families: V6FamilyManifest, corpus_path: Path
) -> None:
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise ValueError("V6 corpus manifest has unknown or missing fields")
    if value.get("schema_version") != V6_CORPUS_VERSION:
        raise ValueError("V6 corpus manifest schema_version is unsupported")
    expected = {
        "sweep_spec_sha256": expected_families.sweep_spec_sha256,
        "family_manifest_sha256": expected_families.manifest_sha256,
        "observable_context": expected_families.observable_context,
        "family_count": expected_families.family_count,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("V6 corpus manifest identity does not match expected families")
    for key in ("window_ticks", "stride_ticks", "family_count", "total_windows"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 1:
            raise ValueError("V6 corpus manifest count is malformed")
    if not isinstance(value["label_counts"], dict):
        raise ValueError("V6 corpus manifest label_counts is malformed")
    if value["corpus_jsonl_sha256"] != _file_sha256(corpus_path):
        raise ValueError("V6 corpus JSONL digest does not match bytes")
    claimed = value["manifest_sha256"]
    without_hash = dict(value)
    without_hash.pop("manifest_sha256")
    if claimed != _sha256(_canonical_json(without_hash)):
        raise ValueError("V6 corpus manifest self digest does not match")


def _validate_row(
    value: object,
    families: dict[str, V6ScenarioFamily],
    expected_context: dict[str, str],
) -> None:
    if not isinstance(value, dict) or set(value) != _ROW_KEYS:
        raise ValueError("V6 corpus row has unknown or missing fields")
    family_id = value["family_id"]
    if not isinstance(family_id, str) or family_id not in families:
        raise ValueError("V6 corpus row family_id is unknown")
    family = families[family_id]
    if (
        value["room_family_id"] != family.room_family_id
        or value["split"] != family.role
        or value["scenario_role"] not in {"reference", "fault"}
    ):
        raise ValueError("V6 corpus row family metadata does not match manifest")
    metadata = {
        "observable_context_version": value["observable_context_version"],
        "selector_sha256": value["selector_sha256"],
        "topology_sha256": value["topology_sha256"],
    }
    if metadata != expected_context or metadata["observable_context_version"] != OBSERVABLE_CONTEXT_VERSION:
        raise ValueError("V6 corpus row observable context does not match manifest")
    for key in ("window_index", "start_tick", "end_tick", "observable_onset_tick"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 0:
            raise ValueError("V6 corpus row tick/index is malformed")
    if value["start_tick"] < 1 or value["end_tick"] < value["start_tick"]:
        raise ValueError("V6 corpus row tick bounds are malformed")
    if not isinstance(value["label"], str):
        raise ValueError("V6 corpus row label is malformed")
    features = value["features"]
    if not isinstance(features, list) or not features or not all(isinstance(row, list) and row for row in features):
        raise ValueError("V6 corpus row features are malformed")
    try:
        feature_values = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V6 corpus features are not numeric: {exc}") from None
    if feature_values.ndim != 2 or not np.isfinite(feature_values).all():
        raise ValueError("V6 corpus features are not finite rank-two context")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
