"""Score window labellers against versioned corpus contracts.

Corpus v1 retains historical declared-start labels and latency. Corpus v2 uses a
validated family manifest as the trusted model-input contract, scores one
explicit split, and measures latency from persisted observable onset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from aeolus.baseline import RuleBaseline
from aeolus.config import HabitatConfig, load_scenario
from aeolus.families import (
    FamilyEvidence,
    build_family_evidence,
    family_window_label,
    load_family_manifest,
)
from aeolus.model_input import MODEL_INPUT_SHAPE, MODEL_INPUT_VERSION

USAGE = (
    "Usage: PYTHONPATH=src python3 -m aeolus.evaluate <corpus.jsonl> "
    "<scenario.json> [scenario.json ...]\n"
    "   or: PYTHONPATH=src python3 -m aeolus.evaluate --v2 <corpus.jsonl> "
    "<families.json> --expected-family-manifest-sha256 <sha256> "
    "[--split train|validation|test|stress]"
)
EXCLUDED_TRANSITION_LABEL = "excluded_transition"
_V2_CONTRACT_KEYS = frozenset(
    {"model_input_version", "selector_sha256", "topology_sha256"}
)
_V2_ROW_KEYS = _V2_CONTRACT_KEYS | {
    "family_id",
    "scenario_role",
    "split",
    "window_index",
    "start_tick",
    "end_tick",
    "observable_onset_tick",
    "label",
    "features",
}
_V2_SPLITS = frozenset({"train", "validation", "test", "stress"})
_V2_SCENARIO_ROLES = frozenset({"reference", "fault"})


@runtime_checkable
class _WindowLabeller(Protocol):
    """A stateful per-window labeller (rule baseline today, classifier later)."""

    def label_window(self, features: list[dict]) -> str: ...


class _FunctionLabeller:
    """Adapt a plain window-labelling function to the labeller protocol."""

    def __init__(self, fn: Callable[[list[dict]], str]) -> None:
        self._fn = fn

    def label_window(self, features: list[dict]) -> str:
        return self._fn(features)


def fault_start_tick(config: HabitatConfig) -> int | None:
    """Return the declared start tick of a scenario's fault, if it has one."""
    for profile in config.fault_profiles:
        return profile.start_tick
    return None


def evaluate(
    rows: list[dict],
    labeller: Callable[[list[dict]], str] | _WindowLabeller,
    *,
    fault_starts: dict[str, int],
) -> dict:
    """Grade a labeller over corpus rows.

    ``fault_starts`` maps scenario name to declared fault start tick and is
    used only for latency, never for accuracy. Stateful labellers (objects
    with a ``reset`` method) are reset whenever the scenario changes.
    """
    ordered = sorted(rows, key=lambda row: (row["scenario"], row["end_tick"]))
    if not isinstance(labeller, _WindowLabeller):
        labeller = _FunctionLabeller(labeller)
    total = 0
    correct = 0
    per_class: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    latencies: dict[str, list[int]] = {}
    detected: set[tuple[str, str]] = set()
    current_scenario: str | None = None

    for row in ordered:
        if row["scenario"] != current_scenario:
            current_scenario = row["scenario"]
            reset = getattr(labeller, "reset", None)
            if callable(reset):
                reset()
        true_label = row["label"]
        predicted = labeller.label_window(row["features"])
        total += 1
        if predicted == true_label:
            correct += 1
        support = per_class.setdefault(true_label, {"support": 0, "correct": 0})
        support["support"] += 1
        if predicted == true_label:
            support["correct"] += 1
        confusion.setdefault(true_label, {})[predicted] = (
            confusion.setdefault(true_label, {}).get(predicted, 0) + 1
        )
        key = (row["scenario"], true_label)
        if (
            true_label != "nominal"
            and predicted == true_label
            and key not in detected
            and row["scenario"] in fault_starts
        ):
            latencies.setdefault(true_label, []).append(
                row["end_tick"] - fault_starts[row["scenario"]]
            )
            detected.add(key)

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "detection_latency_ticks": {
            label: sum(values) / len(values)
            for label, values in sorted(latencies.items())
        },
    }


def evaluate_v2(
    rows: list[dict],
    labeller: Callable[[list[dict]], str] | _WindowLabeller,
    *,
    expected_contract: dict[str, str],
    expected_families: Mapping[str, FamilyEvidence],
    target_split: str,
) -> dict:
    """Score one corpus-v2 split from trusted observable-onset evidence."""
    validate_v2_rows(rows, expected_contract, expected_families)
    if target_split not in _V2_SPLITS:
        raise ValueError("corpus v2 target split is unsupported")
    if not isinstance(labeller, _WindowLabeller):
        labeller = _FunctionLabeller(labeller)

    ordered = sorted(
        (row for row in rows if row["split"] == target_split),
        key=lambda row: (row["family_id"], row["scenario_role"], row["end_tick"]),
    )
    if not ordered:
        raise ValueError("corpus v2 target split has no rows")
    current_stream: tuple[str, str] | None = None
    correct = 0
    scored_total = 0
    excluded_total = 0
    per_class: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    latencies: dict[str, list[int]] = {}
    detected: set[tuple[str, str]] = set()

    for row in ordered:
        stream = (row["family_id"], row["scenario_role"])
        if stream != current_stream:
            current_stream = stream
            reset = getattr(labeller, "reset", None)
            if callable(reset):
                reset()
        predicted = labeller.label_window(row["features"])
        if row["label"] == EXCLUDED_TRANSITION_LABEL:
            excluded_total += 1
            continue

        true_label = row["label"]
        scored_total += 1
        is_correct = predicted == true_label
        correct += int(is_correct)
        support = per_class.setdefault(true_label, {"support": 0, "correct": 0})
        support["support"] += 1
        support["correct"] += int(is_correct)
        confusion.setdefault(true_label, {})[predicted] = (
            confusion.setdefault(true_label, {}).get(predicted, 0) + 1
        )
        detection_key = (row["family_id"], true_label)
        if (
            true_label != "nominal"
            and is_correct
            and detection_key not in detected
        ):
            latencies.setdefault(true_label, []).append(
                row["end_tick"] - row["observable_onset_tick"]
            )
            detected.add(detection_key)

    return {
        "scored_total": scored_total,
        "excluded_transition_total": excluded_total,
        "correct": correct,
        "accuracy": (correct / scored_total) if scored_total else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "detection_latency_ticks": {
            label: sum(values) / len(values)
            for label, values in sorted(latencies.items())
        },
    }


def validate_v2_rows(
    rows: list[dict],
    expected_contract: dict[str, str],
    expected_families: Mapping[str, FamilyEvidence],
) -> int:
    """Validate complete corpus-v2 evidence and return its window length."""
    if not rows:
        raise ValueError("corpus v2 evaluation requires at least one row")
    if set(expected_contract) != _V2_CONTRACT_KEYS or any(
        not isinstance(value, str) for value in expected_contract.values()
    ):
        raise ValueError("expected corpus v2 contract metadata is malformed")
    if (
        expected_contract["model_input_version"] != MODEL_INPUT_VERSION
        or not _is_sha256(expected_contract["selector_sha256"])
        or not _is_sha256(expected_contract["topology_sha256"])
    ):
        raise ValueError("expected corpus v2 contract is incompatible")
    if not expected_families:
        raise ValueError("expected corpus v2 family evidence is empty")
    for family_id, evidence in expected_families.items():
        if not isinstance(evidence, FamilyEvidence) or evidence.family_id != family_id:
            raise ValueError("expected corpus v2 family evidence is malformed")
        if (
            not _is_non_boolean_int(evidence.reference_trace_ticks)
            or not _is_non_boolean_int(evidence.fault_trace_ticks)
        ):
            raise ValueError("expected corpus v2 family evidence lacks replay lengths")

    window_ticks: int | None = None
    family_streams = {family_id: set() for family_id in expected_families}
    stream_rows: dict[tuple[str, str], list[dict]] = {}
    row_identities: set[tuple[str, str, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"corpus v2 row {row_number} must be an object")
        missing = _V2_ROW_KEYS - set(row)
        unexpected = set(row) - _V2_ROW_KEYS
        if missing or unexpected:
            raise ValueError(
                f"corpus v2 row {row_number} schema mismatch: "
                f"missing={sorted(missing)!r} unexpected={sorted(unexpected)!r}"
            )
        actual_contract = {key: row[key] for key in _V2_CONTRACT_KEYS}
        if any(not isinstance(value, str) for value in actual_contract.values()):
            raise ValueError("corpus v2 contract metadata must be strings")
        if actual_contract != expected_contract:
            raise ValueError("corpus v2 contract does not match the expected contract")

        family_id = row["family_id"]
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("corpus v2 family_id is malformed")
        evidence = expected_families.get(family_id)
        if evidence is None:
            raise ValueError("corpus v2 family_id is absent from family evidence")
        scenario_role = row["scenario_role"]
        if not isinstance(scenario_role, str) or scenario_role not in _V2_SCENARIO_ROLES:
            raise ValueError("corpus v2 scenario_role is unsupported")
        family_streams[family_id].add(scenario_role)
        if row["split"] != evidence.split:
            raise ValueError("corpus v2 split does not match family evidence")
        for field_name in ("window_index", "start_tick", "end_tick"):
            if not _is_non_boolean_int(row[field_name]):
                raise ValueError(f"corpus v2 {field_name} must be an integer")
        if row["window_index"] < 0:
            raise ValueError("corpus v2 window_index must not be negative")
        if row["start_tick"] < 1 or row["end_tick"] < 1:
            raise ValueError("corpus v2 window ticks must be positive")
        if row["start_tick"] > row["end_tick"]:
            raise ValueError("corpus v2 start_tick must not exceed end_tick")
        if not _is_non_boolean_int(row["observable_onset_tick"]):
            raise ValueError("corpus v2 observable onset must be an integer")
        if row["observable_onset_tick"] != evidence.observable_onset_tick:
            raise ValueError("corpus v2 observable onset does not match family evidence")
        expected_label = family_window_label(
            scenario_role=scenario_role,
            start_tick=row["start_tick"],
            end_tick=row["end_tick"],
            evidence=evidence,
        )
        if row["label"] != expected_label:
            raise ValueError("corpus v2 label does not match family evidence")
        feature_ticks = _validate_v2_features(row["features"])
        _validate_v2_feature_values(
            row["features"],
            evidence=evidence,
            scenario_role=scenario_role,
            start_tick=row["start_tick"],
            end_tick=row["end_tick"],
        )
        if window_ticks is None:
            window_ticks = feature_ticks
        elif feature_ticks != window_ticks:
            raise ValueError("corpus v2 rows do not share one window shape")
        identity = (family_id, scenario_role, row["window_index"])
        if identity in row_identities:
            raise ValueError("corpus v2 contains a duplicate row identity")
        row_identities.add(identity)
        stream_rows.setdefault((family_id, scenario_role), []).append(row)

    incomplete_streams = {
        family_id: sorted(_V2_SCENARIO_ROLES - streams)
        for family_id, streams in family_streams.items()
        if streams != _V2_SCENARIO_ROLES
    }
    if incomplete_streams:
        raise ValueError(
            f"corpus v2 family evidence is missing streams: {incomplete_streams!r}"
        )
    assert window_ticks is not None
    for family_id, evidence in expected_families.items():
        for scenario_role, trace_ticks in (
            ("reference", evidence.reference_trace_ticks),
            ("fault", evidence.fault_trace_ticks),
        ):
            assert trace_ticks is not None
            _validate_v2_window_sequence(
                stream_rows[(family_id, scenario_role)],
                trace_ticks=trace_ticks,
                window_ticks=window_ticks,
            )
    return window_ticks


def _validate_v2_window_sequence(
    rows: list[dict], *, trace_ticks: int, window_ticks: int
) -> None:
    if not _is_non_boolean_int(trace_ticks) or trace_ticks < window_ticks:
        raise ValueError("expected corpus v2 trace length is malformed")
    ordered = sorted(rows, key=lambda row: row["window_index"])
    actual_indices = [row["window_index"] for row in ordered]
    if actual_indices != list(range(len(ordered))):
        raise ValueError("corpus v2 window sequence is incomplete")
    if len(ordered) == 1:
        only = ordered[0]
        if only["start_tick"] != 1 or only["end_tick"] != trace_ticks:
            raise ValueError("corpus v2 window sequence does not match its ticks")
        return
    stride = ordered[1]["start_tick"] - ordered[0]["start_tick"]
    if stride < 1:
        raise ValueError("corpus v2 window sequence has an invalid stride")
    expected_count = (trace_ticks - window_ticks) // stride + 1
    if len(ordered) != expected_count:
        raise ValueError("corpus v2 window sequence is incomplete")
    for row in ordered:
        expected_start = 1 + row["window_index"] * stride
        expected_end = expected_start + window_ticks - 1
        if row["start_tick"] != expected_start or row["end_tick"] != expected_end:
            raise ValueError("corpus v2 window sequence does not match its ticks")


def _validate_v2_features(features: object) -> int:
    if not isinstance(features, list):
        raise ValueError("corpus v2 features must be a list")
    if not features:
        raise ValueError("corpus v2 features must contain at least one tick")
    for tick_number, vector in enumerate(features, start=1):
        if not isinstance(vector, list) or len(vector) != MODEL_INPUT_SHAPE[0]:
            raise ValueError(f"model-input tick {tick_number} has an unexpected shape")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"model-input tick {tick_number} contains a non-numeric value")
        try:
            with np.errstate(over="raise", invalid="raise"):
                narrowed = np.asarray(vector, dtype=np.float32)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError(f"model-input tick {tick_number} overflows float32") from exc
        if not np.isfinite(narrowed).all():
            raise ValueError(f"model-input tick {tick_number} contains a non-finite value")
    return len(features)


def _validate_v2_feature_values(
    features: list[list[float | int]],
    *,
    evidence: FamilyEvidence,
    scenario_role: str,
    start_tick: int,
    end_tick: int,
) -> None:
    """Reject a finite vector window that differs from its trusted replay slice."""
    expected_trace = (
        evidence.reference_model_input_trace
        if scenario_role == "reference"
        else evidence.fault_model_input_trace
    )
    if expected_trace is None:
        return
    expected_window = expected_trace[start_tick - 1 : end_tick]
    actual_window = tuple(
        tuple(float(value) for value in vector) for vector in features
    )
    if actual_window != expected_window:
        raise ValueError("corpus v2 features do not match the verified replay")


def _is_non_boolean_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def main(argv: list[str]) -> int:
    if argv[:1] == ["--v2"]:
        return _main_v2(argv[1:])
    if len(argv) < 2:
        print(USAGE, file=sys.stderr)
        return 2
    corpus_path, *scenario_paths = argv
    try:
        rows = _load_rows(corpus_path)
        configs = [load_scenario(scenario_path) for scenario_path in scenario_paths]
        baseline_pairs = _loop_connection_pairs(configs[0])
        if any(_loop_connection_pairs(config) != baseline_pairs for config in configs[1:]):
            raise ValueError("evaluation scenarios do not share one validated topology")
        fault_starts = {}
        for scenario_path, config in zip(scenario_paths, configs):
            start = fault_start_tick(config)
            if start is not None:
                fault_starts[Path(scenario_path).stem] = start
        result = evaluate(rows, RuleBaseline(configs[0]), fault_starts=fault_starts)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"cannot evaluate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _main_v2(argv: list[str]) -> int:
    if (
        len(argv) == 4
        and argv[2] == "--expected-family-manifest-sha256"
    ):
        corpus_path, family_manifest_path, _, expected_manifest_sha256 = argv
        target_split = "test"
    elif (
        len(argv) == 6
        and argv[2] == "--expected-family-manifest-sha256"
        and argv[4] == "--split"
    ):
        (
            corpus_path,
            family_manifest_path,
            _,
            expected_manifest_sha256,
            _,
            target_split,
        ) = argv
    else:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        if not _is_sha256(expected_manifest_sha256):
            raise ValueError("expected family manifest hash must be lowercase SHA-256")
        rows = _load_rows(corpus_path)
        manifest = load_family_manifest(Path(family_manifest_path))
        if manifest.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("family manifest does not match the expected SHA-256")
        baseline_config = load_scenario(manifest.families[0].reference_path)
        result = evaluate_v2(
            rows,
            RuleBaseline(baseline_config),
            expected_contract=manifest.contract_metadata,
            expected_families=build_family_evidence(manifest),
            target_split=target_split,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"cannot evaluate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _load_rows(corpus_path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(corpus_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _loop_connection_pairs(config: HabitatConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (config.path_to_processing(zone.id).id, config.path_from_processing(zone.id).id)
        for zone in config.non_processing_zones()
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
