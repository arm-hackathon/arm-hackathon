"""Score window labellers against versioned corpus contracts.

Corpus v1 retains historical declared-start labels and latency. Corpus v2 uses a
validated family manifest as the trusted model-input contract, scores one
explicit split, and measures latency from persisted observable onset.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from aeolus.baseline import RuleBaseline
from aeolus.config import HabitatConfig, load_scenario
from aeolus.families import load_family_manifest
from aeolus.model_input import MODEL_INPUT_SHAPE, MODEL_INPUT_VERSION

USAGE = (
    "Usage: PYTHONPATH=src python3 -m aeolus.evaluate <corpus.jsonl> "
    "<scenario.json> [scenario.json ...]\n"
    "   or: PYTHONPATH=src python3 -m aeolus.evaluate --v2 <corpus.jsonl> "
    "<families.json> [--split train|validation|test]"
)
EXCLUDED_TRANSITION_LABEL = "excluded_transition"
_V2_CONTRACT_KEYS = frozenset(
    {"model_input_version", "selector_sha256", "topology_sha256"}
)


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
    target_split: str,
) -> dict:
    """Score one corpus-v2 split from trusted observable-onset evidence."""
    _validate_v2_contract(rows, expected_contract)
    if target_split not in {"train", "validation", "test"}:
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


def _validate_v2_contract(rows: list[dict], expected_contract: dict[str, str]) -> None:
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
    first = rows[0]
    try:
        expected = {key: first[key] for key in _V2_CONTRACT_KEYS}
    except KeyError as exc:
        raise ValueError("corpus v2 row is missing contract metadata") from exc
    if expected != expected_contract:
        raise ValueError("corpus v2 contract does not match the expected contract")
    if any(not isinstance(value, str) for value in expected.values()):
        raise ValueError("corpus v2 contract metadata must be strings")
    for row in rows:
        try:
            actual = {key: row[key] for key in _V2_CONTRACT_KEYS}
            if actual != expected:
                raise ValueError("corpus v2 rows do not share one model contract")
            if not isinstance(row["observable_onset_tick"], int):
                raise ValueError("corpus v2 observable onset must be an integer")
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError("corpus v2 row split is unsupported")
            _validate_v2_features(row["features"])
            if not isinstance(row["family_id"], str) or not isinstance(
                row["scenario_role"], str
            ):
                raise ValueError("corpus v2 row identity is malformed")
        except KeyError as exc:
            raise ValueError("corpus v2 row is missing required fields") from exc


def _validate_v2_features(features: object) -> None:
    if not isinstance(features, list):
        raise ValueError("corpus v2 features must be a list")
    for tick_number, vector in enumerate(features, start=1):
        if not isinstance(vector, list) or len(vector) != MODEL_INPUT_SHAPE[0]:
            raise ValueError(f"model-input tick {tick_number} has an unexpected shape")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"model-input tick {tick_number} contains a non-numeric value")
            if not math.isfinite(value):
                raise ValueError(f"model-input tick {tick_number} contains a non-finite value")


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
    if len(argv) == 2:
        corpus_path, family_manifest_path = argv
        target_split = "test"
    elif len(argv) == 4 and argv[2] == "--split":
        corpus_path, family_manifest_path, _, target_split = argv
    else:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        rows = _load_rows(corpus_path)
        manifest = load_family_manifest(Path(family_manifest_path))
        baseline_config = load_scenario(manifest.families[0].reference_path)
        result = evaluate_v2(
            rows,
            RuleBaseline(baseline_config),
            expected_contract=manifest.contract_metadata,
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
