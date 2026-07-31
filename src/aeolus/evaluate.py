"""Score window labellers against corpus labels: accuracy and detection latency.

One harness grades every contestant — the rule baseline today, the temporal
classifier later — on the same rows with the same metrics, so the comparison
stays honest. Detection latency is the number of measured ticks between a
fault's declared start and the first window the labeller gets right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from aeolus.baseline import RuleBaseline
from aeolus.config import HabitatConfig, load_scenario

USAGE = (
    "Usage: PYTHONPATH=src python3 -m aeolus.evaluate <corpus.jsonl> "
    "<scenario.json> [scenario.json ...]"
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


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(USAGE, file=sys.stderr)
        return 2
    corpus_path, *scenario_paths = argv
    try:
        rows = [
            json.loads(line)
            for line in Path(corpus_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
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


def _loop_connection_pairs(config: HabitatConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (config.path_to_processing(zone.id).id, config.path_from_processing(zone.id).id)
        for zone in config.non_processing_zones()
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
