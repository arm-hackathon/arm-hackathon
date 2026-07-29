"""Evaluation-harness contracts: accuracy, per-class support and detection latency."""

from __future__ import annotations

import json
from pathlib import Path

from aeolus.baseline import RuleBaseline
from aeolus.config import load_scenario
from aeolus.corpus import generate_corpus
from aeolus.evaluate import evaluate, evaluate_v2, fault_start_tick, main

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "scenarios"
ALL_SCENARIO_PATHS = (
    SCENARIOS / "standard_habitat.json",
    SCENARIOS / "high_demand_healthy.json",
    SCENARIOS / "primary_fan_degradation.json",
    SCENARIOS / "blocked_path.json",
    SCENARIOS / "frozen_sensor.json",
)


def _corpus_rows(tmp_path) -> list[dict]:
    generate_corpus(ALL_SCENARIO_PATHS, tmp_path)
    corpus_path = tmp_path / "corpus.jsonl"
    return [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
    ]


def _fault_starts() -> dict[str, int]:
    starts = {}
    for path in ALL_SCENARIO_PATHS:
        start = fault_start_tick(load_scenario(path))
        if start is not None:
            starts[path.stem] = start
    return starts


def test_fault_start_tick_reads_declared_profiles():
    assert (
        fault_start_tick(load_scenario(SCENARIOS / "primary_fan_degradation.json"))
        == 20
    )
    assert fault_start_tick(load_scenario(SCENARIOS / "blocked_path.json")) == 30
    assert fault_start_tick(load_scenario(SCENARIOS / "frozen_sensor.json")) == 30
    assert fault_start_tick(load_scenario(SCENARIOS / "standard_habitat.json")) is None


def test_rule_baseline_scores_against_corpus_v1(tmp_path):
    rows = _corpus_rows(tmp_path)

    result = evaluate(
        rows,
        RuleBaseline(load_scenario(SCENARIOS / "standard_habitat.json")),
        fault_starts=_fault_starts(),
    )

    assert result["total"] == 115
    assert result["correct"] == 111
    assert result["per_class"] == {
        "nominal": {"support": 57, "correct": 57},
        "gradual_primary_fan_degradation": {"support": 20, "correct": 19},
        "blocked_path": {"support": 19, "correct": 18},
        "frozen_sensor": {"support": 19, "correct": 17},
    }
    # The baseline only fires once a fault fills its persistence window, so it
    # pays a detection-latency cost at every onset boundary.
    assert result["detection_latency_ticks"] == {
        "gradual_primary_fan_degradation": 10.0,
        "blocked_path": 5.0,
        "frozen_sensor": 10.0,
    }


def test_evaluate_reports_confusion_on_synthetic_rows():
    rows = [
        {
            "scenario": "s",
            "window_index": 0,
            "start_tick": 1,
            "end_tick": 10,
            "label": "nominal",
            "features": [],
        },
        {
            "scenario": "s",
            "window_index": 1,
            "start_tick": 6,
            "end_tick": 15,
            "label": "nominal",
            "features": [],
        },
        {
            "scenario": "f",
            "window_index": 0,
            "start_tick": 1,
            "end_tick": 10,
            "label": "frozen_sensor",
            "features": [],
        },
    ]

    def fake_labeller(features):
        return "nominal"

    result = evaluate(rows, fake_labeller, fault_starts={"f": 1})

    assert result["total"] == 3
    assert result["correct"] == 2
    assert result["confusion"] == {
        "nominal": {"nominal": 2},
        "frozen_sensor": {"nominal": 1},
    }
    assert result["detection_latency_ticks"] == {}


def test_evaluate_cli_prints_metrics_json(tmp_path, capsys):
    _corpus_rows(tmp_path)
    corpus_path = tmp_path / "corpus.jsonl"
    exit_code = main([str(corpus_path), *(str(path) for path in ALL_SCENARIO_PATHS)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["total"] == 115
    assert printed["correct"] == 111
    assert printed["detection_latency_ticks"]["blocked_path"] == 5.0


def test_evaluate_cli_rejects_missing_arguments():
    assert main([]) == 2
    assert main(["only-corpus"]) == 2


def test_evaluate_v2_excludes_transition_rows_and_uses_observable_onset():
    metadata = {
        "model_input_version": "model_input_v1",
        "selector_sha256": "a" * 64,
        "topology_sha256": "b" * 64,
    }
    common = {
        **metadata,
        "family_id": "blocked-path-v1",
        "split": "test",
        "observable_onset_tick": 30,
    }
    rows = [
        {
            **common,
            "scenario_role": "reference",
            "start_tick": 1,
            "end_tick": 10,
            "label": "nominal",
            "features": [{"prediction": "nominal"}],
        },
        {
            **common,
            "scenario_role": "fault",
            "start_tick": 26,
            "end_tick": 35,
            "label": "excluded_transition",
            "features": [{"prediction": "blocked_path"}],
        },
        {
            **common,
            "scenario_role": "fault",
            "start_tick": 31,
            "end_tick": 40,
            "label": "blocked_path",
            "features": [{"prediction": "blocked_path"}],
        },
    ]

    result = evaluate_v2(rows, lambda features: features[0]["prediction"])

    assert result["scored_total"] == 2
    assert result["excluded_transition_total"] == 1
    assert result["correct"] == 2
    assert result["accuracy"] == 1.0
    assert result["confusion"] == {"blocked_path": {"blocked_path": 1}, "nominal": {"nominal": 1}}
    assert result["detection_latency_ticks"] == {"blocked_path": 10.0}


def test_evaluate_v2_feeds_excluded_rows_to_stateful_labellers():
    class StatefulLabeller:
        def __init__(self):
            self.seen: list[str] = []

        def label_window(self, features):
            self.seen.append(features[0]["marker"])
            return "nominal"

    metadata = {"model_input_version": "model_input_v1", "selector_sha256": "a" * 64, "topology_sha256": "b" * 64}
    rows = [
        {**metadata, "family_id": "f", "scenario_role": "fault", "observable_onset_tick": 2, "start_tick": 1, "end_tick": 2, "label": "excluded_transition", "features": [{"marker": "transition"}]},
        {**metadata, "family_id": "f", "scenario_role": "fault", "observable_onset_tick": 2, "start_tick": 3, "end_tick": 4, "label": "blocked_path", "features": [{"marker": "scored"}]},
    ]
    labeller = StatefulLabeller()

    evaluate_v2(rows, labeller)

    assert labeller.seen == ["transition", "scored"]
