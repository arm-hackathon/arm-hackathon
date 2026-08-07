"""V6 stateful-evaluation semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeolus.config import load_scenario
from aeolus.evaluate_v6 import V6EvaluationStream, evaluate_v6
from aeolus.scenario import run_scenario

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "scenarios" / "standard_habitat.json"


def _records(count: int = 10):
    return tuple(run_scenario(load_scenario(STANDARD))[:count])


def _stream(
    *,
    role: str,
    fault_class: str | None = None,
    onset: int | None = None,
    family_id: str = "validation-room-transition-heavy-s2300-cabin_a-frozen",
    reference_identity: str | None = None,
):
    return V6EvaluationStream(
        family_id=family_id,
        room_family_id="room-transition-heavy",
        split="validation",
        scenario_role=role,
        records=_records(),
        reference_identity=reference_identity,
        fault_class=fault_class,
        observable_onset_tick=onset,
    )


class _AlwaysUncertain:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def label_window(self, records) -> str:
        return "uncertain"


def test_uncertain_after_fault_onset_is_unresolved_not_detected():
    policy = _AlwaysUncertain()

    report = evaluate_v6(
        [
            _stream(role="reference"),
            _stream(role="fault", fault_class="frozen_sensor", onset=4),
        ],
        policy,
        window_ticks=3,
    )

    assert policy.reset_count == 2
    assert report["fault_stream_count"] == 1
    assert report["named_detection_count"] == 0
    assert report["named_detection_recall"] == 0.0
    assert report["detection_latency_ticks"] == {}
    assert report["post_onset_uncertain_windows"] == 7
    assert report["post_onset_unresolved_stream_count"] == 1
    assert report["supervised_excluded_transition_windows"] == 2
    assert report["named_supervised_total"] == 14
    assert report["class_specific_named_recall"] == {"frozen_sensor": 0.0}
    assert report["specialist_window_metrics"]["sensor_health"] == {
        "support": 5,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 5,
        "precision": 0.0,
        "recall": 0.0,
    }


class _StreamScriptedPolicy:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def label_window(self, records) -> str:
        end_tick = records[-1].tick
        if self.reset_count == 1:
            return "frozen_sensor"
        if end_tick in {3, 4, 5, 7, 8}:
            return "sensor_health_concern"
        if end_tick == 6:
            return "uncertain"
        return "nominal"


def test_healthy_overlapping_alerts_are_deduplicated_into_episodes():
    policy = _StreamScriptedPolicy()

    report = evaluate_v6(
        [
            _stream(role="reference"),
            _stream(role="fault", fault_class="frozen_sensor", onset=4),
        ],
        policy,
        window_ticks=3,
    )

    assert policy.reset_count == 2
    assert report["healthy_eligible_ticks"] == 10
    assert report["healthy_policy_windows"] == 8
    assert report["healthy_alert_episode_count"] == 2
    assert report["healthy_alert_episodes_per_1000_ticks"] == 200.0
    assert report["healthy_alert_stream_count"] == 1
    assert report["healthy_uncertain_windows"] == 1
    assert report["healthy_uncertainty_fraction"] == 0.125
    assert report["named_detection_count"] == 1
    assert report["detection_latency_ticks"] == {"frozen_sensor": 2.0}


def test_shared_healthy_reference_is_counted_once_across_fault_families():
    report = evaluate_v6(
        [
            _stream(role="reference", family_id="family-a", reference_identity="reference-sha"),
            _stream(role="fault", family_id="family-a", fault_class="frozen_sensor", onset=4),
            _stream(role="reference", family_id="family-b", reference_identity="reference-sha"),
            _stream(role="fault", family_id="family-b", fault_class="blocked_path", onset=4),
        ],
        _AlwaysUncertain(),
        window_ticks=3,
    )

    assert report["healthy_stream_count"] == 1
    assert report["healthy_eligible_ticks"] == 10
    assert report["healthy_policy_windows"] == 8
    assert report["fault_stream_count"] == 2


class _InvalidLabelPolicy(_AlwaysUncertain):
    def label_window(self, records) -> str:
        return "all_good_trust_me"


def test_evaluator_rejects_orphaned_family_streams():
    with pytest.raises(ValueError, match="missing reference or fault"):
        evaluate_v6([_stream(role="reference")], _AlwaysUncertain(), window_ticks=3)


def test_evaluator_rejects_unsupported_policy_decisions():
    with pytest.raises(ValueError, match="unsupported decision label"):
        evaluate_v6(
            [
                _stream(role="reference"),
                _stream(role="fault", fault_class="frozen_sensor", onset=4),
            ],
            _InvalidLabelPolicy(),
            window_ticks=3,
        )
