"""Stateful V6 specialist evaluation over observable replay streams.

This module scores policy output; fault class and onset are evaluation metadata and
are never passed to the policy. It is deliberately separate from historical
vector-based evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from aeolus.trace import TickRecord

V6_EVALUATION_VERSION = "aeolus_v6_stateful_evaluation_v1"
_STREAM_ROLES = frozenset({"reference", "fault"})
_SPLITS = frozenset({"fit", "calibration", "validation"})
_NAMED_FAULTS = frozenset(
    {"frozen_sensor", "blocked_path", "gradual_primary_fan_degradation"}
)
_NON_OPERATIONAL_LABELS = frozenset({"nominal", "uncertain"})
_ALLOWED_LABELS = _NON_OPERATIONAL_LABELS | _NAMED_FAULTS | frozenset(
    {"sensor_health_concern", "physical_flow_concern"}
)
_SPECIALIST_LABELS = {
    "frozen_sensor": "sensor_health",
    "sensor_health_concern": "sensor_health",
    "blocked_path": "physical_flow",
    "gradual_primary_fan_degradation": "physical_flow",
    "physical_flow_concern": "physical_flow",
}


class V6WindowPolicy(Protocol):
    """A policy which receives only one observable causal trace window."""

    def reset(self) -> None: ...

    def label_window(self, records: Sequence[TickRecord]) -> str: ...


@dataclass(frozen=True)
class V6EvaluationStream:
    """One complete paired-family replay stream for V6 evaluation."""

    family_id: str
    room_family_id: str
    split: str
    scenario_role: Literal["reference", "fault"]
    records: tuple[TickRecord, ...]
    fault_class: str | None = None
    observable_onset_tick: int | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.family_id, "family_id")
        _require_identifier(self.room_family_id, "room_family_id")
        if self.split not in _SPLITS:
            raise ValueError("V6 evaluation stream split is unsupported")
        if self.scenario_role not in _STREAM_ROLES:
            raise ValueError("V6 evaluation stream scenario_role is unsupported")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("V6 evaluation stream records must be a non-empty tuple")
        previous_tick = 0
        for record in self.records:
            if not isinstance(record, TickRecord):
                raise ValueError("V6 evaluation stream records must be TickRecord values")
            if record.tick != previous_tick + 1:
                raise ValueError("V6 evaluation stream ticks must be contiguous from one")
            previous_tick = record.tick
        if self.scenario_role == "reference":
            if self.fault_class is not None or self.observable_onset_tick is not None:
                raise ValueError("V6 reference stream must not carry fault metadata")
            return
        if self.fault_class not in _NAMED_FAULTS:
            raise ValueError("V6 fault stream fault_class is unsupported")
        if (
            isinstance(self.observable_onset_tick, bool)
            or not isinstance(self.observable_onset_tick, int)
            or not 1 <= self.observable_onset_tick <= previous_tick
        ):
            raise ValueError("V6 fault stream observable onset is malformed")


def evaluate_v6(
    streams: Sequence[V6EvaluationStream], policy: V6WindowPolicy, *, window_ticks: int
) -> dict[str, object]:
    """Replay V6 streams causally and report alert/abstention operational burden."""
    _validate_inputs(streams, policy, window_ticks)
    ordered = sorted(streams, key=lambda item: (item.family_id, item.scenario_role))
    healthy_eligible_ticks = 0
    healthy_alert_episode_count = 0
    healthy_alert_stream_count = 0
    healthy_uncertain_windows = 0
    fault_stream_count = 0
    named_detection_count = 0
    post_onset_windows = 0
    post_onset_uncertain_windows = 0
    post_onset_unresolved_stream_count = 0
    detection_latencies: dict[str, list[int]] = {}
    supervised_total = 0
    supervised_excluded_transition_windows = 0
    supervised_confusion: dict[str, dict[str, int]] = {}
    named_support: dict[str, int] = {}
    named_correct: dict[str, int] = {}
    specialist_counts = {
        name: {"support": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}
        for name in ("sensor_health", "physical_flow")
    }

    for stream in ordered:
        policy.reset()
        labels = _replay_stream(stream.records, policy, window_ticks)
        if stream.scenario_role == "reference":
            healthy_eligible_ticks += len(labels)
            healthy_uncertain_windows += sum(label == "uncertain" for _, label in labels)
            episodes, alerted = _healthy_alert_episodes(labels)
            healthy_alert_episode_count += episodes
            healthy_alert_stream_count += int(alerted)
            for _, label in labels:
                _record_supervised_prediction(
                    supervised_confusion, "nominal", label
                )
                _record_specialist_prediction(specialist_counts, "nominal", label)
                supervised_total += 1
            continue

        fault_stream_count += 1
        assert stream.fault_class is not None
        assert stream.observable_onset_tick is not None
        post_onset = [
            (end_tick, label)
            for end_tick, label in labels
            if end_tick >= stream.observable_onset_tick
        ]
        post_onset_windows += len(post_onset)
        post_onset_uncertain_windows += sum(label == "uncertain" for _, label in post_onset)
        first_named_detection = next(
            (end_tick for end_tick, label in post_onset if label == stream.fault_class),
            None,
        )
        if first_named_detection is None:
            post_onset_unresolved_stream_count += 1
        else:
            named_detection_count += 1
            detection_latencies.setdefault(stream.fault_class, []).append(
                first_named_detection - stream.observable_onset_tick
            )

        for end_tick, label in labels:
            start_tick = end_tick - window_ticks + 1
            if end_tick < stream.observable_onset_tick:
                expected_label = "nominal"
            elif start_tick < stream.observable_onset_tick:
                supervised_excluded_transition_windows += 1
                continue
            else:
                expected_label = stream.fault_class
                named_support[expected_label] = named_support.get(expected_label, 0) + 1
                named_correct[expected_label] = named_correct.get(expected_label, 0) + int(
                    label == expected_label
                )
            _record_supervised_prediction(supervised_confusion, expected_label, label)
            _record_specialist_prediction(specialist_counts, expected_label, label)
            supervised_total += 1

    return {
        "schema_version": V6_EVALUATION_VERSION,
        "healthy_eligible_ticks": healthy_eligible_ticks,
        "healthy_alert_episode_count": healthy_alert_episode_count,
        "healthy_alert_episodes_per_1000_ticks": (
            1000.0 * healthy_alert_episode_count / healthy_eligible_ticks
            if healthy_eligible_ticks
            else 0.0
        ),
        "healthy_alert_stream_count": healthy_alert_stream_count,
        "healthy_stream_count": sum(stream.scenario_role == "reference" for stream in streams),
        "healthy_uncertain_windows": healthy_uncertain_windows,
        "healthy_uncertainty_fraction": (
            healthy_uncertain_windows / healthy_eligible_ticks if healthy_eligible_ticks else 0.0
        ),
        "fault_stream_count": fault_stream_count,
        "named_detection_count": named_detection_count,
        "named_detection_recall": (
            named_detection_count / fault_stream_count if fault_stream_count else 0.0
        ),
        "post_onset_windows": post_onset_windows,
        "post_onset_uncertain_windows": post_onset_uncertain_windows,
        "post_onset_uncertainty_fraction": (
            post_onset_uncertain_windows / post_onset_windows if post_onset_windows else 0.0
        ),
        "post_onset_unresolved_stream_count": post_onset_unresolved_stream_count,
        "named_supervised_total": supervised_total,
        "supervised_excluded_transition_windows": supervised_excluded_transition_windows,
        "named_confusion": supervised_confusion,
        "class_specific_named_recall": {
            label: named_correct[label] / support
            for label, support in sorted(named_support.items())
        },
        "named_fault_macro_f1": _named_macro_f1(supervised_confusion, named_support),
        "specialist_window_metrics": {
            name: _specialist_metrics(counts)
            for name, counts in sorted(specialist_counts.items())
        },
        "detection_latency_ticks": {
            label: sum(latencies) / len(latencies)
            for label, latencies in sorted(detection_latencies.items())
        },
    }


def _validate_inputs(
    streams: Sequence[V6EvaluationStream], policy: object, window_ticks: int
) -> None:
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes)) or not streams:
        raise ValueError("V6 evaluation requires non-empty streams")
    if not isinstance(window_ticks, int) or isinstance(window_ticks, bool) or window_ticks < 1:
        raise ValueError("V6 evaluation window_ticks must be a positive integer")
    if not callable(getattr(policy, "reset", None)) or not callable(
        getattr(policy, "label_window", None)
    ):
        raise ValueError("V6 evaluation policy lacks reset or label_window")
    family_roles: dict[str, set[str]] = {}
    stream_ids: set[tuple[str, str]] = set()
    for stream in streams:
        if not isinstance(stream, V6EvaluationStream):
            raise ValueError("V6 evaluation stream is malformed")
        if len(stream.records) < window_ticks:
            raise ValueError("V6 evaluation stream is shorter than the causal window")
        identity = (stream.family_id, stream.scenario_role)
        if identity in stream_ids:
            raise ValueError("V6 evaluation contains a duplicate family stream")
        stream_ids.add(identity)
        family_roles.setdefault(stream.family_id, set()).add(stream.scenario_role)
    if any(roles != _STREAM_ROLES for roles in family_roles.values()):
        raise ValueError("V6 evaluation family is missing reference or fault stream")


def _replay_stream(
    records: tuple[TickRecord, ...], policy: V6WindowPolicy, window_ticks: int
) -> list[tuple[int, str]]:
    labels: list[tuple[int, str]] = []
    for end_index in range(window_ticks - 1, len(records)):
        label = policy.label_window(records[end_index - window_ticks + 1 : end_index + 1])
        if not isinstance(label, str) or label not in _ALLOWED_LABELS:
            raise ValueError("V6 policy emitted an unsupported decision label")
        labels.append((records[end_index].tick, label))
    return labels


def _healthy_alert_episodes(labels: Sequence[tuple[int, str]]) -> tuple[int, bool]:
    prior_alert = False
    episodes = 0
    for _, label in labels:
        alert = label not in _NON_OPERATIONAL_LABELS
        if alert and not prior_alert:
            episodes += 1
        prior_alert = alert
    return episodes, episodes > 0


def _record_supervised_prediction(
    confusion: dict[str, dict[str, int]], expected: str, predicted: str
) -> None:
    confusion.setdefault(expected, {})[predicted] = (
        confusion.setdefault(expected, {}).get(predicted, 0) + 1
    )


def _named_macro_f1(
    confusion: dict[str, dict[str, int]], named_support: dict[str, int]
) -> float:
    if not named_support:
        return 0.0
    values: list[float] = []
    for label in named_support:
        true_positive = confusion.get(label, {}).get(label, 0)
        false_negative = sum(confusion.get(label, {}).values()) - true_positive
        false_positive = sum(
            predictions.get(label, 0)
            for expected, predictions in confusion.items()
            if expected != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(values) / len(values)


def _record_specialist_prediction(
    counts: dict[str, dict[str, int]], expected_label: str, predicted_label: str
) -> None:
    expected = _SPECIALIST_LABELS.get(expected_label)
    predicted = _SPECIALIST_LABELS.get(predicted_label)
    if expected is not None:
        counts[expected]["support"] += 1
        if predicted == expected:
            counts[expected]["true_positive"] += 1
        else:
            counts[expected]["false_negative"] += 1
    if predicted is not None and predicted != expected:
        counts[predicted]["false_positive"] += 1


def _specialist_metrics(counts: dict[str, int]) -> dict[str, int | float]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    return {
        **counts,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
    }


def _require_identifier(value: object, description: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"V6 evaluation stream {description} must be a non-empty string")
