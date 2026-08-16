"""Feature projections for baseline fitting on the validated Habitat V2 pilot dataset.

This module is deliberately separate from ``baselines.py``: that module is bound
by design to the historical D1 development-fixture manifests, whereas this
module consumes only the completed, custody-validated pilot packet tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .projection import ForecastLayout

TARGET_COUNT: Final = 51
NUMERIC_FEATURE_COUNT: Final = 194
PILOT_INPUT_MANIFEST_SHA256: Final = (
    "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
)
PILOT_TARGET_MANIFEST_SHA256: Final = (
    "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"
)
_ENVIRONMENT_FIELDS: Final = frozenset(
    {
        "temperature_k",
        "pressure_pa",
        "co2_ppm",
        "o2_mole_fraction",
        "relative_humidity",
    }
)
_RESOURCE_IDS: Final = frozenset(
    {
        "battery_state_of_charge",
        "oxygen_store_fraction",
        "sorbent_remaining_fraction",
    }
)


class PilotBaselineError(ValueError):
    """A real-pilot packet cannot be projected into the approved compact view."""


@dataclass(frozen=True, slots=True)
class PilotExample:
    """One model-fit row derived from a custody-validated matched packet."""

    sample_id: str
    cluster_id: str
    action_present: bool
    history_f32: np.ndarray
    action_f32: np.ndarray
    targets_f32: np.ndarray


@dataclass(frozen=True, slots=True)
class ValidatedPilotDataset:
    """One timing view materialised only from a complete custody-verified campaign."""

    campaign_manifest_sha256: str
    window_steps: int
    horizon_steps: int
    examples: tuple[PilotExample, ...]


def _readonly_f32(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _target_source_columns(layout: ForecastLayout) -> tuple[tuple[int, int | None], ...]:
    if (
        type(layout) is not ForecastLayout
        or (
            layout.input_manifest_sha256,
            layout.target_manifest_sha256,
        )
        != (PILOT_INPUT_MANIFEST_SHA256, PILOT_TARGET_MANIFEST_SHA256)
    ):
        raise PilotBaselineError("compact history requires the exact frozen forecast layout")
    operational = layout.operational_descriptors
    targets = layout.target_descriptors
    if len(operational) != 167 or len(targets) != TARGET_COUNT:
        raise PilotBaselineError("forecast layout dimensions drift")

    indexed = {
        (descriptor.get("descriptor_id"), descriptor.get("source_kind")): index
        for index, descriptor in enumerate(operational)
    }
    if len(indexed) != len(operational):
        raise PilotBaselineError("operational descriptor identities are not unique")

    columns: list[tuple[int, int | None]] = []
    for target in targets:
        target_id = target.get("descriptor_id")
        if type(target_id) is not str or not target_id:
            raise PilotBaselineError("target descriptor identity is malformed")
        field = target_id.rsplit("/", 1)[-1]
        if field in _ENVIRONMENT_FIELDS:
            first = indexed.get((target_id, "primary_sensor_head"))
            second = indexed.get((target_id, "secondary_sensor_head"))
            if first is None or second is None:
                raise PilotBaselineError("environmental target lacks both sensor heads")
            columns.append((first, second))
        elif field == "branch_airflow_m3_s":
            zone = target_id.rsplit("/", 1)[0]
            source = indexed.get(
                (f"branch_airflow_m3_s/{zone}", "operational_feedback_instrument")
            )
            if source is None:
                raise PilotBaselineError("airflow target lacks operational feedback")
            columns.append((source, None))
        elif target_id in _RESOURCE_IDS:
            source = indexed.get((target_id, "operational_feedback_instrument"))
            if source is None:
                raise PilotBaselineError("resource target lacks operational feedback")
            columns.append((source, None))
        else:
            raise PilotBaselineError("target descriptor is not in the compact contract")
    if len(columns) != TARGET_COUNT:
        raise PilotBaselineError("compact target mapping has the wrong width")
    return tuple(columns)


def compact_target_history(
    history_numeric_f32: np.ndarray,
    layout: ForecastLayout,
    *,
    operational_available_bool: np.ndarray,
) -> np.ndarray:
    """Return causal 51-target observations from frozen public sources.

    Every environmental target maps to its ordered primary sensor head and,
    only when that head is unavailable, to its named secondary head. Airflow
    and resource targets map to their exact public operational-feedback
    instruments. This is a source-grounded observation mapping, never an
    availability-weighted or averaged target estimate. A row with no available
    mapped source is rejected. No target truth, future field, receipt, ID or
    authority outcome is read here.
    """
    history = np.asarray(history_numeric_f32)
    available = np.asarray(operational_available_bool)
    if (
        history.ndim != 2
        or history.shape[0] not in (4, 8, 16)
        or history.shape[1] != NUMERIC_FEATURE_COUNT
        or history.dtype != np.float32
        or not np.isfinite(history).all()
    ):
        raise PilotBaselineError("numeric history must be finite float32[W,194]")
    if available.shape != (history.shape[0], 167) or available.dtype != np.bool_:
        raise PilotBaselineError(
            "compact history requires bool[W,167] operational availability evidence"
        )

    output = np.empty((history.shape[0], TARGET_COUNT), dtype=np.float32)
    for target_index, (first, second) in enumerate(_target_source_columns(layout)):
        first_available = available[:, first]
        if second is None:
            if not first_available.all():
                raise PilotBaselineError("target source availability is incomplete")
            output[:, target_index] = history[:, first]
            continue
        second_available = available[:, second]
        if not np.logical_or(first_available, second_available).all():
            raise PilotBaselineError("target source availability is incomplete")
        # Ordered, source-grounded fallback: no synthetic averaging of heads.
        output[:, target_index] = np.where(
            first_available,
            history[:, first],
            history[:, second],
        )
    if not np.isfinite(output).all():
        raise PilotBaselineError("compact target history is non-finite")
    output.setflags(write=False)
    return output


def packet_examples(
    *,
    continuation_ids: np.ndarray,
    cluster_ids: np.ndarray,
    action_present: np.ndarray,
    operational_available_bool: np.ndarray,
    history_numeric_f32: np.ndarray,
    proposed_action_f32: np.ndarray,
    targets_f32: np.ndarray,
    layout: ForecastLayout,
    window_steps: int,
    horizon_steps: int,
) -> tuple[PilotExample, ...]:
    """Slice one maximum-context packet into a fixed timing-view example tuple."""
    if window_steps not in (4, 8, 16) or horizon_steps not in (2, 4, 8):
        raise PilotBaselineError("timing view is outside the approved grid")
    identifiers = np.asarray(continuation_ids)
    clusters = np.asarray(cluster_ids)
    present = np.asarray(action_present)
    available = np.asarray(operational_available_bool)
    histories = np.asarray(history_numeric_f32)
    actions = np.asarray(proposed_action_f32)
    targets = np.asarray(targets_f32)
    if (
        identifiers.shape != (5,)
        or clusters.shape != (5,)
        or present.shape != (5,)
        or present.dtype != np.bool_
        or available.shape != (5, 16, 167)
        or available.dtype != np.bool_
        or histories.shape != (5, 16, NUMERIC_FEATURE_COUNT)
        or actions.shape != (5, 27)
        or targets.shape != (5, 8, TARGET_COUNT)
        or histories.dtype != np.float32
        or actions.dtype != np.float32
        or targets.dtype != np.float32
        or not np.isfinite(histories).all()
        or not np.isfinite(actions).all()
        or not np.isfinite(targets).all()
    ):
        raise PilotBaselineError("maximum packet tensors are malformed")
    if (
        any(type(value) is not str or not value for value in identifiers.tolist())
        or len(set(identifiers.tolist())) != 5
        or any(type(value) is not str or not value for value in clusters.tolist())
        or len(set(clusters.tolist())) != 1
    ):
        raise PilotBaselineError("packet identities are malformed or not matched")

    examples: list[PilotExample] = []
    for index in range(5):
        history = compact_target_history(
            histories[index, -window_steps:],
            layout,
            operational_available_bool=available[index, -window_steps:],
        )
        examples.append(
            PilotExample(
                sample_id=str(identifiers[index]),
                cluster_id=str(clusters[index]),
                action_present=bool(present[index]),
                history_f32=history,
                action_f32=_readonly_f32(actions[index]),
                targets_f32=_readonly_f32(targets[index, :horizon_steps]),
            )
        )
    return tuple(examples)


def load_validated_pilot_dataset(
    repo_root: str | Path,
    campaign_root: str | Path,
    *,
    window_steps: int,
    horizon_steps: int,
) -> ValidatedPilotDataset:
    """Refuse archive materialisation until availability evidence is frozen.

    The current packet schema stores numeric values but not the status tensors
    needed to distinguish an observed zero from unavailable telemetry. This
    guard makes that missing evidence a visible data-contract blocker.
    """
    if window_steps not in (4, 8, 16) or horizon_steps not in (2, 4, 8):
        raise PilotBaselineError("timing view is outside the approved grid")
    raise PilotBaselineError(
        "campaign packets omit target-source availability evidence; "
        "a provenance-bound availability artifact is required before loading"
    )


def _validate_example(example: PilotExample) -> None:
    if type(example) is not PilotExample or not example.sample_id or not example.cluster_id:
        raise PilotBaselineError("pilot example identity is malformed")
    history = np.asarray(example.history_f32)
    action = np.asarray(example.action_f32)
    target = np.asarray(example.targets_f32)
    if (
        history.ndim != 2
        or history.shape[0] not in (4, 8, 16)
        or history.shape[1] != TARGET_COUNT
        or action.shape != (27,)
        or target.ndim != 2
        or target.shape[0] not in (2, 4, 8)
        or target.shape[1] != TARGET_COUNT
        or history.dtype != np.float32
        or action.dtype != np.float32
        or target.dtype != np.float32
        or not np.isfinite(history).all()
        or not np.isfinite(action).all()
        or not np.isfinite(target).all()
    ):
        raise PilotBaselineError("pilot example tensors are malformed")


def compact_feature_matrix(
    examples: tuple[PilotExample, ...], *, include_action: bool
) -> np.ndarray:
    """Flatten only compact causal history and optionally the proposed action."""
    if type(include_action) is not bool or not examples:
        raise PilotBaselineError("feature assembly needs examples and an action policy")
    first = examples[0]
    _validate_example(first)
    window_steps = first.history_f32.shape[0]
    horizon_steps = first.targets_f32.shape[0]
    rows: list[np.ndarray] = []
    identities: set[str] = set()
    for item in examples:
        _validate_example(item)
        if (
            item.sample_id in identities
            or item.history_f32.shape[0] != window_steps
            or item.targets_f32.shape[0] != horizon_steps
        ):
            raise PilotBaselineError("examples have duplicate IDs or mixed timing views")
        identities.add(item.sample_id)
        pieces = [item.history_f32.reshape(-1)]
        if include_action:
            pieces.append(item.action_f32)
        rows.append(np.concatenate(pieces).astype(np.float32, copy=False))
    result = np.stack(rows, axis=0).astype(np.float32, copy=False)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CompactRidgeModel:
    """One standardised direct multi-output ridge fitted from pilot examples."""

    window_steps: int
    horizon_steps: int
    include_action: bool
    alpha: float
    feature_mean_f32: np.ndarray
    feature_scale_f32: np.ndarray
    target_mean_f32: np.ndarray
    coefficient_f32: np.ndarray

    def predict(self, history_f32: np.ndarray, action_f32: np.ndarray) -> np.ndarray:
        history = np.asarray(history_f32)
        action = np.asarray(action_f32)
        if (
            history.shape != (self.window_steps, TARGET_COUNT)
            or action.shape != (27,)
            or history.dtype != np.float32
            or action.dtype != np.float32
            or not np.isfinite(history).all()
            or not np.isfinite(action).all()
        ):
            raise PilotBaselineError("prediction inputs drift from the fitted timing view")
        parts = [history.reshape(-1)]
        if self.include_action:
            parts.append(action)
        feature = np.concatenate(parts).astype(np.float32, copy=False)
        prediction = (
            ((feature - self.feature_mean_f32) / self.feature_scale_f32)
            @ self.coefficient_f32
            + self.target_mean_f32
        )
        result = prediction.reshape(self.horizon_steps, TARGET_COUNT).astype(
            np.float32, copy=False
        )
        if not np.isfinite(result).all():
            raise PilotBaselineError("ridge prediction is non-finite")
        result.setflags(write=False)
        return result


def fit_compact_ridge(
    examples: tuple[PilotExample, ...], *, include_action: bool, alpha: float
) -> CompactRidgeModel:
    """Fit a deterministic compact ridge using only the caller's training examples."""
    if type(alpha) is not float or not np.isfinite(alpha) or alpha <= 0.0:
        raise PilotBaselineError("ridge alpha must be a positive finite float")
    features = compact_feature_matrix(examples, include_action=include_action)
    first = examples[0]
    targets = np.stack([item.targets_f32.reshape(-1) for item in examples], axis=0)
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    target_mean = targets.mean(axis=0, dtype=np.float64)
    standardised = (features.astype(np.float64) - mean) / scale
    centred_targets = targets.astype(np.float64) - target_mean
    system = standardised.T @ standardised + alpha * np.eye(features.shape[1])
    coefficient = np.linalg.solve(system, standardised.T @ centred_targets)
    if not np.isfinite(coefficient).all():
        raise PilotBaselineError("ridge fit is non-finite")
    return CompactRidgeModel(
        window_steps=first.history_f32.shape[0],
        horizon_steps=first.targets_f32.shape[0],
        include_action=include_action,
        alpha=alpha,
        feature_mean_f32=_readonly_f32(mean),
        feature_scale_f32=_readonly_f32(scale),
        target_mean_f32=_readonly_f32(target_mean),
        coefficient_f32=_readonly_f32(coefficient),
    )
