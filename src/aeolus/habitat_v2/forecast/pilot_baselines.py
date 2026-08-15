"""Feature projections for baseline fitting on the validated Habitat V2 pilot dataset.

This module is deliberately separate from ``baselines.py``: that module is bound
by design to the historical D1 development-fixture manifests, whereas this
module consumes only the completed, custody-validated pilot packet tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np

from .projection import ForecastLayout


TARGET_COUNT: Final = 51
NUMERIC_FEATURE_COUNT: Final = 194
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
    if type(layout) is not ForecastLayout:
        raise PilotBaselineError("compact history requires the exact forecast layout")
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
    history_numeric_f32: np.ndarray, layout: ForecastLayout
) -> np.ndarray:
    """Return causal 51-target estimates from a finite public numeric history.

    Environmental estimates use the arithmetic mean of the two public sensor
    heads. Airflow and resource estimates use their public operational feedback
    channels. No target truth, future field, receipt, ID or authority outcome is
    read here.
    """
    history = np.asarray(history_numeric_f32)
    if (
        history.ndim != 2
        or history.shape[0] not in (4, 8, 16)
        or history.shape[1] != NUMERIC_FEATURE_COUNT
        or history.dtype != np.float32
        or not np.isfinite(history).all()
    ):
        raise PilotBaselineError("numeric history must be finite float32[W,194]")

    output = np.empty((history.shape[0], TARGET_COUNT), dtype=np.float32)
    for target_index, (first, second) in enumerate(_target_source_columns(layout)):
        output[:, target_index] = (
            history[:, first]
            if second is None
            else (history[:, first] + history[:, second]) / np.float32(2.0)
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
    histories = np.asarray(history_numeric_f32)
    actions = np.asarray(proposed_action_f32)
    targets = np.asarray(targets_f32)
    if (
        identifiers.shape != (5,)
        or clusters.shape != (5,)
        or present.shape != (5,)
        or present.dtype != np.bool_
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
        history = compact_target_history(histories[index, -window_steps:], layout)
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
    """Materialise a timing view only after validating the complete campaign.

    This is intentionally read-only. It reuses the campaign's fail-closed pair
    validator for every packet, checks the self-hashed campaign manifest, and
    refuses incomplete/bounded campaigns.
    """
    from .contracts import canonical_json_bytes, load_forecast_contracts
    from .pilot import load_approved_pilot_design
    from .pilot_campaign import _load_validated_staged_pair
    from .projection import forecast_layout

    if window_steps not in (4, 8, 16) or horizon_steps not in (2, 4, 8):
        raise PilotBaselineError("timing view is outside the approved grid")
    root = Path(repo_root).resolve()
    output = Path(campaign_root).resolve()
    manifest_path = output / "campaign-manifest.json"
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
        declared = manifest.pop("campaign_manifest_sha256")
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise PilotBaselineError("campaign manifest cannot be loaded") from error
    if (
        raw_manifest
        != canonical_json_bytes({**manifest, "campaign_manifest_sha256": declared})
        or type(declared) is not str
        or declared != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        or set(manifest)
        != {
            "schema_version",
            "roster_sha256",
            "profile_action_sha256",
            "preflight_sha256",
            "planned_hmc_runs",
            "worker_count",
            "pairs_completed",
            "hmc_runs_executed",
            "pair_manifests",
        }
        or manifest["schema_version"]
        != "aeolus_habitat_v2_forecast_pilot_campaign_manifest_v1"
        or manifest["pairs_completed"] != 4680
        or manifest["hmc_runs_executed"] != 23400
        or len(manifest["pair_manifests"]) != 4680
    ):
        raise PilotBaselineError("campaign manifest is not the complete frozen campaign")

    design = load_approved_pilot_design(root)
    layout = forecast_layout(load_forecast_contracts(root))
    expected_clusters = {cluster.cluster_id for cluster in design.clusters}
    examples: list[PilotExample] = []
    seen_pairs: set[str] = set()
    seen_samples: set[str] = set()
    for declared_pair in manifest["pair_manifests"]:
        if not isinstance(declared_pair, dict) or set(declared_pair) != {
            "pair_id",
            "manifest_sha256",
            "training_packet_sha256",
            "training_packet_byte_length",
        }:
            raise PilotBaselineError("campaign pair manifest is malformed")
        pair_id = declared_pair["pair_id"]
        if type(pair_id) is not str or pair_id in seen_pairs:
            raise PilotBaselineError("campaign pair identity is malformed or duplicated")
        seen_pairs.add(pair_id)
        validated = _load_validated_staged_pair(output / pair_id, design)
        if validated != declared_pair:
            raise PilotBaselineError("campaign pair record drifts from validated packet")
        try:
            with np.load(output / pair_id / "training.npz", allow_pickle=False) as packet:
                pair_examples = packet_examples(
                    continuation_ids=packet["continuation_ids"],
                    cluster_ids=packet["cluster_ids"],
                    action_present=packet["action_present"],
                    history_numeric_f32=packet["history_numeric_f32"],
                    proposed_action_f32=packet["proposed_action_f32"],
                    targets_f32=packet["targets_f32"],
                    layout=layout,
                    window_steps=window_steps,
                    horizon_steps=horizon_steps,
                )
        except (OSError, ValueError, KeyError) as error:
            raise PilotBaselineError("validated packet tensors cannot be loaded") from error
        for example in pair_examples:
            if example.cluster_id not in expected_clusters or example.sample_id in seen_samples:
                raise PilotBaselineError("dataset cluster/sample identity drifts")
            seen_samples.add(example.sample_id)
            examples.append(example)
    if len(examples) != 23400 or {item.cluster_id for item in examples} != expected_clusters:
        raise PilotBaselineError("complete campaign examples do not cover the frozen design")
    return ValidatedPilotDataset(
        campaign_manifest_sha256=declared,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        examples=tuple(examples),
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
