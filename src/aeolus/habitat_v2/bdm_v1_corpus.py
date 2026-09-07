"""BDM-v1 development corpus: causal features and counterfactual labels.

Issue #72 part 3 (the ``#72c`` corpus deliverable). Builds model-facing
feature rows and true-plant counterfactual labels over the frozen custody
roster, conforming to the Issue #70 input schema and label contract.

Conventions (all declared in ``contracts/habitat_v2_bdm_v1_corpus_protocol_v1.json``):

- One sample per (family, decision step, catalogue candidate action). Feature
  keys are exactly the nineteen contract-declared model-facing field names;
  ``validate_model_input_fields`` rejects anything else fail-closed.
- Features consume only issued, verified operational snapshots completed at
  or before the decision step (causal window of 16 completed steps), the
  requested/achieved command references, operational resource gauges, the
  completed operating mode, and retained HMC proposal dispositions.
- Missingness is explicit: unavailable environmental channels carry forward
  their last available value with ``observed_value_mask`` false and a
  staleness count; a channel never yet observed carries marked zero
  imputation (mask false, staleness equal to the completed step), never
  unmarked zeros. Achieved-feedback and gauge availability is recorded in a
  separate ``missingness_metadata`` block for audit; it is not model input.
  Sensor-stuck defects present as frozen AVAILABLE readings (the
  instrumentation holds the last value); genuine unavailability
  (``DEPENDENCY_UNAVAILABLE``/``MISSING``) exercises the carry-forward path.
- Labels are true-plant outcomes of plant-only counterfactual rollouts from
  the checkpoint causal state at the decision step: the candidate command and
  the hold command, 32 steps, identical prior observations and disturbances.
  Labels are label material only and are never runtime features.
- The sealed BLIND_FINAL partition is rejected fail-closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .bdm_v1_benchmark_contract import validate_model_input_fields
from .bdm_v1_families import DECISION_STEPS, EPISODE_STEPS
from .forecast.contracts import ForecastContracts, canonical_json_bytes
from .forecast.projection import MODE_ORDER, _command_vector
from .forecast_issue55_race import (
    COMFORT_COLUMNS,
    RESOURCE_COLUMNS,
    project_true_targets,
    scenario_zone_order,
    target_bounds,
)
from .hmc import HabitatManagementComputer
from .physics import (
    advance_one_step_with_command,
    command_from_achieved_state,
    initial_state,
    operating_mode_for_application_step,
)
from .scenario import Scenario

BDM_V1_CORPUS_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_corpus_v1"
WINDOW_STEPS = 16
ROLLOUT_HORIZON_STEPS = 32
LABEL_HORIZON_KEYS = (4, 16, 32)
CORPUS_PARTITIONS = ("TRAIN", "DEV", "CALIBRATION")
ENV_CHANNELS = (
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
    "branch_airflow_m3_s",
)
PRIMARY_CHANNELS = ENV_CHANNELS[:5]
DISPOSITION_ORDER = ("NO_PROPOSAL", "VALID", "REJECTED", "REJECTED_INPUT")
RESOURCE_GAUGE_IDS = (
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_remaining_fraction",
)
RESOURCE_GAUGE_FIELD_NAMES = (
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_fraction",
)
COMMAND_DIM = 27
MODE_DIM = len(MODE_ORDER)
LABEL_FIELD_NAMES = (
    "crossing_event",
    "safety_exposure",
    "maximum_crossing",
    "comfort_deviation",
    "resource_composite",
)
FEATURE_FIELD_NAMES = (
    "zone_temperature_k",
    "zone_pressure_pa",
    "zone_co2_ppm",
    "zone_o2_mole_fraction",
    "zone_relative_humidity",
    "zone_branch_airflow_m3_s",
    "observed_value_mask",
    "steps_since_last_valid_observation",
    "requested_command_vector",
    "achieved_actuator_state",
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_fraction",
    "operating_mode_one_hot",
    "prior_proposal_dispositions",
    "topology_configuration_descriptor",
    "candidate_action_command_vector",
    "candidate_action_index",
    "declared_known_future_schedule",
)


class BdmV1CorpusError(ValueError):
    """Raised when corpus construction or validation fails closed."""


@dataclass(frozen=True, slots=True)
class CorpusFamilyMeta:
    """Identity binding for one corpus family (no hidden truth)."""

    family_id: str
    group_id: str
    group_index: int
    partition: str
    template_id: str
    stratum: str
    sensor_variant: str
    scenario_sha256: str

    def __post_init__(self) -> None:
        if self.partition not in CORPUS_PARTITIONS:
            raise BdmV1CorpusError(
                f"corpus rejects partition {self.partition!r}; BLIND_FINAL and "
                "unknown partitions are sealed fail-closed"
            )
        if len(self.scenario_sha256) != 64:
            raise BdmV1CorpusError("corpus family meta requires a 64-hex scenario digest")


def _f32(value: float, label: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise BdmV1CorpusError(f"{label} is non-finite or boolean")
    return float(np.float32(float(value)))


def _sample_value(sample: Mapping[str, Any], label: str) -> tuple[bool, float]:
    availability = sample["availability"]
    if availability == "AVAILABLE":
        return True, _f32(float(sample["value"]), label)
    if availability == "UNAVAILABLE" and sample["value"] is None:
        return False, 0.0
    raise BdmV1CorpusError(f"{label} availability is invalid")


class _CarryTracker:
    """Carry-forward bookkeeping with explicit mask and staleness."""

    def __init__(self) -> None:
        self._last_value: dict[str, float] = {}
        self._last_row: dict[str, int] = {}

    def read(self, key: str, available: bool, value: float, row: int, step: int) -> tuple[float, bool, int]:
        if available:
            self._last_value[key] = value
            self._last_row[key] = row
            return value, True, 0
        if key in self._last_value:
            return self._last_value[key], False, row - self._last_row[key]
        return 0.0, False, step


def _window_rows(
    snapshots: Sequence[Mapping[str, Any]],
    bundle: ForecastContracts,
) -> dict[str, Any]:
    zone_ids = tuple(bundle.topology.zone_ids)
    env = np.zeros((WINDOW_STEPS, len(zone_ids), len(ENV_CHANNELS)), dtype=np.float64)
    mask = np.zeros((WINDOW_STEPS, len(zone_ids), len(ENV_CHANNELS)), dtype=bool)
    stale = np.zeros((WINDOW_STEPS, len(zone_ids), len(ENV_CHANNELS)), dtype=int)
    requested = np.zeros((WINDOW_STEPS, COMMAND_DIM), dtype=np.float64)
    achieved = np.zeros((WINDOW_STEPS, COMMAND_DIM), dtype=np.float64)
    achieved_mask = np.zeros((WINDOW_STEPS, COMMAND_DIM), dtype=bool)
    achieved_stale = np.zeros((WINDOW_STEPS, COMMAND_DIM), dtype=int)
    gauges = np.zeros((WINDOW_STEPS, len(RESOURCE_GAUGE_IDS)), dtype=np.float64)
    gauge_mask = np.zeros((WINDOW_STEPS, len(RESOURCE_GAUGE_IDS)), dtype=bool)
    gauge_stale = np.zeros((WINDOW_STEPS, len(RESOURCE_GAUGE_IDS)), dtype=int)
    modes = np.zeros((WINDOW_STEPS, MODE_DIM), dtype=np.float64)
    dispositions = np.zeros((WINDOW_STEPS, len(DISPOSITION_ORDER)), dtype=np.float64)
    env_tracker = _CarryTracker()
    achieved_tracker = _CarryTracker()
    gauge_tracker = _CarryTracker()

    achieved_names = (
        ["fan_speed_fraction"]
        + [f"damper_position_by_id/{damper}" for _, damper in bundle.topology.branch_pairs]
        + ["scrubber_capture_rate_mol_s", "condenser_removal_rate_mol_s"]
        + [f"cooling_delivery_w/{zone}" for zone in zone_ids]
        + [f"oxygen_delivery_mol_s/{zone}" for zone in zone_ids]
    )
    if len(achieved_names) != COMMAND_DIM:
        raise BdmV1CorpusError("achieved actuator feedback layout drifted")

    for row, snapshot in enumerate(snapshots):
        step = int(snapshot["completed_step"])
        primary = {
            sample["descriptor_id"]: sample
            for sample in snapshot["primary_telemetry"]["samples"]
        }
        feedback = {
            sample["descriptor_id"]: sample
            for sample in snapshot["operational_feedback"]["samples"]
        }
        for zone_index, zone in enumerate(zone_ids):
            for channel_index, channel in enumerate(ENV_CHANNELS):
                descriptor = (
                    f"{zone}/{channel}"
                    if channel in PRIMARY_CHANNELS
                    else f"branch_airflow_m3_s/{zone}"
                )
                source = primary if channel in PRIMARY_CHANNELS else feedback
                available, raw = _sample_value(source[descriptor], f"env {descriptor}")
                value, observed, staleness = env_tracker.read(
                    f"env:{descriptor}", available, raw, row, step
                )
                env[row, zone_index, channel_index] = value
                mask[row, zone_index, channel_index] = observed
                stale[row, zone_index, channel_index] = staleness
        requested[row] = np.asarray(
            _command_vector(bundle, snapshot["command_reference"]["command"], "requested command"),
            dtype=np.float64,
        )
        for column, name in enumerate(achieved_names):
            available, raw = _sample_value(feedback[name], f"achieved {name}")
            value, observed, staleness = achieved_tracker.read(
                f"achieved:{name}", available, raw, row, step
            )
            achieved[row, column] = value
            achieved_mask[row, column] = observed
            achieved_stale[row, column] = staleness
        for column, gauge_id in enumerate(RESOURCE_GAUGE_IDS):
            sample = next(
                item
                for item in snapshot["operational_resource_gauges"]["samples"]
                if item["descriptor_id"] == gauge_id
            )
            available, raw = _sample_value(sample, f"gauge {gauge_id}")
            value, observed, staleness = gauge_tracker.read(
                f"gauge:{gauge_id}", available, raw, row, step
            )
            gauges[row, column] = value
            gauge_mask[row, column] = observed
            gauge_stale[row, column] = staleness
        mode = str(snapshot["completed_operating_mode"])
        if mode not in MODE_ORDER:
            raise BdmV1CorpusError(f"unknown completed operating mode {mode!r}")
        modes[row, MODE_ORDER.index(mode)] = 1.0
        disposition = str(snapshot["proposal_disposition"])
        if disposition not in DISPOSITION_ORDER:
            raise BdmV1CorpusError(f"unknown retained disposition {disposition!r}")
        dispositions[row, DISPOSITION_ORDER.index(disposition)] = 1.0
    return {
        "env": env,
        "mask": mask,
        "stale": stale,
        "requested": requested,
        "achieved": achieved,
        "achieved_mask": achieved_mask,
        "achieved_stale": achieved_stale,
        "gauges": gauges,
        "gauge_mask": gauge_mask,
        "gauge_stale": gauge_stale,
        "modes": modes,
        "dispositions": dispositions,
    }


def rollout_labels(
    scenario: Scenario,
    zone_ids: Sequence[str],
    state: Any,
    command: Mapping[str, Any],
) -> dict[str, Any]:
    """Roll the true plant 32 steps under one declared command; label it."""

    scales, nominals, lowers, uppers = target_bounds()
    rows: list[np.ndarray] = []
    current = state
    for _ in range(ROLLOUT_HORIZON_STEPS):
        result = advance_one_step_with_command(scenario, current, dict(command))
        current = result.state
        rows.append(
            np.asarray(project_true_targets(scenario, zone_ids, current), dtype=np.float64)
        )
    values = np.stack(rows)
    lower_crossing = np.maximum(0.0, lowers[None, :] - values) / scales[None, :]
    upper_crossing = np.maximum(0.0, values - uppers[None, :]) / scales[None, :]
    crossings = lower_crossing + upper_crossing
    per_step = crossings.sum(axis=1)
    occupied_rows = [
        values[offset]
        for offset in range(ROLLOUT_HORIZON_STEPS)
        if operating_mode_for_application_step(scenario, state.step - 1 + offset)
        == "occupied"
    ]
    if occupied_rows:
        comfort = float(
            np.mean(
                np.abs(
                    np.stack(occupied_rows)[:, list(COMFORT_COLUMNS)]
                    - nominals[list(COMFORT_COLUMNS)][None, :]
                )
            )
        )
    else:
        comfort = 0.0
    resource = float(
        sum(
            max(0.0, float(values[0, column]) - float(values[-1, column]))
            for column in RESOURCE_COLUMNS
        )
    )
    return {
        "decision_targets": {
            "crossing_event": 1.0 if bool(np.any(crossings > 0.0)) else 0.0,
            "safety_exposure": _f32(float(per_step.sum()), "rollout safety exposure"),
            "maximum_crossing": _f32(float(per_step.max()), "rollout maximum crossing"),
            "comfort_deviation": _f32(comfort, "rollout comfort deviation"),
            "resource_composite": _f32(resource, "rollout resource composite"),
        },
        "trajectory_targets": {
            str(horizon): [_f32(float(value), f"trajectory target {horizon}") for value in rows[horizon - 1]]
            for horizon in LABEL_HORIZON_KEYS
        },
    }


def collect_family_samples(
    bundle: ForecastContracts,
    scenario: Scenario,
    meta: CorpusFamilyMeta,
    decision_steps: Sequence[int] = DECISION_STEPS,
) -> tuple[dict[str, Any], ...]:
    """Run one hold episode; emit labelled samples per decision and candidate."""

    if type(bundle) is not ForecastContracts or type(scenario) is not Scenario:
        raise BdmV1CorpusError("corpus collection requires contracts and a Scenario")
    steps = tuple(int(step) for step in decision_steps)
    if any(
        isinstance(step, bool) or step < WINDOW_STEPS or step > EPISODE_STEPS - ROLLOUT_HORIZON_STEPS
        for step in steps
    ):
        raise BdmV1CorpusError(
            "decision steps must admit a complete causal window and a full rollout"
        )
    zone_ids = scenario_zone_order(scenario)
    actions = tuple(bundle.actions)
    if len(actions) != 4 or len({action.action_id for action in actions}) != 4:
        raise BdmV1CorpusError("corpus requires the frozen four-action catalogue")
    nonce = hashlib.sha256(b"bdm-v1-corpus|" + meta.family_id.encode("utf-8")).digest()
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    shadow = initial_state(scenario)
    snapshots: dict[int, dict[str, Any]] = {}
    states: dict[int, Any] = {}
    for step in range(EPISODE_STEPS):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise BdmV1CorpusError(f"HMC terminated during corpus episode at step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        receipt = hmc.propose(None, handle)
        receipt_mapping = receipt.to_mapping()
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise BdmV1CorpusError(f"HMC terminated while arbitrating step {step}")
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise BdmV1CorpusError(f"HMC terminated while stepping step {step}")
        result = advance_one_step_with_command(scenario, shadow, dict(arbitration.final_command))
        if result.state.step != step + 1:
            raise BdmV1CorpusError("shadow state drifted from the HMC application step")
        if (
            hashlib.sha256(canonical_json_bytes(result.receipt)).hexdigest()
            != step_receipt.plant_receipt_digest
        ):
            raise BdmV1CorpusError("shadow plant receipt diverges from the HMC receipt")
        shadow = result.state
        mapping = snapshot.to_mapping()
        mapping["proposal_disposition"] = str(receipt_mapping["validation_outcome"])
        states[shadow.step] = shadow
        completed = int(mapping["completed_step"])
        if completed < 1:
            continue
        snapshots[completed] = mapping

    samples: list[dict[str, Any]] = []
    for decision in steps:
        window = [
            snapshots[completed]
            for completed in range(decision - WINDOW_STEPS + 1, decision + 1)
        ]
        rows = _window_rows(window, bundle)
        checkpoint = states[decision]
        hold = dict(command_from_achieved_state(scenario, checkpoint).command.to_mapping())
        hold_labels = rollout_labels(scenario, zone_ids, checkpoint, hold)
        env_by_channel = {
            channel: [
                [
                    _f32(float(step_row[zone_index, channel_index]), f"env {channel}")
                    for zone_index in range(step_row.shape[0])
                ]
                for step_row in rows["env"]
            ]
            for channel_index, channel in enumerate(ENV_CHANNELS)
        }
        mask_rows = [
            [[bool(flag) for flag in zone_row] for zone_row in step_row]
            for step_row in rows["mask"]
        ]
        stale_rows = [
            [[int(value) for value in zone_row] for zone_row in step_row]
            for step_row in rows["stale"]
        ]
        for action_index, action in enumerate(actions):
            labels = rollout_labels(
                scenario, zone_ids, checkpoint, action.command.to_mapping()
            )
            command_vector = [
                _f32(float(value), "candidate command")
                for value in _command_vector(bundle, action.command.to_mapping(), "candidate action")
            ]
            sample = {
                "schema_version": BDM_V1_CORPUS_SCHEMA_VERSION,
                "family_id": meta.family_id,
                "group_id": meta.group_id,
                "group_index": meta.group_index,
                "partition": meta.partition,
                "template_id": meta.template_id,
                "stratum": meta.stratum,
                "sensor_variant": meta.sensor_variant,
                "scenario_sha256": meta.scenario_sha256,
                "decision_step": decision,
                "window_completed_steps": list(range(decision - WINDOW_STEPS + 1, decision + 1)),
                "candidate_action_id": action.action_id,
                "features": {
                    "zone_temperature_k": env_by_channel["temperature_k"],
                    "zone_pressure_pa": env_by_channel["pressure_pa"],
                    "zone_co2_ppm": env_by_channel["co2_ppm"],
                    "zone_o2_mole_fraction": env_by_channel["o2_mole_fraction"],
                    "zone_relative_humidity": env_by_channel["relative_humidity"],
                    "zone_branch_airflow_m3_s": env_by_channel["branch_airflow_m3_s"],
                    "observed_value_mask": mask_rows,
                    "steps_since_last_valid_observation": stale_rows,
                    "requested_command_vector": [
                        [_f32(float(value), "requested command") for value in step_row]
                        for step_row in rows["requested"]
                    ],
                    "achieved_actuator_state": [
                        [_f32(float(value), "achieved state") for value in step_row]
                        for step_row in rows["achieved"]
                    ],
                    "battery_state_of_charge": [
                        _f32(float(value), "battery gauge") for value in rows["gauges"][:, 0]
                    ],
                    "oxygen_store_fraction": [
                        _f32(float(value), "oxygen gauge") for value in rows["gauges"][:, 1]
                    ],
                    "sorbent_fraction": [
                        _f32(float(value), "sorbent gauge") for value in rows["gauges"][:, 2]
                    ],
                    "operating_mode_one_hot": [
                        [_f32(float(value), "mode one-hot") for value in step_row]
                        for step_row in rows["modes"]
                    ],
                    "prior_proposal_dispositions": [
                        [_f32(float(value), "disposition one-hot") for value in step_row]
                        for step_row in rows["dispositions"]
                    ],
                    "topology_configuration_descriptor": {
                        "zone_order": list(zone_ids),
                        "observable_topology_sha256": window[-1]["observable_topology_sha256"],
                    },
                    "candidate_action_command_vector": command_vector,
                    "candidate_action_index": action_index,
                    "declared_known_future_schedule": [],
                },
                "missingness_metadata": {
                    "achieved_actuator_state_mask": [
                        [bool(flag) for flag in step_row] for step_row in rows["achieved_mask"]
                    ],
                    "achieved_actuator_state_staleness": [
                        [int(value) for value in step_row] for step_row in rows["achieved_stale"]
                    ],
                    "resource_gauges_mask": [
                        [bool(flag) for flag in step_row] for step_row in rows["gauge_mask"]
                    ],
                    "resource_gauges_staleness": [
                        [int(value) for value in step_row] for step_row in rows["gauge_stale"]
                    ],
                },
                "labels": labels,
                "hold_reference": hold_labels,
                "action_minus_hold": {
                    name: _f32(
                        float(labels["decision_targets"][name])
                        - float(hold_labels["decision_targets"][name]),
                        f"action-minus-hold {name}",
                    )
                    for name in LABEL_FIELD_NAMES
                },
            }
            digest_payload = dict(sample)
            sample["sample_sha256"] = hashlib.sha256(
                canonical_json_bytes(digest_payload)
            ).hexdigest()
            samples.append(sample)
    return tuple(samples)


def validate_sample_against_contract(
    contract: Mapping[str, Any], sample: Mapping[str, Any]
) -> None:
    """Fail closed unless the sample honours the contract input/label schema."""

    validate_model_input_fields(tuple(sample["features"].keys()), contract)
    if set(sample["features"].keys()) != set(FEATURE_FIELD_NAMES):
        raise BdmV1CorpusError("sample feature keys drifted from the corpus convention")
    for name in LABEL_FIELD_NAMES:
        if name not in sample["labels"]["decision_targets"]:
            raise BdmV1CorpusError(f"sample is missing label {name}")
        if name not in sample["hold_reference"]["decision_targets"]:
            raise BdmV1CorpusError(f"sample hold reference is missing label {name}")
    window = sample["window_completed_steps"]
    if len(window) != WINDOW_STEPS or window != list(range(window[0], window[0] + WINDOW_STEPS)):
        raise BdmV1CorpusError("sample window is not contiguous")
    if window[-1] != sample["decision_step"]:
        raise BdmV1CorpusError("sample window extends past its decision step")
    for step_mask, step_stale in zip(
        sample["features"]["observed_value_mask"],
        sample["features"]["steps_since_last_valid_observation"],
        strict=True,
    ):
        for zone_mask, zone_stale in zip(step_mask, step_stale, strict=True):
            for observed, staleness in zip(zone_mask, zone_stale, strict=True):
                if observed and staleness != 0:
                    raise BdmV1CorpusError("observed channel must carry zero staleness")
                if not observed and staleness <= 0:
                    raise BdmV1CorpusError("unobserved channel must carry positive staleness")


def corpus_manifest_digest(samples: Sequence[Mapping[str, Any]]) -> str:
    payload = [sample["sample_sha256"] for sample in samples]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def corpus_feature_manifest(bundle: ForecastContracts) -> dict[str, Any]:
    """Hashable declaration of the corpus feature/label convention."""

    if type(bundle) is not ForecastContracts:
        raise BdmV1CorpusError("feature manifest requires frozen contracts")
    return {
        "schema_version": BDM_V1_CORPUS_SCHEMA_VERSION,
        "window_steps": WINDOW_STEPS,
        "rollout_horizon_steps": ROLLOUT_HORIZON_STEPS,
        "label_horizon_keys": list(LABEL_HORIZON_KEYS),
        "feature_fields": list(FEATURE_FIELD_NAMES),
        "label_fields": list(LABEL_FIELD_NAMES),
        "env_channels": list(ENV_CHANNELS),
        "disposition_order": list(DISPOSITION_ORDER),
        "missingness": {
            "convention": "carry_forward_with_explicit_mask_and_staleness",
            "never_observed": "marked_zero_imputation_with_staleness_equal_to_completed_step",
            "audit_metadata_is_not_model_input": True,
        },
        "catalogue_binding": {
            "ordering": [action.action_id for action in bundle.actions],
            "catalogue_sha256": bundle.action_catalogue_sha256,
        },
        "prohibited_inputs": [
            "hidden_physical_truth",
            "fault_labels_or_future_fault_schedules",
            "simulator_seeds",
            "internal_noise_or_bias_state",
            "future_measurements",
            "counterfactual_outcomes_as_features",
            "evaluator_only_reserve_or_audit_state",
            "future_hmc_arbitration_results",
            "undeclared_future_crew_or_environmental_loads",
        ],
    }


def load_partition_samples(corpus: Path, partition: str) -> tuple[dict[str, Any], ...]:
    """Load one partition of a built corpus in deterministic sample order.

    Single home for corpus shard loading: every study script imports this
    instead of carrying its own copy. Samples are sorted by family, decision
    step, and candidate index so fits and evaluations are reproducible.
    """

    corpus_path = Path(corpus)
    shard_dir = corpus_path / "shards"
    if not shard_dir.is_dir():
        raise BdmV1CorpusError(f"corpus shards missing under {corpus_path}")
    samples: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line:
                continue
            sample = json.loads(line)
            if sample["partition"] == partition:
                samples.append(sample)
    samples.sort(
        key=lambda item: (
            item["family_id"],
            item["decision_step"],
            item["features"]["candidate_action_index"],
        )
    )
    if not samples:
        raise BdmV1CorpusError(f"no {partition} samples in corpus {corpus_path}")
    return tuple(samples)


__all__ = [
    "BDM_V1_CORPUS_SCHEMA_VERSION",
    "BdmV1CorpusError",
    "CORPUS_PARTITIONS",
    "CorpusFamilyMeta",
    "FEATURE_FIELD_NAMES",
    "LABEL_FIELD_NAMES",
    "ROLLOUT_HORIZON_STEPS",
    "WINDOW_STEPS",
    "collect_family_samples",
    "corpus_feature_manifest",
    "corpus_manifest_digest",
    "load_partition_samples",
    "rollout_labels",
    "validate_sample_against_contract",
]
