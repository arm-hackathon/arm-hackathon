"""Issue #55 three-way controller race: rules vs model-advised vs oracle.

This module is deliberately outside the HMC authority core.  The Habitat
Management Controller remains the sole actuator authority in every arm; the
model and oracle arms may only issue standard advisory proposals that the HMC
is free to reject.  The oracle arm is a measuring instrument that evaluates a
finite full-remaining-episode constant-command schedule and must never be
integrated into any demo or runtime advisor surface.

Protocol: contracts/habitat_v2_forecast_issue_55_preregistration_v2.json
(declared before any run).  Every digest-bearing structure in this module is
canonical-JSON bound and free of wall-clock values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import math
from typing import Any

import numpy as np

from .control_trace import parse_control_trace, replay_control_trace
from .forecast.projection import (
    project_history_window,
    project_proposed_action,
)
from .forecast_issue52 import _command_vector, extend_scenario_for_issue52
from .hmc import HabitatManagementComputer
from .hmc_contract import canonical_json_bytes
from .physics import (
    InfeasibleActionError,
    ScenarioValidationError,
    advance_one_step_with_command,
    initial_state,
    operating_mode_for_application_step,
    validate_external_command,
)
from .scenario import Scenario
from .telemetry import derive_observable_topology


RACE_SCHEMA_VERSION = "aeolus_habitat_v2_race_issue_55_v2"
PREREGISTRATION_ID = "habitat_v2_forecast_issue_55_preregistration_v2"
ADVISORY_RANKING_METRIC_ID = "issue55-advisory-point-ranking-v1"
ORACLE_SELECTION_METRIC_ID = "issue55-oracle-full-horizon-v2"
MODEL_SOURCE_TYPE = "issue55-model-advisory-v1"
ORACLE_SOURCE_TYPE = "issue55-oracle-instrument-v2"
HMC_IMPLEMENTATION_GIT_SHA = "3bc5da3d716212cac6524b088a963b6abf47a0ef"

ARMS = ("rules_only", "model_advised", "oracle_instrument")
ADVISORY_ARMS = ("model_advised", "oracle_instrument")
CORPUS_ID = "issue55_race_v2"
FAMILY_COUNT = 32
EPISODE_STEPS = 96
DECISION_START_STEP = 16
DECISION_CADENCE_STEPS = 4
HISTORY_WINDOW_STEPS = 16
MODEL_HORIZON_STEPS = 8
HORIZON_STEPS = MODEL_HORIZON_STEPS
TARGET_COUNT = 51
ZONE_FIELD_COUNT = 6
N_ZONES = 8
RESOURCE_COLUMNS = (48, 49, 50)
ORACLE_RESOURCE_WEIGHT = 0.1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 550055
GAP_DENOMINATOR_FLOOR = 1e-12

OPERATING_CONDITIONS: tuple[tuple[str, str, float, float, float, float, float, float, float], ...] = (
    ("nominal_occupied", "occupied", 1.00, 5500.0, 0.0, 0.0, 0.30, 0.45, 72000.0),
    ("high_load_occupied", "occupied", 1.30, 4500.0, 1.5, 160.0, 0.27, 0.55, 72000.0),
    ("eva_transition", "eva_transition", 0.75, 6000.0, -0.5, -80.0, 0.25, 0.40, 68000.0),
    ("contingency", "contingency", 1.20, 3500.0, 2.0, 260.0, 0.23, 0.60, 70000.0),
)
PLANT_CONDITIONS: tuple[tuple[str, str | None, float, float], ...] = (
    ("nominal_plant", None, 1.00, 1.00),
    ("fan_degradation", "fan_speed_degradation", 0.95, 0.95),
    ("laboratory_resistance", "branch_resistance_increase", 1.90, 1.00),
    ("equipment_cooling_loss", "cooling_delivery_degradation", 0.85, 0.90),
)
SENSOR_VARIANTS: tuple[tuple[str, int], ...] = (
    ("sensor_seed_a", 0),
    ("sensor_seed_b", 17),
)

ZONE_FIELD_BOUNDS: tuple[tuple[str, float, float, float, float], ...] = (
    ("temperature_k", 10.0, 295.15, 250.0, 330.0),
    ("pressure_pa", 1000.0, 101325.0, 50000.0, 150000.0),
    ("co2_ppm", 800.0, 800.0, 300.0, 5000.0),
    ("o2_mole_fraction", 0.005, 0.2095, 0.15, 0.30),
    ("relative_humidity", 0.25, 0.45, 0.0, 1.0),
    ("branch_airflow_m3_s", 0.1, 0.05, 0.0, 1.0),
)
RESOURCE_FIELD_BOUNDS: tuple[tuple[str, float, float, float, float], ...] = (
    ("battery_state_of_charge", 0.25, 0.75, 0.0, 1.0),
    ("oxygen_store_fraction", 0.25, 0.75, 0.0, 1.0),
    ("sorbent_remaining_fraction", 0.25, 0.75, 0.0, 1.0),
)
COMFORT_COLUMNS: tuple[int, ...] = tuple(
    zone * ZONE_FIELD_COUNT + offset
    for zone in range(N_ZONES)
    for offset in (0, 2, 4)
)


class Issue55RaceError(ValueError):
    """Raised when a race episode or its evidence violates the frozen protocol."""


def target_bounds() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the preregistered (scale, nominal, lower, upper) float64[51] tables."""

    scales: list[float] = []
    nominals: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for _ in range(N_ZONES):
        for _, scale, nominal, lower, upper in ZONE_FIELD_BOUNDS:
            scales.append(scale)
            nominals.append(nominal)
            lowers.append(lower)
            uppers.append(upper)
    for _, scale, nominal, lower, upper in RESOURCE_FIELD_BOUNDS:
        scales.append(scale)
        nominals.append(nominal)
        lowers.append(lower)
        uppers.append(upper)
    result = (
        np.array(scales, dtype=np.float64),
        np.array(nominals, dtype=np.float64),
        np.array(lowers, dtype=np.float64),
        np.array(uppers, dtype=np.float64),
    )
    if any(array.shape != (TARGET_COUNT,) for array in result):
        raise Issue55RaceError("preregistered bound tables have drifted")
    return result


def deterministic_family_ids(count: int, corpus_id: str = CORPUS_ID) -> tuple[str, ...]:
    """Derive preregistered family identities from the corpus id and index."""

    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 < count <= FAMILY_COUNT
    ):
        raise Issue55RaceError("family count must be between one and 32")
    if type(corpus_id) is not str or not corpus_id:
        raise Issue55RaceError("corpus id is invalid")
    return tuple(
        "issue55f"
        + hashlib.sha256(
            f"issue55-family-v2|{corpus_id}|{index:04d}".encode("utf-8")
        ).hexdigest()[:16]
        for index in range(count)
    )


def _family_variant_indices(family_index: int) -> tuple[int, int, int]:
    if (
        isinstance(family_index, bool)
        or not isinstance(family_index, int)
        or not 0 <= family_index < FAMILY_COUNT
    ):
        raise Issue55RaceError("family index is outside the preregistered roster")
    return family_index // 8, (family_index // 2) % 4, family_index % 2


def family_condition_descriptor(family_index: int) -> dict[str, Any]:
    """Return the immutable roster row for one preregistered family index."""

    operation_index, plant_index, sensor_index = _family_variant_indices(family_index)
    operation = OPERATING_CONDITIONS[operation_index]
    plant = PLANT_CONDITIONS[plant_index]
    sensor = SENSOR_VARIANTS[sensor_index]
    return {
        "family_index": family_index,
        "operating_condition": operation[0],
        "operating_mode": operation[1],
        "plant_condition": plant[0],
        "fault_type": plant[1],
        "sensor_condition": sensor[0],
        "sensor_seed_offset": sensor[1],
    }


def build_family_scenario(base_scenario: Scenario, family_index: int) -> Scenario:
    """Create one preregistered operating, plant and sensor-condition family."""

    if type(base_scenario) is not Scenario:
        raise Issue55RaceError("family construction requires the base Scenario")
    operation_index, plant_index, sensor_index = _family_variant_indices(family_index)
    (
        operation_name,
        operating_mode,
        load_scale,
        generation_w,
        temperature_delta,
        co2_delta,
        oxygen_fraction,
        relative_humidity,
        pressure_pa,
    ) = OPERATING_CONDITIONS[operation_index]
    plant_name, fault_type, fault_multiplier, resource_factor = PLANT_CONDITIONS[
        plant_index
    ]
    sensor_name, sensor_offset = SENSOR_VARIANTS[sensor_index]
    data: dict[str, Any] = {
        key: _copy_json(value) for key, value in base_scenario.data.items()
    }
    sensor_model = data.get("sensor_model")
    if not isinstance(sensor_model, Mapping) or "random_seed" not in sensor_model:
        raise Issue55RaceError("base scenario has no sensor seed to vary")
    sensor_model["random_seed"] = (
        int(sensor_model["random_seed"]) + family_index * 1000 + sensor_offset
    )
    data["name"] = (
        f"{data.get('name', 'dev')}-issue55-f{family_index:04d}-"
        f"{operation_name}-{plant_name}-{sensor_name}"
    )
    for zone in data["zones"]:
        initial = zone["initial"]
        initial["temperature_k"] = float(initial["temperature_k"]) + temperature_delta
        initial["co2_ppm"] = max(0.0, float(initial["co2_ppm"]) + co2_delta)
        initial["o2_mole_fraction"] = oxygen_fraction
        initial["relative_humidity"] = relative_humidity
        initial["pressure_pa"] = pressure_pa
    utility = data["initial_utility"]
    utility["battery_energy_wh"] = float(utility["battery_energy_wh"]) * resource_factor
    utility["oxygen_store_mol"] = float(utility["oxygen_store_mol"]) * resource_factor
    utility["co2_sorbent_remaining_mol"] = (
        float(utility["co2_sorbent_remaining_mol"]) * resource_factor
    )
    for segment in data["timeline"]:
        segment["operating_mode"] = operating_mode
        segment["generation_w"] = generation_w
        for load in segment["loads"].values():
            for field_name in (
                "co2_generation_mol_s",
                "o2_consumption_mol_s",
                "sensible_heat_w",
                "water_vapor_generation_mol_s",
            ):
                load[field_name] = float(load[field_name]) * load_scale
    fault_profiles: list[dict[str, Any]] = []
    if fault_type == "fan_speed_degradation":
        fault_profiles.append(
            {
                "id": f"issue55-f{family_index:04d}-fan-degradation",
                "type": fault_type,
                "start_step": 32,
                "end_step": 80,
                "start_multiplier": fault_multiplier,
                "end_multiplier": fault_multiplier,
            }
        )
    elif fault_type == "branch_resistance_increase":
        fault_profiles.append(
            {
                "id": f"issue55-f{family_index:04d}-laboratory-resistance",
                "type": fault_type,
                "zone_id": "laboratory",
                "start_step": 32,
                "end_step": 80,
                "start_multiplier": fault_multiplier,
                "end_multiplier": fault_multiplier,
            }
        )
    elif fault_type == "cooling_delivery_degradation":
        fault_profiles.append(
            {
                "id": f"issue55-f{family_index:04d}-equipment-cooling",
                "type": fault_type,
                "zone_id": "equipment_power_bay",
                "start_step": 32,
                "end_step": 80,
                "start_multiplier": fault_multiplier,
                "end_multiplier": fault_multiplier,
            }
        )
    try:
        variant = Scenario.from_mapping(data)
    except ScenarioValidationError as error:
        raise Issue55RaceError("family scenario failed V5 validation") from error
    try:
        extended = extend_scenario_for_issue52(variant, minimum_steps=EPISODE_STEPS)
        extended_data = {
            key: _copy_json(value) for key, value in extended.data.items()
        }
        extended_data["fault_profiles"] = fault_profiles
        return Scenario.from_mapping(extended_data)
    except (ValueError, ScenarioValidationError) as error:
        raise Issue55RaceError("family scenario cannot be extended to 96 steps") from error


def episode_nonce(family_id: str) -> bytes:
    """Family-bound, arm-independent episode nonce for paired comparison."""

    if type(family_id) is not str or not family_id:
        raise Issue55RaceError("family id is invalid")
    return hashlib.sha256(
        b"issue55-episode-nonce-v1|" + family_id.encode("utf-8")
    ).digest()


def decision_steps(episode_steps: int = EPISODE_STEPS) -> tuple[int, ...]:
    """Preregistered model-decision steps with complete 8-step predictions."""

    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, int)
        or episode_steps <= DECISION_START_STEP + MODEL_HORIZON_STEPS
    ):
        raise Issue55RaceError("episode is too short for the preregistered cadence")
    return tuple(
        step
        for step in range(DECISION_START_STEP, episode_steps, DECISION_CADENCE_STEPS)
        if step + MODEL_HORIZON_STEPS <= episode_steps - 1
    )


def scenario_zone_order(scenario: Scenario) -> tuple[str, ...]:
    """Deterministic zone ordering bound to the observable topology."""

    if type(scenario) is not Scenario:
        raise Issue55RaceError("zone order requires Scenario")
    zone_ids = tuple(derive_observable_topology(scenario).zone_ids)
    if len(zone_ids) != N_ZONES:
        raise Issue55RaceError("race requires exactly eight observable zones")
    return zone_ids


def project_true_targets(
    scenario: Scenario,
    zone_ids: Sequence[str],
    state: Any,
) -> np.ndarray:
    """Project one true plant state onto the preregistered float32[51] layout."""

    if type(scenario) is not Scenario:
        raise Issue55RaceError("target projection requires Scenario")
    if len(tuple(zone_ids)) != N_ZONES:
        raise Issue55RaceError("target projection requires the eight zone ids")
    zones_config = {str(zone["id"]): zone for zone in scenario.data["zones"]}
    equipment = scenario.data["equipment"]
    initial_oxygen = float(scenario.data["initial_utility"]["oxygen_store_mol"])
    values: list[float] = []
    for zone_id in zone_ids:
        if zone_id not in state.zones:
            raise Issue55RaceError("plant state is missing a projected zone")
        telemetry = state.zones[zone_id].telemetry(
            volume_m3=float(zones_config[zone_id]["volume_m3"])
        )
        for field_name, _, _, _, _ in ZONE_FIELD_BOUNDS[:-1]:
            values.append(
                _finite_target(
                    float(telemetry[field_name]),
                    f"{zone_id}/{field_name}",
                )
            )
        values.append(
            _finite_target(
                float(state.utility.actual_airflow_m3_s[zone_id]),
                f"{zone_id}/branch_airflow_m3_s",
            )
        )
    values.append(
        _finite_target(
            float(state.utility.battery_energy_wh)
            / float(equipment["battery_capacity_wh"]),
            "battery_state_of_charge",
        )
    )
    values.append(
        _finite_target(
            float(state.utility.oxygen_store_mol) / max(initial_oxygen, 1e-12),
            "oxygen_store_fraction",
        )
    )
    values.append(
        _finite_target(
            float(state.utility.co2_sorbent_remaining_mol)
            / float(equipment["scrubber_capacity_mol"]),
            "sorbent_remaining_fraction",
        )
    )
    row = np.asarray(values, dtype=np.float32)
    if row.shape != (TARGET_COUNT,) or not np.isfinite(row).all():
        raise Issue55RaceError("true target projection is malformed")
    row.setflags(write=False)
    return row


def _finite_target(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise Issue55RaceError(f"true target {label} is non-finite")
    return value


def _crossings(values: np.ndarray) -> np.ndarray:
    scales, _, lowers, uppers = target_bounds()
    lower_crossing = np.maximum(0.0, lowers[None, :] - values) / scales[None, :]
    upper_crossing = np.maximum(0.0, values - uppers[None, :]) / scales[None, :]
    return lower_crossing + upper_crossing


def compute_race_metrics(
    scenario: Scenario,
    zone_ids: Sequence[str],
    initial_row: np.ndarray,
    states: Sequence[Any],
) -> dict[str, float | int]:
    """Preregistered true-plant metrics over one episode."""

    _, nominals, _, _ = target_bounds()
    if type(scenario) is not Scenario or not states:
        raise Issue55RaceError("metrics require a scenario and observed states")
    expected_step = 1
    rows: list[np.ndarray] = []
    occupied_rows: list[np.ndarray] = []
    for state in states:
        if state.step != expected_step:
            raise Issue55RaceError("metrics require contiguous states from step 1")
        expected_step += 1
        rows.append(project_true_targets(scenario, zone_ids, state).astype(np.float64))
        if (
            operating_mode_for_application_step(scenario, state.step - 1)
            == "occupied"
        ):
            occupied_rows.append(rows[-1])
    values = np.stack(rows)
    crossings = _crossings(values)
    safety_exposure = float(np.sum(crossings))
    violation_steps = int(np.count_nonzero(np.any(crossings > 0.0, axis=1)))
    if occupied_rows:
        comfort_rows = np.stack(occupied_rows)[:, list(COMFORT_COLUMNS)]
        comfort_deviation = float(
            np.mean(
                np.abs(comfort_rows - nominals[list(COMFORT_COLUMNS)][None, :])
            )
        )
    else:
        comfort_deviation = 0.0
    final_row = rows[-1]
    resource_components = [
        float(max(0.0, float(initial_row[column]) - float(final_row[column])))
        for column in RESOURCE_COLUMNS
    ]
    metrics: dict[str, float | int] = {
        "safety_exposure": safety_exposure,
        "safety_violation_steps": violation_steps,
        "comfort_deviation": comfort_deviation,
        "resource_battery_fraction": resource_components[0],
        "resource_oxygen_fraction": resource_components[1],
        "resource_sorbent_fraction": resource_components[2],
        "resource_composite": float(sum(resource_components)),
    }
    for key, value in metrics.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise Issue55RaceError(f"metric {key} is non-finite or negative")
    return metrics


@dataclass(frozen=True, slots=True)
class AdvisoryScore:
    """Point-prediction score for one catalogue action (advisory arm)."""

    action_id: str
    score: float
    hard_ineligible: bool
    safety: float
    tracking: float
    intervention: float
    reason: str | None


def score_point_prediction(
    action_id: str,
    prediction: np.ndarray,
    current_command_vec: np.ndarray,
    candidate_command_vec: np.ndarray,
) -> AdvisoryScore:
    """`issue55-advisory-point-ranking-v1` on one action's [8,51] prediction."""

    if type(action_id) is not str or not action_id:
        raise Issue55RaceError("advisory action id is invalid")
    scales, nominals, lowers, uppers = target_bounds()
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != (HORIZON_STEPS, TARGET_COUNT) or not np.isfinite(pred).all():
        raise Issue55RaceError("advisory prediction must be finite float64[8,51]")
    current = np.asarray(current_command_vec, dtype=np.float64)
    candidate = np.asarray(candidate_command_vec, dtype=np.float64)
    if current.shape != candidate.shape or current.ndim != 1 or current.size == 0:
        raise Issue55RaceError("advisory command vectors are malformed")
    if not np.isfinite(current).all() or not np.isfinite(candidate).all():
        raise Issue55RaceError("advisory command vectors must be finite")
    tracking = float(np.mean(np.abs(pred - nominals[None, :]) / scales[None, :]))
    safety = float(np.mean(_crossings(pred)))
    intervention = float(np.mean(np.abs(candidate - current)))
    hard = bool(np.any(pred < lowers[None, :]) or np.any(pred > uppers[None, :]))
    score = tracking + 0.5 * safety + 0.05 * intervention
    return AdvisoryScore(
        action_id=action_id,
        score=score if not hard else math.inf,
        hard_ineligible=hard,
        safety=safety,
        tracking=tracking,
        intervention=intervention,
        reason="predicted_hard_bound_crossing" if hard else None,
    )


def rank_actions_advisory(scores: Sequence[AdvisoryScore]) -> AdvisoryScore | None:
    """Select the advisory action or None when every candidate is hard-ineligible."""

    if not scores:
        raise Issue55RaceError("advisory ranking requires at least one score")
    ids = [item.action_id for item in scores]
    if len(set(ids)) != len(ids):
        raise Issue55RaceError("advisory ranking received duplicate action ids")
    eligible = [item for item in scores if not item.hard_ineligible]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item.score,
            item.safety,
            item.intervention,
            0 if "hold" in item.action_id else 1,
            item.action_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class OracleScore:
    """Full-remaining-episode true-plant score for one catalogue action."""

    action_id: str
    score: float
    safety: float
    comfort: float
    resource: float
    feasible_steps: int
    horizon_steps: int
    excluded: bool


def oracle_full_horizon_scores(
    scenario: Scenario,
    zone_ids: Sequence[str],
    state: Any,
    actions: Sequence[Any],
) -> tuple[OracleScore, ...]:
    """`issue55-oracle-full-horizon-v2`: repeat one action to episode end."""

    if type(scenario) is not Scenario:
        raise Issue55RaceError("oracle full-horizon score requires Scenario")
    if not actions:
        raise Issue55RaceError("oracle full-horizon score requires catalogue actions")
    base_row = project_true_targets(scenario, zone_ids, state).astype(np.float64)
    _, nominals, _, _ = target_bounds()
    horizon_steps = int(scenario.data["steps"]) - int(state.step)
    if horizon_steps <= 0:
        raise Issue55RaceError("oracle full-horizon score requires remaining steps")
    results: list[OracleScore] = []
    for action in actions:
        command_mapping = action.command.to_mapping()
        cursor = state
        rows: list[np.ndarray] = []
        feasible_steps = 0
        try:
            for _ in range(horizon_steps):
                stepped = advance_one_step_with_command(
                    scenario, cursor, command_mapping
                )
                cursor = stepped.state
                rows.append(
                    project_true_targets(scenario, zone_ids, cursor).astype(np.float64)
                )
                feasible_steps += 1
        except (InfeasibleActionError, ScenarioValidationError):
            rows = []
        if feasible_steps < horizon_steps or not rows:
            results.append(
                OracleScore(
                    action_id=action.action_id,
                    score=math.inf,
                    safety=math.inf,
                    comfort=math.inf,
                    resource=math.inf,
                    feasible_steps=feasible_steps,
                    horizon_steps=horizon_steps,
                    excluded=True,
                )
            )
            continue
        values = np.stack(rows)
        safety = float(np.sum(_crossings(values)))
        occupied_rows = [
            row
            for offset, row in enumerate(rows)
            if operating_mode_for_application_step(scenario, state.step + offset)
            == "occupied"
        ]
        comfort = float(
            np.mean(
                np.abs(
                    np.stack(occupied_rows)[:, list(COMFORT_COLUMNS)]
                    - nominals[list(COMFORT_COLUMNS)][None, :]
                )
            )
            if occupied_rows
            else 0.0
        )
        final_row = rows[-1]
        resource = float(
            sum(
                max(0.0, float(base_row[column]) - float(final_row[column]))
                for column in RESOURCE_COLUMNS
            )
        )
        score = safety + comfort + ORACLE_RESOURCE_WEIGHT * resource
        results.append(
            OracleScore(
                action_id=action.action_id,
                score=score,
                safety=safety,
                comfort=comfort,
                resource=resource,
                feasible_steps=feasible_steps,
                horizon_steps=horizon_steps,
                excluded=False,
            )
        )
    ids = [item.action_id for item in results]
    if len(set(ids)) != len(ids):
        raise Issue55RaceError("oracle full-horizon score produced duplicate action ids")
    return tuple(results)


def select_oracle_action(scores: Sequence[OracleScore]) -> OracleScore | None:
    """Argmin of the preregistered full-horizon score."""

    if not scores:
        raise Issue55RaceError("oracle selection requires at least one score")
    eligible = [item for item in scores if not item.excluded]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (item.score, item.safety, item.comfort, item.action_id),
    )


def build_advisory_proposal(
    hmc: HabitatManagementComputer,
    snapshot_sha256: str,
    step: int,
    command_mapping: Mapping[str, Any],
    action_id: str,
    source_type: str,
) -> dict[str, Any]:
    """Standard advisory control proposal; the HMC remains free to reject it."""

    if source_type not in (MODEL_SOURCE_TYPE, ORACLE_SOURCE_TYPE):
        raise Issue55RaceError("advisory proposal source type is not preregistered")
    body = {
        "schema_version": "aeolus_habitat_v2_control_proposal_v1",
        "control_run_id": hmc.control_run_id,
        "authority_epoch": hmc.authority_epoch,
        "source_id": action_id,
        "source_type": source_type,
        "completed_observation_step": step,
        "observation_snapshot_sha256": snapshot_sha256,
        "requested_application_step": step,
        "observable_topology_sha256": hmc.observable_topology_sha256,
        "proposed_command": _copy_json(dict(command_mapping)),
        "confidence": None,
    }
    return {**body, "proposal_sha256": _sha256_bytes(canonical_json_bytes(body))}


def _copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_EPISODE_DIGEST_EXCLUDED = "episode_sha256"


def _episode_digest(record: "RaceEpisodeRecord") -> str:
    payload = {
        item.name: getattr(record, item.name)
        for item in fields(record)
        if item.name != _EPISODE_DIGEST_EXCLUDED
    }
    return _sha256_bytes(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class RaceEpisodeRecord:
    """Digest-bound, wall-clock-free record of one raced episode."""

    schema_version: str
    arm: str
    family_id: str
    family_index: int
    scenario_sha256: str
    episode_steps: int
    decision_steps: tuple[int, ...]
    decision_actions: tuple[str | None, ...]
    proposal_count: int
    abstention_count: int
    admitted_proposal_count: int
    hmc_rejection_count: int
    safety_exposure: float
    safety_violation_steps: int
    comfort_deviation: float
    resource_battery_fraction: float
    resource_oxygen_fraction: float
    resource_sorbent_fraction: float
    resource_composite: float
    control_run_id: str
    trace_sha256: str
    replay_committed_steps: int
    replay_final_state_sha256: str
    episode_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != RACE_SCHEMA_VERSION:
            raise Issue55RaceError("episode record schema version is invalid")
        if self.arm not in ARMS:
            raise Issue55RaceError("episode record arm is invalid")
        if type(self.family_id) is not str or not self.family_id:
            raise Issue55RaceError("episode record family id is invalid")
        if (
            isinstance(self.family_index, bool)
            or not isinstance(self.family_index, int)
            or self.family_index < 0
        ):
            raise Issue55RaceError("episode record family index is invalid")
        if (
            type(self.scenario_sha256) is not str
            or len(self.scenario_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.scenario_sha256)
        ):
            raise Issue55RaceError("episode record scenario identity is invalid")
        if self.decision_steps != decision_steps(self.episode_steps):
            raise Issue55RaceError("episode record decision steps drifted")
        if type(self.decision_actions) is not tuple or len(
            self.decision_actions
        ) != len(self.decision_steps):
            raise Issue55RaceError("episode record decision actions drifted")
        if any(
            entry is not None and (type(entry) is not str or not entry)
            for entry in self.decision_actions
        ):
            raise Issue55RaceError("episode record decision actions are malformed")
        if sum(entry is not None for entry in self.decision_actions) != self.proposal_count:
            raise Issue55RaceError("decision actions do not match proposal count")
        if self.arm == "rules_only":
            if any(
                getattr(self, name) != 0
                for name in (
                    "proposal_count",
                    "abstention_count",
                    "admitted_proposal_count",
                    "hmc_rejection_count",
                )
            ):
                raise Issue55RaceError("rules-only episode must never propose")
        else:
            if self.proposal_count + self.abstention_count != len(self.decision_steps):
                raise Issue55RaceError(
                    "every preregistered decision must propose or abstain"
                )
            if sum(entry is None for entry in self.decision_actions) != self.abstention_count:
                raise Issue55RaceError("decision abstentions do not match count")
            if self.proposal_count != self.admitted_proposal_count:
                raise Issue55RaceError("every issued proposal must be admitted cleanly")
            if not 0 <= self.hmc_rejection_count <= self.admitted_proposal_count:
                raise Issue55RaceError("HMC rejection count is inconsistent")
        for name in (
            "safety_exposure",
            "comfort_deviation",
            "resource_battery_fraction",
            "resource_oxygen_fraction",
            "resource_sorbent_fraction",
            "resource_composite",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise Issue55RaceError(f"episode metric {name} is non-finite or negative")
        if (
            isinstance(self.safety_violation_steps, bool)
            or not isinstance(self.safety_violation_steps, int)
            or self.safety_violation_steps < 0
            or self.safety_violation_steps > self.episode_steps
        ):
            raise Issue55RaceError("violation step count is inconsistent")
        if self.replay_committed_steps != self.episode_steps:
            raise Issue55RaceError("episode record replay did not commit every step")
        if self.episode_sha256 != _episode_digest(self):
            raise Issue55RaceError("episode record digest is inconsistent")

    def to_mapping(self) -> dict[str, Any]:
        return {
            item.name: _copy_json(getattr(self, item.name))
            for item in fields(self)
        }


def run_race_episode(
    bundle: Any,
    scenario: Scenario,
    arm: str,
    family_id: str,
    family_index: int,
    teacher: Any | None,
) -> RaceEpisodeRecord:
    """Race one family under one arm with strict HMC authority and replay checks."""

    if arm not in ARMS:
        raise Issue55RaceError("race arm is invalid")
    if type(scenario) is not Scenario:
        raise Issue55RaceError("race episode requires Scenario")
    if arm == "model_advised":
        if teacher is None or not hasattr(teacher, "predictor"):
            raise Issue55RaceError("model-advised episode requires the frozen teacher")
    elif teacher is not None:
        raise Issue55RaceError("only the model-advised arm may carry a teacher")
    contract = bundle.hmc_contract
    actions = tuple(bundle.actions)
    if len(actions) != 4 or len({action.action_id for action in actions}) != 4:
        raise Issue55RaceError("race requires exactly four unique catalogue actions")
    total_steps = int(scenario.data["steps"])
    if total_steps != EPISODE_STEPS:
        raise Issue55RaceError("race episodes must be exactly 96 steps")
    decisions = decision_steps(total_steps)
    zone_ids = scenario_zone_order(scenario)
    hmc = HabitatManagementComputer.reset(scenario, contract, episode_nonce(family_id))
    shadow = initial_state(scenario)
    initial_row = project_true_targets(scenario, zone_ids, shadow)
    states: dict[int, Any] = {0: shadow}
    snapshots: dict[int, tuple[Any, Any]] = {}
    last_command_mapping: dict[str, Any] | None = None

    proposal_count = 0
    abstention_count = 0
    admitted_proposal_count = 0
    hmc_rejection_count = 0
    decision_actions: list[str | None] = []
    for step in range(total_steps):
        observed = hmc.observe()
        if type(observed) is not tuple:
            raise Issue55RaceError(f"HMC terminated at step {step}")
        snapshot, verification = observed
        handle = hmc.verify_snapshot(snapshot, verification)
        if step > 0:
            snapshots[step] = (snapshot, verification)

        proposal = None
        if step in decisions and arm in ADVISORY_ARMS:
            if last_command_mapping is None:
                raise Issue55RaceError("decision reached without any applied command")
            if arm == "model_advised":
                proposal = _model_advisory_proposal(
                    bundle,
                    scenario,
                    hmc,
                    teacher,
                    snapshots,
                    snapshot,
                    step,
                    last_command_mapping,
                )
            else:
                proposal = _oracle_proposal(
                    scenario, zone_ids, hmc, actions, snapshot, step, shadow
                )
        proposal_receipt = hmc.propose(proposal, handle)
        receipt_mapping = proposal_receipt.to_mapping()
        proposed_command_sha: str | None = None
        if step in decisions and arm in ADVISORY_ARMS:
            if proposal is None:
                if receipt_mapping["validation_outcome"] != "NO_PROPOSAL":
                    raise Issue55RaceError(
                        f"abstention at step {step} was not recorded cleanly"
                    )
                abstention_count += 1
                decision_actions.append(None)
            else:
                if (
                    receipt_mapping["attempt_class"],
                    receipt_mapping["validation_outcome"],
                ) != ("CANONICAL_PROPOSAL", "VALID"):
                    raise Issue55RaceError(
                        f"decision proposal at step {step} was not admitted cleanly"
                    )
                admitted_proposal_count += 1
                proposal_count += 1
                decision_actions.append(str(proposal["source_id"]))
                proposed_command_sha = str(
                    validate_external_command(
                        scenario, proposal["proposed_command"]
                    ).sha256
                )
        elif receipt_mapping["validation_outcome"] != "NO_PROPOSAL":
            raise Issue55RaceError("proposal issued outside the preregistered decisions")
        arbitration = hmc.arbitrate()
        if not hasattr(arbitration, "final_command"):
            raise Issue55RaceError(f"HMC terminated while arbitrating step {step}")
        if (
            proposed_command_sha is not None
            and arbitration.final_command_sha256 != proposed_command_sha
        ):
            hmc_rejection_count += 1
        last_command_mapping = dict(arbitration.final_command)
        step_receipt = hmc.step()
        if not hasattr(step_receipt, "plant_receipt_digest"):
            raise Issue55RaceError(f"HMC terminated while stepping step {step}")
        shadow_result = advance_one_step_with_command(
            scenario, shadow, arbitration.final_command
        )
        if shadow_result.state.step != step + 1:
            raise Issue55RaceError("shadow state drifted from the HMC application step")
        shadow_digest = _sha256_bytes(canonical_json_bytes(shadow_result.receipt))
        if shadow_digest != step_receipt.plant_receipt_digest:
            raise Issue55RaceError("shadow plant receipt diverges from the HMC receipt")
        shadow = shadow_result.state
        states[shadow.step] = shadow

    trace = hmc.export_control_trace(HMC_IMPLEMENTATION_GIT_SHA)
    parsed_trace = parse_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=contract
    )
    replay = replay_control_trace(
        trace.canonical_bytes, scenario=scenario, contract=contract
    )
    if (
        parsed_trace.footer["terminal_status"] != "COMPLETED"
        or replay.committed_step_count != total_steps
        or replay.final_state_sha256 != parsed_trace.footer["final_state_sha256"]
    ):
        raise Issue55RaceError("raced episode failed strict trace replay")

    metrics = compute_race_metrics(
        scenario,
        zone_ids,
        initial_row,
        [states[step] for step in range(1, total_steps + 1)],
    )
    record = object.__new__(RaceEpisodeRecord)
    record_values = {
        "schema_version": RACE_SCHEMA_VERSION,
        "arm": arm,
        "family_id": family_id,
        "family_index": family_index,
        "scenario_sha256": scenario.scenario_sha256,
        "episode_steps": total_steps,
        "decision_steps": decisions,
        "decision_actions": tuple(decision_actions)
        if arm in ADVISORY_ARMS
        else tuple([None] * len(decisions)),
        "proposal_count": proposal_count,
        "abstention_count": abstention_count,
        "admitted_proposal_count": admitted_proposal_count,
        "hmc_rejection_count": hmc_rejection_count,
        "safety_exposure": float(metrics["safety_exposure"]),
        "safety_violation_steps": int(metrics["safety_violation_steps"]),
        "comfort_deviation": float(metrics["comfort_deviation"]),
        "resource_battery_fraction": float(metrics["resource_battery_fraction"]),
        "resource_oxygen_fraction": float(metrics["resource_oxygen_fraction"]),
        "resource_sorbent_fraction": float(metrics["resource_sorbent_fraction"]),
        "resource_composite": float(metrics["resource_composite"]),
        "control_run_id": hmc.control_run_id,
        "trace_sha256": _sha256_bytes(trace.canonical_bytes),
        "replay_committed_steps": int(replay.committed_step_count),
        "replay_final_state_sha256": str(replay.final_state_sha256),
    }
    for name, value in record_values.items():
        object.__setattr__(record, name, value)
    object.__setattr__(record, _EPISODE_DIGEST_EXCLUDED, _episode_digest(record))
    record.__post_init__()
    return record


def _model_advisory_proposal(
    bundle: Any,
    scenario: Scenario,
    hmc: HabitatManagementComputer,
    teacher: Any,
    snapshots: Mapping[int, tuple[Any, Any]],
    snapshot: Any,
    step: int,
    last_command_mapping: Mapping[str, Any],
) -> dict[str, Any] | None:
    pairs = [
        snapshots[earlier]
        for earlier in range(step - HISTORY_WINDOW_STEPS + 1, step + 1)
    ]
    if len(pairs) != HISTORY_WINDOW_STEPS:
        raise Issue55RaceError("advisory history window is incomplete")
    history = project_history_window(bundle, pairs, window_steps=HISTORY_WINDOW_STEPS)
    current_vec = _command_vector(scenario, last_command_mapping)
    actions = tuple(bundle.actions)
    scores: list[AdvisoryScore] = []
    for action in actions:
        proposed_action = project_proposed_action(bundle, action.command)
        prediction = teacher.predictor.predict(history, proposed_action)
        candidate_vec = _command_vector(scenario, action.command.to_mapping())
        scores.append(
            score_point_prediction(
                action.action_id, prediction, current_vec, candidate_vec
            )
        )
    selected = rank_actions_advisory(scores)
    if selected is None:
        return None
    command_mapping = next(
        action.command.to_mapping()
        for action in actions
        if action.action_id == selected.action_id
    )
    return build_advisory_proposal(
        hmc,
        snapshot.snapshot_sha256,
        step,
        command_mapping,
        selected.action_id,
        MODEL_SOURCE_TYPE,
    )


def _oracle_proposal(
    scenario: Scenario,
    zone_ids: tuple[str, ...],
    hmc: HabitatManagementComputer,
    actions: tuple[Any, ...],
    snapshot: Any,
    step: int,
    shadow: Any,
) -> dict[str, Any] | None:
    scores = oracle_full_horizon_scores(scenario, zone_ids, shadow, actions)
    selected = select_oracle_action(scores)
    if selected is None:
        return None
    command_mapping = next(
        action.command.to_mapping()
        for action in actions
        if action.action_id == selected.action_id
    )
    return build_advisory_proposal(
        hmc,
        snapshot.snapshot_sha256,
        step,
        command_mapping,
        selected.action_id,
        ORACLE_SOURCE_TYPE,
    )


def bootstrap_gap_closure(
    rules_values: Sequence[float],
    model_values: Sequence[float],
    oracle_values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | str | bool | int]:
    """Preregistered gap-closure point estimate and family-bootstrap CI."""

    for label, values in (
        ("rules", rules_values),
        ("model", model_values),
        ("oracle", oracle_values),
    ):
        if len(values) != len(rules_values) or not values:
            raise Issue55RaceError("bootstrap inputs must be equal-length families")
        if any(not math.isfinite(float(value)) for value in values):
            raise Issue55RaceError(f"bootstrap {label} values must be finite")
    rules = np.asarray(rules_values, dtype=np.float64)
    model = np.asarray(model_values, dtype=np.float64)
    oracle = np.asarray(oracle_values, dtype=np.float64)
    count = rules.size

    def _closure(
        rule_sample: np.ndarray, model_sample: np.ndarray, oracle_sample: np.ndarray
    ) -> float:
        denominator = float(np.mean(rule_sample) - np.mean(oracle_sample))
        if abs(denominator) < GAP_DENOMINATOR_FLOOR:
            return math.nan
        numerator = float(np.mean(rule_sample)) - float(np.mean(model_sample))
        return numerator / denominator

    point = _closure(rules, model, oracle)
    degenerate = (
        abs(float(np.mean(rules) - np.mean(oracle))) < GAP_DENOMINATOR_FLOOR
    )
    valid: list[float] = []
    if not degenerate:
        label = f"issue55-bootstrap-v1|{seed}".encode("utf-8")
        for resample in range(resamples):
            indices = []
            for position in range(count):
                digest = hashlib.sha256(
                    label
                    + resample.to_bytes(8, "big")
                    + position.to_bytes(8, "big")
                ).digest()
                indices.append(int.from_bytes(digest[:8], "big") % count)
            picked = np.asarray(indices, dtype=np.int64)
            value = _closure(rules[picked], model[picked], oracle[picked])
            if not math.isnan(value):
                valid.append(value)
    if len(valid) >= 2:
        lower = float(np.percentile(np.asarray(valid), 2.5))
        upper = float(np.percentile(np.asarray(valid), 97.5))
    else:
        lower = math.nan
        upper = math.nan
    return {
        "point_estimate": None if math.isnan(point) else point,
        "ci_lower": None if math.isnan(lower) else lower,
        "ci_upper": None if math.isnan(upper) else upper,
        "degenerate_gap": degenerate,
        "status": "DEGENERATE_GAP" if degenerate else "ESTIMATED",
        "resamples": resamples,
        "seed": seed,
    }


def aggregate_race_results(records: Sequence[RaceEpisodeRecord]) -> dict[str, Any]:
    """Per-arm summaries and preregistered gap closures over family means."""

    if not records:
        raise Issue55RaceError("aggregation requires episode records")
    families = sorted({record.family_id for record in records})
    expected_pairs = {(arm, family) for arm in ARMS for family in families}
    actual_pairs = {(record.arm, record.family_id) for record in records}
    if actual_pairs != expected_pairs or len(records) != len(expected_pairs):
        raise Issue55RaceError("aggregation requires one complete record per arm/family")
    metrics = (
        "safety_exposure",
        "safety_violation_steps",
        "comfort_deviation",
        "resource_composite",
    )
    summaries: dict[str, Any] = {}
    for arm in ARMS:
        arm_records = sorted(
            (record for record in records if record.arm == arm),
            key=lambda item: item.family_id,
        )
        summaries[arm] = {
            "family_count": len(arm_records),
            "proposal_count": sum(record.proposal_count for record in arm_records),
            "abstention_count": sum(record.abstention_count for record in arm_records),
            "admitted_proposal_count": sum(
                record.admitted_proposal_count for record in arm_records
            ),
            "hmc_rejection_count": sum(
                record.hmc_rejection_count for record in arm_records
            ),
            "family_means": {
                metric: float(
                    np.mean([float(getattr(record, metric)) for record in arm_records])
                )
                for metric in metrics
            },
            "totals": {
                metric: (
                    int(sum(getattr(record, metric) for record in arm_records))
                    if metric == "safety_violation_steps"
                    else float(
                        np.sum([float(getattr(record, metric)) for record in arm_records])
                    )
                )
                for metric in metrics
            },
        }
    ordered = list(families)
    by_pair = {(record.arm, record.family_id): record for record in records}
    closures: dict[str, Any] = {}
    for metric in ("safety_exposure", "comfort_deviation", "resource_composite"):
        rules_values = [
            float(getattr(by_pair[("rules_only", family)], metric))
            for family in ordered
        ]
        model_values = [
            float(getattr(by_pair[("model_advised", family)], metric))
            for family in ordered
        ]
        oracle_values = [
            float(getattr(by_pair[("oracle_instrument", family)], metric))
            for family in ordered
        ]
        closures[metric] = bootstrap_gap_closure(
            rules_values, model_values, oracle_values
        )
    return {
        "schema_version": RACE_SCHEMA_VERSION,
        "preregistration_id": PREREGISTRATION_ID,
        "corpus_id": CORPUS_ID,
        "family_count": len(families),
        "arm_summaries": summaries,
        "gap_closures": closures,
    }


__all__ = [
    "ADVISORY_RANKING_METRIC_ID",
    "ARMS",
    "AdvisoryScore",
    "CORPUS_ID",
    "EPISODE_STEPS",
    "Issue55RaceError",
    "ORACLE_SELECTION_METRIC_ID",
    "OracleScore",
    "PREREGISTRATION_ID",
    "RACE_SCHEMA_VERSION",
    "RaceEpisodeRecord",
    "aggregate_race_results",
    "bootstrap_gap_closure",
    "build_advisory_proposal",
    "build_family_scenario",
    "compute_race_metrics",
    "decision_steps",
    "deterministic_family_ids",
    "episode_nonce",
    "family_condition_descriptor",
    "oracle_full_horizon_scores",
    "project_true_targets",
    "rank_actions_advisory",
    "run_race_episode",
    "scenario_zone_order",
    "score_point_prediction",
    "select_oracle_action",
    "target_bounds",
]
