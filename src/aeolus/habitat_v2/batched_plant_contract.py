"""Research interface for the fixed-topology batched thermal and gas plant step.

This module is the executable contract for issue 92.  It declares the inputs a
batched *research* plant step accepts, the supported domain, the mechanisms
that are excluded, and the tolerances any backend must meet against the
authoritative scalar plant in ``physics.py``.

It deliberately contains no numerical kernel, no Torch import, no airflow solve
and no batch-layout machinery.  Arrays are plain NumPy ``float64``:

* per-zone quantities have shape ``batch_shape + (ZONE_COUNT,)`` with the zone
  axis last and in ``sorted(zone_id)`` order, which is the order the scalar
  plant iterates in;
* per-world quantities have shape ``batch_shape``.

Everything here is simulator truth for evaluator-side counterfactual work.  No
field is a model-facing observation, and a result produced through this
interface is not an authoritative HMC episode: it must be replayed through the
scalar plant and the deterministic HMC before any claim about an admitted
intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .scenario import SCENARIO_SCHEMA_VERSION_V5, Scenario
from .state import PlantState

INTERFACE_VERSION = "aeolus_habitat_v2_batched_plant_research_interface_v1"

SUPPORTED_SCENARIO_SCHEMA_VERSION = SCENARIO_SCHEMA_VERSION_V5

# Zone axis order.  This is ``sorted()`` of the eight zone ids of the fixed
# Habitat V2 topology; the scalar plant applies its recirculation-heat closure
# correction to the last entry.
SUPPORTED_ZONE_IDS: tuple[str, ...] = (
    "air_processing_bay",
    "airlock_suitport",
    "common_galley",
    "crew_quarters_a",
    "crew_quarters_b",
    "equipment_power_bay",
    "hygiene_medical",
    "laboratory",
)
ZONE_COUNT = len(SUPPORTED_ZONE_IDS)

SPECIES_FIELDS: tuple[str, ...] = (
    "co2_mol",
    "o2_mol",
    "water_vapor_mol",
    "inert_mol",
)

# Murphy-Koop liquid-water saturation pressure is stated for 123 K < T < 332 K.
# The supported band is narrower than that and wider than the scenario
# validator's initial-temperature range (250 K to 330 K).
SUPPORTED_TEMPERATURE_K: tuple[float, float] = (200.0, 332.0)
SUPPORTED_DT_SECONDS: tuple[float, float] = (1.0, 600.0)

# Slack allowed on the total achieved oxygen injection against the equipment
# cap.  The scalar actuator projects strictly below the cap with ``fsum``; a
# plain left-to-right sum of the same values can land a few ulps above it.
OXYGEN_CAP_RELATIVE_SLACK = 1e-12

# The scalar plant can leave the battery marginally outside [0, capacity] and
# then keeps stepping: a discharge is accepted up to
# ``max(1e-12, 1e-10 * max(1, served_load_wh))`` beyond the available energy,
# and a headroom-limited charge can round one ulp above capacity.  A state the
# plant produces must be a legal next input, so the battery domain carries this
# declared slack on both sides.
BATTERY_STATE_SLACK_WH = 1e-6

# Per-world outcome codes, in the order the scalar plant evaluates them.  The
# scalar plant raises ``InfeasibleActionError`` before the state advances, so a
# world with a nonzero code keeps its input state.
FEASIBLE = 0
INFEASIBLE_OXYGEN_LOAD = 1
INFEASIBLE_OXYGEN_STORE = 2
INFEASIBLE_ELECTRICAL = 3
# Not a scalar-plant refusal.  The step's result for this world would be
# nonfinite or outside the declared state domain (for example a temperature
# outside ``SUPPORTED_TEMPERATURE_K``), where the scalar plant would either
# continue outside the validated range or fail with a math error.  The world is
# held and reported so that it cannot abort or contaminate its neighbours.
OUTSIDE_SUPPORTED_DOMAIN = 4
INFEASIBLE_CODE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        FEASIBLE: "feasible",
        INFEASIBLE_OXYGEN_LOAD: "oxygen_load_exceeds_zone_inventory",
        INFEASIBLE_OXYGEN_STORE: "oxygen_injection_exceeds_store",
        INFEASIBLE_ELECTRICAL: "electrical_demand_exceeds_supply",
        OUTSIDE_SUPPORTED_DOMAIN: "result_outside_supported_domain",
    }
)

# Mechanisms that are NOT inside the research step.  Each stays on the scalar
# reference path (or outside the plant entirely) and must be resolved there
# before values enter this interface.
EXCLUDED_MECHANISMS: Mapping[str, str] = MappingProxyType({
    "requested_command_slew": (
        "Requested commands and rate limits for fan, dampers, scrubber, "
        "condenser, cooling and oxygen are resolved by physics.py and "
        "actuators.py; this interface accepts achieved values only."
    ),
    "oxygen_total_cap_projection": (
        "The fsum/nextafter projection of achieved oxygen injection onto the "
        "equipment cap stays in actuators.achieve_actuator_state."
    ),
    "fault_schedule_evaluation": (
        "Fault profiles are evaluated by faults.physical_fault_effects; only "
        "their per-step effect enters here, as effective delivery and "
        "ability values."
    ),
    "air_network_solve": (
        "Per-zone airflow and fan electrical power are inputs.  They come "
        "from the scalar solver or, after issue 91 merges, its airflow "
        "backend."
    ),
    "damper_jam": "Part of achieved damper position on the reference path.",
    "sensor_and_feedback_models": (
        "Sensor noise, sensor faults, operational feedback and seeds are "
        "measurement-side and never enter the plant step."
    ),
    "hmc_arbitration_and_preflight": (
        "HMC proposal, arbitration, preflight and final-command digests are "
        "outside the numerical step."
    ),
    "scenario_parsing_hashing_trace_writing": (
        "Closed-schema parsing, canonical bytes, hashing, accounting-receipt "
        "validation and JSONL trace writing stay in the authoritative path."
    ),
    "legacy_scenario_schemas": (
        "Scenario schema v1 to v4 and any topology other than the fixed "
        "eight zones are rejected."
    ),
    "per_world_plant_parameters": (
        "Plant parameters are shared by every world in a batch in this "
        "interface version."
    ),
})


class BatchedPlantDomainError(ValueError):
    """Raised when an input is outside the declared research-step domain."""


@dataclass(frozen=True)
class Tolerance:
    """Accept ``|candidate - reference| <= absolute + relative * |reference|``."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        for value in (self.absolute, self.relative):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BatchedPlantDomainError("tolerance terms must be real numbers")
            if not np.isfinite(value) or value < 0.0:
                raise BatchedPlantDomainError(
                    "tolerance terms must be finite and nonnegative"
                )

    def allowed(self, reference: np.ndarray) -> np.ndarray:
        return self.absolute + self.relative * np.abs(reference)


# Predeclared per-variable tolerances against the scalar plant.  They were
# fixed before any Torch or FP32 backend comparison existed and must not be
# loosened to make a comparison pass; a change is a recorded interface
# revision.  Compare with ``numpy.all(error <= tolerance.allowed(reference))``
# in float64, so that a NaN error fails.
#
# FP64 one-step: 1e-12 absolute plus 1e-12 relative for inventories, the
# plant's own exchange-fraction and slew exactness scale.  FP64 rollout: one
# hundred times the one-step terms over at most ``REFERENCE_ROLLOUT_MAX_STEPS``
# steps, which is deliberately tighter than 240 accumulated worst-case
# one-step errors.
#
# FP32 (provisional until measured on real hardware): one-step terms are a
# small multiple of float32 resolution at the magnitudes of the development
# fixture, chosen so that a candidate returning its input unchanged still
# fails; rollout terms follow the repository's 1e-4 scaled FP32 parity
# convention.
REFERENCE_ROLLOUT_MAX_STEPS = 240


def _table(
    inventory: Tolerance,
    temperature: Tolerance,
    battery: Tolerance,
    heat: Tolerance,
) -> Mapping[str, Tolerance]:
    return MappingProxyType(
        {
            "co2_mol": inventory,
            "o2_mol": inventory,
            "water_vapor_mol": inventory,
            "inert_mol": inventory,
            "temperature_k": temperature,
            "co2_sorbent_remaining_mol": inventory,
            "captured_co2_mol": inventory,
            "condensed_water_mol": inventory,
            "oxygen_store_mol": inventory,
            "battery_energy_wh": battery,
            "external_heat_rejected_j": heat,
            "external_heat_received_j": heat,
        }
    )


_REFERENCE_TOLERANCES: Mapping[str, Mapping[str, Mapping[str, Tolerance]]] = (
    MappingProxyType(
        {
            "float64": MappingProxyType(
                {
                    "one_step": _table(
                        inventory=Tolerance(1e-12, 1e-12),
                        temperature=Tolerance(1e-10, 0.0),
                        battery=Tolerance(1e-9, 1e-12),
                        heat=Tolerance(1e-6, 1e-12),
                    ),
                    "rollout": _table(
                        inventory=Tolerance(1e-9, 1e-10),
                        temperature=Tolerance(1e-8, 0.0),
                        battery=Tolerance(1e-7, 1e-10),
                        heat=Tolerance(1e-4, 1e-10),
                    ),
                }
            ),
            "float32": MappingProxyType(
                {
                    "one_step": _table(
                        inventory=Tolerance(1e-5, 2e-6),
                        temperature=Tolerance(2e-4, 0.0),
                        battery=Tolerance(1e-3, 2e-6),
                        heat=Tolerance(1e-1, 2e-6),
                    ),
                    "rollout": _table(
                        inventory=Tolerance(1e-3, 1e-4),
                        temperature=Tolerance(2e-2, 0.0),
                        battery=Tolerance(1e-1, 1e-4),
                        heat=Tolerance(1e1, 1e-4),
                    ),
                }
            ),
        }
    )
)


def reference_tolerances(precision: str, horizon: str) -> Mapping[str, Tolerance]:
    """Return the predeclared per-variable tolerance table."""

    try:
        if type(precision) is not str or type(horizon) is not str:
            raise KeyError(precision)
        return _REFERENCE_TOLERANCES[precision][horizon]
    except KeyError as exc:
        raise BatchedPlantDomainError(
            f"no predeclared tolerance for precision={precision!r} horizon={horizon!r}"
        ) from exc


# --------------------------------------------------------------------------
# Array validation helpers
# --------------------------------------------------------------------------


def _require_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise BatchedPlantDomainError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.float64:
        raise BatchedPlantDomainError(
            f"{name} must have dtype float64, got {value.dtype}"
        )
    if value.shape != shape:
        raise BatchedPlantDomainError(
            f"{name} must have shape {shape}, got {value.shape}"
        )
    if not bool(np.all(np.isfinite(value))):
        raise BatchedPlantDomainError(f"{name} contains a nonfinite value")
    return value


def _require_range(
    name: str,
    value: np.ndarray,
    *,
    low: float | None = None,
    high: float | None = None,
) -> None:
    if low is not None and bool(np.any(value < low)):
        raise BatchedPlantDomainError(f"{name} has a value below {low!r}")
    if high is not None and bool(np.any(value > high)):
        raise BatchedPlantDomainError(f"{name} has a value above {high!r}")


def _require_scalar(
    name: str,
    value: Any,
    *,
    low: float | None = None,
    high: float | None = None,
    strict_low: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchedPlantDomainError(f"{name} must be a real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise BatchedPlantDomainError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise BatchedPlantDomainError(f"{name} must be finite")
    if low is not None and (number <= low if strict_low else number < low):
        raise BatchedPlantDomainError(f"{name} is below its supported range")
    if high is not None and number > high:
        raise BatchedPlantDomainError(f"{name} is above its supported range")
    return number


def _check_layout(
    owner: str,
    instance: Any,
    zone_fields: tuple[str, ...],
    world_fields: tuple[str, ...],
) -> tuple[int, ...]:
    declared = {item.name for item in fields(instance)}
    if declared != set(zone_fields) | set(world_fields):
        raise AssertionError(f"{owner} field layout is inconsistent")
    first = getattr(instance, zone_fields[0])
    if type(first) is not np.ndarray or first.ndim < 1:
        raise BatchedPlantDomainError(
            f"{owner}.{zone_fields[0]} must be a numpy.ndarray with a zone axis"
        )
    if first.shape[-1] != ZONE_COUNT:
        raise BatchedPlantDomainError(
            f"{owner}.{zone_fields[0]} must have {ZONE_COUNT} zones on its last axis"
        )
    batch_shape = tuple(first.shape[:-1])
    if 0 in batch_shape:
        raise BatchedPlantDomainError(f"{owner} must contain at least one world")
    for name in zone_fields:
        _require_array(
            f"{owner}.{name}", getattr(instance, name), batch_shape + (ZONE_COUNT,)
        )
    for name in world_fields:
        _require_array(f"{owner}.{name}", getattr(instance, name), batch_shape)
    return batch_shape


def zone_sum(values: np.ndarray) -> np.ndarray:
    """Left-to-right sum over the zone axis.

    ``numpy.sum`` may use pairwise or vectorised accumulation whose rounding
    depends on memory layout.  A fixed left-to-right order matches the scalar
    plant and keeps each world independent of how the batch is arranged.
    """

    total = values[..., 0].copy()
    for index in range(1, values.shape[-1]):
        total = total + values[..., index]
    return total


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

_ZONE_PARAMETER_KEYS: tuple[str, ...] = (
    "volume_m3",
    "thermal_capacity_j_per_k",
    "passive_thermal_conductance_w_per_k",
    "sink_temperature_k",
)
_EQUIPMENT_PARAMETER_KEYS: tuple[str, ...] = (
    "air_density_kg_m3",
    "air_specific_heat_j_kg_k",
    "scrubber_max_co2_mol_s",
    "condenser_max_water_mol_s",
    "oxygen_injection_max_total_mol_s",
    "cooling_max_thermal_w_per_zone",
    "base_load_w",
    "scrubber_power_w_full",
    "condenser_power_w_full",
    "cooling_coefficient_of_performance",
    "oxygen_injection_power_w_per_mol_s",
    "battery_capacity_wh",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "battery_max_charge_w",
    "battery_max_discharge_w",
)


@dataclass(frozen=True)
class PlantParameters:
    """Plant parameters shared by every world.  Units are in the field names."""

    zone_ids: tuple[str, ...]
    volume_m3: np.ndarray
    thermal_capacity_j_per_k: np.ndarray
    passive_thermal_conductance_w_per_k: np.ndarray
    sink_temperature_k: np.ndarray
    air_density_kg_m3: float
    air_specific_heat_j_kg_k: float
    scrubber_max_co2_mol_s: float
    condenser_max_water_mol_s: float
    oxygen_injection_max_total_mol_s: float
    cooling_max_thermal_w_per_zone: float
    base_load_w: float
    scrubber_power_w_full: float
    condenser_power_w_full: float
    cooling_coefficient_of_performance: float
    oxygen_injection_power_w_per_mol_s: float
    battery_capacity_wh: float
    battery_charge_efficiency: float
    battery_discharge_efficiency: float
    battery_max_charge_w: float
    battery_max_discharge_w: float

    def __post_init__(self) -> None:
        if type(self.zone_ids) is not tuple or self.zone_ids != SUPPORTED_ZONE_IDS:
            raise BatchedPlantDomainError(
                "zone_ids must be the fixed eight-zone topology in sorted order"
            )
        for name in _ZONE_PARAMETER_KEYS:
            array = _require_array(
                f"PlantParameters.{name}", getattr(self, name), (ZONE_COUNT,)
            )
            if name == "passive_thermal_conductance_w_per_k":
                _require_range(f"PlantParameters.{name}", array, low=0.0)
            elif name == "sink_temperature_k":
                _require_range(f"PlantParameters.{name}", array, low=150.0, high=330.0)
            elif bool(np.any(array <= 0.0)):
                raise BatchedPlantDomainError(f"PlantParameters.{name} must be > 0")
        strictly_positive = {
            "air_density_kg_m3",
            "air_specific_heat_j_kg_k",
            "cooling_coefficient_of_performance",
            "battery_capacity_wh",
            "battery_charge_efficiency",
            "battery_discharge_efficiency",
        }
        for name in _EQUIPMENT_PARAMETER_KEYS:
            _require_scalar(
                f"PlantParameters.{name}",
                getattr(self, name),
                low=0.0,
                strict_low=name in strictly_positive,
                high=(1.0 if name.endswith("_efficiency") else None),
            )

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "PlantParameters":
        require_supported_scenario(scenario)
        zone_configs = {str(zone["id"]): zone for zone in scenario.data["zones"]}
        equipment = scenario.data["equipment"]
        zone_arrays = {
            name: np.array(
                [float(zone_configs[zone_id][name]) for zone_id in SUPPORTED_ZONE_IDS],
                dtype=np.float64,
            )
            for name in _ZONE_PARAMETER_KEYS
        }
        return cls(
            zone_ids=SUPPORTED_ZONE_IDS,
            **zone_arrays,
            **{name: float(equipment[name]) for name in _EQUIPMENT_PARAMETER_KEYS},
        )


def require_supported_scenario(scenario: Scenario) -> None:
    """Fail closed unless the scenario is inside the declared research domain."""

    if type(scenario) is not Scenario:
        raise BatchedPlantDomainError("scenario must be a validated Scenario")
    try:
        scenario.validate_contract_identities()
    except Exception as exc:
        raise BatchedPlantDomainError(
            "scenario does not match its validated canonical identity"
        ) from exc
    if scenario.scenario_schema_version != SUPPORTED_SCENARIO_SCHEMA_VERSION:
        raise BatchedPlantDomainError(
            "only scenario schema v5 is supported by the batched research step"
        )
    zone_ids = tuple(sorted(str(zone["id"]) for zone in scenario.data["zones"]))
    if zone_ids != SUPPORTED_ZONE_IDS:
        raise BatchedPlantDomainError(
            "only the fixed eight-zone Habitat V2 topology is supported"
        )
    branch_zone_ids = tuple(
        sorted(
            str(branch["zone_id"]) for branch in scenario.data["air_network"]["branches"]
        )
    )
    if branch_zone_ids != SUPPORTED_ZONE_IDS:
        raise BatchedPlantDomainError(
            "air network must have exactly one branch per supported zone"
        )
    require_supported_dt(scenario.data["dt_seconds"])


def require_supported_dt(dt_seconds: Any) -> float:
    low, high = SUPPORTED_DT_SECONDS
    return _require_scalar("dt_seconds", dt_seconds, low=low, high=high)


# --------------------------------------------------------------------------
# State, achieved control and disturbance
# --------------------------------------------------------------------------

_STATE_ZONE_FIELDS = SPECIES_FIELDS + ("temperature_k",)
_STATE_WORLD_FIELDS: tuple[str, ...] = (
    "co2_sorbent_remaining_mol",
    "captured_co2_mol",
    "condensed_water_mol",
    "oxygen_store_mol",
    "battery_energy_wh",
    "external_heat_rejected_j",
    "external_heat_received_j",
)
STATE_FIELDS: tuple[str, ...] = _STATE_ZONE_FIELDS + _STATE_WORLD_FIELDS


@dataclass(frozen=True)
class PlantStateArrays:
    """Thermal, gas and store state.  Actuator state is not part of it.

    Achieved actuator positions evolve on the scalar reference path and enter
    each step through :class:`AchievedControl`.  Measurement state
    (``last_operational_feedback``) is evaluator-side and is never carried.
    """

    co2_mol: np.ndarray
    o2_mol: np.ndarray
    water_vapor_mol: np.ndarray
    inert_mol: np.ndarray
    temperature_k: np.ndarray
    co2_sorbent_remaining_mol: np.ndarray
    captured_co2_mol: np.ndarray
    condensed_water_mol: np.ndarray
    oxygen_store_mol: np.ndarray
    battery_energy_wh: np.ndarray
    external_heat_rejected_j: np.ndarray
    external_heat_received_j: np.ndarray

    def __post_init__(self) -> None:
        _check_layout(
            "PlantStateArrays", self, _STATE_ZONE_FIELDS, _STATE_WORLD_FIELDS
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.temperature_k.shape[:-1])

    @classmethod
    def from_plant_states(cls, states: tuple[PlantState, ...]) -> "PlantStateArrays":
        """Stack scalar plant states into one leading world axis."""

        if not states:
            raise BatchedPlantDomainError("at least one PlantState is required")
        for state in states:
            if type(state) is not PlantState:
                raise BatchedPlantDomainError("states must be PlantState values")
            if tuple(sorted(state.zones)) != SUPPORTED_ZONE_IDS:
                raise BatchedPlantDomainError(
                    "PlantState does not have the supported eight-zone topology"
                )
        zone_values = {
            name: np.array(
                [
                    [float(getattr(state.zones[zone_id], name)) for zone_id in SUPPORTED_ZONE_IDS]
                    for state in states
                ],
                dtype=np.float64,
            )
            for name in _STATE_ZONE_FIELDS
        }
        world_values = {
            name: np.array(
                [float(getattr(state.utility, name)) for state in states],
                dtype=np.float64,
            )
            for name in _STATE_WORLD_FIELDS
        }
        return cls(**zone_values, **world_values)


_CONTROL_ZONE_FIELDS: tuple[str, ...] = (
    "zone_airflow_m3_s",
    "achieved_cooling_removed_w",
    "effective_cooling_delivery_w",
    "achieved_oxygen_injection_mol_s",
    "effective_oxygen_delivery_mol_s",
)
_CONTROL_WORLD_FIELDS: tuple[str, ...] = (
    "fan_electrical_power_w",
    "achieved_scrubber_duty",
    "achieved_condenser_duty",
    "scrubber_capture_ability",
    "condenser_removal_ability",
)
CONTROL_FIELDS: tuple[str, ...] = _CONTROL_ZONE_FIELDS + _CONTROL_WORLD_FIELDS


@dataclass(frozen=True)
class AchievedControl:
    """Achieved and effective actuation for one step.

    There is no field for a requested or HMC-final command: this interface
    cannot be handed one by mistake.  ``achieved_*`` values are post-slew
    actuator outputs and set electrical load.  ``effective_*`` values and the
    two abilities include physical-fault effects and set what reaches the air.
    ``zone_airflow_m3_s`` and ``fan_electrical_power_w`` are the air-network
    solution at the effective fan speed and achieved damper positions.
    """

    zone_airflow_m3_s: np.ndarray
    achieved_cooling_removed_w: np.ndarray
    effective_cooling_delivery_w: np.ndarray
    achieved_oxygen_injection_mol_s: np.ndarray
    effective_oxygen_delivery_mol_s: np.ndarray
    fan_electrical_power_w: np.ndarray
    achieved_scrubber_duty: np.ndarray
    achieved_condenser_duty: np.ndarray
    scrubber_capture_ability: np.ndarray
    condenser_removal_ability: np.ndarray

    def __post_init__(self) -> None:
        _check_layout(
            "AchievedControl", self, _CONTROL_ZONE_FIELDS, _CONTROL_WORLD_FIELDS
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.zone_airflow_m3_s.shape[:-1])


_DISTURBANCE_ZONE_FIELDS: tuple[str, ...] = (
    "co2_generation_mol_s",
    "o2_consumption_mol_s",
    "water_vapor_generation_mol_s",
    "sensible_heat_w",
)
_DISTURBANCE_WORLD_FIELDS: tuple[str, ...] = ("generation_w",)
DISTURBANCE_FIELDS: tuple[str, ...] = (
    _DISTURBANCE_ZONE_FIELDS + _DISTURBANCE_WORLD_FIELDS
)


@dataclass(frozen=True)
class Disturbance:
    """Explicit per-world loads and generation for one step.

    The step never looks anything up by step index and draws no random
    numbers: a stochastic disturbance must be sampled by the caller, per
    world, and passed in here.
    """

    co2_generation_mol_s: np.ndarray
    o2_consumption_mol_s: np.ndarray
    water_vapor_generation_mol_s: np.ndarray
    sensible_heat_w: np.ndarray
    generation_w: np.ndarray

    def __post_init__(self) -> None:
        _check_layout(
            "Disturbance", self, _DISTURBANCE_ZONE_FIELDS, _DISTURBANCE_WORLD_FIELDS
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.sensible_heat_w.shape[:-1])


def validate_step_inputs(
    parameters: PlantParameters,
    state: PlantStateArrays,
    control: AchievedControl,
    disturbance: Disturbance,
    dt_seconds: Any,
) -> float:
    """Fail closed on any type, shape, finiteness or domain violation.

    Returns the validated timestep.  Infeasible but in-domain inputs (for
    example an oxygen load larger than a zone inventory) are *not* rejected
    here; the step reports them per world.
    """

    if type(parameters) is not PlantParameters:
        raise BatchedPlantDomainError("parameters must be PlantParameters")
    if type(state) is not PlantStateArrays:
        raise BatchedPlantDomainError("state must be PlantStateArrays")
    if type(control) is not AchievedControl:
        raise BatchedPlantDomainError(
            "control must be AchievedControl; requested commands are not accepted"
        )
    if type(disturbance) is not Disturbance:
        raise BatchedPlantDomainError("disturbance must be Disturbance")
    dt = require_supported_dt(dt_seconds)
    # The containers hold writable arrays, so everything checked at
    # construction is checked again here.
    parameters.__post_init__()
    state.__post_init__()
    control.__post_init__()
    disturbance.__post_init__()
    if not (state.batch_shape == control.batch_shape == disturbance.batch_shape):
        raise BatchedPlantDomainError(
            "state, control and disturbance must share one batch shape"
        )

    for name in STATE_FIELDS:
        low, high = state_field_domain(parameters, name)
        _require_range(
            f"PlantStateArrays.{name}", getattr(state, name), low=low, high=high
        )

    for name in CONTROL_FIELDS:
        _require_range(f"AchievedControl.{name}", getattr(control, name), low=0.0)
    for name in (
        "achieved_scrubber_duty",
        "achieved_condenser_duty",
        "scrubber_capture_ability",
        "condenser_removal_ability",
    ):
        _require_range(f"AchievedControl.{name}", getattr(control, name), high=1.0)
    _require_range(
        "AchievedControl.achieved_cooling_removed_w",
        control.achieved_cooling_removed_w,
        high=parameters.cooling_max_thermal_w_per_zone,
    )
    if bool(
        np.any(control.effective_cooling_delivery_w > control.achieved_cooling_removed_w)
    ):
        raise BatchedPlantDomainError(
            "effective cooling delivery exceeds achieved cooling"
        )
    if bool(
        np.any(
            control.effective_oxygen_delivery_mol_s
            > control.achieved_oxygen_injection_mol_s
        )
    ):
        raise BatchedPlantDomainError(
            "effective oxygen delivery exceeds achieved oxygen injection"
        )
    cap = parameters.oxygen_injection_max_total_mol_s
    total_oxygen = zone_sum(control.achieved_oxygen_injection_mol_s)
    if bool(np.any(total_oxygen > cap * (1.0 + OXYGEN_CAP_RELATIVE_SLACK))):
        raise BatchedPlantDomainError(
            "total achieved oxygen injection exceeds the equipment cap"
        )

    for name in DISTURBANCE_FIELDS:
        _require_range(f"Disturbance.{name}", getattr(disturbance, name), low=0.0)
    return dt


def state_field_domain(
    parameters: PlantParameters, name: str
) -> tuple[float, float | None]:
    """Closed ``(low, high)`` domain of one state field; ``None`` is unbounded.

    The same bounds decide whether an input state is accepted and whether a
    step result is reported as ``OUTSIDE_SUPPORTED_DOMAIN``.
    """

    if name == "temperature_k":
        return SUPPORTED_TEMPERATURE_K
    if name == "battery_energy_wh":
        return (
            -BATTERY_STATE_SLACK_WH,
            parameters.battery_capacity_wh + BATTERY_STATE_SLACK_WH,
        )
    if name in STATE_FIELDS:
        return (0.0, None)
    raise BatchedPlantDomainError(f"unknown state field {name!r}")


__all__ = [
    "AchievedControl",
    "BATTERY_STATE_SLACK_WH",
    "BatchedPlantDomainError",
    "CONTROL_FIELDS",
    "DISTURBANCE_FIELDS",
    "Disturbance",
    "EXCLUDED_MECHANISMS",
    "FEASIBLE",
    "INFEASIBLE_CODE_NAMES",
    "INFEASIBLE_ELECTRICAL",
    "INFEASIBLE_OXYGEN_LOAD",
    "INFEASIBLE_OXYGEN_STORE",
    "INTERFACE_VERSION",
    "OUTSIDE_SUPPORTED_DOMAIN",
    "OXYGEN_CAP_RELATIVE_SLACK",
    "PlantParameters",
    "PlantStateArrays",
    "REFERENCE_ROLLOUT_MAX_STEPS",
    "SPECIES_FIELDS",
    "STATE_FIELDS",
    "SUPPORTED_DT_SECONDS",
    "SUPPORTED_SCENARIO_SCHEMA_VERSION",
    "SUPPORTED_TEMPERATURE_K",
    "SUPPORTED_ZONE_IDS",
    "Tolerance",
    "ZONE_COUNT",
    "reference_tolerances",
    "require_supported_dt",
    "require_supported_scenario",
    "state_field_domain",
    "validate_step_inputs",
    "zone_sum",
]
