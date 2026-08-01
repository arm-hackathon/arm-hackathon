"""Rule-based fault detection over model-feature windows.

The rule baseline is the honest bar the temporal classifier must beat. It
reads only ``model_feature_row``-shaped windows — sensor readings, actuator
fields and requested/delivered/residual airflow — never hidden simulator
truth. Every threshold is a declared constant so the rules stay auditable.

Detection is streaming: a scenario's windows arrive in order, and the
detector remembers the largest residual jump it has seen per connection
until :meth:`RuleBaseline.reset` is called. That memory is what separates a
blockage (a sudden onset, visible only while the onset sits inside the
window) from a gradual degradation.

Two healthy states can produce a persistent residual without any fault, and
both are handled explicitly:

- shared-capacity contention cuts every loop proportionally, so no single
  loop stands out — faults are only called when one loop is *isolated* above
  the others;
- static path health would produce a constant residual from tick one; no
  shipped scenario exercises it, so the detector does not baseline-subtract
  yet. If a scenario with health < 1.0 lands, add an establishment window.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

from aeolus.config import HabitatConfig
from aeolus.model_input import build_model_input_contract

DELIVERY_RESIDUAL_RATIO = 0.05
"""Relative residual (residual / requested) above which a loop is losing delivery."""

DELIVERY_PERSISTENCE_TICKS = 3
"""Consecutive window-end ticks the loss must persist before it is a fault."""

ISOLATION_MARGIN = 0.05
"""How far one loop's loss must sit above the others to be a fault, not contention."""

BLOCK_JUMP_RATIO = 0.2
"""Single-tick jump in relative residual that marks a sudden (blocked) onset."""

FROZEN_RUN_TICKS = 10
"""Default consecutive readings used to distinguish a frozen sensor."""

FROZEN_NORMALIZED_RANGE = 0.0
"""Default tail range, normalised by controller scale, for a frozen sensor."""

RULE_PARAMETER_GRID = {
    "residual_threshold": (0.04, 0.06, 0.08, 0.10),
    "isolation_margin": (0.03, 0.05, 0.08),
    "blockage_jump": (0.10, 0.20, 0.30),
    "frozen_normalized_range": (0.005, 0.01, 0.02),
    "persistence_ticks": (3, 5),
}

_REQUESTED_EPSILON = 1e-9


@dataclass(frozen=True, order=True)
class RuleParameters:
    """Validation-tuned thresholds for the observable rule detector."""

    residual_threshold: float = DELIVERY_RESIDUAL_RATIO
    isolation_margin: float = ISOLATION_MARGIN
    blockage_jump: float = BLOCK_JUMP_RATIO
    frozen_normalized_range: float = FROZEN_NORMALIZED_RANGE
    persistence_ticks: int = DELIVERY_PERSISTENCE_TICKS

    def __post_init__(self) -> None:
        fractions = (
            self.residual_threshold,
            self.isolation_margin,
            self.blockage_jump,
            self.frozen_normalized_range,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in fractions
        ):
            raise ValueError("rule thresholds must be finite and non-negative")
        if (
            isinstance(self.persistence_ticks, bool)
            or not isinstance(self.persistence_ticks, int)
            or self.persistence_ticks < 2
        ):
            raise ValueError("rule persistence_ticks must be an integer of at least two")

    def as_dict(self) -> dict[str, float | int]:
        """Return a stable JSON-ready parameter record."""
        return asdict(self)


def rule_parameter_grid() -> tuple[RuleParameters, ...]:
    """Return the committed finite calibration grid in lexicographic order."""
    keys = tuple(RULE_PARAMETER_GRID)
    return tuple(
        RuleParameters(**dict(zip(keys, values)))
        for values in itertools.product(*(RULE_PARAMETER_GRID[key] for key in keys))
    )


class RuleBaseline:
    """Streaming rule detector over consecutive windows of one scenario run."""

    def __init__(
        self, config: HabitatConfig, parameters: RuleParameters | None = None
    ) -> None:
        self.parameters = parameters or RuleParameters()
        self._frozen_persistence_ticks = (
            self.parameters.persistence_ticks
            if parameters is not None
            else FROZEN_RUN_TICKS
        )
        self._model_input_contract = build_model_input_contract(config)
        self._processing_zone_id = config.processing_zone().id
        self._loop_connection_pairs = tuple(
            (config.path_to_processing(zone.id).id, config.path_from_processing(zone.id).id)
            for zone in config.non_processing_zones()
        )
        self._connection_ids = tuple(
            connection_id
            for outbound_id, return_id in self._loop_connection_pairs
            for connection_id in (outbound_id, return_id)
        )
        self._expected_connection_ids = frozenset(self._connection_ids)
        self._expected_zone_ids = frozenset(zone.id for zone in config.zones)
        self._expected_actuator_ids = frozenset(
            zone.id for zone in config.non_processing_zones()
        )
        self._co2_scale = config.control.upper_threshold
        self.reset()

    def reset(self) -> None:
        """Forget all scenario history (call between scenario runs)."""
        self._max_jump: dict[str, float] = {}

    def label_window(self, features: list[dict]) -> str:
        """Label one window: nominal, frozen_sensor, blocked_path or degradation."""
        if not features:
            raise ValueError("a window must contain at least one tick")
        if not isinstance(features[0], dict):
            features = self._expand_model_input_window(features)
        self._validate_window_topology(features)
        ratios_by_connection = {
            connection_id: [_residual_ratio(tick, connection_id) for tick in features]
            for connection_id in self._connection_ids
        }
        for connection_id, ratios in ratios_by_connection.items():
            jumps = [later - earlier for earlier, later in zip(ratios, ratios[1:])]
            if jumps:
                self._max_jump[connection_id] = max(
                    self._max_jump.get(connection_id, 0.0), max(jumps)
                )
        if _has_frozen_sensor(
            features,
            normalized_range=self.parameters.frozen_normalized_range,
            persistence_ticks=self._frozen_persistence_ticks,
            scale=self._co2_scale,
        ):
            return "frozen_sensor"
        faulty = self._isolated_faulty_connection(ratios_by_connection)
        if faulty is None:
            return "nominal"
        if self._max_jump.get(faulty, 0.0) > self.parameters.blockage_jump:
            return "blocked_path"
        return "gradual_primary_fan_degradation"

    def _expand_model_input_window(self, vectors: list[object]) -> list[dict]:
        """Adapt exact v1 vectors to the baseline's observable rule representation."""
        ticks: list[dict] = []
        for tick_number, vector in enumerate(vectors, start=1):
            if not isinstance(vector, list) or len(vector) != len(self._model_input_contract.fields):
                raise ValueError(f"model-input tick {tick_number} has an unexpected shape")
            tick = {
                "zones": {self._processing_zone_id: {"sensor_co2_concentration": 0.0}},
                "actuators": {},
                "connections": {
                    connection_id: {
                        "requested_airflow": 0.0,
                        "delivered_airflow": 0.0,
                        "airflow_residual": 0.0,
                    }
                    for _, connection_id in self._loop_connection_pairs
                },
            }
            for field, value in zip(self._model_input_contract.fields, vector):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"model-input tick {tick_number} contains a non-numeric value")
                group = tick[field.group]
                entity = group.setdefault(field.entity_id, {})
                entity[field.field] = float(value)
            ticks.append(tick)
        return ticks

    def _loop_groups(
        self, ratios_by_connection: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Select graph-paired outbound legs after checking topology compatibility."""
        self._validate_connection_ids(set(ratios_by_connection), "ratio collection")
        return {
            outbound_id: ratios_by_connection[outbound_id]
            for outbound_id, _ in self._loop_connection_pairs
        }

    def _validate_window_topology(self, features: list[dict]) -> None:
        for tick_number, tick in enumerate(features, start=1):
            if not isinstance(tick, dict):
                raise ValueError(f"feature tick {tick_number} must be an object")
            context = f"tick {tick_number}"
            for group, expected in (
                ("zones", self._expected_zone_ids),
                ("actuators", self._expected_actuator_ids),
                ("connections", self._expected_connection_ids),
            ):
                values = tick.get(group)
                if not isinstance(values, dict):
                    raise ValueError(f"feature {context} {group} must be an object")
                self._validate_entity_ids(set(values), expected, group, context)

    def _validate_connection_ids(self, actual: set[object], context: str) -> None:
        self._validate_entity_ids(
            actual, self._expected_connection_ids, "connections", context
        )

    def _validate_entity_ids(
        self,
        actual: set[object],
        expected: frozenset[str],
        group: str,
        context: str,
    ) -> None:
        if actual == expected:
            return
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected, key=repr)
        raise ValueError(
            "feature window topology does not match RuleBaseline config "
            f"at {context} for {group}: missing={missing!r}, unexpected={unexpected!r}"
        )

    def _isolated_faulty_connection(
        self, ratios_by_connection: dict[str, list[float]]
    ) -> str | None:
        loops = self._loop_groups(ratios_by_connection)
        if len(next(iter(loops.values()), [])) < self.parameters.persistence_ticks:
            return None
        # Isolation needs a reference: a one-loop habitat offers no way to
        # tell a faulted loop from shared contention, so it stays nominal.
        if len(loops) < 2:
            return None
        candidates: list[tuple[float, str]] = []
        for connection_id, ratios in loops.items():
            tail = ratios[-self.parameters.persistence_ticks:]
            others = [
                ratio
                for other_id, other_ratios in loops.items()
                if other_id != connection_id
                for ratio in other_ratios[-self.parameters.persistence_ticks:]
            ]
            others_peak = max(others, default=0.0)
            if all(
                ratio > self.parameters.residual_threshold
                and ratio - others_peak > self.parameters.isolation_margin
                for ratio in tail
            ):
                candidates.append((tail[-1], connection_id))
        if not candidates:
            return None
        return max(candidates)[1]


def _has_frozen_sensor(
    features: list[dict],
    *,
    normalized_range: float,
    persistence_ticks: int,
    scale: float,
) -> bool:
    # Zones with actuators are the real sensor loops; the processing bay has a
    # constant zero reading by construction and must never count as frozen.
    if len(features) < persistence_ticks:
        return False
    sensor_zones = [
        zone_id
        for zone_id in features[0]["zones"]
        if zone_id in features[0]["actuators"]
    ]
    for zone_id in sensor_zones:
        readings = [
            tick["zones"][zone_id]["sensor_co2_concentration"]
            for tick in features[-persistence_ticks:]
        ]
        reading_range = max(readings) - min(readings)
        normalizer = max(abs(scale), max(abs(value) for value in readings), 1e-9)
        if reading_range / normalizer <= normalized_range:
            return True
    return False


def _residual_ratio(tick: dict, connection_id: str) -> float:
    entry = tick["connections"][connection_id]
    requested = entry["requested_airflow"]
    if requested <= _REQUESTED_EPSILON:
        return 0.0
    return entry["airflow_residual"] / requested
