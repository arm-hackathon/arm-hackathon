"""Bounded recovery governor for AEOLUS.

The governor is a causal, observable-only decision maker. It reads the same
``model_input_v1`` vectors the fault detector sees — never hidden fault
effectiveness, health, seeds, or schedules — and emits bounded per-zone
actuator commands with structured rationale.

Every threshold is a declared constant in :class:`ResponseSettings`, so the
response policy stays auditable, deterministic and reproducible.

Policy
------
1. **Proportional demand** — each zone starts from the same bounded
   proportional command the baseline controller would issue for its latest
   sensor reading.
2. **Frozen-sensor hold** — a zone whose reading is flat across its window is
   held at its last good command instead of chasing a stale reading.
3. **Degraded-loop spare-capacity release** — a loop with a severe, isolated,
   sustained delivery residual is treated as degraded. Its commanded demand is
   released back to shared capacity **only while that zone has spare comfort
   (its reading is below the comfort threshold)**. A zone that actually needs
   air always keeps its full proportional command, so the governor never
   under-drives a hot zone and never wastes the ability of another loop.
4. **Rate and energy bounds** — per-zone commands may move by at most
   ``max_command_delta`` per tick and always remain in 0.0..1.0, so the
   response is bounded by construction.

The governor acts one tick behind the physics: it decides for the next tick
from completed-tick observations, which is the causal contract an embedded
decision maker must respect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aeolus.config import HabitatConfig
from aeolus.model_input import build_model_input_contract

REQUESTED_EPSILON = 1e-9


@dataclass(frozen=True)
class ResponseSettings:
    """Declared, validated constants of the bounded response policy."""

    window_ticks: int = 10
    max_command_delta: float = 0.1
    degraded_residual_threshold: float = 0.4
    degradation_isolation_margin: float = 0.2
    degradation_persistence_ticks: int = 3
    min_requested_fraction: float = 0.05
    frozen_normalized_range: float = 0.02
    frozen_persistence_ticks: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.window_ticks, bool) or not isinstance(
            self.window_ticks, int
        ):
            raise ValueError("response window_ticks must be an integer")
        if self.window_ticks < 1:
            raise ValueError("response window_ticks must be positive")
        fractions = (
            self.max_command_delta,
            self.degraded_residual_threshold,
            self.degradation_isolation_margin,
            self.min_requested_fraction,
            self.frozen_normalized_range,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= value <= 1.0
            for value in fractions
        ):
            raise ValueError("response thresholds must be finite numbers in 0.0..1.0")
        for name in ("degradation_persistence_ticks", "frozen_persistence_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(
                    f"response {name} must be an integer of at least two"
                )


class BoundedRecoveryGovernor:
    """Deterministic, no-loss bounded-response controller over observed windows."""

    def __init__(
        self, config: HabitatConfig, settings: ResponseSettings | None = None
    ) -> None:
        self.config = config
        self.settings = settings or ResponseSettings()
        self._contract = build_model_input_contract(config)
        self._control = config.control
        self._zone_ids = tuple(zone.id for zone in config.non_processing_zones())
        self._index: dict[tuple[str, str, str], int] = {}
        for index, field in enumerate(self._contract.fields):
            self._index[(field.group, field.entity_id, field.field)] = index
        self._feature_count = len(self._contract.fields)
        self._outbound_id: dict[str, str] = {}
        self._max_airflow: dict[str, float] = {}
        for zone_id in self._zone_ids:
            connection = config.path_to_processing(zone_id)
            self._outbound_id[zone_id] = connection.id
            self._max_airflow[zone_id] = connection.max_airflow
        self.reset()

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Forget all scenario history and logs (call between runs)."""
        self._window: list[list[float]] = []
        self._last_commands: dict[str, float] = {}
        self._last_good: dict[str, float] = {}
        self.command_history: list[dict[str, float]] = []
        self.rationale_history: list[dict[str, dict[str, Any]]] = []

    def observe(self, vector: list[float]) -> None:
        """Feed one completed tick's exact ``model_input_v1`` vector."""
        if (
            not isinstance(vector, list)
            or len(vector) != self._feature_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            )
        ):
            raise ValueError("governor observation must be a finite model-input vector")
        self._window.append([float(value) for value in vector])
        del self._window[:-self.settings.window_ticks]

    def next_commands(self) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Return bounded commands and structured rationale for the next tick."""
        commands: dict[str, float] = {}
        rationale: dict[str, dict[str, Any]] = {}
        ratios = self._residual_ratios()
        degraded_zone = self._isolated_degraded_loop(ratios)
        loss_ratio = (
            self._estimated_loss(zone_id=degraded_zone)
            if degraded_zone is not None
            else None
        )
        has_spare = (
            self._reading(degraded_zone) <= self._control.upper_threshold
            if degraded_zone is not None
            else False
        )

        for zone_id in self._zone_ids:
            base = self._proportional_command(zone_id)
            if self._is_frozen(zone_id):
                held = self._last_good.get(zone_id, base)
                commands[zone_id] = held
                rationale[zone_id] = {
                    "reason": "frozen_hold",
                    "base_command": base,
                    "held_command": held,
                }
                continue
            self._last_good[zone_id] = base
            if degraded_zone == zone_id and has_spare:
                # The loop cannot turn more of the request stream into
                # airflow, and its zone is not comfort-pressured right now,
                # so the request beyond the deliverable is released.
                assert loss_ratio is not None
                target = base * max(0.0, 1.0 - loss_ratio)
                commands[zone_id] = _clamp(target, 0.0, 1.0)
                rationale[zone_id] = {
                    "reason": "degraded_spare_release",
                    "base_command": base,
                    "commanded": commands[zone_id],
                    "estimated_loss": loss_ratio,
                }
            else:
                commands[zone_id] = base
                rationale[zone_id] = {
                    "reason": "nominal",
                    "base_command": base,
                    "commanded": commands[zone_id],
                }

        for zone_id in self._zone_ids:
            if self._is_frozen(zone_id):
                continue
            previous = self._last_commands.get(zone_id)
            if previous is not None:
                commands[zone_id] = _clamp(
                    commands[zone_id],
                    previous - self.settings.max_command_delta,
                    previous + self.settings.max_command_delta,
                )
            commands[zone_id] = _clamp(commands[zone_id], 0.0, 1.0)
            entry = rationale[zone_id]
            if entry["reason"] == "nominal" and not math.isclose(
                entry["commanded"], commands[zone_id], abs_tol=1e-12
            ):
                entry = {**entry, "reason": "bounded_rate", "commanded": commands[zone_id]}
            elif not math.isclose(
                entry.get("commanded", commands[zone_id]),
                commands[zone_id],
                abs_tol=1e-12,
            ):
                entry = {**entry, "commanded": commands[zone_id]}
            rationale[zone_id] = entry
        self._last_commands = dict(commands)
        self.command_history.append(dict(commands))
        self.rationale_history.append(
            {zone_id: dict(entry) for zone_id, entry in rationale.items()}
        )
        return dict(commands), {
            zone_id: dict(entry) for zone_id, entry in rationale.items()
        }

    # -- decision primitives -------------------------------------------------

    def _proportional_command(self, zone_id: str) -> float:
        reading = self._reading(zone_id)
        span = self._control.upper_threshold - self._control.lower_threshold
        demand = _clamp((reading - self._control.lower_threshold) / span, 0.0, 1.0)
        return self._control.minimum_command + demand * (
            self._control.maximum_command - self._control.minimum_command
        )

    def _reading(self, zone_id: str) -> float:
        tick = self._window[-1] if self._window else None
        if tick is None:
            return 0.0
        return tick[self._index[("zones", zone_id, "sensor_co2_concentration")]]

    def _is_frozen(self, zone_id: str) -> bool:
        persistence = self.settings.frozen_persistence_ticks
        if len(self._window) < persistence:
            return False
        readings = [
            tick[self._index[("zones", zone_id, "sensor_co2_concentration")]]
            for tick in self._window[-persistence:]
        ]
        reading_range = max(readings) - min(readings)
        normalizer = max(
            abs(self._control.upper_threshold),
            max(abs(value) for value in readings),
            1e-9,
        )
        return reading_range / normalizer <= self.settings.frozen_normalized_range

    def _residual_ratios(self) -> dict[str, list[float]]:
        ratios: dict[str, list[float]] = {}
        for zone_id in self._zone_ids:
            requested_index = self._index[
                ("connections", self._outbound_id[zone_id], "requested_airflow")
            ]
            residual_index = self._index[
                ("connections", self._outbound_id[zone_id], "airflow_residual")
            ]
            ratios[zone_id] = []
            for tick in self._window:
                requested = tick[requested_index]
                if requested <= REQUESTED_EPSILON:
                    ratios[zone_id].append(0.0)
                    continue
                if requested < self.settings.min_requested_fraction * self._max_airflow[
                    zone_id
                ]:
                    ratios[zone_id].append(0.0)
                    continue
                ratios[zone_id].append(tick[residual_index] / requested)
        return ratios

    def _isolated_degraded_loop(self, ratios: dict[str, list[float]]) -> str | None:
        if (
            len(self._zone_ids) < 2
            or len(self._window) < self.settings.degradation_persistence_ticks
        ):
            return None
        candidates: list[tuple[float, str]] = []
        for zone_id, series in ratios.items():
            tail = series[-self.settings.degradation_persistence_ticks :]
            others_peak = max(
                (
                    ratio
                    for other_id, other in ratios.items()
                    if other_id != zone_id
                    for ratio in other[-self.settings.degradation_persistence_ticks :]
                ),
                default=0.0,
            )
            if all(
                ratio > self.settings.degraded_residual_threshold
                and ratio - others_peak > self.settings.degradation_isolation_margin
                for ratio in tail
            ):
                candidates.append((tail[-1], zone_id))
        if not candidates:
            return None
        return max(candidates)[1]

    def _estimated_loss(self, zone_id: str) -> float:
        ratios = self._residual_ratios()
        series = ratios[zone_id]
        tail = series[-3:] if len(series) >= 3 else series
        mean_residual = sum(tail) / len(tail)
        return max(0.0, min(1.0, mean_residual))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))