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

DELIVERY_RESIDUAL_RATIO = 0.05
"""Relative residual (residual / requested) above which a loop is losing delivery."""

DELIVERY_PERSISTENCE_TICKS = 3
"""Consecutive window-end ticks the loss must persist before it is a fault."""

ISOLATION_MARGIN = 0.05
"""How far one loop's loss must sit above the others to be a fault, not contention."""

BLOCK_JUMP_RATIO = 0.2
"""Single-tick jump in relative residual that marks a sudden (blocked) onset."""

FROZEN_RUN_TICKS = 10
"""Identical consecutive sensor readings that mark a frozen sensor."""

_REQUESTED_EPSILON = 1e-9


class RuleBaseline:
    """Streaming rule detector over consecutive windows of one scenario run."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget all scenario history (call between scenario runs)."""
        self._max_jump: dict[str, float] = {}

    def label_window(self, features: list[dict]) -> str:
        """Label one window: nominal, frozen_sensor, blocked_path or degradation."""
        if not features:
            raise ValueError("a window must contain at least one tick")
        ratios_by_connection = {
            connection_id: [_residual_ratio(tick, connection_id) for tick in features]
            for connection_id in features[0]["connections"]
        }
        for connection_id, ratios in ratios_by_connection.items():
            jumps = [later - earlier for earlier, later in zip(ratios, ratios[1:])]
            if jumps:
                self._max_jump[connection_id] = max(
                    self._max_jump.get(connection_id, 0.0), max(jumps)
                )
        if _has_frozen_sensor(features):
            return "frozen_sensor"
        faulty = self._isolated_faulty_connection(ratios_by_connection)
        if faulty is None:
            return "nominal"
        if self._max_jump.get(faulty, 0.0) > BLOCK_JUMP_RATIO:
            return "blocked_path"
        return "gradual_primary_fan_degradation"

    def _loop_groups(
        self, ratios_by_connection: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Group connections into loops via the hub naming convention.

        Every zone has an outbound ``<zone>_to_processing`` leg and a paired
        ``processing_to_<zone>`` return leg reporting identical airflow. The
        outbound leg represents the loop. Legs are paired by name, never by
        series equality — loops in identical states (healthy, or jointly
        contended) must stay separate so isolation keeps a reference.
        """
        loops: dict[str, list[float]] = {}
        inbound_mates: set[str] = set()
        for connection_id in sorted(ratios_by_connection):
            if connection_id in inbound_mates:
                continue
            if connection_id.endswith("_to_processing"):
                mate = "processing_to_" + connection_id[: -len("_to_processing")]
                if mate in ratios_by_connection:
                    inbound_mates.add(mate)
            loops[connection_id] = ratios_by_connection[connection_id]
        return loops

    def _isolated_faulty_connection(
        self, ratios_by_connection: dict[str, list[float]]
    ) -> str | None:
        loops = self._loop_groups(ratios_by_connection)
        if len(next(iter(loops.values()), [])) < DELIVERY_PERSISTENCE_TICKS:
            return None
        # Isolation needs a reference: a one-loop habitat offers no way to
        # tell a faulted loop from shared contention, so it stays nominal.
        if len(loops) < 2:
            return None
        candidates: list[tuple[float, str]] = []
        for connection_id, ratios in loops.items():
            tail = ratios[-DELIVERY_PERSISTENCE_TICKS:]
            others = [
                ratio
                for other_id, other_ratios in loops.items()
                if other_id != connection_id
                for ratio in other_ratios[-DELIVERY_PERSISTENCE_TICKS:]
            ]
            others_peak = max(others, default=0.0)
            if all(
                ratio > DELIVERY_RESIDUAL_RATIO
                and ratio - others_peak > ISOLATION_MARGIN
                for ratio in tail
            ):
                candidates.append((tail[-1], connection_id))
        if not candidates:
            return None
        return max(candidates)[1]


def _has_frozen_sensor(features: list[dict]) -> bool:
    # Zones with actuators are the real sensor loops; the processing bay has a
    # constant zero reading by construction and must never count as frozen.
    if len(features) < FROZEN_RUN_TICKS:
        return False
    sensor_zones = [
        zone_id
        for zone_id in features[0]["zones"]
        if zone_id in features[0]["actuators"]
    ]
    for zone_id in sensor_zones:
        readings = [
            tick["zones"][zone_id]["sensor_co2_concentration"]
            for tick in features[-FROZEN_RUN_TICKS:]
        ]
        if len(set(readings)) == 1:
            return True
    return False


def _residual_ratio(tick: dict, connection_id: str) -> float:
    entry = tick["connections"][connection_id]
    requested = entry["requested_airflow"]
    if requested <= _REQUESTED_EPSILON:
        return 0.0
    return entry["airflow_residual"] / requested
