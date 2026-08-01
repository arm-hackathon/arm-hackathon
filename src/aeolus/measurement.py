"""Stateless deterministic measurement effects for replayable simulations."""

from __future__ import annotations

import hashlib
import math

from aeolus.config import HabitatConfig

DRIFT_ANCHOR_TICKS = 20


def deterministic_measurement_sample(
    config: HabitatConfig, entity_id: str, channel: str, tick: int
) -> float:
    """Return a stable uniform sample in ``[-1, 1)``.

    The key includes seed, entity, channel and tick, so samples do not depend
    on traversal order or mutable random-generator state.
    """
    key = (
        f"aeolus:measurement:{config.simulation.random_seed}:"
        f"{entity_id}:{channel}:{tick}"
    ).encode("utf-8")
    unit_integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
    return 2.0 * (unit_integer / 2**53) - 1.0


def deterministic_measurement_drift(
    config: HabitatConfig,
    entity_id: str,
    channel: str,
    tick: int,
    *,
    anchor_ticks: int = DRIFT_ANCHOR_TICKS,
) -> float:
    """Interpolate deterministic samples into bounded piecewise-linear drift."""
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise ValueError("measurement drift tick must be an integer")
    if isinstance(anchor_ticks, bool) or not isinstance(anchor_ticks, int) or anchor_ticks < 1:
        raise ValueError("measurement drift anchor_ticks must be positive")
    lower_anchor = math.floor(tick / anchor_ticks) * anchor_ticks
    upper_anchor = lower_anchor + anchor_ticks
    progress = (tick - lower_anchor) / anchor_ticks
    lower = deterministic_measurement_sample(
        config, entity_id, f"{channel}:drift-anchor", lower_anchor
    )
    upper = deterministic_measurement_sample(
        config, entity_id, f"{channel}:drift-anchor", upper_anchor
    )
    return lower + progress * (upper - lower)
