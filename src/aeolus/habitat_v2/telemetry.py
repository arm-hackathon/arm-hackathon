from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .scenario import SCENARIO_SCHEMA_VERSION_V5, Scenario, ScenarioValidationError

OBSERVABLE_TOPOLOGY_SCHEMA_V1 = "aeolus_habitat_v2_observable_topology_v1"
ENVIRONMENTAL_CHANNELS = (
    "temperature_k",
    "pressure_pa",
    "co2_ppm",
    "o2_mole_fraction",
    "relative_humidity",
)
OPERATIONAL_FEEDBACK_CHANNELS = (
    "fan_speed_fraction",
    "fan_dc_bus_current_a",
    "damper_position_by_id",
    "branch_airflow_m3_s",
    "branch_differential_pressure_pa",
    "scrubber_capture_rate_mol_s",
    "condenser_removal_rate_mol_s",
    "cooling_delivery_w",
    "oxygen_delivery_mol_s",
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_remaining_fraction",
)
OPERATIONAL_RESOURCE_GAUGE_CHANNELS = (
    "battery_state_of_charge",
    "oxygen_store_fraction",
    "sorbent_remaining_fraction",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ObservableTopology:
    schema_version: str
    zone_ids: tuple[str, ...]
    fan_id: str
    branch_pairs: tuple[tuple[str, str], ...]
    environmental_channels: tuple[str, ...]
    operational_feedback_channels: tuple[str, ...]
    operational_resource_gauge_channels: tuple[str, ...]
    canonical_bytes: bytes
    sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def derive_observable_topology(scenario: Scenario) -> ObservableTopology:
    if type(scenario) is not Scenario:
        raise ScenarioValidationError(
            "observable topology requires the exact Scenario type"
        )
    if scenario.scenario_schema_version != SCENARIO_SCHEMA_VERSION_V5:
        raise ScenarioValidationError("observable topology requires a V5 scenario")
    zone_ids = tuple(sorted(str(zone["id"]) for zone in scenario.data["zones"]))
    if len(zone_ids) != len(set(zone_ids)):
        raise ScenarioValidationError("observable topology contains duplicate zone ids")
    network = scenario.data["air_network"]
    fan_id = str(network["fan"]["id"])
    branch_pairs = tuple(
        sorted(
            (str(branch["zone_id"]), str(branch["damper_id"]))
            for branch in network["branches"]
        )
    )
    if {zone_id for zone_id, _ in branch_pairs} != set(zone_ids):
        raise ScenarioValidationError("observable branch topology does not match zones")
    if len({damper_id for _, damper_id in branch_pairs}) != len(branch_pairs):
        raise ScenarioValidationError(
            "observable topology contains duplicate damper ids"
        )
    content = {
        "schema_version": OBSERVABLE_TOPOLOGY_SCHEMA_V1,
        "zone_ids": list(zone_ids),
        "fan_id": fan_id,
        "branches": [
            {"zone_id": zone_id, "damper_id": damper_id}
            for zone_id, damper_id in branch_pairs
        ],
        "environmental_channels": list(ENVIRONMENTAL_CHANNELS),
        "operational_feedback_channels": list(OPERATIONAL_FEEDBACK_CHANNELS),
        "operational_resource_gauge_channels": list(
            OPERATIONAL_RESOURCE_GAUGE_CHANNELS
        ),
    }
    canonical = _canonical_bytes(content)
    return ObservableTopology(
        schema_version=OBSERVABLE_TOPOLOGY_SCHEMA_V1,
        zone_ids=zone_ids,
        fan_id=fan_id,
        branch_pairs=branch_pairs,
        environmental_channels=ENVIRONMENTAL_CHANNELS,
        operational_feedback_channels=OPERATIONAL_FEEDBACK_CHANNELS,
        operational_resource_gauge_channels=OPERATIONAL_RESOURCE_GAUGE_CHANNELS,
        canonical_bytes=canonical,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )
