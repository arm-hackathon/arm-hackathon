from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

GAS_CONSTANT_J_PER_MOL_K = 8.314_462_618_153_24


def saturation_vapor_pressure_pa(temperature_k: float) -> float:
    """Murphy-Koop liquid-water vapour-pressure approximation in pascals."""
    log_temperature = math.log(temperature_k)
    correction = math.tanh(0.0415 * (temperature_k - 218.8)) * (
        53.878
        - (1331.22 / temperature_k)
        - (9.44523 * log_temperature)
        + (0.014025 * temperature_k)
    )
    log_pressure = (
        54.842763
        - (6763.22 / temperature_k)
        - (4.210 * log_temperature)
        + (0.000367 * temperature_k)
        + correction
    )
    return math.exp(log_pressure)


@dataclass(frozen=True)
class ZoneState:
    co2_mol: float
    o2_mol: float
    water_vapor_mol: float
    inert_mol: float
    temperature_k: float

    @property
    def total_moles(self) -> float:
        return self.co2_mol + self.o2_mol + self.water_vapor_mol + self.inert_mol

    def telemetry(self, *, volume_m3: float) -> dict[str, float]:
        total_moles = self.total_moles
        pressure_pa = (
            total_moles * GAS_CONSTANT_J_PER_MOL_K * self.temperature_k / volume_m3
        )
        water_partial_pressure_pa = (self.water_vapor_mol / total_moles) * pressure_pa
        relative_humidity = water_partial_pressure_pa / saturation_vapor_pressure_pa(
            self.temperature_k
        )
        return {
            "temperature_k": self.temperature_k,
            "pressure_pa": pressure_pa,
            "co2_ppm": 1_000_000.0 * self.co2_mol / total_moles,
            "o2_mole_fraction": self.o2_mol / total_moles,
            "relative_humidity": relative_humidity,
            "water_vapor_mol": self.water_vapor_mol,
            "co2_mol": self.co2_mol,
            "o2_mol": self.o2_mol,
            "inert_mol": self.inert_mol,
        }


@dataclass(frozen=True)
class UtilityState:
    co2_sorbent_remaining_mol: float
    captured_co2_mol: float
    condensed_water_mol: float
    oxygen_store_mol: float
    battery_energy_wh: float
    actual_airflow_m3_s: Mapping[str, float]
    actual_scrubber_duty: float
    actual_condenser_duty: float
    external_heat_rejected_j: float
    external_heat_received_j: float
    actual_fan_speed_fraction: float | None = None
    actual_damper_position_by_id: Mapping[str, float] = field(default_factory=dict)
    actual_cooling_removed_w: Mapping[str, float] = field(default_factory=dict)
    actual_oxygen_injection_mol_s: Mapping[str, float] = field(default_factory=dict)
    effective_scrubber_capture_ability: float = 1.0
    effective_condenser_removal_ability: float = 1.0
    effective_cooling_delivery_by_zone: Mapping[str, float] = field(default_factory=dict)
    effective_oxygen_delivery_by_zone: Mapping[str, float] = field(default_factory=dict)
    last_operational_feedback: Mapping[str, object] | None = None

    @property
    def achieved_cooling_removed_w(self) -> Mapping[str, float]:
        """V5 name for the immutable cooling actuator state."""

        return self.actual_cooling_removed_w

    @property
    def achieved_oxygen_injection_mol_s(self) -> Mapping[str, float]:
        """V5 name for the immutable oxygen actuator state."""

        return self.actual_oxygen_injection_mol_s


@dataclass(frozen=True)
class PlantState:
    step: int
    zones: Mapping[str, ZoneState]
    utility: UtilityState
