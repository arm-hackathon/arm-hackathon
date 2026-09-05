"""Independent numerical reference checks for Habitat V2 physics (Issue #71).

Every expected value in this file is re-derived from first principles with
hand-entered constants and float64 arithmetic written inside the test. The
production helpers are called only to obtain the values under test; no
expected value is produced by the production code being checked. Tolerances
follow `conservation_tolerances` in
`contracts/habitat_v2_physics_provenance_v1.json`: production physics runs in
float64, so agreement is asserted at 1e-9 absolute / 1e-12 relative levels,
which cover summation-order differences only.
"""

from __future__ import annotations

import math

import pytest

from aeolus.habitat_v2.air_network import (
    AirNetworkSpec,
    BranchSpec,
    FanSpec,
    solve_air_network,
)
from aeolus.habitat_v2.actuators import achieve_actuator_state
from aeolus.habitat_v2.physics import (
    InfeasibleActionError,
    _apply_passive_condensation,
    _apply_zone_thermal_balance,
    _electrical_balance,
    _recirculate,
    _recirculation_heat_transfer_j,
)
from aeolus.habitat_v2.state import ZoneState

# Constants re-entered by hand from CODATA and Murphy & Koop (2005); these
# literals are the independent oracle, not imports of production values.
R_GAS_ORACLE = 8.31446261815324


def murphy_koop_psat_oracle(t_k: float) -> float:
    ln_t = math.log(t_k)
    correction = math.tanh(0.0415 * (t_k - 218.8)) * (
        53.878 - 1331.22 / t_k - 9.44523 * ln_t + 0.014025 * t_k
    )
    ln_p = 54.842763 - 6763.22 / t_k - 4.210 * ln_t + 0.000367 * t_k + correction
    return math.exp(ln_p)


def _zone(co2: float, o2: float, water: float, inert: float, t_k: float) -> ZoneState:
    return ZoneState(
        co2_mol=co2, o2_mol=o2, water_vapor_mol=water, inert_mol=inert, temperature_k=t_k
    )


def test_ideal_gas_and_telemetry_oracle() -> None:
    volume = 40.0
    temperature = 296.0
    zone = _zone(8.0, 60.0, 2.0, 130.0, temperature)
    telemetry = zone.telemetry(volume_m3=volume)

    n_total = 8.0 + 60.0 + 2.0 + 130.0
    expected_pressure = n_total * R_GAS_ORACLE * temperature / volume
    water_partial = (2.0 / n_total) * expected_pressure
    expected_rh = water_partial / murphy_koop_psat_oracle(temperature)
    expected_co2_ppm = 1.0e6 * 8.0 / n_total

    assert telemetry["pressure_pa"] == pytest.approx(expected_pressure, rel=1e-12)
    assert telemetry["co2_ppm"] == pytest.approx(expected_co2_ppm, rel=1e-12)
    assert telemetry["o2_mole_fraction"] == pytest.approx(60.0 / n_total, rel=1e-12)
    assert telemetry["relative_humidity"] == pytest.approx(expected_rh, rel=1e-10)


def test_well_mixed_exchange_oracle() -> None:
    zones = {
        "alpha": _zone(10.0, 70.0, 3.0, 150.0, 295.0),
        "beta": _zone(4.0, 40.0, 1.0, 90.0, 297.0),
    }
    configs = {"alpha": {"volume_m3": 40.0}, "beta": {"volume_m3": 25.0}}
    airflow = {"alpha": 0.05, "beta": 0.02}
    equipment = {
        "scrubber_max_co2_mol_s": 0.004,
        "condenser_max_water_mol_s": 0.012,
    }
    dt = 60.0

    mixed, receipt = _recirculate(
        zones,
        zone_configs=configs,
        airflow_m3_s=airflow,
        scrubber_duty=0.0,
        condenser_duty=0.0,
        equipment=equipment,
        sorbent_remaining_mol=1000.0,
        dt_seconds=dt,
    )

    species = ("co2_mol", "o2_mol", "water_vapor_mol", "inert_mol")
    fraction_alpha = 1.0 - math.exp(-0.05 * dt / 40.0)
    fraction_beta = 1.0 - math.exp(-0.02 * dt / 25.0)
    pool = {name: 0.0 for name in species}
    extracted = {}
    for zone_id, fraction in (("alpha", fraction_alpha), ("beta", fraction_beta)):
        extracted[zone_id] = {
            name: getattr(zones[zone_id], name) * fraction for name in species
        }
        for name in species:
            pool[name] += extracted[zone_id][name]
    total_alpha = sum(extracted["alpha"].values())
    total_beta = sum(extracted["beta"].values())
    total = total_alpha + total_beta
    for zone_id, zone_total in (("alpha", total_alpha), ("beta", total_beta)):
        share = zone_total / total
        for name in species:
            expected = (
                getattr(zones[zone_id], name)
                - extracted[zone_id][name]
                + pool[name] * share
            )
            actual = getattr(mixed[zone_id], name)
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

    for name in species:
        assert receipt["species_residual_mol"][name] == pytest.approx(0.0, abs=1e-9)
    before = sum(getattr(z, name) for z in zones.values() for name in species)
    after = sum(getattr(z, name) for z in mixed.values() for name in species)
    assert after == pytest.approx(before, rel=1e-12)


def test_scrubber_capture_oracle() -> None:
    zones = {"alpha": _zone(2.0, 70.0, 0.5, 150.0, 295.0)}
    configs = {"alpha": {"volume_m3": 40.0}}
    equipment = {
        "scrubber_max_co2_mol_s": 0.004,
        "condenser_max_water_mol_s": 0.012,
    }
    dt = 60.0
    mixed, receipt = _recirculate(
        zones,
        zone_configs=configs,
        airflow_m3_s={"alpha": 0.05},
        scrubber_duty=0.5,
        condenser_duty=0.0,
        equipment=equipment,
        sorbent_remaining_mol=0.05,
        dt_seconds=dt,
    )
    fraction = 1.0 - math.exp(-0.05 * dt / 40.0)
    pool_co2 = 2.0 * fraction
    rate_limited = 0.004 * 0.5 * 1.0 * dt
    expected_capture = min(pool_co2, rate_limited, 0.05)
    assert receipt["co2_captured_mol"] == pytest.approx(expected_capture, rel=1e-12)
    assert mixed["alpha"].co2_mol == pytest.approx(
        2.0 - pool_co2 + (pool_co2 - expected_capture), rel=1e-12
    )


def test_passive_condensation_oracle() -> None:
    temperature = 295.15
    volume = 40.0
    saturated_mol = murphy_koop_psat_oracle(temperature) * volume / (
        R_GAS_ORACLE * temperature
    )
    zones = {
        "wet": _zone(5.0, 60.0, saturated_mol * 1.4, 120.0, temperature),
        "dry": _zone(5.0, 60.0, saturated_mol * 0.5, 120.0, temperature),
    }
    configs = {"wet": {"volume_m3": volume}, "dry": {"volume_m3": volume}}
    updated, condensed = _apply_passive_condensation(zones, zone_configs=configs)

    expected_condensed_wet = saturated_mol * 1.4 - saturated_mol
    assert condensed["wet"] == pytest.approx(expected_condensed_wet, rel=1e-12)
    assert updated["wet"].water_vapor_mol == pytest.approx(saturated_mol, rel=1e-12)
    assert condensed["dry"] == 0.0
    assert updated["dry"].water_vapor_mol == pytest.approx(
        saturated_mol * 0.5, rel=1e-12
    )
    assert updated["wet"].temperature_k == temperature


def test_lumped_thermal_oracle() -> None:
    zones = {
        "cold": _zone(5.0, 60.0, 1.0, 120.0, 294.0),
        "warm": _zone(5.0, 60.0, 1.0, 120.0, 297.0),
    }
    airflow = {"cold": 0.05, "warm": 0.03}
    equipment = {"air_density_kg_m3": 0.85, "air_specific_heat_j_kg_k": 1005.0}
    dt = 60.0

    transfers = _recirculation_heat_transfer_j(
        zones, airflow_m3_s=airflow, equipment=equipment, dt_seconds=dt
    )

    t_mix = (0.05 * 294.0 + 0.03 * 297.0) / 0.08
    q_cold = 0.85 * 1005.0 * 0.05 * (t_mix - 294.0) * dt
    q_warm = 0.85 * 1005.0 * 0.03 * (t_mix - 297.0) * dt
    assert q_cold + q_warm == pytest.approx(0.0, abs=1e-9)
    assert transfers["cold"] == pytest.approx(q_cold, rel=1e-12)
    assert transfers["warm"] == pytest.approx(q_warm, rel=1e-9, abs=1e-9)
    assert sum(transfers.values()) == pytest.approx(0.0, abs=1e-9)

    zero = _recirculation_heat_transfer_j(
        zones,
        airflow_m3_s={"cold": 0.0, "warm": 0.0},
        equipment=equipment,
        dt_seconds=dt,
    )
    assert zero == {"cold": 0.0, "warm": 0.0}


def test_zone_thermal_balance_oracle() -> None:
    zones = {"alpha": _zone(5.0, 60.0, 1.0, 120.0, 300.0)}
    configs = {
        "alpha": {
            "thermal_capacity_j_per_k": 8.0e6,
            "passive_thermal_conductance_w_per_k": 16.0,
            "sink_temperature_k": 280.0,
        }
    }
    loads = {"alpha": {"sensible_heat_w": 250.0}}
    cooling = {"alpha": 100.0}
    recirculation = {"alpha": 1500.0}
    dt = 60.0

    updated, receipt = _apply_zone_thermal_balance(
        zones,
        zone_configs=configs,
        loads=loads,
        cooling_removed_w=cooling,
        recirculation_heat_added_j=recirculation,
        dt_seconds=dt,
    )

    passive_signed = 16.0 * (280.0 - 300.0) * dt
    expected_delta_j = 250.0 * dt + 1500.0 + max(0.0, passive_signed) - max(
        0.0, -passive_signed
    ) - 100.0 * dt
    expected_temperature = 300.0 + expected_delta_j / 8.0e6
    assert updated["alpha"].temperature_k == pytest.approx(expected_temperature, rel=1e-12)
    zone_receipt = receipt["zones"]["alpha"]
    assert zone_receipt["zone_thermal_residual_j"] == pytest.approx(0.0, abs=1e-6)
    assert receipt["external_heat_rejected_j"] == pytest.approx(
        -passive_signed + 100.0 * dt, rel=1e-12
    )
    assert receipt["external_heat_received_j"] == 0.0


def test_air_network_operating_point_oracle_single_branch() -> None:
    spec = AirNetworkSpec(
        fan=FanSpec(
            component_id="fan",
            rated_free_delivery_m3_s=0.75,
            rated_shutoff_pressure_pa=900.0,
            total_efficiency=0.68,
        ),
        shared_resistance_pa_s2_m6=1000.0,
        air_density_kg_m3=0.85,
        branches=(
            BranchSpec(
                zone_id="alpha",
                damper_id="alpha_damper",
                open_supply_resistance_pa_s2_m6=8000.0,
                return_resistance_pa_s2_m6=2000.0,
                damper_leak_fraction=0.08,
            ),
        ),
    )
    speed = 0.8
    result = solve_air_network(
        spec, fan_speed_fraction=speed, damper_position_by_id={"alpha_damper": 0.5}
    )

    area_fraction = 0.08 + (1.0 - 0.08) * 0.5
    r_branch = 8000.0 / area_fraction**2 + 2000.0
    shutoff = 900.0 * speed**2
    free_flow = 0.75 * speed
    q_squared = shutoff / (1000.0 + r_branch + shutoff / free_flow**2)
    q_expected = math.sqrt(q_squared)
    p_branch_expected = r_branch * q_squared
    p_fan_expected = shutoff * (1.0 - q_squared / free_flow**2)

    assert result.total_flow_m3_s == pytest.approx(q_expected, rel=1e-9)
    assert result.branch_pressure_loss_pa["alpha"] == pytest.approx(
        p_branch_expected, rel=1e-9
    )
    assert result.fan_pressure_rise_pa == pytest.approx(p_fan_expected, rel=1e-9)
    assert result.shared_pressure_loss_pa == pytest.approx(
        1000.0 * q_squared, rel=1e-9
    )
    assert result.fan_air_power_w == pytest.approx(
        p_fan_expected * q_expected, rel=1e-9
    )
    assert result.fan_electrical_power_w == pytest.approx(
        p_fan_expected * q_expected / 0.68, rel=1e-9
    )
    assert result.operating_point_residual_pa == pytest.approx(0.0, abs=1e-6)
    assert result.zone_mass_flow_kg_s["alpha"] == pytest.approx(
        q_expected * 0.85, rel=1e-9
    )


def test_air_network_operating_point_oracle_two_branch() -> None:
    spec = AirNetworkSpec(
        fan=FanSpec(
            component_id="fan",
            rated_free_delivery_m3_s=0.75,
            rated_shutoff_pressure_pa=900.0,
            total_efficiency=0.68,
        ),
        shared_resistance_pa_s2_m6=450.0,
        air_density_kg_m3=0.85,
        branches=(
            BranchSpec("alpha", "d_a", 8000.0, 2000.0, 0.08),
            BranchSpec("beta", "d_b", 5500.0, 3000.0, 0.08),
        ),
    )
    speed = 1.0
    positions = {"d_a": 1.0, "d_b": 0.25}
    result = solve_air_network(
        spec, fan_speed_fraction=speed, damper_position_by_id=positions
    )

    area_a = 0.08 + 0.92 * 1.0
    area_b = 0.08 + 0.92 * 0.25
    r_a = 8000.0 / area_a**2 + 2000.0
    r_b = 5500.0 / area_b**2 + 3000.0
    shutoff = 900.0
    free_flow = 0.75

    def residual(p_branch: float) -> float:
        q_total = math.sqrt(p_branch / r_a) + math.sqrt(p_branch / r_b)
        p_fan = shutoff * max(0.0, 1.0 - (q_total / free_flow) ** 2)
        return p_fan - (450.0 * q_total**2 + p_branch)

    lo, hi = 0.0, shutoff
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    p_oracle = (lo + hi) / 2.0
    q_oracle = math.sqrt(p_oracle / r_a) + math.sqrt(p_oracle / r_b)

    assert result.total_flow_m3_s == pytest.approx(q_oracle, rel=1e-9)
    assert result.branch_pressure_loss_pa["alpha"] == pytest.approx(p_oracle, rel=1e-9)
    assert result.zone_flow_m3_s["alpha"] == pytest.approx(
        math.sqrt(p_oracle / r_a), rel=1e-9
    )
    assert result.zone_flow_m3_s["beta"] == pytest.approx(
        math.sqrt(p_oracle / r_b), rel=1e-9
    )

    stopped = solve_air_network(
        spec, fan_speed_fraction=0.0, damper_position_by_id=positions
    )
    assert stopped.total_flow_m3_s == 0.0
    assert stopped.fan_air_power_w == 0.0


def test_actuator_slew_oracle() -> None:
    achievement = achieve_actuator_state(
        current_cooling_removed_w={"alpha": 100.0},
        current_oxygen_injection_mol_s={"alpha": 0.0},
        requested_cooling_removed_w={"alpha": 400.0},
        requested_oxygen_injection_mol_s={"alpha": 0.001},
        cooling_slew_w_per_s=5.0,
        oxygen_slew_mol_s2=1.0e-5,
        dt_seconds=60.0,
    )
    assert achievement.cooling_removed_w["alpha"] == pytest.approx(400.0, rel=1e-12)
    assert achievement.oxygen_injection_mol_s["alpha"] == pytest.approx(
        6.0e-4, rel=1e-12
    )

    limited = achieve_actuator_state(
        current_cooling_removed_w={"alpha": 100.0},
        current_oxygen_injection_mol_s={"alpha": 0.0},
        requested_cooling_removed_w={"alpha": 900.0},
        requested_oxygen_injection_mol_s={"alpha": 0.0},
        cooling_slew_w_per_s=5.0,
        oxygen_slew_mol_s2=1.0e-5,
        dt_seconds=60.0,
    )
    assert limited.cooling_removed_w["alpha"] == pytest.approx(400.0, rel=1e-12)

    capped = achieve_actuator_state(
        current_cooling_removed_w={"a": 0.0, "b": 0.0},
        current_oxygen_injection_mol_s={"a": 0.0, "b": 0.0},
        requested_cooling_removed_w={"a": 0.0, "b": 0.0},
        requested_oxygen_injection_mol_s={"a": 0.001, "b": 0.001},
        cooling_slew_w_per_s=5.0,
        oxygen_slew_mol_s2=1.0e-4,
        dt_seconds=60.0,
        oxygen_max_total_mol_s=0.001,
    )
    total = math.fsum(capped.oxygen_injection_mol_s.values())
    assert total <= 0.001
    assert total == pytest.approx(0.001, rel=1e-9)


def test_electrical_balance_discharge_oracle() -> None:
    equipment = {
        "base_load_w": 800.0,
        "fan_power_w_per_m3_s": 1000.0,
        "scrubber_power_w_full": 700.0,
        "condenser_power_w_full": 600.0,
        "cooling_coefficient_of_performance": 3.0,
        "oxygen_injection_power_w_per_mol_s": 100000.0,
        "battery_capacity_wh": 20000.0,
        "battery_charge_efficiency": 0.95,
        "battery_discharge_efficiency": 0.95,
        "battery_max_charge_w": 4000.0,
        "battery_max_discharge_w": 5000.0,
    }
    command = {
        "cooling_removed_w": {"alpha": 300.0},
        "oxygen_injection_mol_s": {"alpha": 0.001},
    }
    hours = 60.0 / 3600.0
    served_wh = (
        800.0 * hours
        + 600.0 * hours
        + 700.0 * 0.5 * hours
        + 600.0 * 0.25 * hours
        + (300.0 / 3.0) * hours
        + 100000.0 * 0.001 * hours
    )
    generated_wh = 2000.0 * hours
    deficit_wh = served_wh - generated_wh
    withdrawn_wh = deficit_wh / 0.95
    expected_next = 10000.0 - withdrawn_wh

    next_energy, receipt = _electrical_balance(
        battery_energy_wh=10000.0,
        equipment=equipment,
        command=command,
        actual_airflow_m3_s={"alpha": 0.05},
        fan_electrical_power_w=600.0,
        actual_scrubber_duty=0.5,
        actual_condenser_duty=0.25,
        generation_w=2000.0,
        dt_seconds=60.0,
    )
    assert next_energy == pytest.approx(expected_next, rel=1e-12)
    assert receipt["served_load_wh"] == pytest.approx(served_wh, rel=1e-12)
    assert receipt["battery_withdrawn_wh"] == pytest.approx(withdrawn_wh, rel=1e-12)
    assert receipt["battery_charge_input_wh"] == 0.0
    assert receipt["residual_wh"] == pytest.approx(0.0, abs=1e-9)


def test_electrical_balance_charge_oracle() -> None:
    equipment = {
        "base_load_w": 800.0,
        "fan_power_w_per_m3_s": 1000.0,
        "scrubber_power_w_full": 700.0,
        "condenser_power_w_full": 600.0,
        "cooling_coefficient_of_performance": 3.0,
        "oxygen_injection_power_w_per_mol_s": 100000.0,
        "battery_capacity_wh": 20000.0,
        "battery_charge_efficiency": 0.95,
        "battery_discharge_efficiency": 0.95,
        "battery_max_charge_w": 4000.0,
        "battery_max_discharge_w": 5000.0,
    }
    command = {
        "cooling_removed_w": {"alpha": 300.0},
        "oxygen_injection_mol_s": {"alpha": 0.001},
    }
    hours = 60.0 / 3600.0
    served_wh = (
        800.0 * hours
        + 600.0 * hours
        + 700.0 * 0.5 * hours
        + 600.0 * 0.25 * hours
        + (300.0 / 3.0) * hours
        + 100000.0 * 0.001 * hours
    )
    generated_wh = 5000.0 * hours
    surplus_wh = generated_wh - served_wh
    max_input_wh = min(4000.0 * hours, (20000.0 - 10000.0) / 0.95)
    input_wh = min(surplus_wh, max_input_wh)
    stored_wh = input_wh * 0.95
    expected_next = 10000.0 + stored_wh

    next_energy, receipt = _electrical_balance(
        battery_energy_wh=10000.0,
        equipment=equipment,
        command=command,
        actual_airflow_m3_s={"alpha": 0.05},
        fan_electrical_power_w=600.0,
        actual_scrubber_duty=0.5,
        actual_condenser_duty=0.25,
        generation_w=5000.0,
        dt_seconds=60.0,
    )
    assert next_energy == pytest.approx(expected_next, rel=1e-12)
    assert receipt["battery_charge_input_wh"] == pytest.approx(input_wh, rel=1e-12)
    assert receipt["battery_charge_stored_wh"] == pytest.approx(stored_wh, rel=1e-12)
    assert receipt["charge_conversion_loss_wh"] == pytest.approx(
        input_wh - stored_wh, rel=1e-12
    )
    assert receipt["curtailed_generation_wh"] == pytest.approx(
        surplus_wh - input_wh, rel=1e-12, abs=1e-12
    )
    assert receipt["residual_wh"] == pytest.approx(0.0, abs=1e-9)


def test_electrical_balance_infeasible_oracle() -> None:
    equipment = {
        "base_load_w": 800.0,
        "fan_power_w_per_m3_s": 1000.0,
        "scrubber_power_w_full": 700.0,
        "condenser_power_w_full": 600.0,
        "cooling_coefficient_of_performance": 3.0,
        "oxygen_injection_power_w_per_mol_s": 100000.0,
        "battery_capacity_wh": 20000.0,
        "battery_charge_efficiency": 0.95,
        "battery_discharge_efficiency": 0.95,
        "battery_max_charge_w": 4000.0,
        "battery_max_discharge_w": 50.0,
    }
    command = {
        "cooling_removed_w": {"alpha": 300.0},
        "oxygen_injection_mol_s": {"alpha": 0.001},
    }
    hours = 60.0 / 3600.0
    served_wh = (
        800.0 * hours
        + 600.0 * hours
        + 350.0 * hours
        + 150.0 * hours
        + 100.0 * hours
        + 100.0 * hours
    )
    deficit_wh = served_wh - 100.0 * hours
    required_withdrawal_wh = deficit_wh / 0.95
    assert required_withdrawal_wh > min(5000.0, 50.0 * hours)
    with pytest.raises(InfeasibleActionError):
        _electrical_balance(
            battery_energy_wh=5000.0,
            equipment=equipment,
            command=command,
            actual_airflow_m3_s={"alpha": 0.05},
            fan_electrical_power_w=600.0,
            actual_scrubber_duty=0.5,
            actual_condenser_duty=0.25,
            generation_w=100.0,
            dt_seconds=60.0,
        )


def test_resource_depletion_monotone_oracle() -> None:
    zones = {"alpha": _zone(6.0, 70.0, 1.0, 150.0, 295.0)}
    configs = {"alpha": {"volume_m3": 40.0}}
    equipment = {
        "scrubber_max_co2_mol_s": 0.004,
        "condenser_max_water_mol_s": 0.012,
    }
    sorbent = 10.0
    total_captured = 0.0
    for _ in range(5):
        zones, receipt = _recirculate(
            zones,
            zone_configs=configs,
            airflow_m3_s={"alpha": 0.05},
            scrubber_duty=1.0,
            condenser_duty=1.0,
            equipment=equipment,
            sorbent_remaining_mol=sorbent,
            dt_seconds=60.0,
        )
        captured = receipt["co2_captured_mol"]
        assert captured >= 0.0
        sorbent -= captured
        total_captured += captured
        assert sorbent >= -1e-12
    assert sorbent == pytest.approx(10.0 - total_captured, rel=1e-12)
    assert sorbent < 10.0
