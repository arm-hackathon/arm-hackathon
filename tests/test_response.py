"""Bounded-recovery-governor unit tests."""

import copy
import math
from types import SimpleNamespace

import pytest

from aeolus.config import load_scenario
from aeolus.model_input import build_model_input_contract, model_input_v1
from aeolus.response import BoundedRecoveryGovernor, ResponseSettings
from aeolus.scenario import RunSpec, run_governed_scenario, run_scenario


@pytest.fixture
def config(standard_scenario_path):
    return load_scenario(standard_scenario_path)


@pytest.fixture
def base_vectors(config, standard_scenario_path):
    """Real nominal model-input vectors reused as synthetic building blocks."""
    records = run_scenario(config)
    contract = build_model_input_contract(config)
    return [model_input_v1(record, contract).tolist() for record in records]


def _index(contract):
    return {field: i for i, field in enumerate(contract.fields)}


def _with_vector(vector, index, value):
    altered = copy.copy(vector)
    altered[index] = value
    return altered


class _WarmupProbeGovernor:
    def __init__(self, config, settings=None):
        self.observation_count = 0
        self._commands = {
            zone.id: 0.0 for zone in config.non_processing_zones()
        }
        if settings is not None:
            self.settings = settings

    def reset(self) -> None:
        self.observation_count = 0

    def observe(self, vector: list[float]) -> None:
        self.observation_count += 1

    def next_commands(self) -> tuple[dict[str, float], dict[str, object]]:
        return dict(self._commands), {}


class TestSettingsValidation:
    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="window_ticks"):
            ResponseSettings(window_ticks=0)

    def test_rejects_non_finite_threshold(self):
        with pytest.raises(ValueError, match="0.0..1.0"):
            ResponseSettings(degraded_residual_threshold=math.nan)

    def test_rejects_small_persistence(self):
        with pytest.raises(ValueError, match="persistence_ticks"):
            ResponseSettings(degradation_persistence_ticks=1)


class TestGovernorLifecycle:
    def test_observe_rejects_bad_vector(self, config):
        governor = BoundedRecoveryGovernor(config)
        with pytest.raises(ValueError, match="model-input"):
            governor.observe([math.nan])

    def test_observe_rejects_wrong_width(self, config, base_vectors):
        governor = BoundedRecoveryGovernor(config)
        vector = base_vectors[10]
        with pytest.raises(ValueError, match="model-input"):
            governor.observe(vector[:-1])

    def test_reset_clears_history(self, config, base_vectors):
        governor = BoundedRecoveryGovernor(config)
        for vector in base_vectors[:12]:
            governor.observe(vector)
        governor.next_commands()
        assert governor.command_history
        governor.reset()
        assert governor.command_history == []
        assert governor.rationale_history == []

    @pytest.mark.parametrize(
        "settings",
        [
            None,
            object(),
            SimpleNamespace(window_ticks=-2),
            SimpleNamespace(window_ticks=True),
        ],
    )
    def test_alternative_governor_uses_run_warmup_for_missing_or_bad_settings(
        self, config, settings
    ):
        run = RunSpec(
            total_ticks=2, warmup_ticks=4, crew_cabin_co2_concentration_ceiling=0.30
        )
        governor = _WarmupProbeGovernor(config, settings=settings)

        records = run_governed_scenario(config, governor, run=run)

        assert len(records) == run.total_ticks
        assert governor.observation_count == run.warmup_ticks + run.total_ticks


class TestBoundedness:
    def test_commands_always_in_unit_interval(self, config, standard_scenario_path):
        records = run_governed_scenario(config, BoundedRecoveryGovernor(config))
        for record in records:
            for zone_id, actuator in record.actuators.items():
                assert 0.0 <= actuator["setpoint"] <= 1.0, zone_id

    def test_commands_are_rate_limited(self, config, base_vectors):
        settings = ResponseSettings(max_command_delta=0.05)
        governor = BoundedRecoveryGovernor(config, settings=settings)
        previous = None
        for vector in base_vectors:
            governor.observe(vector)
            commands, _ = governor.next_commands()
            if previous is not None:
                for zone_id in governor._zone_ids:
                    assert abs(commands[zone_id] - previous[zone_id]) <= 0.05 + 1e-12
            previous = commands

    def test_deterministic_across_runs(self, config, standard_scenario_path):
        first = run_governed_scenario(config, BoundedRecoveryGovernor(config))
        second = run_governed_scenario(config, BoundedRecoveryGovernor(config))
        assert first == second

    def test_frozen_hold_respects_rate_limit_after_high_base(self, config, base_vectors):
        settings = ResponseSettings(max_command_delta=0.05, frozen_persistence_ticks=2)
        governor = BoundedRecoveryGovernor(config, settings=settings)
        zone_id = governor._zone_ids[0]
        reading_index = _contract_index(
            governor, "zones", zone_id, "sensor_co2_concentration"
        )
        low = _with_vector(base_vectors[0], reading_index, 0.0)
        high = _with_vector(base_vectors[1], reading_index, 1.0)

        governor.observe(low)
        governor.next_commands()
        governor.observe(high)
        previous, _ = governor.next_commands()
        governor.observe(high)
        commands, rationale = governor.next_commands()

        assert rationale[zone_id]["reason"] == "frozen_hold"
        assert abs(commands[zone_id] - previous[zone_id]) <= settings.max_command_delta + 1e-12
        assert rationale[zone_id]["commanded"] == commands[zone_id]


class TestPolicyRules:
    def test_frozen_hold(self, config, base_vectors):
        contract = build_model_input_contract(config)
        governor = BoundedRecoveryGovernor(config, settings=ResponseSettings())
        zone_id = governor._zone_ids[0]
        reading_index = next(
            i
            for i, field in enumerate(contract.fields)
            if field.group == "zones"
            and field.entity_id == zone_id
            and field.field == "sensor_co2_concentration"
        )
        for vector in base_vectors[:12]:
            governor.observe(vector)
        commands, rationale = governor.next_commands()
        baseline_before = commands[zone_id]
        frozen = _with_vector(base_vectors[12], reading_index, 0.30)
        for _ in range(governor.settings.frozen_persistence_ticks):
            governor.observe(_with_vector(frozen, reading_index, value=0.30))
        commands, rationale = governor.next_commands()
        assert rationale[zone_id]["reason"] == "frozen_hold"
        assert abs(commands[zone_id] - baseline_before) <= 1e-9

    def test_degraded_spare_release_below_pressure(self, config, base_vectors):
        settings = ResponseSettings()
        governor = BoundedRecoveryGovernor(config, settings=settings)
        zone_id = governor._zone_ids[0]
        outbound = config.path_to_processing(zone_id)
        requested_index = _contract_index(
            governor, "connections", outbound.id, "requested_airflow"
        )
        residual_index = _contract_index(
            governor, "connections", outbound.id, "airflow_residual"
        )
        reading_index = _contract_index(
            governor, "zones", zone_id, "sensor_co2_concentration"
        )
        for vector in base_vectors[:8]:
            governor.observe(vector)
        release_vectors = []
        for index, vector in enumerate(base_vectors[8:14]):
            altered = _with_vector(vector, requested_index, 0.5)
            altered = _with_vector(altered, residual_index, 0.4)
            altered = _with_vector(altered, reading_index, 0.05 + 0.002 * index)
            release_vectors.append(altered)
        for vector in release_vectors:
            governor.observe(vector)
        commands, rationale = governor.next_commands()
        assert rationale[zone_id]["reason"] == "degraded_spare_release"
        assert commands[zone_id] < rationale[zone_id]["base_command"]
        assert 0.0 < rationale[zone_id]["estimated_loss"] < 1.0
        for other in governor._zone_ids[1:]:
            assert rationale[other]["reason"] == "nominal"

    def test_degraded_zone_under_pressure_keeps_full_demand(self, config, base_vectors):
        settings = ResponseSettings()
        governor = BoundedRecoveryGovernor(config, settings=settings)
        zone_id = governor._zone_ids[0]
        outbound = config.path_to_processing(zone_id)
        requested_index = _contract_index(
            governor, "connections", outbound.id, "requested_airflow"
        )
        residual_index = _contract_index(
            governor, "connections", outbound.id, "airflow_residual"
        )
        reading_index = _contract_index(
            governor, "zones", zone_id, "sensor_co2_concentration"
        )
        for vector in base_vectors[:8]:
            governor.observe(vector)
        pressure_vectors = []
        for index, vector in enumerate(base_vectors[8:14]):
            altered = _with_vector(vector, requested_index, 0.5)
            altered = _with_vector(altered, residual_index, 0.4)
            altered = _with_vector(altered, reading_index, 0.25 + 0.002 * index)
            pressure_vectors.append(altered)
        for vector in pressure_vectors:
            governor.observe(vector)
        commands, rationale = governor.next_commands()
        assert rationale[zone_id]["reason"] == "nominal"
        assert commands[zone_id] >= rationale[zone_id]["base_command"]


def _contract_index(governor, group, entity_id, field):
    return governor._index[(group, entity_id, field)]