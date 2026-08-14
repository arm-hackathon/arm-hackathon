from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_rejects_tampered_topology_bundle() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        ForecastProjectionError,
        forecast_layout,
    )

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    tampered_topology = replace(
        bundle.topology,
        zone_ids=tuple(reversed(bundle.topology.zone_ids)),
    )

    with pytest.raises(ForecastProjectionError, match="frozen contract bundle"):
        forecast_layout(replace(bundle, topology=tampered_topology))


def test_layout_hashes_bind_frozen_authority_identity() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import forecast_layout

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    original = forecast_layout(bundle)
    substituted = forecast_layout(replace(bundle, binding_sha256="0" * 64))

    assert substituted.input_manifest_sha256 != original.input_manifest_sha256
    assert substituted.target_manifest_sha256 != original.target_manifest_sha256


def _completed_pairs(bundle: object, count: int) -> list[tuple[object, object]]:
    from aeolus.habitat_v2.hmc import HabitatManagementComputer

    hmc = HabitatManagementComputer.reset(
        bundle.development_scenario, bundle.hmc_contract, b"f" * 32
    )
    pairs: list[tuple[object, object]] = []
    for _ in range(count):
        observed = hmc.observe()
        assert isinstance(observed, tuple)
        snapshot, receipt = observed
        hmc.verify_snapshot(snapshot, receipt)
        if snapshot.to_mapping()["completed_step"] > 0:
            pairs.append((snapshot, receipt))
        handle = hmc.verify_snapshot(snapshot, receipt)
        hmc.propose(None, handle)
        hmc.arbitrate()
        hmc.step()
    return pairs


def test_projects_completed_issued_history_and_catalogue_action() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        project_history_window,
        project_proposed_action,
    )

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    history = project_history_window(
        bundle, _completed_pairs(bundle, 5), window_steps=4
    )
    action = project_proposed_action(bundle, bundle.actions[0].command)

    assert history.numeric_f32.shape == (4, 194)
    assert history.status_f32.shape == (4, 167, 5)
    assert history.mode_f32.shape == (4, 4)
    assert history.health_f32.shape == (4, 4)
    assert history.alarm_lifecycle_f32.shape == (4, 287, 4)
    assert history.numeric_f32.dtype == np.float32
    assert action.shape == (27,)
    assert action.dtype == np.float32
    assert not history.numeric_f32.flags.writeable
    assert not action.flags.writeable


def test_rejects_valid_but_uncatalogued_requested_action() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        ForecastProjectionError,
        project_proposed_action,
    )

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    mapping = bundle.actions[0].command.to_mapping()
    mapping["fan_speed_fraction"] = float(mapping["fan_speed_fraction"]) + 0.001

    with pytest.raises(ForecastProjectionError, match="catalogue"):
        project_proposed_action(bundle, mapping)


def _project_tampered_snapshot(
    bundle: object,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    from aeolus.habitat_v2.forecast import projection

    pair = _completed_pairs(bundle, 2)[0]
    mapping = copy.deepcopy(pair[0].to_mapping())
    mutate(mapping)
    monkeypatch.setattr(projection, "_snapshot_mapping", lambda *_: mapping)
    projection.project_history_window(bundle, [pair], window_steps=1)


def test_rejects_reset_snapshot_row() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        ForecastProjectionError,
        project_history_window,
    )
    from aeolus.habitat_v2.hmc import HabitatManagementComputer

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    hmc = HabitatManagementComputer.reset(
        bundle.development_scenario,
        bundle.hmc_contract,
        b"r" * 32,
    )
    observed = hmc.observe()
    assert isinstance(observed, tuple)

    with pytest.raises(ForecastProjectionError, match="reset"):
        project_history_window(bundle, [observed], window_steps=1)


def test_rejects_noncontiguous_history() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        ForecastProjectionError,
        project_history_window,
    )

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    pairs = _completed_pairs(bundle, 6)
    gapped = [pairs[0], pairs[1], pairs[3], pairs[4]]

    with pytest.raises(ForecastProjectionError, match="contiguous"):
        project_history_window(bundle, gapped, window_steps=4)


@pytest.mark.parametrize("mutation", ["reorder", "duplicate"])
def test_rejects_reordered_or_duplicate_descriptor(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import ForecastProjectionError

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])

    def mutate(mapping: dict[str, object]) -> None:
        samples = mapping["primary_telemetry"]["samples"]
        if mutation == "reorder":
            samples[0], samples[1] = samples[1], samples[0]
        else:
            samples[1]["descriptor_id"] = samples[0]["descriptor_id"]

    with pytest.raises(ForecastProjectionError, match="descriptor ordering"):
        _project_tampered_snapshot(bundle, monkeypatch, mutate)


def test_rejects_duplicate_resource_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import ForecastProjectionError

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])

    def mutate(mapping: dict[str, object]) -> None:
        sample = mapping["operational_resource_gauges"]["samples"][0]
        sample["value"] = float(sample["value"]) + 0.01

    with pytest.raises(ForecastProjectionError, match="resource gauges"):
        _project_tampered_snapshot(bundle, monkeypatch, mutate)


def test_rejects_unknown_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import ForecastProjectionError

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])

    def mutate(mapping: dict[str, object]) -> None:
        mapping["active_operational_alarms"]["alarms"].append(
            {
                "alarm_id": "invented/alarm/critical",
                "family": "invented",
                "target": "alarm",
                "severity": "critical",
                "lifecycle": "ACTIVE",
            }
        )

    with pytest.raises(ForecastProjectionError, match="unknown or duplicate"):
        _project_tampered_snapshot(bundle, monkeypatch, mutate)


def test_rejects_noncompleted_command_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import ForecastProjectionError

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])

    def mutate(mapping: dict[str, object]) -> None:
        mapping["command_reference"]["command_reference_kind"] = "RESET_REFERENCE"

    with pytest.raises(ForecastProjectionError, match="completed final command"):
        _project_tampered_snapshot(bundle, monkeypatch, mutate)


@pytest.mark.parametrize("value", [float("nan"), 1.0e39])
def test_rejects_nonfinite_or_float32_overflow(
    value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import ForecastProjectionError

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])

    def mutate(mapping: dict[str, object]) -> None:
        mapping["primary_telemetry"]["samples"][0]["value"] = value

    with pytest.raises(ForecastProjectionError, match="finite|overflows"):
        _project_tampered_snapshot(bundle, monkeypatch, mutate)


def test_rejects_bad_issued_snapshot_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        ForecastProjectionError,
        project_history_window,
    )
    from aeolus.habitat_v2.snapshot import OperationalSnapshot

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    pair = _completed_pairs(bundle, 2)[0]
    original = OperationalSnapshot.to_mapping

    def tampered(snapshot: OperationalSnapshot) -> dict[str, object]:
        mapping = original(snapshot)
        mapping["snapshot_sha256"] = "0" * 64
        return mapping

    monkeypatch.setattr(OperationalSnapshot, "to_mapping", tampered)
    with pytest.raises(ForecastProjectionError, match="canonical bytes|self hash"):
        project_history_window(bundle, [pair], window_steps=1)


def test_projects_physical_targets_in_frozen_order() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.projection import (
        forecast_layout,
        project_physical_targets,
    )
    from aeolus.habitat_v2.physics import advance_one_step_with_command, initial_state

    bundle = load_forecast_contracts(Path(__file__).resolve().parents[2])
    state = initial_state(bundle.development_scenario)
    states = []
    for _ in range(2):
        result = advance_one_step_with_command(
            bundle.development_scenario,
            state,
            bundle.actions[0].command.to_mapping(),
        )
        state = result.state
        states.append(state)

    targets = project_physical_targets(bundle, states, horizon_steps=2)
    first_zone = bundle.topology.zone_ids[0]
    zone_spec = next(
        item
        for item in bundle.development_scenario.data["zones"]
        if item["id"] == first_zone
    )
    telemetry = (
        states[0].zones[first_zone].telemetry(volume_m3=float(zone_spec["volume_m3"]))
    )
    expected_first_six = [
        telemetry["temperature_k"],
        telemetry["pressure_pa"],
        telemetry["co2_ppm"],
        telemetry["o2_mole_fraction"],
        telemetry["relative_humidity"],
        states[0].utility.actual_airflow_m3_s[first_zone],
    ]
    layout = forecast_layout(bundle)

    assert targets.shape == (2, 51)
    np.testing.assert_allclose(targets[0, :6], expected_first_six, rtol=1e-6)
    assert layout.target_descriptors[0]["descriptor_id"] == (
        f"{first_zone}/temperature_k"
    )
    assert layout.target_descriptors[-1]["descriptor_id"] == (
        "sorbent_remaining_fraction"
    )
    assert not targets.flags.writeable
