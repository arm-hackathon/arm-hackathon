from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _measurements():
    from aeolus.habitat_v2.forecast.pilot_benchmark import RunMeasurement

    return (
        RunMeasurement(
            wall_time_seconds=10.0,
            peak_rss_bytes=500_000_000,
            artifact_bytes=200_000,
        ),
        RunMeasurement(
            wall_time_seconds=12.0,
            peak_rss_bytes=600_000_000,
            artifact_bytes=220_000,
        ),
    )


def _ceilings(**overrides):
    from aeolus.habitat_v2.forecast.pilot_benchmark import BenchmarkCeilings

    values = {
        "wall_time_seconds": 1_000_000.0,
        "peak_rss_bytes": 4_000_000_000,
        "artifact_bytes": 500_000_000_000,
        "disk_reserve_bytes": 1_000_000_000,
    }
    values.update(overrides)
    return BenchmarkCeilings(**values)


def test_preflight_receipt_round_trips_through_pinned_loader(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pilot import load_resource_preflight
    from aeolus.habitat_v2.forecast.pilot_benchmark import (
        build_preflight_receipt,
        write_preflight_receipt,
    )

    receipt = build_preflight_receipt(
        measurements=_measurements(),
        ceilings=_ceilings(),
        free_disk_bytes=600_000_000_000,
    )
    assert receipt["schema_version"] == (
        "aeolus_habitat_v2_forecast_pilot_resource_preflight_v1"
    )
    assert receipt["planned_hmc_runs"] == 23_400
    assert receipt["benchmark_hmc_runs"] == 2
    assert receipt["measured_wall_time_seconds"] == pytest.approx(22.0)
    assert receipt["measured_peak_rss_bytes"] == 600_000_000
    assert receipt["measured_artifact_bytes"] == 420_000
    assert receipt["projected_wall_time_seconds"] == pytest.approx(22.0 / 2 * 23_400)
    assert receipt["projected_peak_rss_bytes"] == 600_000_000
    assert receipt["projected_artifact_bytes"] == 420_000 * 23_400 // 2
    assert receipt["runtime_within_ceiling"] is True
    assert receipt["memory_within_ceiling"] is True
    assert receipt["disk_reserve_preserved"] is True
    assert receipt["verdict"] == "PASS"

    path = tmp_path / "preflight.json"
    write_preflight_receipt(path, receipt)
    loaded = load_resource_preflight(
        path,
        expected_preflight_sha256=receipt["preflight_sha256"],
        expected_preflight_bytes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert loaded.planned_hmc_runs == 23_400
    assert loaded.benchmark_hmc_runs == 2
    assert loaded.verdict == "PASS"


def test_failed_ceiling_receipt_is_rejected_by_loader(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        PilotContractError,
        load_resource_preflight,
    )
    from aeolus.habitat_v2.forecast.pilot_benchmark import (
        build_preflight_receipt,
        write_preflight_receipt,
    )

    receipt = build_preflight_receipt(
        measurements=_measurements(),
        ceilings=_ceilings(wall_time_seconds=1.0),
        free_disk_bytes=600_000_000_000,
    )
    assert receipt["runtime_within_ceiling"] is False
    assert receipt["verdict"] != "PASS"

    path = tmp_path / "failed.json"
    write_preflight_receipt(path, receipt)
    with pytest.raises(PilotContractError, match="does not authorize"):
        load_resource_preflight(
            path,
            expected_preflight_sha256=receipt["preflight_sha256"],
            expected_preflight_bytes_sha256=hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        )


def test_benchmark_refuses_empty_measurements_and_overwrite(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pilot_benchmark import (
        PilotBenchmarkError,
        build_preflight_receipt,
        write_preflight_receipt,
    )

    with pytest.raises(PilotBenchmarkError, match="at least one"):
        build_preflight_receipt(
            measurements=(),
            ceilings=_ceilings(),
            free_disk_bytes=600_000_000_000,
        )
    receipt = build_preflight_receipt(
        measurements=_measurements(),
        ceilings=_ceilings(),
        free_disk_bytes=600_000_000_000,
    )
    path = tmp_path / "existing.json"
    write_preflight_receipt(path, receipt)
    with pytest.raises(PilotBenchmarkError, match="already exists"):
        write_preflight_receipt(path, receipt)



def test_measure_continuations_returns_real_replay_verified_metrics() -> None:
    from aeolus.habitat_v2.forecast.pilot import (
        iter_pilot_continuations,
        load_approved_pilot_design,
    )
    from aeolus.habitat_v2.forecast.pilot_benchmark import (
        build_preflight_receipt,
        measure_continuations,
    )

    design = load_approved_pilot_design(ROOT)
    continuations = iter_pilot_continuations(design)
    control = next(continuations)
    action = next(
        item for item in continuations if item.variant == "ACTION_PROPOSAL"
    )

    measurements = measure_continuations(ROOT, design, (control, action))

    assert len(measurements) == 2
    for measurement in measurements:
        assert measurement.wall_time_seconds > 0
        assert measurement.peak_rss_bytes > 0
        assert measurement.artifact_bytes > 0
    receipt = build_preflight_receipt(
        measurements=measurements,
        ceilings=_ceilings(),
        free_disk_bytes=600_000_000_000,
    )
    assert receipt["benchmark_hmc_runs"] == 2
    assert receipt["verdict"] == "PASS"
