"""Four-arm deterministic recovery-evidence integration tests."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from aeolus.recovery_evidence import (
    RECOVERY_ARMS,
    _canonical_sha256,
    main,
    reproduce_recovery_evidence,
    run_recovery_evidence,
)
from aeolus.scenario import RunSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mini_recovery_sweep(tmp_path: Path) -> Path:
    base_name = "recovery_habitat.json"
    shutil.copy(REPO_ROOT / "scenarios" / base_name, tmp_path / base_name)
    telemetry = {
        "airflow_noise_fraction": 0.0,
        "airflow_bias_fraction": 0.0,
        "airflow_drift_fraction": 0.0,
        "actuator_position_noise_fraction": 0.0,
        "co2_sensor_noise_fraction": 0.0,
        "co2_sensor_bias_fraction": 0.0,
        "co2_sensor_drift_fraction": 0.0,
    }

    def split(seed: int) -> dict:
        return {
            "seeds": [seed],
            "fault_start_ticks": [5],
            "operating_profiles": [
                {
                    "id": "low-noise",
                    "source_multiplier": 1.0,
                    "shared_airflow_capacity": 24.0,
                    "telemetry": telemetry,
                }
            ],
            "gradual_profiles": [{"duration_ticks": 10, "end_effectiveness": 0.75}],
            "blocked_effectiveness": [0.65],
            "transient_blocked_profiles": [
                {"blocked_effectiveness": 0.65, "duration_ticks": 10}
            ],
            "transient_gradual_profiles": [
                {
                    "start_effectiveness": 1.0,
                    "end_effectiveness": 0.75,
                    "duration_ticks": 10,
                }
            ],
        }

    document = {
        "schema_version": "aeolus_sweep_v4",
        "base_scenario": base_name,
        "targets": ["cabin_a"],
        "suite_role": "development",
        "splits": {"train": split(211), "validation": split(601)},
    }
    path = tmp_path / "mini-recovery.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_recovery_evidence_runs_exactly_four_write_once_arms_per_family(tmp_path):
    receipt = run_recovery_evidence(
        _mini_recovery_sweep(tmp_path),
        tmp_path / "evidence",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )

    assert RECOVERY_ARMS == (
        "reference_reserve_off",
        "reference_governed",
        "fault_reserve_off",
        "fault_governed",
    )
    assert receipt["families_evaluated"] == 10
    assert len(receipt["per_family"]) == 10
    for family in receipt["per_family"]:
        assert tuple(family["arms"]) == RECOVERY_ARMS
        for arm in family["arms"].values():
            assert len(arm["trace_sha256"]) == 64
            assert set(arm["metrics"]) >= {
                "primary_requested_airflow_integral",
                "primary_delivered_airflow_integral",
                "reserve_delivered_airflow_integral",
                "integrated_physical_co2",
                "states",
                "invariant_violation_count",
            }


def test_recovery_evidence_emits_paired_safety_and_benefit_gates(tmp_path):
    receipt = run_recovery_evidence(
        _mini_recovery_sweep(tmp_path),
        tmp_path / "evidence",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )

    safety = receipt["gates"]["safety"]
    benefit = receipt["gates"]["benefit"]
    assert isinstance(safety["passed"], bool)
    assert isinstance(benefit["passed"], bool)
    assert set(safety) >= {
        "zero_invariant_violations",
        "reserve_off_zero_delivery",
        "healthy_governed_no_protect",
        "frozen_sensor_no_protect",
        "preactivation_physical_parity",
        "transient_handback_acknowledged",
        "failed_reserve_no_rearm",
    }
    assert set(benefit) >= {
        "physical_reserve_delivery_for_benefit",
        "median_excess_improvement",
        "validation_improvement_fraction",
        "median_total_delivery_non_regression",
        "healthy_reference_non_regression",
    }
    for family in receipt["per_family"]:
        assert family["paired_metrics"]["integrated_excess_improvement_fraction"][
            "status"
        ] in {"defined", "undefined_zero_denominator"}
        assert family["paired_metrics"]["steady_state_restoration_fraction"][
            "status"
        ] in {"defined", "not_applicable", "undefined_zero_denominator"}


def test_recovery_evidence_counts_full_reserve_request_as_saturation(tmp_path):
    sweep_path = _mini_recovery_sweep(tmp_path)
    document = json.loads(sweep_path.read_text(encoding="utf-8"))
    for split in document["splits"].values():
        split["blocked_effectiveness"] = [0.01]
    sweep_path.write_text(json.dumps(document), encoding="utf-8")

    receipt = run_recovery_evidence(
        sweep_path,
        tmp_path / "saturated",
        run=RunSpec(
            total_ticks=80,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )
    blocked = next(
        family
        for family in receipt["per_family"]
        if family["fault_class"] == "blocked_path"
        and family["split"] == "validation"
    )
    assert blocked["arms"]["fault_governed"]["metrics"]["reserve_saturation_ticks"] > 0


def test_recovery_evidence_handles_schema_valid_high_noise_and_drift(tmp_path):
    sweep_path = _mini_recovery_sweep(tmp_path)
    document = json.loads(sweep_path.read_text(encoding="utf-8"))
    for split in document["splits"].values():
        for profile in split["operating_profiles"]:
            profile["telemetry"] = {
                "airflow_noise_fraction": 0.75,
                "airflow_bias_fraction": 0.75,
                "airflow_drift_fraction": 0.75,
                "actuator_position_noise_fraction": 0.75,
                "co2_sensor_noise_fraction": 0.75,
                "co2_sensor_bias_fraction": 0.75,
                "co2_sensor_drift_fraction": 0.75,
            }
    sweep_path.write_text(json.dumps(document), encoding="utf-8")

    receipt = run_recovery_evidence(
        sweep_path,
        tmp_path / "high-noise",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )

    assert receipt["families_evaluated"] == 10
    assert isinstance(receipt["gates"]["safety"]["passed"], bool)
    assert isinstance(receipt["gates"]["benefit"]["passed"], bool)


def test_recovery_evidence_reports_zero_shortfall_denominator_explicitly(tmp_path):
    receipt = run_recovery_evidence(
        _mini_recovery_sweep(tmp_path),
        tmp_path / "zero-denominator",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )

    metrics = receipt["per_family"][0]["arms"]["reference_reserve_off"]["metrics"]
    assert metrics["reserve_shortfall_coverage_fraction"] == {
        "status": "undefined_zero_denominator",
        "value": None,
    }


def test_recovery_evidence_duplicate_relocation_is_byte_identical(tmp_path):
    comparison = reproduce_recovery_evidence(
        _mini_recovery_sweep(tmp_path),
        tmp_path / "first",
        tmp_path / "second",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )

    assert comparison["byte_identical"] is True
    assert comparison["first_evidence_sha256"] == comparison["second_evidence_sha256"]
    assert comparison["trace_count"] == 40


def test_recovery_evidence_refuses_existing_output_without_mutation(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "preserve.txt"
    sentinel.write_bytes(b"immutable")

    with pytest.raises(FileExistsError, match="already exists"):
        run_recovery_evidence(
            _mini_recovery_sweep(tmp_path),
            output,
            run=RunSpec(
                total_ticks=30,
                warmup_ticks=10,
                crew_cabin_co2_concentration_ceiling=0.30,
            ),
            require_clean_source=False,
        )

    assert sentinel.read_bytes() == b"immutable"
    assert list(output.iterdir()) == [sentinel]


def test_canonical_recovery_evidence_clean_gate_precedes_output_creation(
    tmp_path, monkeypatch
):
    output = tmp_path / "canonical"

    def dirty_status(*args: str) -> str:
        assert args == ("status", "--porcelain")
        return "?? controlled-dirty-probe"

    monkeypatch.setattr("aeolus.recovery_evidence._git_output", dirty_status)

    with pytest.raises(ValueError, match="clean source worktree"):
        run_recovery_evidence(_mini_recovery_sweep(tmp_path), output)

    assert not output.exists()


def test_recovery_evidence_canonical_hash_rejects_non_finite_values():
    with pytest.raises(ValueError, match="Out of range float values"):
        _canonical_sha256({"metric": math.nan})


def test_recovery_evidence_cli_rejects_wrong_arguments(capsys):
    assert main([]) == 2
    assert "Usage:" in capsys.readouterr().err
