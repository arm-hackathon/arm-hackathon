"""Four-arm deterministic recovery-evidence integration tests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path

import pytest

import aeolus.recovery_evidence as recovery_evidence
from aeolus.recovery import RecoverySettings
from aeolus.recovery_evidence import (
    RECOVERY_ARMS,
    _canonical_sha256,
    _evaluate_gates,
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
    environment = receipt["environment"]
    assert environment["uv_lock_sha256"] == hashlib.sha256(
        (REPO_ROOT / "uv.lock").read_bytes()
    ).hexdigest()
    assert environment["runtime_packages"] == {
        "aeolus": importlib.metadata.version("aeolus"),
        "numpy": importlib.metadata.version("numpy"),
    }
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
                "lifecycle",
                "invariant_violation_count",
            }
            assert set(arm["metrics"]["lifecycle"]) == {
                "protect_target_zone_ids",
                "protect_entry_count",
                "handback_recurrence_count",
                "handback_timeout_count",
                "reserve_failure_count",
                "final_physical_zero",
            }


def test_recovery_evidence_admits_final_suite_without_changing_four_arm_contract(
    tmp_path,
):
    sweep_path = _mini_recovery_sweep(tmp_path)
    document = json.loads(sweep_path.read_text(encoding="utf-8"))
    document["suite_role"] = "final"
    document["splits"] = {"final": document["splits"]["validation"]}
    sweep_path.write_text(json.dumps(document), encoding="utf-8")

    receipt = run_recovery_evidence(
        sweep_path,
        tmp_path / "final-evidence",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
    )

    assert receipt["sweep"]["suite_role"] == "final"
    assert {family["split"] for family in receipt["per_family"]} == {"final"}
    assert all(tuple(family["arms"]) == RECOVERY_ARMS for family in receipt["per_family"])
    assert receipt["gates"]["benefit"]["evaluation_split"] == "final"


def test_recovery_evidence_rejects_malformed_settings_before_output(tmp_path):
    output = tmp_path / "malformed-settings"

    with pytest.raises(TypeError, match="settings are malformed"):
        run_recovery_evidence(
            _mini_recovery_sweep(tmp_path),
            output,
            require_clean_source=False,
            settings=object(),
        )

    assert not output.exists()


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
        "harmful_physical_fault_protected",
        "fault_governed_expected_target_only",
        "transient_single_protect_episode",
        "transient_zero_handback_recurrence",
        "transient_zero_handback_timeout",
        "transient_final_physical_zero",
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


def test_final_verdict_lifecycle_gates_fail_closed(tmp_path):
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
    rows = receipt["per_family"]
    physical = next(
        row
        for row in rows
        if row["paired_metrics"]["eligible_physical_airflow_fault"]
    )
    target = physical["paired_metrics"]["target_zone_id"]
    physical["arms"]["fault_reserve_off"]["metrics"][
        "integrated_physical_co2_excess"
    ][target] = 1.0
    physical["arms"]["fault_governed"]["metrics"]["states"][
        "first_protect_tick"
    ] = None
    physical["arms"]["fault_governed"]["metrics"]["lifecycle"][
        "protect_target_zone_ids"
    ] = ["not-the-target"]

    transient = next(row for row in rows if row["fault_class"].startswith("transient_"))
    lifecycle = transient["arms"]["fault_governed"]["metrics"]["lifecycle"]
    lifecycle["protect_entry_count"] = 2
    lifecycle["handback_recurrence_count"] = 1
    lifecycle["handback_timeout_count"] = 1
    lifecycle["final_physical_zero"] = False

    safety = _evaluate_gates(rows, evaluation_split="validation")["safety"]

    assert safety["passed"] is False
    assert safety["harmful_physical_fault_protected"]["passed"] is False
    assert safety["fault_governed_expected_target_only"]["passed"] is False
    assert safety["transient_single_protect_episode"]["passed"] is False
    assert safety["transient_zero_handback_recurrence"]["passed"] is False
    assert safety["transient_zero_handback_timeout"]["passed"] is False
    assert safety["transient_final_physical_zero"]["passed"] is False


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
    assert comparison["first_file_count"] == comparison["second_file_count"]
    assert comparison["first_file_count"] > comparison["trace_count"] + 1
    assert comparison["mismatched_files"] == []


def test_recovery_evidence_duplicate_detects_generated_corpus_mutation(
    tmp_path, monkeypatch
):
    real_run = recovery_evidence.run_recovery_evidence
    calls = 0

    def run_then_mutate(sweep_path, output_dir, **kwargs):
        nonlocal calls
        receipt = real_run(sweep_path, output_dir, **kwargs)
        calls += 1
        if calls == 2:
            manifest = Path(output_dir) / "corpus" / "families.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        return receipt

    monkeypatch.setattr(recovery_evidence, "run_recovery_evidence", run_then_mutate)
    comparison = recovery_evidence.reproduce_recovery_evidence(
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

    assert comparison["byte_identical"] is False
    assert comparison["mismatched_files"] == ["corpus/families.json"]


def test_recovery_evidence_rejects_source_change_during_run(tmp_path, monkeypatch):
    real_provenance = recovery_evidence._source_provenance
    calls = 0

    def changing_provenance(*, require_clean_source: bool):
        nonlocal calls
        calls += 1
        provenance = real_provenance(require_clean_source=False)
        if calls == 2:
            provenance["source"]["manifest_sha256"] = "0" * 64
        return provenance

    monkeypatch.setattr(recovery_evidence, "_source_provenance", changing_provenance)
    output = tmp_path / "evidence"

    with pytest.raises(ValueError, match="source provenance changed during recovery evidence"):
        recovery_evidence.run_recovery_evidence(
            _mini_recovery_sweep(tmp_path),
            output,
            run=RunSpec(
                total_ticks=30,
                warmup_ticks=10,
                crew_cabin_co2_concentration_ceiling=0.30,
            ),
            require_clean_source=False,
        )

    assert output.exists()
    assert not (output / "recovery-evidence.json").exists()


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


def test_recovery_evidence_records_and_applies_explicit_settings(tmp_path):
    sweep_path = _mini_recovery_sweep(tmp_path)
    document = json.loads(sweep_path.read_text(encoding="utf-8"))
    for split in document["splits"].values():
        split["blocked_effectiveness"] = [0.84]
    sweep_path.write_text(json.dumps(document), encoding="utf-8")
    settings = RecoverySettings(
        entry_residual_ratio=0.10,
        entry_isolation_margin=0.05,
        entry_persistence_ticks=2,
        exit_residual_ratio=0.05,
        handback_abort_residual_ratio=0.08,
    )

    receipt = run_recovery_evidence(
        sweep_path,
        tmp_path / "explicit-settings",
        run=RunSpec(
            total_ticks=30,
            warmup_ticks=10,
            crew_cabin_co2_concentration_ceiling=0.30,
        ),
        require_clean_source=False,
        settings=settings,
    )

    assert receipt["recovery_settings"]["entry_residual_ratio"] == 0.10
    assert receipt["recovery_settings"]["entry_isolation_margin"] == 0.05
    assert receipt["recovery_settings"]["entry_persistence_ticks"] == 2
    blocked = next(
        family
        for family in receipt["per_family"]
        if family["fault_class"] == "blocked_path"
        and family["split"] == "validation"
    )
    governed = blocked["arms"]["fault_governed"]["metrics"]
    assert governed["states"]["first_protect_tick"] is not None
    assert governed["reserve_delivered_airflow_integral"] > 0.0


def test_recovery_evidence_canonical_hash_rejects_non_finite_values():
    with pytest.raises(ValueError, match="Out of range float values"):
        _canonical_sha256({"metric": math.nan})


def test_recovery_evidence_cli_rejects_wrong_arguments(capsys):
    assert main([]) == 2
    assert "Usage:" in capsys.readouterr().err
