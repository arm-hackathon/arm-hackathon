from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cli_module():
    import importlib.util

    path = _repo_root() / "scripts/benchmark_habitat_v2_fp32.py"
    spec = importlib.util.spec_from_file_location("benchmark_habitat_v2_fp32", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fp32_optimisation_preserves_contract_and_runtime_precision(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.arm_optimization import optimise_ridge_fp32
    from aeolus.habitat_v2.forecast.live_demo import load_live_ridge_model

    root = _repo_root()
    source = root / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
    destination = tmp_path / "action-aware-ridge-fp32.npz"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    receipt = optimise_ridge_fp32(
        source,
        destination,
        expected_source_sha256=source_sha256,
    )

    with np.load(destination, allow_pickle=False) as archive:
        assert str(archive["schema_version"].item()) == (
            "aeolus_habitat_v2_forecast_demo_model_fp32_v1"
        )
        for field in ("feature_mean", "feature_scale", "target_mean", "coef"):
            assert archive[field].dtype == np.float32

    model = load_live_ridge_model(
        destination,
        expected_sha256=receipt["candidate_model_sha256"],
    )
    predictor = model.predictor

    assert model.model_kind == "action_aware_ridge_fp32"
    assert model.actuator_authority is False
    assert predictor.feature_mean.dtype == np.float32
    assert predictor.feature_scale.dtype == np.float32
    assert predictor.target_mean.dtype == np.float32
    assert predictor.coef.dtype == np.float32
    assert receipt["source_model_sha256"] == source_sha256
    assert receipt["candidate_raw_array_bytes"] * 2 == receipt["source_raw_array_bytes"]
    assert receipt["candidate_raw_array_bytes_reduction_fraction"] == 0.5


def test_fp32_candidate_passes_live_drift_gate_and_emits_comparable_timings(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.arm_optimization import (
        benchmark_fp64_vs_fp32,
        optimise_ridge_fp32,
    )

    root = _repo_root()
    source = root / "artifacts/demo-only/habitat-v2-forecast/action-aware-ridge.npz"
    candidate = tmp_path / "action-aware-ridge-fp32.npz"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    conversion = optimise_ridge_fp32(
        source,
        candidate,
        expected_source_sha256=source_sha256,
    )

    receipt = benchmark_fp64_vs_fp32(
        root,
        source,
        candidate,
        expected_source_sha256=source_sha256,
        expected_candidate_sha256=conversion["candidate_model_sha256"],
        warmup_iterations=2,
        measured_iterations=8,
    )

    assert receipt["prediction_parity"]["gate"] == (
        "max_abs_drift_div_max_abs_reference_or_one_lte_1e-4"
    )
    assert receipt["prediction_parity"]["passed"] is True
    assert receipt["prediction_parity"]["maximum_normalised_drift"] <= 1e-4
    assert receipt["workload"]["candidate_action_count"] == 4
    assert receipt["workload"]["prediction_shape"] == [8, 51]
    assert receipt["models"]["fp64"]["precision"] == "float64"
    assert receipt["models"]["fp32"]["precision"] == "float32"
    assert (
        receipt["models"]["fp32"]["raw_array_bytes"] * 2
        == (receipt["models"]["fp64"]["raw_array_bytes"])
    )
    assert receipt["timing"]["fp64"]["sample_count"] == 8
    assert receipt["timing"]["fp32"]["sample_count"] == 8
    assert receipt["timing"]["fp64"]["median_ns"] > 0
    assert receipt["timing"]["fp32"]["median_ns"] > 0
    assert receipt["claims"]["actuator_authority"] is False
    assert receipt["claims"]["arm_specific_operator_optimisation"] is False


def test_fp32_candidate_and_static_receipt_are_exactly_reproducible(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.arm_optimization import optimise_ridge_fp32
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    root = _repo_root()
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    source = artifact_root / "action-aware-ridge.npz"
    committed_candidate = artifact_root / "action-aware-ridge-fp32.npz"
    committed_receipt = artifact_root / "fp32-conversion-receipt.json"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    first_receipt = optimise_ridge_fp32(
        source, first, expected_source_sha256=source_sha256
    )
    second_receipt = optimise_ridge_fp32(
        source, second, expected_source_sha256=source_sha256
    )

    assert first.read_bytes() == second.read_bytes() == committed_candidate.read_bytes()
    assert first_receipt == second_receipt
    assert committed_receipt.read_bytes() == canonical_json_bytes(first_receipt)
    assert json.loads(committed_receipt.read_bytes()) == first_receipt


def test_cli_benchmarks_exact_existing_candidate_without_rewriting_it(
    tmp_path: Path,
) -> None:
    root = _repo_root()
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    candidate = artifact_root / "action-aware-ridge-fp32.npz"
    conversion = artifact_root / "fp32-conversion-receipt.json"
    candidate_before = candidate.read_bytes()
    conversion_before = conversion.read_bytes()
    benchmark = tmp_path / "benchmark.json"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), existing_pythonpath))
    )

    subprocess.run(
        (
            sys.executable,
            "scripts/benchmark_habitat_v2_fp32.py",
            "--use-existing-candidate",
            "--candidate",
            str(candidate),
            "--conversion-receipt",
            str(conversion),
            "--benchmark-receipt",
            str(benchmark),
            "--warmup-iterations",
            "2",
            "--measured-iterations",
            "8",
        ),
        cwd=root,
        env=environment,
        check=True,
    )

    receipt = json.loads(benchmark.read_bytes())
    assert receipt["prediction_parity"]["passed"] is True
    assert (
        receipt["models"]["fp32"]["sha256"]
        == hashlib.sha256(candidate_before).hexdigest()
    )
    assert candidate.read_bytes() == candidate_before
    assert conversion.read_bytes() == conversion_before


def test_cli_returns_nonzero_and_writes_canonical_receipt_when_parity_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    module = _load_cli_module()
    root = _repo_root()
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    benchmark_path = tmp_path / "failed-benchmark.json"
    failed = {"prediction_parity": {"passed": False}, "timing": {}}
    monkeypatch.setattr(
        module, "benchmark_fp64_vs_fp32", lambda *_args, **_kwargs: failed
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_habitat_v2_fp32.py",
            "--use-existing-candidate",
            "--source",
            str(artifact_root / "action-aware-ridge.npz"),
            "--candidate",
            str(artifact_root / "action-aware-ridge-fp32.npz"),
            "--conversion-receipt",
            str(artifact_root / "fp32-conversion-receipt.json"),
            "--benchmark-receipt",
            str(benchmark_path),
        ],
    )

    assert module.main() == 1
    stored = json.loads(benchmark_path.read_bytes())
    assert stored["prediction_parity"]["passed"] is False
    assert benchmark_path.read_bytes() == canonical_json_bytes(stored)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("noncanonical", "canonical JSON"),
        ("unknown_field", "fields drift"),
        ("wrong_source_sha", "semantics or model identity"),
        ("wrong_candidate_sha", "semantics or model identity"),
        ("wrong_candidate_size", "semantics or model identity"),
        ("qualified", "semantics or model identity"),
    ),
)
def test_existing_conversion_receipt_validation_is_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    module = _load_cli_module()
    root = _repo_root()
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    source = artifact_root / "action-aware-ridge.npz"
    candidate = artifact_root / "action-aware-ridge-fp32.npz"
    receipt = json.loads((artifact_root / "fp32-conversion-receipt.json").read_bytes())
    if mutation == "unknown_field":
        receipt["unexpected"] = True
    elif mutation == "wrong_source_sha":
        receipt["source_model_sha256"] = "0" * 64
    elif mutation == "wrong_candidate_sha":
        receipt["candidate_model_sha256"] = "0" * 64
    elif mutation == "wrong_candidate_size":
        receipt["candidate_model_file_bytes"] += 1
    elif mutation == "qualified":
        receipt["qualified_model"] = True
    receipt_path = tmp_path / "conversion.json"
    raw = canonical_json_bytes(receipt)
    if mutation == "noncanonical":
        raw += b"\n"
    receipt_path.write_bytes(raw)

    with pytest.raises(ValueError, match=message):
        module._load_existing_conversion(source, candidate, receipt_path)


def test_existing_conversion_rejects_candidate_byte_mismatch(tmp_path: Path) -> None:
    module = _load_cli_module()
    root = _repo_root()
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    candidate = tmp_path / "candidate.npz"
    candidate.write_bytes(
        (artifact_root / "action-aware-ridge-fp32.npz").read_bytes() + b"tamper"
    )

    with pytest.raises(ValueError, match="semantics or model identity"):
        module._load_existing_conversion(
            artifact_root / "action-aware-ridge.npz",
            candidate,
            artifact_root / "fp32-conversion-receipt.json",
        )


def test_repeated_benchmarks_emit_provenance_and_median_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_cli_module()
    root = _repo_root()
    artifact_root = root / "artifacts/demo-only/habitat-v2-forecast"
    output = tmp_path / "series.json"
    medians = iter(((900, 500), (1000, 400), (1100, 550)))

    def benchmark(*_args: object, **_kwargs: object) -> dict[str, object]:
        fp64, fp32 = next(medians)
        return {
            "prediction_parity": {"passed": True},
            "timing": {
                "fp64": {"median_ns": fp64},
                "fp32": {"median_ns": fp32},
                "median_speedup_fp64_over_fp32": fp64 / fp32,
            },
        }

    monkeypatch.setattr(module, "benchmark_fp64_vs_fp32", benchmark)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_habitat_v2_fp32.py",
            "--use-existing-candidate",
            "--source",
            str(artifact_root / "action-aware-ridge.npz"),
            "--candidate",
            str(artifact_root / "action-aware-ridge-fp32.npz"),
            "--conversion-receipt",
            str(artifact_root / "fp32-conversion-receipt.json"),
            "--benchmark-receipt",
            str(output),
            "--benchmark-repetitions",
            "3",
        ],
    )

    assert module.main() == 0
    series = json.loads(output.read_bytes())
    assert series["run_count"] == 3
    assert series["all_prediction_parity_passed"] is True
    assert series["median_distribution"]["fp64_median_ns_by_run"] == [900, 1000, 1100]
    assert series["median_distribution"]["fp32_median_ns_by_run"] == [500, 400, 550]
    assert series["median_distribution"]["fp64_median_of_run_medians_ns"] == 1000
    assert series["provenance"]["github_sha"] == "a" * 40
    assert series["provenance"]["github_run_id"] == "123"
    assert len(series["provenance"]["conversion_receipt_sha256"]) == 64
