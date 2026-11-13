"""Tests for the Issue #91 batched air-network benchmark command.

The NumPy-only path runs everywhere. The PyTorch path is skipped (not passed)
when PyTorch is absent; no test fabricates a CUDA measurement. Receipt logic
that does not need hardware (crossover, timing loop, prerequisites) is tested
directly on synthetic inputs.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tracemalloc
from types import ModuleType

import pytest

from aeolus.habitat_v2.air_network_analytic import (
    COMPARISON_VARIABLES,
    air_network_batch_sha256,
    generate_air_network_cases,
    load_air_network_batched_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmark_issue91_air_network_batched.py"
TORCH_SKIP_REASON = "PyTorch not installed: optional backend test NOT run"
SMALL = ["--batch-sizes", "1", "8", "--warmup-iterations", "1", "--measured-iterations", "4",
         "--reference-iterations", "4"]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_issue91_air_network_batched", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_run(module: ModuleType, **overrides: object) -> dict:
    arguments: dict[str, object] = {
        "batch_sizes": (1, 8),
        "warmup_iterations": 1,
        "measured_iterations": 4,
        "reference_iterations": 4,
    }
    arguments.update(overrides)
    return module.run_benchmark(**arguments)


def test_numpy_only_receipt_is_complete_honest_and_canonical() -> None:
    module = _load_script()
    receipt = _small_run(module)

    assert receipt["schema_version"] == module.RECEIPT_SCHEMA_VERSION
    # Two fixture-band batches vs the reviewed solver plus one stress sample vs
    # the high-precision root.
    assert receipt["gate"]["passed"] is True
    assert receipt["gate"]["gated_comparison_count"] == 3
    assert [entry["batch_size"] for entry in receipt["grid"]] == [1, 8]
    for entry in receipt["grid"]:
        assert len(entry["input_sha256"]) == 64
        assert entry["torch"] == {}
        numpy_entry = entry["numpy_fp64"]
        assert numpy_entry["accuracy_passed"] is True
        assert set(numpy_entry["accuracy_vs_scalar_reference"]) == set(COMPARISON_VARIABLES)
        for route in ("validated", "kernel_only"):
            summary = numpy_entry[route]
            assert summary["sample_count"] == 4
            assert 0 < summary["minimum_ns"] <= summary["median_ns"] <= summary["maximum_ns"]
        assert entry["scalar_reference_loop"]["sample_count"] == 4
        assert numpy_entry["peak_host_bytes"] > 0

    stress = receipt["stress_accuracy"]
    assert stress["numpy_fp64_vs_high_precision_root"]["gated"] is True
    assert stress["numpy_fp64_vs_high_precision_root"]["accuracy_passed"] is True
    assert stress["numpy_fp64_vs_high_precision_root"]["world_count"] == 8
    assert stress["numpy_fp64_vs_scalar_reference"]["gated"] is False
    assert stress["torch"] == {}

    claims = receipt["claims"]
    assert claims["torch_backend_evaluated"] is False
    assert claims["cuda_device_measured"] is False
    assert claims["hardware_identity_verified_by_script"] is False
    assert claims["cuda_receipt_prerequisites_met"] is False
    assert "no CUDA device was measured" in claims["cuda_receipt_prerequisites_missing"]
    assert claims["actuator_authority"] is False
    assert claims["canonical_replay_or_hash_authority"] is False
    assert claims["end_to_end_simulator_speedup_claimed"] is False
    environment = receipt["environment"]
    assert environment["torch"] is None
    assert environment["torch_num_threads_requested"] is None
    assert environment["container_image_digest"] == {"value": None, "source": "not_provided"}
    assert environment["timing_conditions"] == {
        "value": "shared_host_provisional",
        "source": "operator_declared_not_verified",
        "provisional": True,
    }
    assert all(
        crossover["status"] == "not_evaluated"
        for crossover in receipt["crossover_vs_numpy_fp64"].values()
    )

    raw = module._canonical_json_bytes(receipt)
    assert raw == module._canonical_json_bytes(json.loads(raw))


def test_receipt_binds_executed_source_contract_and_inputs_without_host_details() -> None:
    module = _load_script()
    receipt = _small_run(module, batch_sizes=(4,))

    contract, contract_sha256 = load_air_network_batched_contract(REPO_ROOT)
    assert receipt["contract"]["canonical_sha256"] == contract_sha256
    assert receipt["grid"][0]["input_sha256"] == air_network_batch_sha256(
        generate_air_network_cases(
            contract["case_profiles"]["fixture_band"], seed=91, batch_size=4, branch_count=8
        )
    )
    assert receipt["stress_accuracy"]["input_sha256"] == air_network_batch_sha256(
        generate_air_network_cases(
            contract["case_profiles"]["supported_domain_stress"], seed=91, batch_size=4,
            branch_count=8,
        )
    )
    source = receipt["source"]
    assert source["executed_modules_verified_from_checkout"] is True
    files = {entry["relative_path"]: entry for entry in source["source_files"]}
    assert set(files) == set(module.SOURCE_PATHS)
    reference = files["src/aeolus/habitat_v2/air_network.py"]
    assert reference["git_blob_sha1"] == contract["reference_solver"]["git_blob_sha1"]
    assert reference["sha256_lf_normalised"] == contract["reference_solver"]["sha256_lf_normalised"]
    for relative, entry in files.items():
        normalised = (REPO_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        assert entry["sha256_lf_normalised"] == hashlib.sha256(normalised).hexdigest()
        assert entry["byte_length_lf_normalised"] == len(normalised)

    text = module._canonical_json_bytes(receipt).decode("utf-8")
    for private in (str(REPO_ROOT), REPO_ROOT.as_posix(), str(Path.home()), Path.home().as_posix()):
        assert private not in text


def test_run_refuses_when_measured_modules_are_not_the_recorded_checkout(tmp_path: Path) -> None:
    module = _load_script()
    # A different root holding byte-identical copies is still not what was imported.
    for relative in module.SOURCE_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())

    with pytest.raises(module.AirNetworkBenchmarkError, match="was not imported from the checkout"):
        _small_run(module, repo_root=tmp_path)


def test_same_seed_reproduces_inputs_and_accuracy_but_not_timings() -> None:
    module = _load_script()
    first, second = _small_run(module), _small_run(module)

    def deterministic(receipt: dict) -> dict:
        stable = copy.deepcopy(receipt)
        for entry in stable["grid"]:
            del entry["scalar_reference_loop"]
            for route in ("validated", "kernel_only", "peak_host_bytes"):
                del entry["numpy_fp64"][route]
        del stable["crossover_vs_numpy_fp64"]
        # These depend on whether uncommitted work exists while the suite runs.
        del stable["source"]["source_worktree_dirty"]
        del stable["claims"]["cuda_receipt_prerequisites_missing"]
        return stable

    assert deterministic(first) == deterministic(second)
    other = _small_run(module, seed=92)
    assert other["grid"][0]["input_sha256"] != first["grid"][0]["input_sha256"]


def _biased(module: ModuleType, field: str, change):
    genuine = module.solve_air_network_analytic

    def solve(batch):
        result = genuine(batch)
        values = {name: getattr(result, name) for name in result.__dataclass_fields__}
        values[field] = change(values[field], batch)
        return type(result)(**values)

    return solve


def test_gate_failure_is_reported_and_exits_nonzero_with_a_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module, "solve_air_network_analytic",
        _biased(module, "total_flow_m3_s", lambda values, batch: values * (1.0 + 1e-9)),
    )
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "failing"

    assert module.main(["--output", str(output), *SMALL]) == 1

    receipt = json.loads((output / module.RECEIPT_NAME).read_bytes())
    assert receipt["gate"]["passed"] is False
    flow = receipt["grid"][1]["numpy_fp64"]["accuracy_vs_scalar_reference"]["total_flow_m3_s"]
    assert flow["passed"] is False and flow["violation_count"] > 0
    assert receipt["stress_accuracy"]["numpy_fp64_vs_high_precision_root"]["accuracy_passed"] is False
    assert "an accuracy gate failed" in receipt["claims"]["cuda_receipt_prerequisites_missing"]


def test_an_error_confined_to_tiny_speeds_fails_the_stress_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def double_tiny_speed_flow(values, batch):
        tiny = (batch.fan_speed_fraction > 0.0) & (batch.fan_speed_fraction < 1e-3)
        return values * (1.0 + tiny)

    monkeypatch.setattr(
        module, "solve_air_network_analytic",
        _biased(module, "total_flow_m3_s", double_tiny_speed_flow),
    )
    receipt = _small_run(module, batch_sizes=(64,))

    # The fixture band has no such speeds, so only the stress gate can see it.
    assert receipt["grid"][0]["numpy_fp64"]["accuracy_passed"] is True
    assert receipt["stress_accuracy"]["numpy_fp64_vs_high_precision_root"]["accuracy_passed"] is False
    assert receipt["gate"]["passed"] is False


def test_fan_off_bias_still_yields_a_canonical_failing_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    # An additive error is nonzero where the allowance is exactly zero (fan off).
    monkeypatch.setattr(
        module, "solve_air_network_analytic",
        _biased(module, "total_flow_m3_s", lambda values, batch: values + 1e-6),
    )
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "fan-off-bias"

    assert module.main(
        ["--output", str(output), "--batch-sizes", "256", "--warmup-iterations", "1",
         "--measured-iterations", "4", "--reference-iterations", "4"]
    ) == 1

    raw = (output / module.RECEIPT_NAME).read_bytes()
    receipt = json.loads(raw)
    flow = receipt["grid"][0]["numpy_fp64"]["accuracy_vs_scalar_reference"]["total_flow_m3_s"]
    assert flow["passed"] is False
    assert raw == module._canonical_json_bytes(receipt)


def test_command_writes_one_canonical_receipt_and_refuses_to_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "issue91" / "run-1"
    arguments = ["--output", str(output), *SMALL]

    assert module.main(arguments) == 0
    raw = (output / module.RECEIPT_NAME).read_bytes()
    assert [path.name for path in output.iterdir()] == [module.RECEIPT_NAME]
    assert raw == module._canonical_json_bytes(json.loads(raw))
    printed = capsys.readouterr().out
    assert f"receipt sha256: {hashlib.sha256(raw).hexdigest()}" in printed
    assert "receipt: out/issue91/run-1/benchmark-receipt.json" in printed
    assert str(tmp_path) not in printed

    assert module.main(arguments) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    # A new run may not be nested inside an existing receipt directory either.
    assert module.main(["--output", str(output / "nested"), *SMALL]) == 2
    assert "existing receipt directory" in capsys.readouterr().err
    assert [path.name for path in output.iterdir()] == [module.RECEIPT_NAME]
    assert (output / module.RECEIPT_NAME).read_bytes() == raw


def test_output_must_be_a_new_directory_below_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")

    for target in (tmp_path / "elsewhere", tmp_path / "out", tmp_path / "out" / ".." / "escape"):
        assert module.main(["--output", str(target), "--batch-sizes", "1"]) == 2
        assert "out/" in capsys.readouterr().err
        assert not (target / module.RECEIPT_NAME).exists()
    assert not (tmp_path / "elsewhere").exists()
    assert _load_script().OUT_ROOT == REPO_ROOT / "out"


def test_every_failure_exits_two_leaves_nothing_and_prints_no_host_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    calls = []
    genuine = module.run_benchmark
    monkeypatch.setattr(
        module, "run_benchmark", lambda **arguments: calls.append(1) or genuine(**arguments)
    )

    # An ancestor that is a regular file is detected before anything is measured.
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a-file").write_bytes(b"x")
    assert module.main(["--output", str(tmp_path / "out" / "a-file" / "run"), *SMALL]) == 2
    assert calls == []
    assert str(tmp_path) not in capsys.readouterr().err

    # An unexpected exception inside the run is status 2, never the documented 1,
    # and the claimed directory is removed.
    def explode(**arguments):
        raise KeyError("unexpected")

    monkeypatch.setattr(module, "run_benchmark", explode)
    output = tmp_path / "out" / "exploded"
    assert module.main(["--output", str(output), *SMALL]) == 2
    assert not output.exists()
    assert "benchmark not run" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"batch_sizes": ()}, "batch sizes"),
        ({"batch_sizes": (8, 8)}, "batch sizes"),
        ({"batch_sizes": (0,)}, "batch sizes"),
        ({"batch_sizes": (True,)}, "batch sizes"),
        ({"warmup_iterations": 0}, "warmup"),
        ({"measured_iterations": 3}, "warmup"),
        ({"reference_iterations": 3}, "warmup"),
        ({"measured_iterations": 4.0}, "warmup"),
        ({"torch_devices": ("cpu", "cpu")}, "torch devices"),
        ({"precisions": ("fp16",)}, "torch devices"),
        ({"precisions": ("fp32", "fp32")}, "precisions must be distinct"),
        ({"torch_devices": ("cpu",), "precisions": ()}, "precision"),
        ({"torch_num_threads": 1}, "thread count"),
        ({"torch_devices": ("cpu",), "torch_num_threads": 0}, "thread count"),
        ({"torch_devices": ("cpu",), "torch_num_threads": True}, "thread count"),
        ({"image_digest": "latest"}, "image digest"),
        ({"image_digest": "sha256:" + "A" * 64}, "image digest"),
        ({"hardware_label": "C:\\Users\\someone\\box"}, "hardware label"),
        ({"hardware_label": "/home/someone"}, "hardware label"),
        ({"hardware_label": ""}, "hardware label"),
        ({"timing_conditions": "quiet"}, "timing conditions"),
    ),
)
def test_invalid_benchmark_requests_fail_closed(overrides: dict[str, object], message: str) -> None:
    module = _load_script()
    with pytest.raises(module.AirNetworkBenchmarkError, match=message):
        _small_run(module, **overrides)


def test_summary_statistics_are_computed_from_the_samples() -> None:
    module = _load_script()
    summary = module._summarise([4_000, 1_000, 3_000, 2_000, 10_000], batch_size=50)

    assert summary["sample_count"] == 5
    assert summary["minimum_ns"] == 1_000 and summary["maximum_ns"] == 10_000
    assert summary["median_ns"] == 3_000
    assert summary["mean_ns"] == 4_000
    assert summary["p95_ns"] == pytest.approx(8_800) and summary["p99_ns"] == pytest.approx(9_760)
    # 50 worlds in 3 microseconds.
    assert summary["worlds_per_second_at_median"] == pytest.approx(50 / 3e-6)


def test_timing_loop_warms_up_and_synchronizes_inside_every_sample() -> None:
    module = _load_script()
    events: list[str] = []

    samples = module._time_call(
        lambda: events.append("call"),
        warmup_iterations=2, measured_iterations=3,
        synchronize=lambda: events.append("sync"),
    )

    assert len(samples) == 3 and all(sample >= 0 for sample in samples)
    assert events == ["call", "call"] + ["sync", "call", "sync"] * 3
    assert module._time_call(lambda: None, warmup_iterations=1, measured_iterations=4) != []


def _grid(module: ModuleType, rows: list[tuple[int, float, float, float | None, float | None]]) -> list[dict]:
    """rows: batch size, numpy validated, numpy kernel, torch transfer, torch kernel medians."""

    grid = []
    for batch_size, validated, kernel, torch_transfer, torch_kernel in rows:
        torch_entries = {}
        if torch_transfer is not None:
            torch_entries["cuda/fp32"] = {
                "transfer_inclusive": {"median_ns": torch_transfer},
                "kernel_only": {"median_ns": torch_kernel},
            }
        grid.append(
            {
                "batch_size": batch_size,
                "numpy_fp64": {"validated": {"median_ns": validated}, "kernel_only": {"median_ns": kernel}},
                "torch": torch_entries,
            }
        )
    return grid


def test_crossover_compares_like_with_like_and_reports_only_what_was_measured() -> None:
    module = _load_script()

    # Torch kernel sits between NumPy kernel-only and NumPy validated: it must
    # not be called faster, because the kernel route is compared with a kernel.
    between = _grid(module, [(8, 40.0, 16.0, 200.0, 30.0), (2048, 250.0, 170.0, 300.0, 200.0)])
    kernel = module._crossover(between, "cuda/fp32", "kernel_only")
    assert kernel["status"] == "no_crossover_in_tested_grid"
    assert kernel["numpy_fp64_baseline_route"] == "kernel_only"
    assert kernel["first_faster_batch_size"] is None and kernel["faster_batch_sizes"] == []
    transfer = module._crossover(between, "cuda/fp32", "transfer_inclusive")
    assert transfer["status"] == "no_crossover_in_tested_grid"
    assert transfer["numpy_fp64_baseline_route"] == "validated"

    crossing = _grid(
        module,
        [(2048, 250.0, 170.0, 90.0, 20.0), (8, 40.0, 16.0, 200.0, 30.0), (512, 90.0, 60.0, 80.0, 25.0)],
    )
    measured = module._crossover(crossing, "cuda/fp32", "transfer_inclusive")
    assert measured["status"] == "measured_crossover"
    assert measured["first_faster_batch_size"] == 512
    assert measured["faster_batch_sizes"] == [512, 2048]
    assert [row["batch_size"] for row in measured["compared"]] == [8, 512, 2048]

    always = _grid(module, [(8, 40.0, 16.0, 30.0, 10.0), (2048, 250.0, 170.0, 90.0, 20.0)])
    assert module._crossover(always, "cuda/fp32", "kernel_only")["status"] == (
        "faster_at_every_tested_size"
    )
    mixed = _grid(
        module,
        [(8, 40.0, 16.0, 30.0, 10.0), (512, 90.0, 60.0, 95.0, 70.0), (2048, 250.0, 170.0, 90.0, 20.0)],
    )
    mixed_outcome = module._crossover(mixed, "cuda/fp32", "transfer_inclusive")
    assert mixed_outcome["status"] == "mixed_no_clean_crossover"
    assert mixed_outcome["first_faster_batch_size"] is None

    absent = _grid(module, [(8, 40.0, 16.0, None, None)])
    assert module._crossover(absent, "cuda/fp32", "kernel_only")["status"] == "not_evaluated"
    assert module._crossover(always, "cuda/fp64", "kernel_only")["status"] == "not_evaluated"


def test_cuda_receipt_prerequisites_name_every_missing_requirement() -> None:
    module = _load_script()
    complete = {
        "cuda_devices": ["cuda"],
        "batch_sizes": module.DEFAULT_BATCH_SIZES,
        "precisions": ("fp32", "fp64"),
        "image_digest": "sha256:" + "a" * 64,
        "hardware_label": "declared label",
        "source_worktree_dirty": False,
        "source_commit": "c" * 40,
        "gates_passed": True,
        "timing_conditions": "dedicated_host",
    }
    assert module._cuda_receipt_prerequisites(**complete) == []

    cases = {
        "cuda_devices": ([], "no CUDA device was measured"),
        "batch_sizes": ((1,), "the default batch grid was not fully measured"),
        "precisions": (("fp32",), "both precisions were not measured"),
        "image_digest": (None, "no container image digest was declared"),
        "hardware_label": (None, "no hardware label was declared"),
        "source_worktree_dirty": (True, "source worktree is dirty or its state is unknown"),
        "source_commit": (None, "source commit is unknown"),
        "gates_passed": (False, "an accuracy gate failed"),
        "timing_conditions": ("shared_host_provisional", "timings were taken on a shared host"),
    }
    for key, (value, reason) in cases.items():
        assert module._cuda_receipt_prerequisites(**{**complete, key: value}) == [reason], key
    assert module._cuda_receipt_prerequisites(**{**complete, "source_worktree_dirty": None}) == [
        "source worktree is dirty or its state is unknown"
    ]


def test_host_peak_measurement_is_per_call_and_leaves_caller_tracing_on() -> None:
    module = _load_script()
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        ballast = bytearray(8_000_000)
        del ballast  # raises the process peak far above any solve
        peak = module._host_peak_bytes(lambda: bytearray(100_000))
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()

    assert 100_000 <= peak < 1_000_000
    assert module._host_peak_bytes(lambda: bytearray(100_000)) >= 100_000
    assert not tracemalloc.is_tracing()


def test_requested_backend_that_is_missing_stops_the_run_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    output = tmp_path / "out" / "no-torch"

    exit_code = module.main(["--output", str(output), "--batch-sizes", "1", "--torch-devices", "cuda"])

    assert exit_code == 2
    assert "requires PyTorch" in capsys.readouterr().err
    assert not output.exists()


def test_requested_cuda_that_is_unavailable_stops_the_run_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here: the unavailable-CUDA path is not reachable")
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "no-cuda"
    threads = torch.get_num_threads()

    exit_code = module.main(
        ["--output", str(output), "--batch-sizes", "1", "--torch-devices", "cpu", "cuda",
         "--torch-num-threads", "3"]
    )

    assert exit_code == 2
    assert "CUDA is not available" in capsys.readouterr().err
    assert not output.exists()
    assert torch.get_num_threads() == threads


def test_torch_cpu_measurements_are_gated_and_never_claimed_as_cuda() -> None:
    torch = pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    module = _load_script()
    threads = torch.get_num_threads()
    receipt = _small_run(
        module, torch_devices=("cpu",), image_digest="sha256:" + "a" * 64,
        hardware_label="declared-label", torch_num_threads=1,
        batch_sizes=module.DEFAULT_BATCH_SIZES, timing_conditions="dedicated_host",
    )

    assert torch.get_num_threads() == threads, "the benchmark must restore the thread count"
    assert receipt["environment"]["torch_num_threads_requested"] == 1
    assert receipt["environment"]["torch"]["cpu"]["torch_num_threads"] == 1
    assert receipt["gate"]["passed"] is True
    # 6 batch sizes x (numpy + 2 torch precisions) + stress (1 numpy + 2 torch).
    assert receipt["gate"]["gated_comparison_count"] == 21
    for entry in receipt["grid"]:
        assert set(entry["torch"]) == {"cpu/fp32", "cpu/fp64"}
        for measured in entry["torch"].values():
            assert measured["accuracy_passed"] is True
            assert measured["kernel_only"]["sample_count"] == 4
            assert measured["transfer_inclusive"]["sample_count"] == 4
            assert measured["peak_memory"]["peak_device_bytes"] is None
    assert set(receipt["stress_accuracy"]["torch"]) == {"cpu/fp32", "cpu/fp64"}
    claims = receipt["claims"]
    assert claims["torch_backend_evaluated"] is True
    assert claims["cuda_device_measured"] is False
    # Every other prerequisite is declared, yet a CPU run is never a hardware receipt.
    assert claims["cuda_receipt_prerequisites_met"] is False
    assert "no CUDA device was measured" in claims["cuda_receipt_prerequisites_missing"]
    assert claims["hardware_identity_verified_by_script"] is False
    assert receipt["environment"]["hardware_label"]["source"] == "operator_declared_not_verified"
    assert receipt["environment"]["torch"]["cpu"]["device_name"] is None
    crossovers = receipt["crossover_vs_numpy_fp64"]
    assert set(crossovers) == {
        "cpu/fp32/transfer_inclusive", "cpu/fp32/kernel_only",
        "cpu/fp64/transfer_inclusive", "cpu/fp64/kernel_only",
    }
    for key, crossover in crossovers.items():
        assert crossover["status"] != "not_evaluated", key
        assert len(crossover["compared"]) == 6


def test_torch_gate_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch", reason=TORCH_SKIP_REASON)
    module = _load_script()
    genuine = module.torch_module.solve_air_network_torch

    def biased(batch, *, precision, device="cpu"):
        result = genuine(batch, precision=precision, device=device)
        values = {name: getattr(result, name) for name in result.__dataclass_fields__}
        if precision == "fp32":
            values["zone_flow_m3_s"] = values["zone_flow_m3_s"] * 1.001
        return type(result)(**values)

    monkeypatch.setattr(module.torch_module, "solve_air_network_torch", biased)
    receipt = _small_run(module, torch_devices=("cpu",))

    assert receipt["gate"]["passed"] is False
    assert receipt["grid"][0]["torch"]["cpu/fp32"]["accuracy_passed"] is False
    assert receipt["grid"][0]["torch"]["cpu/fp64"]["accuracy_passed"] is True
    assert receipt["stress_accuracy"]["torch"]["cpu/fp32"]["accuracy_passed"] is False
