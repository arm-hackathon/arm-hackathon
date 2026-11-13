"""Tests for the Issue #91 batched air-network benchmark command.

The NumPy-only path runs everywhere. The PyTorch path is skipped (not passed)
when PyTorch is absent; no test fabricates a CUDA measurement.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmark_issue91_air_network_batched.py"


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
        "reference_iterations": 1,
    }
    arguments.update(overrides)
    return module.run_benchmark(**arguments)


def test_numpy_only_receipt_is_complete_honest_and_canonical() -> None:
    module = _load_script()
    receipt = _small_run(module)

    assert receipt["schema_version"] == module.RECEIPT_SCHEMA_VERSION
    assert receipt["gate"] == {
        "description": "every gated per-variable comparison within its frozen tolerance",
        "passed": True,
        "gated_comparison_count": 2,
    }
    assert [entry["batch_size"] for entry in receipt["grid"]] == [1, 8]
    for entry in receipt["grid"]:
        assert len(entry["input_sha256"]) == 64
        assert entry["torch"] == {}
        numpy_entry = entry["numpy_fp64"]
        assert numpy_entry["accuracy_passed"] is True
        assert set(numpy_entry["accuracy_vs_scalar_reference"]) == set(
            receipt_variables(module)
        )
        for route in ("validated", "kernel_only"):
            summary = numpy_entry[route]
            assert summary["sample_count"] == 4
            assert 0 < summary["minimum_ns"] <= summary["median_ns"] <= summary["maximum_ns"]
        assert entry["scalar_reference_loop"]["sample_count"] == 4
        assert numpy_entry["peak_host_bytes"] > 0

    claims = receipt["claims"]
    assert claims["torch_backend_evaluated"] is False
    assert claims["cuda_hardware_evaluated"] is False
    assert claims["dgx_hardware_receipt"] is False
    assert "no CUDA device was measured" in claims["dgx_hardware_receipt_missing_requirements"]
    assert claims["actuator_authority"] is False
    assert claims["canonical_replay_or_hash_authority"] is False
    assert claims["end_to_end_simulator_speedup_claimed"] is False
    assert receipt["environment"]["torch"] is None
    assert receipt["environment"]["torch_num_threads_requested"] is None
    assert receipt["environment"]["container_image_digest"] == {
        "value": None, "source": "not_provided",
    }
    assert all(
        crossover["status"] == "not_evaluated"
        for crossover in receipt["crossover_vs_numpy_fp64_validated"].values()
    )
    # The stress report against the bisection reference is recorded, not gated.
    assert set(receipt["stress_accuracy"]) == {"scalar_reference_reported_not_gated", "torch"}

    raw = module._canonical_json_bytes(receipt)
    assert raw == module._canonical_json_bytes(json.loads(raw))


def receipt_variables(module: ModuleType) -> tuple[str, ...]:
    from aeolus.habitat_v2.air_network_analytic import COMPARISON_VARIABLES

    return COMPARISON_VARIABLES


def test_receipt_binds_source_contract_and_inputs_without_host_details() -> None:
    module = _load_script()
    receipt = _small_run(module, batch_sizes=(4,))

    from aeolus.habitat_v2.air_network_analytic import (
        air_network_batch_sha256,
        generate_air_network_cases,
        load_air_network_batched_contract,
    )

    contract, contract_sha256 = load_air_network_batched_contract(REPO_ROOT)
    assert receipt["contract"]["canonical_sha256"] == contract_sha256
    assert receipt["grid"][0]["input_sha256"] == air_network_batch_sha256(
        generate_air_network_cases(
            contract["case_profiles"]["fixture_band"], seed=91, batch_size=4, branch_count=8
        )
    )
    files = {entry["relative_path"]: entry for entry in receipt["source"]["source_files"]}
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


def test_same_seed_reproduces_inputs_and_accuracy_but_not_timings() -> None:
    module = _load_script()
    first, second = _small_run(module), _small_run(module)

    def deterministic(receipt: dict) -> dict:
        stable = copy.deepcopy(receipt)
        for entry in stable["grid"]:
            del entry["scalar_reference_loop"]
            for route in ("validated", "kernel_only", "peak_host_bytes"):
                del entry["numpy_fp64"][route]
        del stable["crossover_vs_numpy_fp64_validated"]
        # Depends on whether uncommitted work exists while the suite runs.
        del stable["source"]["source_worktree_dirty"]
        del stable["claims"]["dgx_hardware_receipt_missing_requirements"]
        return stable

    assert deterministic(first) == deterministic(second)
    other = _small_run(module, seed=92)
    assert other["grid"][0]["input_sha256"] != first["grid"][0]["input_sha256"]


def test_gate_failure_is_reported_and_exits_nonzero_with_a_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    genuine = module.solve_air_network_analytic

    def biased(batch):
        result = genuine(batch)
        values = {name: getattr(result, name) for name in result.__dataclass_fields__}
        values["total_flow_m3_s"] = values["total_flow_m3_s"] * (1.0 + 1e-9)
        return type(result)(**values)

    monkeypatch.setattr(module, "solve_air_network_analytic", biased)
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "failing"

    exit_code = module.main(
        ["--output", str(output), "--batch-sizes", "8", "--warmup-iterations", "1",
         "--measured-iterations", "4", "--reference-iterations", "1"]
    )

    assert exit_code == 1
    receipt = json.loads((output / module.RECEIPT_NAME).read_bytes())
    assert receipt["gate"]["passed"] is False
    flow = receipt["grid"][0]["numpy_fp64"]["accuracy_vs_scalar_reference"]["total_flow_m3_s"]
    assert flow["passed"] is False and flow["violation_count"] > 0
    assert "an accuracy gate failed" in (
        receipt["claims"]["dgx_hardware_receipt_missing_requirements"]
    )


def test_command_writes_one_canonical_receipt_and_refuses_to_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "issue91" / "run-1"
    arguments = ["--output", str(output), "--batch-sizes", "1", "8", "--warmup-iterations", "1",
                 "--measured-iterations", "4", "--reference-iterations", "1"]

    assert module.main(arguments) == 0
    raw = (output / module.RECEIPT_NAME).read_bytes()
    assert [path.name for path in output.iterdir()] == [module.RECEIPT_NAME]
    assert raw == module._canonical_json_bytes(json.loads(raw))
    printed = capsys.readouterr().out
    assert f"receipt sha256: {hashlib.sha256(raw).hexdigest()}" in printed
    assert str(tmp_path) not in printed

    assert module.main(arguments) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
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
    assert module.OUT_ROOT == tmp_path / "out"
    assert _load_script().OUT_ROOT == REPO_ROOT / "out"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"batch_sizes": ()}, "batch sizes"),
        ({"batch_sizes": (8, 8)}, "batch sizes"),
        ({"batch_sizes": (0,)}, "batch sizes"),
        ({"batch_sizes": (True,)}, "batch sizes"),
        ({"warmup_iterations": 0}, "warmup"),
        ({"measured_iterations": 3}, "warmup"),
        ({"reference_iterations": 0}, "warmup"),
        ({"torch_devices": ("cpu", "cpu")}, "torch devices"),
        ({"precisions": ("fp16",)}, "torch devices"),
        ({"torch_devices": ("cpu",), "precisions": ()}, "precision"),
        ({"torch_num_threads": 1}, "thread count"),
        ({"torch_devices": ("cpu",), "torch_num_threads": 0}, "thread count"),
        ({"torch_devices": ("cpu",), "torch_num_threads": True}, "thread count"),
    ),
)
def test_invalid_benchmark_requests_fail_closed(overrides: dict[str, object], message: str) -> None:
    module = _load_script()
    with pytest.raises(module.AirNetworkBenchmarkError, match=message):
        _small_run(module, **overrides)


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
    torch = pytest.importorskip(
        "torch", reason="PyTorch not installed: optional backend test NOT run"
    )
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here: the unavailable-CUDA path is not reachable")
    module = _load_script()
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path / "out")
    output = tmp_path / "out" / "no-cuda"

    exit_code = module.main(
        ["--output", str(output), "--batch-sizes", "1", "--torch-devices", "cpu", "cuda"]
    )

    assert exit_code == 2
    assert "CUDA is not available" in capsys.readouterr().err
    assert not output.exists()


def test_torch_cpu_measurements_are_gated_and_never_claimed_as_cuda() -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: optional backend test NOT run")
    module = _load_script()
    receipt = _small_run(
        module, torch_devices=("cpu",), image_digest="sha256:" + "a" * 64,
        hardware_label="declared-label", torch_num_threads=1,
    )
    assert receipt["environment"]["torch_num_threads_requested"] == 1
    assert receipt["environment"]["torch"]["cpu"]["torch_num_threads"] == 1

    assert receipt["gate"]["passed"] is True
    # 2 batch sizes x (numpy + 2 torch precisions) + 2 stress torch comparisons.
    assert receipt["gate"]["gated_comparison_count"] == 8
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
    assert claims["cuda_hardware_evaluated"] is False
    # Declaring a label and digest cannot turn a CPU run into a hardware receipt.
    assert claims["dgx_hardware_receipt"] is False
    assert "no CUDA device was measured" in claims["dgx_hardware_receipt_missing_requirements"]
    assert receipt["environment"]["torch"]["cpu"]["device_name"] is None
    crossovers = receipt["crossover_vs_numpy_fp64_validated"]
    assert set(crossovers) == {
        "cpu/fp32/transfer_inclusive", "cpu/fp32/kernel_only",
        "cpu/fp64/transfer_inclusive", "cpu/fp64/kernel_only",
    }
    assert all(
        crossover["status"] in {"measured_crossover", "no_crossover_in_tested_grid"}
        for crossover in crossovers.values()
    )
