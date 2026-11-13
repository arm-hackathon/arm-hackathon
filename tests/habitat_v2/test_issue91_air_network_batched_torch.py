"""Optional PyTorch backend tests for the batched air network (Issue #91).

PyTorch is not a declared or locked AEOLUS dependency. Tests that need it are
*skipped* (never passed) when it is absent, and CUDA tests are skipped with a
distinct reason when no CUDA device is visible. A skipped test is not evidence.
The unavailable-backend tests at the top run in every environment.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from aeolus.habitat_v2 import air_network_batched_torch as backend
from aeolus.habitat_v2.air_network_analytic import (
    BRANCH_FIELDS,
    COMPARISON_VARIABLES,
    SCALAR_FIELDS,
    AirNetworkAnalyticError,
    AirNetworkBatch,
    compare_results,
    generate_air_network_cases,
    load_air_network_batched_contract,
    solve_air_network_analytic,
)
from aeolus.habitat_v2.air_network_batched_torch import (
    AirNetworkBackendUnavailableError,
    air_network_operating_point_torch,
    require_device,
    solve_air_network_torch,
    to_numpy_result,
    to_torch_inputs,
    torch_runtime_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT, _ = load_air_network_batched_contract(REPO_ROOT)
PROFILES = CONTRACT["case_profiles"]
TOLERANCES = CONTRACT["tolerances"]
CUDA_SKIP_REASON = "CUDA device not available: GPU test NOT run (skipped, not passed)"


def _torch() -> Any:
    return pytest.importorskip(
        "torch", reason="PyTorch not installed: optional backend test NOT run"
    )


def _cuda() -> Any:
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip(CUDA_SKIP_REASON)
    return torch


def _cases(profile: str, seed: int, batch_size: int = 256, branch_count: int = 8) -> AirNetworkBatch:
    return generate_air_network_cases(
        PROFILES[profile], seed=seed, batch_size=batch_size, branch_count=branch_count
    )


def _failures(report: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {name: entry for name, entry in report.items() if not entry["passed"]}


def _take(batch: AirNetworkBatch, indices: np.ndarray) -> AirNetworkBatch:
    return AirNetworkBatch(
        **{
            name: np.ascontiguousarray(getattr(batch, name)[indices])
            for name in SCALAR_FIELDS + BRANCH_FIELDS
        }
    )


def test_missing_torch_fails_with_an_explicit_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ``None`` entry makes ``import torch`` raise ImportError even if installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    batch = _cases("fixture_band", seed=1, batch_size=4)

    for call in (
        lambda: backend.require_torch(),
        lambda: solve_air_network_torch(batch, precision="fp32"),
        lambda: to_torch_inputs(batch, precision="fp64"),
        lambda: torch_runtime_identity(),
        lambda: require_device("cuda"),
    ):
        with pytest.raises(AirNetworkBackendUnavailableError, match="requires PyTorch"):
            call()


def test_importing_the_backend_module_does_not_import_torch_at_module_level() -> None:
    source = Path(backend.__file__).read_text(encoding="utf-8")
    module_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]

    assert not any("torch" in line and "air_network" not in line for line in module_level)
    assert "from .hmc" not in source and "from .physics" not in source


def test_cuda_request_without_cuda_fails_clearly_and_never_falls_back() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here: the unavailable-CUDA path is not reachable")
    batch = _cases("fixture_band", seed=2, batch_size=4)

    for device in ("cuda", "cuda:0"):
        with pytest.raises(AirNetworkBackendUnavailableError, match="CUDA is not available"):
            solve_air_network_torch(batch, precision="fp32", device=device)
        with pytest.raises(AirNetworkBackendUnavailableError, match="CUDA is not available"):
            torch_runtime_identity(device)


@pytest.mark.parametrize("device", ("gpu", "mps", "CUDA", "", "cpu:0"))
def test_unsupported_device_names_are_rejected(device: str) -> None:
    _torch()
    with pytest.raises(AirNetworkBackendUnavailableError, match="unsupported device"):
        require_device(device)


def test_unsupported_precision_is_rejected() -> None:
    _torch()
    batch = _cases("fixture_band", seed=3, batch_size=4)
    for precision in ("fp16", "float32", "", "FP32"):
        with pytest.raises(AirNetworkBackendUnavailableError, match="unsupported precision"):
            solve_air_network_torch(batch, precision=precision)


def test_host_validation_runs_before_any_tensor_is_created() -> None:
    _torch()
    batch = _cases("fixture_band", seed=4, batch_size=4)
    speed = batch.fan_speed_fraction.copy()
    speed[1] = np.nan
    invalid = AirNetworkBatch(
        **{
            **{name: getattr(batch, name) for name in SCALAR_FIELDS + BRANCH_FIELDS},
            "fan_speed_fraction": speed,
        }
    )

    for precision in ("fp32", "fp64"):
        with pytest.raises(AirNetworkAnalyticError, match="fan_speed_fraction in world 1"):
            solve_air_network_torch(invalid, precision=precision)


@pytest.mark.parametrize("branch_count", (1, 3, 8, 16))
@pytest.mark.parametrize("profile", ("fixture_band", "supported_domain_stress"))
def test_cpu_fp64_backend_matches_numpy_candidate(profile: str, branch_count: int) -> None:
    _torch()
    batch = _cases(profile, seed=9100 + branch_count, branch_count=branch_count)
    result = solve_air_network_torch(batch, precision="fp64")

    assert result.total_flow_m3_s.dtype == np.float64
    report = compare_results(
        result, solve_air_network_analytic(batch), batch,
        TOLERANCES["torch_fp64_vs_numpy_fp64"]["variables"],
    )
    assert _failures(report) == {}


@pytest.mark.parametrize("branch_count", (1, 3, 8, 16))
@pytest.mark.parametrize("profile", ("fixture_band", "supported_domain_stress"))
def test_cpu_fp32_backend_matches_numpy_candidate(profile: str, branch_count: int) -> None:
    _torch()
    batch = _cases(profile, seed=9200 + branch_count, branch_count=branch_count)
    result = solve_air_network_torch(batch, precision="fp32")

    assert result.total_flow_m3_s.dtype == np.float32
    assert result.zone_flow_m3_s.shape == (256, branch_count)
    report = compare_results(
        result, solve_air_network_analytic(batch), batch,
        TOLERANCES["torch_fp32_vs_numpy_fp64"]["variables"],
    )
    assert _failures(report) == {}


def test_fp32_fan_off_is_exact_zero_and_tiny_speeds_stay_finite() -> None:
    _torch()
    batch = _cases("fixture_band", seed=5, batch_size=6)
    speeds = np.array([0.0, -0.0, 1e-12, 1e-20, 1e-30, 1e-44])
    tiny = AirNetworkBatch(
        **{
            **{name: getattr(batch, name) for name in SCALAR_FIELDS + BRANCH_FIELDS},
            "fan_speed_fraction": speeds,
        }
    )
    result = solve_air_network_torch(tiny, precision="fp32")
    reference = solve_air_network_analytic(tiny)

    for name in COMPARISON_VARIABLES:
        values = getattr(result, name)
        assert bool(np.isfinite(values).all()), name
        assert not values[:2].any() and not np.signbit(values[:2]).any(), name
    # Flow stays linear in speed while it is an FP32 normal number ...
    np.testing.assert_allclose(
        result.total_flow_m3_s[2:5], reference.total_flow_m3_s[2:5], rtol=1e-5
    )
    # ... but FP32 pressure and power underflow at speeds FP64 still resolves.
    # This is a declared limitation covered by the absolute tolerance term.
    assert reference.fan_pressure_rise_pa[4] > 0.0
    assert result.fan_pressure_rise_pa[4] == 0.0
    assert result.fan_electrical_power_w[3] == 0.0 < reference.fan_electrical_power_w[3]
    report = compare_results(
        result, reference, tiny, TOLERANCES["torch_fp32_vs_numpy_fp64"]["variables"]
    )
    assert _failures(report) == {}


@pytest.mark.parametrize("precision", ("fp32", "fp64"))
def test_worlds_are_isolated_from_batch_size_order_and_chunking(precision: str) -> None:
    _torch()
    batch = _cases("supported_domain_stress", seed=9700, batch_size=96)
    full = solve_air_network_torch(batch, precision=precision)
    tolerances = TOLERANCES["torch_fp64_vs_numpy_fp64"]["variables"]

    order = np.random.Generator(np.random.PCG64(7)).permutation(96)
    pieces = {
        "permuted": (order, solve_air_network_torch(_take(batch, order), precision=precision)),
        "single": (np.array([41]), solve_air_network_torch(_take(batch, np.array([41])), precision=precision)),
        "chunk": (np.arange(32, 64), solve_air_network_torch(_take(batch, np.arange(32, 64)), precision=precision)),
    }
    for label, (indices, piece) in pieces.items():
        for name in COMPARISON_VARIABLES:
            expected = getattr(full, name)[indices]
            actual = getattr(piece, name)
            if precision == "fp64":
                np.testing.assert_allclose(
                    actual, expected,
                    rtol=tolerances[name]["relative_to_reference"] or 1e-13,
                    atol=1e-12 if name == "operating_point_residual_pa" else 0.0,
                    err_msg=f"{label}:{name}",
                )
            else:
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-6,
                    atol=1e-5 if name == "operating_point_residual_pa" else 1e-37,
                    err_msg=f"{label}:{name}",
                )


def test_kernel_keeps_inputs_untouched_and_outputs_on_the_input_device() -> None:
    torch = _torch()
    batch = _cases("fixture_band", seed=6, batch_size=16)
    inputs = to_torch_inputs(batch, precision="fp32")
    before = [tensor.clone() for tensor in inputs]
    result = air_network_operating_point_torch(inputs)

    assert all(torch.equal(a, b) for a, b in zip(inputs, before))
    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in result)
    assert result.zone_flow_m3_s.shape == (16, 8) and result.total_flow_m3_s.shape == (16,)
    host = to_numpy_result(result)
    assert np.array_equal(
        host.total_flow_m3_s, solve_air_network_torch(batch, precision="fp32").total_flow_m3_s
    )
    # The host batch is copied, not aliased: mutating a tensor leaves it alone.
    original = batch.fan_speed_fraction.copy()
    to_torch_inputs(batch, precision="fp64").fan_speed_fraction.zero_()
    assert np.array_equal(batch.fan_speed_fraction, original)


def test_runtime_identity_records_versions_without_host_details() -> None:
    torch = _torch()
    identity = torch_runtime_identity("cpu")

    assert identity["torch_version"] == torch.__version__
    assert identity["device"] == "cpu" and identity["device_name"] is None
    assert identity["cuda_available"] is bool(torch.cuda.is_available())
    assert set(identity) == {
        "torch_version", "torch_cuda_build_version", "cuda_available", "cudnn_version",
        "device", "device_name", "torch_num_threads",
    }


@pytest.mark.parametrize("precision", ("fp32", "fp64"))
@pytest.mark.parametrize("profile", ("fixture_band", "supported_domain_stress"))
def test_cuda_backend_matches_numpy_candidate(profile: str, precision: str) -> None:
    _cuda()
    batch = _cases(profile, seed=9300, batch_size=2048)
    result = solve_air_network_torch(batch, precision=precision, device="cuda")
    comparison = "torch_fp32_vs_numpy_fp64" if precision == "fp32" else "torch_fp64_vs_numpy_fp64"

    report = compare_results(
        result, solve_air_network_analytic(batch), batch, TOLERANCES[comparison]["variables"]
    )
    assert _failures(report) == {}


def test_cuda_kernel_outputs_stay_on_the_cuda_device() -> None:
    _cuda()
    inputs = to_torch_inputs(_cases("fixture_band", seed=8, batch_size=32), precision="fp32", device="cuda")
    result = air_network_operating_point_torch(inputs)

    assert all(tensor.device.type == "cuda" for tensor in result)
    assert torch_runtime_identity("cuda")["device_name"]
