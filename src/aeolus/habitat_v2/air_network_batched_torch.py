"""Optional eager PyTorch backend for the batched air-network kernel (Issue #91).

Research-only. PyTorch is not a declared AEOLUS dependency and is imported
lazily inside functions, so importing this module (or any core module) never
imports Torch. The algebra is the closed form documented in
``air_network_analytic``; inputs are validated on the host in FP64 before any
tensor is created. Results agree with the authoritative bisection solver only
to declared tolerances and must not enter canonical traces, receipts or hashes.

Plain eager tensor operations only: no CUDA extension, no compilation, no
autograd requirement and no in-place mutation of caller tensors.
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

import numpy as np

from .air_network_analytic import (
    BRANCH_FIELDS,
    SCALAR_FIELDS,
    AirNetworkBatch,
    AirNetworkBatchResult,
    validate_air_network_batch,
)

PRECISIONS: Final = ("fp32", "fp64")


class AirNetworkBackendUnavailableError(RuntimeError):
    """The optional PyTorch backend or the requested device is unavailable."""


class TorchAirNetworkInputs(NamedTuple):
    """Device tensors in the documented layout: six [B] then four [B, N]."""

    rated_free_delivery_m3_s: Any
    rated_shutoff_pressure_pa: Any
    total_efficiency: Any
    shared_resistance_pa_s2_m6: Any
    air_density_kg_m3: Any
    fan_speed_fraction: Any
    open_supply_resistance_pa_s2_m6: Any
    return_resistance_pa_s2_m6: Any
    damper_leak_fraction: Any
    damper_position_fraction: Any


class TorchAirNetworkResult(NamedTuple):
    """Device tensors mirroring ``AirNetworkBatchResult``."""

    fan_pressure_rise_pa: Any
    shared_pressure_loss_pa: Any
    branch_pressure_loss_pa: Any
    total_flow_m3_s: Any
    fan_air_power_w: Any
    fan_electrical_power_w: Any
    operating_point_residual_pa: Any
    zone_flow_m3_s: Any
    zone_mass_flow_kg_s: Any


def require_torch() -> Any:
    """Return the ``torch`` module or fail with an explicit message."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise AirNetworkBackendUnavailableError(
            "the batched air-network backend requires PyTorch, which is not a "
            "declared aeolus dependency; install it separately"
        ) from error
    return torch


def require_device(device: str) -> Any:
    """Return a ``torch.device``; fail clearly if CUDA is requested but absent."""

    torch = require_torch()
    if not isinstance(device, str) or not (
        device == "cpu" or device == "cuda" or device.startswith("cuda:")
    ):
        raise AirNetworkBackendUnavailableError(
            f"unsupported device {device!r}; expected 'cpu', 'cuda' or 'cuda:N'"
        )
    if device != "cpu":
        if not torch.cuda.is_available():
            raise AirNetworkBackendUnavailableError(
                f"device {device!r} was requested but CUDA is not available to "
                "this PyTorch build; no CPU fallback is substituted"
            )
        resolved = torch.device(device)
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise AirNetworkBackendUnavailableError(
                f"device {device!r} was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return resolved
    return torch.device("cpu")


def _torch_dtype(precision: str) -> Any:
    torch = require_torch()
    if precision not in PRECISIONS:
        raise AirNetworkBackendUnavailableError(
            f"unsupported precision {precision!r}; expected one of {PRECISIONS}"
        )
    return torch.float32 if precision == "fp32" else torch.float64


def to_torch_inputs(
    batch: AirNetworkBatch, *, precision: str, device: str = "cpu"
) -> TorchAirNetworkInputs:
    """Validate ``batch`` on the host in FP64, then copy it to ``device``.

    Validation happens before the cast, so an FP32 tensor may differ from the
    validated FP64 value by one FP32 rounding.
    """

    torch = require_torch()
    validate_air_network_batch(batch)
    dtype = _torch_dtype(precision)
    target = require_device(device)
    return TorchAirNetworkInputs(
        *(
            torch.from_numpy(np.array(getattr(batch, name), dtype=np.float64)).to(
                device=target, dtype=dtype
            )
            for name in SCALAR_FIELDS + BRANCH_FIELDS
        )
    )


def air_network_operating_point_torch(
    inputs: TorchAirNetworkInputs,
) -> TorchAirNetworkResult:
    """Unvalidated device-resident kernel; same algebra as the NumPy candidate.

    Callers must pass tensors produced by ``to_torch_inputs`` or otherwise
    already inside the supported domain. Outputs stay on the input device.
    """

    require_torch()
    (
        rated_free_delivery,
        rated_shutoff_pressure,
        total_efficiency,
        shared_resistance,
        air_density,
        fan_speed,
        open_supply_resistance,
        return_resistance,
        damper_leak,
        damper_position,
    ) = inputs
    speed = fan_speed + 0.0
    area_fraction = damper_leak + (1.0 - damper_leak) * damper_position
    branch_resistance = (
        open_supply_resistance / (area_fraction * area_fraction) + return_resistance
    )
    conductance = 1.0 / branch_resistance.sqrt()
    conductance_sum = conductance.sum(dim=1)
    equivalent_resistance = 1.0 / (conductance_sum * conductance_sum)
    fan_curve_resistance = rated_shutoff_pressure / (
        rated_free_delivery * rated_free_delivery
    )
    system_resistance = shared_resistance + equivalent_resistance
    total_resistance = fan_curve_resistance + system_resistance

    unit_flow_squared = rated_shutoff_pressure / total_resistance
    unit_flow = unit_flow_squared.sqrt()
    total_flow = speed * unit_flow
    zone_flow = total_flow[:, None] * (conductance / conductance_sum[:, None])
    branch_pressure = (speed * (unit_flow_squared * equivalent_resistance)) * speed
    shared_pressure = (speed * (unit_flow_squared * shared_resistance)) * speed
    fan_pressure = (speed * (unit_flow_squared * system_resistance)) * speed
    fan_air_power = fan_pressure * total_flow

    normalized_flow = unit_flow / rated_free_delivery
    fan_curve_pressure = (speed * rated_shutoff_pressure) * speed * (
        1.0 - normalized_flow * normalized_flow
    )
    return TorchAirNetworkResult(
        fan_pressure_rise_pa=fan_pressure,
        shared_pressure_loss_pa=shared_pressure,
        branch_pressure_loss_pa=branch_pressure,
        total_flow_m3_s=total_flow,
        fan_air_power_w=fan_air_power,
        fan_electrical_power_w=fan_air_power / total_efficiency,
        operating_point_residual_pa=(
            fan_curve_pressure - shared_pressure - branch_pressure
        ),
        zone_flow_m3_s=zone_flow,
        zone_mass_flow_kg_s=zone_flow * air_density[:, None],
    )


def to_numpy_result(result: TorchAirNetworkResult) -> AirNetworkBatchResult:
    """Copy device tensors back to host NumPy arrays (dtype preserved)."""

    return AirNetworkBatchResult(
        *(tensor.detach().cpu().numpy() for tensor in result)
    )


def solve_air_network_torch(
    batch: AirNetworkBatch, *, precision: str, device: str = "cpu"
) -> AirNetworkBatchResult:
    """Validated, transfer-inclusive solve: host FP64 in, host arrays out."""

    torch = require_torch()
    inputs = to_torch_inputs(batch, precision=precision, device=device)
    with torch.no_grad():
        result = air_network_operating_point_torch(inputs)
    return to_numpy_result(result)


def torch_runtime_identity(device: str = "cpu") -> dict[str, object]:
    """Framework/device versions for receipts. Never includes host paths."""

    torch = require_torch()
    cuda_available = bool(torch.cuda.is_available())
    identity: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_build_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn_version": (
            torch.backends.cudnn.version() if cuda_available else None
        ),
        "device": device,
        "device_name": None,
        "torch_num_threads": int(torch.get_num_threads()),
    }
    if device != "cpu":
        resolved = require_device(device)
        identity["device_name"] = str(torch.cuda.get_device_name(resolved))
    return identity


__all__ = [
    "AirNetworkBackendUnavailableError",
    "PRECISIONS",
    "TorchAirNetworkInputs",
    "TorchAirNetworkResult",
    "air_network_operating_point_torch",
    "require_device",
    "require_torch",
    "solve_air_network_torch",
    "to_numpy_result",
    "to_torch_inputs",
    "torch_runtime_identity",
]
