"""Closed-form batched operating point for the Habitat V2 air network (Issue #91).

Research-only FP64 NumPy candidate for the *existing* fixed topology solved by
``air_network.solve_air_network``: one quadratic fan, one shared quadratic
resistance and N parallel quadratic branches with a leak-floored damper. It is
not a general duct-network solver and it adds no physics.

The authoritative simulator, HMC and replay paths keep using the reviewed
bisection solver. Results from this module agree with it only to declared
numerical tolerances, so they must never be written into canonical traces,
receipts or hashes. This module imports NumPy only; the optional PyTorch
backend lives in ``air_network_batched_torch``.

Algebra (SI units; ``s`` is the dimensionless fan speed fraction):

    a_i   = leak_i + (1 - leak_i) * position_i               [-]
    R_i   = R_open_i / a_i**2 + R_return_i                   [Pa s^2 m^-6]
    g_i   = R_i**-0.5                                        [m^3 s^-1 Pa^-0.5]
    G     = sum_i g_i
    R_eq  = 1 / G**2            (parallel branches)          [Pa s^2 m^-6]
    k     = P_shutoff / Q_free**2   (fan curve slope)        [Pa s^2 m^-6]
    R_tot = k + R_shared + R_eq
    Q     = s * sqrt(P_shutoff / R_tot)                      [m^3 s^-1]
    q_i   = Q * g_i / G                                      [m^3 s^-1]
    p_b   = s**2 * P_shutoff * R_eq / R_tot                  [Pa]
    p_s   = s**2 * P_shutoff * R_shared / R_tot              [Pa]
    p_fan = s**2 * P_shutoff * (R_shared + R_eq) / R_tot     [Pa]
    W_air = p_fan * Q                                        [W]
    W_el  = W_air / efficiency                               [W]

``k`` does not depend on speed because the fan curve scales shutoff pressure
with ``s**2`` and free delivery with ``s``. Flow is therefore exactly linear in
speed, pressure quadratic and power cubic, and no division by a tiny speed is
needed. The fan-curve clamp ``max(0, 1 - (Q / Q_free)**2)`` is never active at
the operating point because ``R_tot > k`` implies ``Q < Q_free``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from .air_network import AirNetworkResult, AirNetworkSpec


class AirNetworkAnalyticError(ValueError):
    """The batched air-network research contract was violated."""


# Batch layout. Every per-world field has shape [B]; every per-branch field has
# shape [B, N]. The branch axis is in ascending ``zone_id`` order, which is the
# key order of the reference solver's result mappings.
SCALAR_FIELDS: Final = (
    "rated_free_delivery_m3_s",
    "rated_shutoff_pressure_pa",
    "total_efficiency",
    "shared_resistance_pa_s2_m6",
    "air_density_kg_m3",
    "fan_speed_fraction",
)
BRANCH_FIELDS: Final = (
    "open_supply_resistance_pa_s2_m6",
    "return_resistance_pa_s2_m6",
    "damper_leak_fraction",
    "damper_position_fraction",
)
RESULT_SCALAR_FIELDS: Final = (
    "fan_pressure_rise_pa",
    "shared_pressure_loss_pa",
    "branch_pressure_loss_pa",
    "total_flow_m3_s",
    "fan_air_power_w",
    "fan_electrical_power_w",
    "operating_point_residual_pa",
)
RESULT_BRANCH_FIELDS: Final = ("zone_flow_m3_s", "zone_mass_flow_kg_s")

# Closed intervals. This is a strict subset of what the reference solver
# accepts (any positive finite value): it is the box inside which the FP64 and
# FP32 candidates are tested and neither overflows nor divides by zero.
SUPPORTED_DOMAIN: Final[Mapping[str, tuple[float, float]]] = {
    "rated_free_delivery_m3_s": (1e-4, 1e4),
    "rated_shutoff_pressure_pa": (1e-2, 1e7),
    "total_efficiency": (1e-3, 1.0),
    "shared_resistance_pa_s2_m6": (0.0, 1e9),
    "air_density_kg_m3": (1e-3, 1e3),
    "fan_speed_fraction": (0.0, 1.0),
    "open_supply_resistance_pa_s2_m6": (1e-3, 1e9),
    "return_resistance_pa_s2_m6": (1e-3, 1e9),
    "damper_leak_fraction": (1e-5, 1.0),
    "damper_position_fraction": (0.0, 1.0),
}
MIN_BRANCH_COUNT: Final = 1
MAX_BRANCH_COUNT: Final = 16

CONTRACT_RELATIVE_PATH: Final = "contracts/habitat_v2_air_network_batched_v1.json"
CONTRACT_SCHEMA_VERSION: Final = "aeolus_habitat_v2_air_network_batched_contract_v1"
_CONTRACT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract_id",
        "status",
        "issue",
        "authority",
        "reference_solver",
        "layout",
        "supported_domain",
        "branch_count_range",
        "case_profiles",
        "comparison_form",
        "comparison_scales",
        "tolerances",
        "non_claims",
    }
)
_TOLERANCE_FIELDS: Final = frozenset(
    {"absolute", "relative_to_reference", "relative_to_scale"}
)
COMPARISON_VARIABLES: Final = RESULT_SCALAR_FIELDS + RESULT_BRANCH_FIELDS


@dataclass(frozen=True)
class AirNetworkBatch:
    """Struct-of-arrays inputs for B independent worlds with N branches each."""

    rated_free_delivery_m3_s: np.ndarray
    rated_shutoff_pressure_pa: np.ndarray
    total_efficiency: np.ndarray
    shared_resistance_pa_s2_m6: np.ndarray
    air_density_kg_m3: np.ndarray
    fan_speed_fraction: np.ndarray
    open_supply_resistance_pa_s2_m6: np.ndarray
    return_resistance_pa_s2_m6: np.ndarray
    damper_leak_fraction: np.ndarray
    damper_position_fraction: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.fan_speed_fraction.shape[0])

    @property
    def branch_count(self) -> int:
        return int(self.damper_position_fraction.shape[1])


@dataclass(frozen=True)
class AirNetworkBatchResult:
    """Struct-of-arrays operating points; shapes are [B] and [B, N].

    ``branch_pressure_loss_pa`` is the single pressure common to all parallel
    branches of a world (the reference solver repeats it once per zone). The
    reference ``mass_balance_residual_kg_s`` is identically zero and is not
    carried. ``operating_point_residual_pa`` is the fan curve evaluated at the
    solved flow minus the shared and branch losses; it is a rounding residual,
    not a solver convergence measure.
    """

    fan_pressure_rise_pa: np.ndarray
    shared_pressure_loss_pa: np.ndarray
    branch_pressure_loss_pa: np.ndarray
    total_flow_m3_s: np.ndarray
    fan_air_power_w: np.ndarray
    fan_electrical_power_w: np.ndarray
    operating_point_residual_pa: np.ndarray
    zone_flow_m3_s: np.ndarray
    zone_mass_flow_kg_s: np.ndarray


def validate_air_network_batch(batch: AirNetworkBatch) -> None:
    """Fail closed on any layout, dtype, finiteness or supported-domain error."""

    if not isinstance(batch, AirNetworkBatch):
        raise AirNetworkAnalyticError("batch must be an AirNetworkBatch")
    arrays = {name: getattr(batch, name) for name in SCALAR_FIELDS + BRANCH_FIELDS}
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise AirNetworkAnalyticError(f"{name} must be a float64 numpy array")
    batch_size = arrays[SCALAR_FIELDS[0]].shape[0] if arrays[SCALAR_FIELDS[0]].ndim == 1 else -1
    for name in SCALAR_FIELDS:
        if arrays[name].ndim != 1 or arrays[name].shape[0] != batch_size or batch_size < 1:
            raise AirNetworkAnalyticError(
                f"{name} must have shape [B] with one shared B >= 1"
            )
    branch_shape = arrays[BRANCH_FIELDS[0]].shape
    for name in BRANCH_FIELDS:
        shape = arrays[name].shape
        if (
            len(shape) != 2
            or shape != branch_shape
            or shape[0] != batch_size
            or not MIN_BRANCH_COUNT <= shape[1] <= MAX_BRANCH_COUNT
        ):
            raise AirNetworkAnalyticError(
                f"{name} must have shape [B, N] with the shared B and "
                f"{MIN_BRANCH_COUNT} <= N <= {MAX_BRANCH_COUNT}"
            )
    for name, array in arrays.items():
        low, high = SUPPORTED_DOMAIN[name]
        # NaN fails both comparisons, so nonfinite values are rejected here too.
        valid = (array >= low) & (array <= high)
        if not bool(valid.all()):
            world = int(np.argwhere(~valid)[0][0])
            raise AirNetworkAnalyticError(
                f"{name} in world {world} is nonfinite or outside the supported "
                f"domain [{low!r}, {high!r}]"
            )


def air_network_operating_point_numpy(batch: AirNetworkBatch) -> AirNetworkBatchResult:
    """Unvalidated closed-form kernel; callers must validate ``batch`` first.

    Exposed so that kernel time can be measured separately from validation.
    Outside the supported domain it may return nonfinite values silently.
    """

    rated_free_delivery = batch.rated_free_delivery_m3_s
    rated_shutoff_pressure = batch.rated_shutoff_pressure_pa
    total_efficiency = batch.total_efficiency
    shared_resistance = batch.shared_resistance_pa_s2_m6
    air_density = batch.air_density_kg_m3
    fan_speed = batch.fan_speed_fraction
    open_supply_resistance = batch.open_supply_resistance_pa_s2_m6
    return_resistance = batch.return_resistance_pa_s2_m6
    damper_leak = batch.damper_leak_fraction
    damper_position = batch.damper_position_fraction
    speed = fan_speed + 0.0  # maps -0.0 to +0.0 so fan-off outputs are +0.0
    area_fraction = damper_leak + (1.0 - damper_leak) * damper_position
    branch_resistance = (
        open_supply_resistance / (area_fraction * area_fraction) + return_resistance
    )
    conductance = 1.0 / np.sqrt(branch_resistance)
    conductance_sum = conductance.sum(axis=1)
    equivalent_resistance = 1.0 / (conductance_sum * conductance_sum)
    fan_curve_resistance = rated_shutoff_pressure / (
        rated_free_delivery * rated_free_delivery
    )
    system_resistance = shared_resistance + equivalent_resistance
    total_resistance = fan_curve_resistance + system_resistance

    # Unit-speed quantities first, then scale by speed. ``(s * x) * s`` keeps a
    # representable product from underflowing through an intermediate ``s * s``.
    unit_flow_squared = rated_shutoff_pressure / total_resistance
    unit_flow = np.sqrt(unit_flow_squared)
    total_flow = speed * unit_flow
    zone_flow = total_flow[:, None] * (conductance / conductance_sum[:, None])
    branch_pressure = (speed * (unit_flow_squared * equivalent_resistance)) * speed
    shared_pressure = (speed * (unit_flow_squared * shared_resistance)) * speed
    fan_pressure = (speed * (unit_flow_squared * system_resistance)) * speed
    fan_air_power = fan_pressure * total_flow

    # Independent fan-side evaluation: Q / Q_free equals unit_flow / rated free
    # delivery at every speed, so the fan curve needs no division by speed.
    normalized_flow = unit_flow / rated_free_delivery
    fan_curve_pressure = (speed * rated_shutoff_pressure) * speed * (
        1.0 - normalized_flow * normalized_flow
    )
    return AirNetworkBatchResult(
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


def solve_air_network_analytic(batch: AirNetworkBatch) -> AirNetworkBatchResult:
    """Validate ``batch`` and return its FP64 closed-form operating points."""

    validate_air_network_batch(batch)
    return air_network_operating_point_numpy(batch)


def batch_from_operating_points(
    points: Sequence[tuple[AirNetworkSpec, float, Mapping[str, float]]],
) -> tuple[AirNetworkBatch, tuple[str, ...]]:
    """Pack reference-solver inputs into one batch.

    Each point is ``(spec, fan_speed_fraction, damper_position_by_id)``. All
    points must declare the same zone ids. Returns the batch and the ascending
    zone-id order of its branch axis.
    """

    if not points:
        raise AirNetworkAnalyticError("at least one operating point is required")
    zone_ids: tuple[str, ...] | None = None
    scalars: list[list[float]] = []
    branches: list[list[list[float]]] = []
    for spec, fan_speed_fraction, damper_position_by_id in points:
        ordered = sorted(spec.branches, key=lambda branch: branch.zone_id)
        point_zone_ids = tuple(branch.zone_id for branch in ordered)
        damper_ids = [branch.damper_id for branch in ordered]
        if (
            len(set(point_zone_ids)) != len(point_zone_ids)
            or len(set(damper_ids)) != len(damper_ids)
            or not all(point_zone_ids)
            or not all(damper_ids)
        ):
            raise AirNetworkAnalyticError(
                "branch zone and damper ids must be unique and non-empty"
            )
        if set(damper_position_by_id) != set(damper_ids):
            raise AirNetworkAnalyticError(
                "damper command ids must exactly match configured damper ids"
            )
        if zone_ids is None:
            zone_ids = point_zone_ids
        elif point_zone_ids != zone_ids:
            raise AirNetworkAnalyticError(
                "all operating points in a batch must declare the same zone ids"
            )
        values = (
            spec.fan.rated_free_delivery_m3_s,
            spec.fan.rated_shutoff_pressure_pa,
            spec.fan.total_efficiency,
            spec.shared_resistance_pa_s2_m6,
            spec.air_density_kg_m3,
            fan_speed_fraction,
        )
        branch_values = [
            [
                branch.open_supply_resistance_pa_s2_m6,
                branch.return_resistance_pa_s2_m6,
                branch.damper_leak_fraction,
                damper_position_by_id[branch.damper_id],
            ]
            for branch in ordered
        ]
        for value in (*values, *(item for row in branch_values for item in row)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AirNetworkAnalyticError(
                    "air-network parameters and commands must be real numbers"
                )
        scalars.append([float(value) for value in values])
        branches.append(branch_values)
    scalar_array = np.array(scalars, dtype=np.float64)
    branch_array = np.array(branches, dtype=np.float64)
    assert zone_ids is not None
    if branch_array.ndim != 3 or branch_array.shape[1] == 0:
        raise AirNetworkAnalyticError("air network must declare at least one branch")
    batch = AirNetworkBatch(
        **{
            name: np.ascontiguousarray(scalar_array[:, index])
            for index, name in enumerate(SCALAR_FIELDS)
        },
        **{
            name: np.ascontiguousarray(branch_array[:, :, index])
            for index, name in enumerate(BRANCH_FIELDS)
        },
    )
    return batch, zone_ids


def solve_air_network_analytic_scalar(
    spec: AirNetworkSpec,
    *,
    fan_speed_fraction: float,
    damper_position_by_id: Mapping[str, float],
) -> AirNetworkResult:
    """Scalar convenience wrapper with the reference solver's result type."""

    batch, zone_ids = batch_from_operating_points(
        [(spec, fan_speed_fraction, damper_position_by_id)]
    )
    result = solve_air_network_analytic(batch)
    branch_pressure = float(result.branch_pressure_loss_pa[0])
    return AirNetworkResult(
        fan_pressure_rise_pa=float(result.fan_pressure_rise_pa[0]),
        shared_pressure_loss_pa=float(result.shared_pressure_loss_pa[0]),
        branch_pressure_loss_pa={zone_id: branch_pressure for zone_id in zone_ids},
        total_flow_m3_s=float(result.total_flow_m3_s[0]),
        zone_flow_m3_s={
            zone_id: float(result.zone_flow_m3_s[0, index])
            for index, zone_id in enumerate(zone_ids)
        },
        zone_mass_flow_kg_s={
            zone_id: float(result.zone_mass_flow_kg_s[0, index])
            for index, zone_id in enumerate(zone_ids)
        },
        fan_air_power_w=float(result.fan_air_power_w[0]),
        fan_electrical_power_w=float(result.fan_electrical_power_w[0]),
        total_efficiency=spec.fan.total_efficiency,
        air_density_kg_m3=spec.air_density_kg_m3,
        operating_point_residual_pa=float(result.operating_point_residual_pa[0]),
        mass_balance_residual_kg_s={zone_id: 0.0 for zone_id in zone_ids},
    )


def comparison_scales(batch: AirNetworkBatch) -> dict[str, np.ndarray]:
    """Per-world physical scale of every compared variable.

    Flows are scaled by the free delivery at the commanded speed, pressures by
    the shutoff pressure at that speed and powers by their product. Per-branch
    variables get a trailing axis so the scales broadcast against [B, N].
    """

    speed = batch.fan_speed_fraction
    flow = batch.rated_free_delivery_m3_s * speed
    pressure = (speed * batch.rated_shutoff_pressure_pa) * speed
    power = pressure * flow
    return {
        "fan_pressure_rise_pa": pressure,
        "shared_pressure_loss_pa": pressure,
        "branch_pressure_loss_pa": pressure,
        "operating_point_residual_pa": pressure,
        "total_flow_m3_s": flow,
        "fan_air_power_w": power,
        "fan_electrical_power_w": power / batch.total_efficiency,
        "zone_flow_m3_s": flow[:, None],
        "zone_mass_flow_kg_s": (flow * batch.air_density_kg_m3)[:, None],
    }


def compare_results(
    candidate: AirNetworkBatchResult,
    reference: AirNetworkBatchResult,
    batch: AirNetworkBatch,
    tolerances: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    """Apply the declared per-variable comparison and report every variable.

    A variable passes when, for every element,
    ``abs(c - r) <= absolute + relative_to_reference * abs(r)
    + relative_to_scale * scale``. Nonfinite candidates never pass.
    """

    if set(tolerances) != set(COMPARISON_VARIABLES):
        raise AirNetworkAnalyticError("tolerance variable set drifted")
    scales = comparison_scales(batch)
    report: dict[str, dict[str, Any]] = {}
    for name in COMPARISON_VARIABLES:
        tolerance = tolerances[name]
        if set(tolerance) != _TOLERANCE_FIELDS:
            raise AirNetworkAnalyticError(f"tolerance field set drifted for {name}")
        candidate_values = np.asarray(getattr(candidate, name), dtype=np.float64)
        reference_values = np.asarray(getattr(reference, name), dtype=np.float64)
        if candidate_values.shape != reference_values.shape:
            raise AirNetworkAnalyticError(f"{name} shapes differ")
        error = np.abs(candidate_values - reference_values)
        allowed = (
            float(tolerance["absolute"])
            + float(tolerance["relative_to_reference"]) * np.abs(reference_values)
            + float(tolerance["relative_to_scale"]) * scales[name]
        )
        finite = bool(np.isfinite(candidate_values).all())
        within = np.isfinite(error) & (error <= allowed)
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.where(error == 0.0, 0.0, error / allowed)
        excess = np.where(np.isfinite(error), excess, np.inf)
        report[name] = {
            "passed": finite and bool(within.all()),
            "candidate_finite": finite,
            "max_absolute_error": float(np.max(np.where(np.isfinite(error), error, 0.0))),
            "max_error_over_allowed": float(np.max(excess)) if finite else None,
            "violation_count": int((~within).sum()),
            "element_count": int(within.size),
        }
    return report


_DECADES: Final = {exponent: float(f"1e{exponent}") for exponent in range(-12, 10)}


def generate_air_network_cases(
    profile: Mapping[str, Any], *, seed: int, batch_size: int, branch_count: int
) -> AirNetworkBatch:
    """Deterministic synthetic inputs for one declared case profile.

    Only ``Generator.random`` draws, exactly-rounded decade constants and single
    IEEE multiplications/additions are used, so the bytes do not depend on a
    platform math library. Draw order is the declared field order.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AirNetworkAnalyticError("seed must be a non-negative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise AirNetworkAnalyticError("batch_size must be a positive integer")
    if (
        isinstance(branch_count, bool)
        or not isinstance(branch_count, int)
        or not MIN_BRANCH_COUNT <= branch_count <= MAX_BRANCH_COUNT
    ):
        raise AirNetworkAnalyticError("branch_count is outside the supported range")
    variables = profile["variables"]
    if set(variables) != set(SCALAR_FIELDS + BRANCH_FIELDS):
        raise AirNetworkAnalyticError("case profile variable set drifted")
    generator = np.random.Generator(np.random.PCG64(seed))
    arrays: dict[str, np.ndarray] = {}
    for name in SCALAR_FIELDS + BRANCH_FIELDS:
        shape = (batch_size,) if name in SCALAR_FIELDS else (batch_size, branch_count)
        rule = variables[name]
        kind = rule["distribution"]
        first = generator.random(shape)
        second = generator.random(shape)
        selector = generator.random(shape)
        if kind == "uniform":
            low, high = float(rule["low"]), float(rule["high"])
            values = low + (high - low) * first
        elif kind == "decade_uniform":
            low_exponent, high_exponent = rule["low_exponent"], rule["high_exponent"]
            span = high_exponent - low_exponent
            decade = low_exponent + np.minimum(
                np.floor(second * span).astype(np.int64), span - 1
            )
            scale = np.array([_DECADES[int(value)] for value in decade.ravel()])
            values = (1.0 + 9.0 * first) * scale.reshape(shape)
        else:
            raise AirNetworkAnalyticError(f"unknown distribution {kind!r} for {name}")
        zero_probability = float(rule.get("exact_zero_probability", 0.0))
        one_probability = float(rule.get("exact_one_probability", 0.0))
        values = np.where(selector < zero_probability, 0.0, values)
        values = np.where(selector >= 1.0 - one_probability, 1.0, values)
        arrays[name] = np.ascontiguousarray(values, dtype=np.float64)
    batch = AirNetworkBatch(**arrays)
    validate_air_network_batch(batch)
    return batch


def air_network_batch_sha256(batch: AirNetworkBatch) -> str:
    """Input identity: SHA-256 over shapes and little-endian float64 bytes."""

    validate_air_network_batch(batch)
    digest = hashlib.sha256()
    for name in SCALAR_FIELDS + BRANCH_FIELDS:
        array = getattr(batch, name)
        digest.update(f"{name}:{list(array.shape)}\n".encode("ascii"))
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def validate_air_network_batched_contract(contract: Any) -> dict[str, Any]:
    """Fail closed unless the frozen contract matches this module's layout."""

    if type(contract) is not dict or set(contract) != _CONTRACT_FIELDS:
        raise AirNetworkAnalyticError("batched air-network contract field set drifted")
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise AirNetworkAnalyticError("unsupported batched air-network contract schema")
    layout = contract["layout"]
    if (
        layout.get("world_fields") != list(SCALAR_FIELDS)
        or layout.get("branch_fields") != list(BRANCH_FIELDS)
        or layout.get("result_world_fields") != list(RESULT_SCALAR_FIELDS)
        or layout.get("result_branch_fields") != list(RESULT_BRANCH_FIELDS)
    ):
        raise AirNetworkAnalyticError("contract layout drifts from the module layout")
    declared_domain = {
        name: tuple(bounds) for name, bounds in contract["supported_domain"].items()
    }
    if declared_domain != dict(SUPPORTED_DOMAIN) or contract["branch_count_range"] != [
        MIN_BRANCH_COUNT,
        MAX_BRANCH_COUNT,
    ]:
        raise AirNetworkAnalyticError(
            "contract supported domain drifts from the module supported domain"
        )
    if set(contract["comparison_scales"]) != set(COMPARISON_VARIABLES):
        raise AirNetworkAnalyticError("contract comparison scale set drifted")
    for comparison_id, comparison in contract["tolerances"].items():
        variables = comparison.get("variables")
        if type(variables) is not dict or set(variables) != set(COMPARISON_VARIABLES):
            raise AirNetworkAnalyticError(
                f"tolerance variable set drifted for {comparison_id}"
            )
        for name, tolerance in variables.items():
            if type(tolerance) is not dict or set(tolerance) != _TOLERANCE_FIELDS:
                raise AirNetworkAnalyticError(
                    f"tolerance field set drifted for {comparison_id}.{name}"
                )
            for value in tolerance.values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) < 1.0
                ):
                    raise AirNetworkAnalyticError(
                        f"tolerance for {comparison_id}.{name} must be in [0, 1)"
                    )
        if comparison.get("case_profile") not in contract["case_profiles"]:
            raise AirNetworkAnalyticError(
                f"tolerance {comparison_id} names an undeclared case profile"
            )
    for profile_id, profile in contract["case_profiles"].items():
        variables = profile.get("variables")
        if type(variables) is not dict or set(variables) != set(
            SCALAR_FIELDS + BRANCH_FIELDS
        ):
            raise AirNetworkAnalyticError(
                f"case profile {profile_id} variable set drifted"
            )
    return contract


def load_air_network_batched_contract(root: str | Path) -> tuple[dict[str, Any], str]:
    """Load the frozen contract; return it with its canonical-JSON SHA-256.

    The digest is taken over canonical JSON rather than raw bytes so that it
    does not depend on checkout line endings.
    """

    path = Path(root) / CONTRACT_RELATIVE_PATH
    try:
        contract = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise AirNetworkAnalyticError(
            "batched air-network contract is missing or not valid JSON"
        ) from error
    validate_air_network_batched_contract(contract)
    return contract, hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()


__all__ = [
    "AirNetworkAnalyticError",
    "AirNetworkBatch",
    "AirNetworkBatchResult",
    "BRANCH_FIELDS",
    "COMPARISON_VARIABLES",
    "CONTRACT_RELATIVE_PATH",
    "MAX_BRANCH_COUNT",
    "MIN_BRANCH_COUNT",
    "RESULT_BRANCH_FIELDS",
    "RESULT_SCALAR_FIELDS",
    "SCALAR_FIELDS",
    "SUPPORTED_DOMAIN",
    "air_network_batch_sha256",
    "air_network_operating_point_numpy",
    "batch_from_operating_points",
    "compare_results",
    "comparison_scales",
    "generate_air_network_cases",
    "load_air_network_batched_contract",
    "solve_air_network_analytic",
    "solve_air_network_analytic_scalar",
    "validate_air_network_batch",
    "validate_air_network_batched_contract",
]
