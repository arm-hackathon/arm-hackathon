"""Frozen V6 development boundary for conditional specialist research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aeolus.model_cycle_v5 import (
    V5_CALIBRATION_SEEDS,
    V5_FIT_SEEDS,
    V5_PROHIBITED_HISTORICAL_SEEDS,
    V5_VALIDATION_SEEDS,
)

V6_SWEEP_SCHEMA_VERSION = "aeolus_sweep_v6"
V6_FIT_SEEDS = (2100, 2101, 2102, 2103)
V6_CALIBRATION_SEEDS = (2104, 2105)
V6_VALIDATION_SEEDS = (2300, 2301, 2302, 2303, 2304, 2305)
V6_PROHIBITED_HISTORICAL_SEEDS = V5_PROHIBITED_HISTORICAL_SEEDS | frozenset(
    (*V5_FIT_SEEDS, *V5_CALIBRATION_SEEDS, *V5_VALIDATION_SEEDS)
)


@dataclass(frozen=True)
class V6DevelopmentRequest:
    """The fixed non-deployment boundary required by the future V6 runner."""

    schema_version: str
    suite_role: str
    fit_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    output_dir: Path
    authorize_final_suite: bool = False
    authorize_response_integration: bool = False


def validate_v6_development_request(request: V6DevelopmentRequest) -> None:
    """Reject stale, mutable, or deployment-capable V6 execution requests."""
    if not isinstance(request, V6DevelopmentRequest):
        raise ValueError("V6 development request is malformed")
    if request.schema_version != V6_SWEEP_SCHEMA_VERSION:
        raise ValueError("V6 development request has an unsupported schema")
    if request.suite_role != "development":
        raise ValueError("V6 request must have development suite role")
    if request.authorize_final_suite:
        raise ValueError("V6 development cannot authorize a final-suite run")
    if request.authorize_response_integration:
        raise ValueError("V6 development cannot authorize response integration")

    seed_groups = (
        request.fit_seeds,
        request.calibration_seeds,
        request.validation_seeds,
    )
    if any(not _valid_seed_group(group) for group in seed_groups):
        raise ValueError("V6 seed groups must be non-empty integer tuples")
    requested_seeds = set().union(*map(set, seed_groups))
    if requested_seeds & V6_PROHIBITED_HISTORICAL_SEEDS:
        raise ValueError("V6 request contains prohibited historical seeds")
    if len(requested_seeds) != sum(len(group) for group in seed_groups):
        raise ValueError("V6 fit, calibration, and validation seed groups must be disjoint")
    if request.fit_seeds != V6_FIT_SEEDS:
        raise ValueError("V6 fit seeds must match the predeclared protocol")
    if request.calibration_seeds != V6_CALIBRATION_SEEDS:
        raise ValueError("V6 calibration seeds must match the predeclared protocol")
    if request.validation_seeds != V6_VALIDATION_SEEDS:
        raise ValueError("V6 validation seeds must match the predeclared protocol")

    if not isinstance(request.output_dir, Path):
        raise ValueError("V6 output directory must be a pathlib Path")
    if request.output_dir.exists() and any(request.output_dir.iterdir()):
        raise ValueError("V6 output directory must be empty")


def _valid_seed_group(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in value)
    )
