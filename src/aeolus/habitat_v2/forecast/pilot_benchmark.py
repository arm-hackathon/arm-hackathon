"""D2 pilot resource benchmark and preflight receipt boundary.

Measures real pilot-continuation executions, projects the frozen 23,400-run
campaign sequentially, and writes the exact ``load_resource_preflight``
receipt contract.  Ceilings are explicit required inputs — this module never
invents resource budgets and never marks a failing campaign as PASS.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import canonical_json_bytes
from .pilot import (
    APPROVED_PROFILE_ACTION_SHA256,
    APPROVED_ROSTER_SHA256,
    PilotContinuation,
    PilotDesign,
)

PLANNED_HMC_RUNS: int = 23_400
PREFLIGHT_SCHEMA_VERSION: str = (
    "aeolus_habitat_v2_forecast_pilot_resource_preflight_v1"
)
V2_PREFLIGHT_SCHEMA_VERSION: str = (
    "aeolus_habitat_v2_forecast_pilot_resource_preflight_v2"
)


class PilotBenchmarkError(ValueError):
    """The benchmark evidence or receipt is outside its contract."""


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    """Measured cost of one replay-verified pilot run."""

    wall_time_seconds: float
    peak_rss_bytes: int
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class BenchmarkCeilings:
    """Human-ratified campaign resource budgets; never invented by the tool."""

    wall_time_seconds: float
    peak_rss_bytes: int
    artifact_bytes: int
    disk_reserve_bytes: int


V2_RESOURCE_CEILINGS = BenchmarkCeilings(
    wall_time_seconds=48.0 * 60.0 * 60.0,
    peak_rss_bytes=4_294_967_296,
    artifact_bytes=34_359_738_368,
    disk_reserve_bytes=21_474_836_480,
)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise PilotBenchmarkError(f"preflight source cannot be read: {path}") from error


def build_v2_source_manifest(repo_root: str | Path) -> dict[str, str]:
    """Bind every tracked Habitat V2 source/config byte in the reviewed scope."""
    root = Path(repo_root).resolve()
    pathspecs = (
        "src/aeolus/habitat_v2",
        "contracts/habitat_v2*.json",
        "scenarios/habitat_v2*.json",
        "scenarios/habitat_v2_observability/*.json",
        "docs/plans/2026-08-14-habitat-v2-forecast-*.json",
        "docs/plans/2026-08-15-habitat-v2-forecast-pilot-v2-generation-amendment-v1.json",
        "docs/plans/2026-08-16-habitat-v2-qualified-model-protocol-v1.json",
        "docs/plans/2026-08-16-habitat-v2-qualified-model-protocol-v2-amendment-v1.json",
        "scripts/run_v2_fitcal_qualified.py",
        "scripts/qual_v2_prepare.py",
    )
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", *pathspecs],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PilotBenchmarkError("Git cannot enumerate the reviewed v2 source") from error
    relative_paths = tuple(sorted(line for line in result.stdout.splitlines() if line))
    if not relative_paths:
        raise PilotBenchmarkError("reviewed v2 source manifest is empty")
    return {relative: _sha256_file(root / relative) for relative in relative_paths}


def build_v2_contract_identities(repo_root: str | Path) -> dict[str, dict[str, str]]:
    """Return raw-byte and canonical-semantic identities for every HMC contract."""
    root = Path(repo_root).resolve()
    result: dict[str, dict[str, str]] = {}
    for path in sorted((root / "contracts").glob("habitat_v2*.json")):
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if type(value) is not dict:
                raise ValueError("contract is not an object")
            semantic = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise PilotBenchmarkError(f"contract identity cannot be built: {path}") from error
        relative = path.relative_to(root).as_posix()
        result[relative] = {
            "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "semantic_sha256": semantic,
        }
    if not result:
        raise PilotBenchmarkError("v2 contract identity set is empty")
    return result


def build_v2_runtime_identity() -> dict[str, str]:
    """Bind the interpreter, platform and measurement-library runtime."""
    try:
        psutil_version = importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        psutil_version = "UNAVAILABLE"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "psutil_version": psutil_version,
    }


def _require_measurement(value: Any) -> RunMeasurement:
    if type(value) is not RunMeasurement:
        raise PilotBenchmarkError("benchmark measurement has an invalid type")
    if (
        type(value.wall_time_seconds) not in {int, float}
        or not math.isfinite(float(value.wall_time_seconds))
        or value.wall_time_seconds <= 0
        or type(value.peak_rss_bytes) is not int
        or value.peak_rss_bytes <= 0
        or type(value.artifact_bytes) is not int
        or value.artifact_bytes <= 0
    ):
        raise PilotBenchmarkError("benchmark measurement values are not positive")
    return value


def _require_ceilings(value: Any) -> BenchmarkCeilings:
    if type(value) is not BenchmarkCeilings:
        raise PilotBenchmarkError("benchmark ceilings must be provided explicitly")
    if (
        type(value.wall_time_seconds) not in {int, float}
        or not math.isfinite(float(value.wall_time_seconds))
        or value.wall_time_seconds <= 0
        or type(value.peak_rss_bytes) is not int
        or value.peak_rss_bytes <= 0
        or type(value.artifact_bytes) is not int
        or value.artifact_bytes <= 0
        or type(value.disk_reserve_bytes) is not int
        or value.disk_reserve_bytes <= 0
    ):
        raise PilotBenchmarkError("benchmark ceiling values are not positive")
    return value


def build_preflight_receipt(
    *,
    measurements: tuple[RunMeasurement, ...],
    ceilings: BenchmarkCeilings,
    free_disk_bytes: int,
) -> dict[str, Any]:
    """Assemble the exact preflight receipt; verdict is computed, never set."""
    items = tuple(measurements)
    if not items:
        raise PilotBenchmarkError("benchmark requires at least one measurement")
    for item in items:
        _require_measurement(item)
    limits = _require_ceilings(ceilings)
    if type(free_disk_bytes) is not int or free_disk_bytes < 0:
        raise PilotBenchmarkError("free disk bytes must be a non-negative integer")

    runs = len(items)
    measured_wall = float(sum(item.wall_time_seconds for item in items))
    measured_peak_rss = max(item.peak_rss_bytes for item in items)
    measured_artifact = sum(item.artifact_bytes for item in items)
    projected_wall = measured_wall * PLANNED_HMC_RUNS / runs
    # Sequential execution: peak RSS does not accumulate across runs.
    projected_peak_rss = measured_peak_rss
    projected_artifact = -(-(measured_artifact * PLANNED_HMC_RUNS) // runs)

    runtime_ok = projected_wall <= limits.wall_time_seconds
    memory_ok = projected_peak_rss <= limits.peak_rss_bytes
    disk_ok = (
        free_disk_bytes - projected_artifact >= limits.disk_reserve_bytes
        and projected_artifact <= limits.artifact_bytes
    )
    verdict = (
        "PASS" if runtime_ok and memory_ok and disk_ok else "FAIL_RESOURCE_CEILING"
    )
    body: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "roster_sha256": APPROVED_ROSTER_SHA256,
        "profile_action_sha256": APPROVED_PROFILE_ACTION_SHA256,
        "planned_hmc_runs": PLANNED_HMC_RUNS,
        "benchmark_hmc_runs": runs,
        "measured_wall_time_seconds": measured_wall,
        "measured_peak_rss_bytes": measured_peak_rss,
        "measured_artifact_bytes": measured_artifact,
        "projected_wall_time_seconds": projected_wall,
        "projected_peak_rss_bytes": projected_peak_rss,
        "projected_artifact_bytes": projected_artifact,
        "runtime_within_ceiling": runtime_ok,
        "memory_within_ceiling": memory_ok,
        "disk_reserve_preserved": disk_ok,
        "verdict": verdict,
    }
    body["preflight_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def build_v2_preflight_receipt(
    repo_root: str | Path,
    design: PilotDesign,
    contracts: Any,
    *,
    measurements: tuple[RunMeasurement, ...],
    ceilings: BenchmarkCeilings,
    free_disk_bytes: int,
) -> dict[str, Any]:
    """Build a fresh v2 receipt bound to all reviewed executable inputs."""
    from .contracts import ForecastContracts

    if type(design) is not PilotDesign or type(contracts) is not ForecastContracts:
        raise PilotBenchmarkError("v2 preflight requires exact design and contracts")
    if design.roster_sha256 != APPROVED_ROSTER_SHA256 or (
        design.profile_action_sha256 != APPROVED_PROFILE_ACTION_SHA256
    ):
        raise PilotBenchmarkError("v2 preflight design identity drifts")
    limits = _require_ceilings(ceilings)
    if limits != V2_RESOURCE_CEILINGS:
        raise PilotBenchmarkError("v2 preflight ceilings drift from the ratified amendment")

    root = Path(repo_root).resolve()
    legacy = build_preflight_receipt(
        measurements=measurements,
        ceilings=limits,
        free_disk_bytes=free_disk_bytes,
    )
    legacy.pop("preflight_sha256")
    source_manifest = build_v2_source_manifest(root)
    body: dict[str, Any] = {
        **legacy,
        "schema_version": V2_PREFLIGHT_SCHEMA_VERSION,
        "roster_bytes_sha256": design.roster_bytes_sha256,
        "profile_action_bytes_sha256": design.profile_action_bytes_sha256,
        "source_manifest": source_manifest,
        "source_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(source_manifest)
        ).hexdigest(),
        "contract_identities": build_v2_contract_identities(root),
        "config_raw_sha256": {
            name: _sha256_file(root / name) for name in ("pyproject.toml", "uv.lock")
        },
        "runtime_identity": build_v2_runtime_identity(),
        "free_disk_bytes": free_disk_bytes,
        "ceilings": {
            "wall_time_seconds": float(limits.wall_time_seconds),
            "peak_rss_bytes": limits.peak_rss_bytes,
            "artifact_bytes": limits.artifact_bytes,
            "disk_reserve_bytes": limits.disk_reserve_bytes,
        },
    }
    body["preflight_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def write_preflight_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    """Write canonical receipt bytes once; existing destinations are refused."""
    if type(receipt) is not dict or "preflight_sha256" not in receipt:
        raise PilotBenchmarkError("preflight receipt is malformed")
    body = dict(receipt)
    declared = body.pop("preflight_sha256")
    if declared != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise PilotBenchmarkError("preflight receipt self-hash is invalid")
    target = Path(path)
    if target.exists():
        raise PilotBenchmarkError("preflight destination already exists")
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(receipt))
    except OSError as error:
        raise PilotBenchmarkError("preflight receipt cannot be written") from error


def measure_continuations(
    repo_root: str | Path,
    design: PilotDesign,
    continuations: tuple[PilotContinuation, ...],
) -> tuple[RunMeasurement, ...]:
    """Execute each continuation in-process and measure wall, RSS and bytes.

    RSS is the absolute resident set sampled around each run.  In-process
    sampling can overstate isolated-run usage because the interpreter retains
    allocator high-water memory; that direction is conservative for ceiling
    decisions and is recorded honestly in the resulting receipt.
    """
    try:
        import psutil
    except ImportError as error:
        raise PilotBenchmarkError(
            "psutil is required in the benchmark execution environment; it is "
            "deliberately not a package dependency because pyproject.toml bytes "
            "are pinned by the reviewed-source binding"
        ) from error

    from .pilot_execution import (
        run_pilot_action_continuation,
        run_pilot_control_continuation,
    )

    items = tuple(continuations)
    if not items:
        raise PilotBenchmarkError("benchmark requires at least one continuation")
    process = psutil.Process()
    results: list[RunMeasurement] = []
    for continuation in items:
        if type(continuation) is not PilotContinuation:
            raise PilotBenchmarkError("benchmark continuation has an invalid type")
        runner = (
            run_pilot_control_continuation
            if continuation.variant == "MATCHED_CONTROL"
            else run_pilot_action_continuation
        )
        process.memory_info()  # prime the sampler
        started = time.perf_counter()
        before = process.memory_info().rss
        bundle = runner(repo_root, design, continuation)
        wall = time.perf_counter() - started
        peak = int(max(before, process.memory_info().rss))
        artifact = len(bundle.trace_canonical_bytes) + len(
            canonical_json_bytes(list(bundle.witnesses))
        )
        anchor = getattr(bundle, "anchor", None)
        if anchor is not None:
            artifact += len(canonical_json_bytes(dict(anchor)))
        results.append(
            RunMeasurement(
                wall_time_seconds=wall,
                peak_rss_bytes=peak,
                artifact_bytes=artifact,
            )
        )
    return tuple(results)
