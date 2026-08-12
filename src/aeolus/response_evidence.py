"""Deterministic response-evidence harness for the bounded recovery governor.

Runs every family of a validated v3 sweep twice — once with the baseline
proportional controller and once with the bounded recovery governor — and
records the same bounded, deterministic metrics for both:

* **time-above-ceiling** — measured ticks where any controlled zone exceeds
  the declared crew-cabin CO2 ceiling, the primary comfort success metric;
* **max excursion** — the largest single-tick exceedance above that ceiling;
* **response latency** — ticks from fault onset until the governor leaves its
  nominal policy on the affected loop;
* **energy** — cumulative actuator power over the measured run, so any
  mitigation's energy overhead is explicit;
* **invariant violations** — measured ticks where total delivered airflow
  exceeds shared capacity (the conservation invariant).

Reference (healthy) runs are evaluated under both controllers to verify the
bounded response never degrades healthy operation.

The emitted JSON receipt records the experiment environment using the same
source/provenance fields as the experiment receipt. It binds exact source,
project-configuration, base-scenario and sweep bytes with SHA-256, while also
recording canonical hashes for the sweep and generated scenario manifest. All
keys are repository-relative logical names; output paths are never hashed.
The `source_worktree_dirty` flag is the repository's actual Git status and is
never inferred as clean from a generated receipt.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from aeolus.families import ScenarioFamily, load_family_manifest
from aeolus.response import BoundedRecoveryGovernor, ResponseSettings
from aeolus.scenario import RunSpec, STANDARD_RUN, run_governed_scenario, run_scenario
from aeolus.sweep import generate_sweep, load_sweep_spec

EVIDENCE_VERSION = "aeolus_response_evidence_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
USAGE = (
    "Usage: PYTHONPATH=src python -m aeolus.response_evidence "
    "<sweep.json> <output-dir>"
)
GovernorFactory = Callable[[Any], BoundedRecoveryGovernor]


def default_governor_factory(config: Any) -> BoundedRecoveryGovernor:
    """The policy under test: a governor over the default response settings."""
    return BoundedRecoveryGovernor(config, settings=ResponseSettings())


def _factory_name(governor_factory: GovernorFactory) -> str:
    return getattr(governor_factory, "__name__", type(governor_factory).__name__)


def _evaluated_response_settings(
    *, sweep_spec: Any, governor_factory: GovernorFactory
) -> dict[str, Any]:
    """Capture settings from the factory instance used by the evidence run."""
    from aeolus.config import load_scenario

    governor = governor_factory(load_scenario(sweep_spec.base_scenario_path))
    return _governor_response_settings(
        governor, factory_name=_factory_name(governor_factory)
    )


def _governor_response_settings(
    governor: Any, *, factory_name: str
) -> dict[str, Any]:
    """Return the declared settings of one governor instance."""
    settings = getattr(governor, "settings", None)
    if settings is None:
        return {"governor_factory": factory_name, "settings_status": "unavailable"}
    if not isinstance(settings, ResponseSettings):
        raise ValueError(
            "response evidence governor settings must be ResponseSettings or unavailable"
        )
    return {"governor_factory": factory_name, **asdict(settings)}


def _receipt_bound_governor_factory(
    governor_factory: GovernorFactory, response_settings: dict[str, Any]
) -> GovernorFactory:
    """Reject factory instances whose declared settings differ from the receipt."""
    factory_name = _factory_name(governor_factory)

    def bound_factory(config: Any) -> BoundedRecoveryGovernor:
        governor = governor_factory(config)
        actual_settings = _governor_response_settings(
            governor, factory_name=factory_name
        )
        if actual_settings != response_settings:
            raise ValueError(
                "response evidence governor factory settings differ from "
                "receipt-bound settings"
            )
        return governor

    return bound_factory


def _provenance_response_settings(response_settings: dict[str, Any]) -> dict[str, Any]:
    """Preserve the provenance schema while binding evaluated factory settings."""
    if response_settings.get("settings_status") == "unavailable":
        return dict(response_settings)
    return {
        "governor_factory": response_settings["governor_factory"],
        "settings": {
            key: value
            for key, value in response_settings.items()
            if key != "governor_factory"
        },
    }


def _git_output(*args: str) -> str:
    """Return a repository-local Git command result for an evidence receipt."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    """Hash one exact file byte stream."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(document: object) -> str:
    """Hash a path-independent canonical JSON representation."""
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_file_hashes() -> dict[str, str]:
    """Hash every repository source module that can affect a replay."""
    source_root = REPOSITORY_ROOT / "src" / "aeolus"
    paths = sorted(
        path for path in source_root.rglob("*.py") if path.is_file()
    )
    if not paths:
        raise OSError(f"canonical response evidence requires source files: {source_root}")
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256_file(path)
        for path in paths
    }


def _required_project_file_hashes() -> dict[str, str]:
    """Hash the locked project configuration using stable logical names."""
    paths = {
        "pyproject.toml": REPOSITORY_ROOT / "pyproject.toml",
        "uv.lock": REPOSITORY_ROOT / "uv.lock",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise OSError(
            "canonical response evidence requires project files: "
            + ", ".join(missing)
        )
    return {name: _sha256_file(path) for name, path in paths.items()}


def _generated_scenario_hashes(corpus_dir: Path) -> dict[str, str]:
    """Hash generated scenario bytes without including the corpus path."""
    paths = sorted(
        path
        for path in corpus_dir.rglob("*.json")
        if path.is_file() and path.name != "families.json"
    )
    return {
        path.relative_to(corpus_dir).as_posix(): _sha256_file(path)
        for path in paths
    }


def _receipt_provenance(
    *,
    sweep_spec: Any,
    corpus_dir: Path,
    manifest: Any,
    response_settings: dict[str, Any],
) -> dict[str, Any]:
    """Return reproducibility provenance with no output-path-dependent values."""
    source_files = _source_file_hashes()
    project_files = _required_project_file_hashes()
    run_spec = asdict(STANDARD_RUN)
    provenance_response_settings = _provenance_response_settings(response_settings)
    generated_scenarios = _generated_scenario_hashes(corpus_dir)
    lock_hash = project_files["uv.lock"]
    return {
        "environment": {
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": importlib.metadata.version("numpy"),
            "uv_lock_sha256": lock_hash,
            "source_commit": _git_output("rev-parse", "HEAD"),
            "source_worktree_dirty": bool(_git_output("status", "--porcelain")),
        },
        "source": {
            "files_sha256": source_files,
            "manifest_sha256": _canonical_sha256(source_files),
        },
        "config": {
            "project_files_sha256": project_files,
            "base_scenario_bytes_sha256": _sha256_file(
                sweep_spec.base_scenario_path
            ),
            "run_spec": run_spec,
            "run_spec_sha256": _canonical_sha256(run_spec),
            "response_settings": provenance_response_settings,
            "response_settings_sha256": _canonical_sha256(
                provenance_response_settings
            ),
        },
        "sweep": {
            "bytes_sha256": _sha256_file(sweep_spec.source_path),
            "canonical_sha256": sweep_spec.sha256,
            "generated_scenarios_sha256": generated_scenarios,
            "generated_scenarios_manifest_sha256": _canonical_sha256(
                generated_scenarios
            ),
            "family_manifest_sha256": manifest.manifest_sha256,
        },
    }


def metrics_for_records(
    records,
    *,
    ceiling: float,
    zone_ids: Sequence[str],
) -> dict[str, float]:
    """Collapse one run's tick records into bounded scalar metrics."""
    time_above_ceiling = 0
    max_excursion = 0.0
    energy = 0.0
    invariant_violations = 0
    for record in records:
        for zone_id in zone_ids:
            concentration = record.zones[zone_id]["co2_concentration"]
            if concentration > ceiling:
                time_above_ceiling += 1
                max_excursion = max(max_excursion, concentration - ceiling)
        for actuator in record.actuators.values():
            energy += actuator["power"]
        if (
            record.system["total_delivered_airflow"]
            > record.system["shared_airflow_capacity"] + 1e-9
        ):
            invariant_violations += 1
    return {
        "time_above_ceiling": time_above_ceiling,
        "max_excursion": max_excursion,
        "energy": energy,
        "invariant_violations": invariant_violations,
    }


def response_latency_ticks(
    rationale_history: Sequence[dict[str, dict[str, Any]]],
    onset_tick: int | None,
    affected_zone_ids: Sequence[str] = (),
) -> int | None:
    """Return ticks between onset and the first mitigation on an affected loop.

    Only rationale entries for zone IDs that the fault profile actually touches
    count, so rate limits or holds on unrelated healthy loops are not mistaken
    for a response to the degraded loop. With no affected zone IDs the caller
    opts into the loose any-zone definition.
    """
    scope = tuple(affected_zone_ids) or None
    if onset_tick is None:
        return None
    for tick_index, rationale in enumerate(rationale_history):
        measured_tick = tick_index + 1
        if measured_tick < onset_tick:
            continue
        zones = scope if scope is not None else tuple(rationale)
        if any(
            rationale[zone_id]["reason"] != "nominal"
            for zone_id in zones
            if zone_id in rationale
        ):
            return measured_tick - onset_tick
    return None


def _affected_command_zones(config: Any) -> tuple[str, ...]:
    """Commanded zone IDs whose loop a fault profile actually degrades.

    Connection faults affect the zone served by that connection (matched on
    either the outbound or the return path); sensor faults affect the zone
    whose sensor is frozen. Falls back to every commanded zone when no match
    is found so the metric never silently goes silent.
    """
    commanded = tuple(zone.id for zone in config.non_processing_zones())
    affected: set[str] = set()
    for fault in config.connection_faults():
        for zone_id in commanded:
            if config.path_to_processing(zone_id).id == fault.connection_id:
                affected.add(zone_id)
            if config.path_from_processing(zone_id).id == fault.connection_id:
                affected.add(zone_id)
    for fault in config.sensor_faults():
        if fault.zone_id in commanded:
            affected.add(fault.zone_id)
    return tuple(sorted(affected)) or commanded


def evaluate_family(
    family: ScenarioFamily,
    *,
    corpus_dir: Path,
    governor_factory: GovernorFactory,
    run: RunSpec = STANDARD_RUN,
) -> dict[str, Any]:
    """Evaluate one family under baseline and governed controllers."""
    fault_config = _load_scenario(corpus_dir, family.fault_path)
    reference_config = _load_scenario(corpus_dir, family.reference_path)
    zone_ids = [zone.id for zone in fault_config.non_processing_zones()]

    baseline_fault = run_scenario(fault_config, run=run)
    baseline_reference = run_scenario(reference_config, run=run)

    governed = governor_factory(fault_config)
    governed_fault = run_governed_scenario(
        fault_config, governed, run=run
    )
    governed_fault_rationale = list(governed.rationale_history)
    governed_reference = run_governed_scenario(
        reference_config, governor_factory(reference_config), run=run
    )

    reason_counts: dict[str, int] = {}
    for tick in governed_fault_rationale:
        for entry in tick.values():
            reason = entry["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "family_id": family.family_id,
        "fault_class": family.fault_class,
        "split": family.split,
        "onset_tick": _fault_onset_tick(fault_config),
        "fault": _compare_metrics(
            metrics_for_records(
                baseline_fault,
                ceiling=run.crew_cabin_co2_concentration_ceiling,
                zone_ids=zone_ids,
            ),
            metrics_for_records(
                governed_fault,
                ceiling=run.crew_cabin_co2_concentration_ceiling,
                zone_ids=zone_ids,
            ),
        ),
        "reference": _compare_metrics(
            metrics_for_records(
                baseline_reference,
                ceiling=run.crew_cabin_co2_concentration_ceiling,
                zone_ids=zone_ids,
            ),
            metrics_for_records(
                governed_reference,
                ceiling=run.crew_cabin_co2_concentration_ceiling,
                zone_ids=zone_ids,
            ),
        ),
        "governed_action_ticks": {
            reason: count for reason, count in sorted(reason_counts.items())
        },
        "response_latency_ticks": response_latency_ticks(
            governed_fault_rationale,
            _fault_onset_tick(fault_config),
            affected_zone_ids=_affected_command_zones(fault_config),
        ),
    }


def run_response_evidence(
    sweep_path: str | Path,
    output_dir: str | Path,
    *,
    governor_factory: GovernorFactory = default_governor_factory,
) -> dict[str, Any]:
    """Generate the sweep corpus and evaluate every family under both policies."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"response evidence output directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    sweep_spec = load_sweep_spec(sweep_path)
    response_settings = _evaluated_response_settings(
        sweep_spec=sweep_spec, governor_factory=governor_factory
    )
    bound_governor_factory = _receipt_bound_governor_factory(
        governor_factory, response_settings
    )
    corpus_dir = destination / "corpus"
    corpus_dir.mkdir()
    generate_sweep(sweep_path, corpus_dir)
    manifest = load_family_manifest(corpus_dir / "families.json")

    rows = [
        evaluate_family(
            family, corpus_dir=corpus_dir, governor_factory=bound_governor_factory
        )
        for family in manifest.families
    ]
    provenance = _receipt_provenance(
        sweep_spec=sweep_spec,
        corpus_dir=corpus_dir,
        manifest=manifest,
        response_settings=response_settings,
    )
    receipt = {
        "evidence_version": EVIDENCE_VERSION,
        **provenance,
        "sweep_spec_sha256": sweep_spec.sha256,
        "family_manifest_sha256": manifest.manifest_sha256,
        "response_settings": response_settings,
        "families_evaluated": len(rows),
        "per_family": rows,
        "aggregate": _aggregate(rows),
        "conclusion": _conclusion(rows),
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    receipt["evidence_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(destination / "response-evidence.json", receipt)
    return receipt


def _compare_metrics(baseline: dict[str, float], governed: dict[str, float]) -> dict:
    comparison = {}
    for metric, baseline_value in baseline.items():
        governed_value = governed[metric]
        delta = governed_value - baseline_value
        entry: dict[str, Any] = {
            "baseline": baseline_value,
            "governed": governed_value,
            "delta": delta,
        }
        if metric == "energy" and baseline_value > 0.0:
            entry["overhead_fraction"] = delta / baseline_value
        comparison[metric] = entry
    return comparison


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    time_deltas = [row["fault"]["time_above_ceiling"]["delta"] for row in rows]
    excursion_deltas = [row["fault"]["max_excursion"]["delta"] for row in rows]
    energy_overheads = [
        row["fault"]["energy"].get("overhead_fraction", 0.0) for row in rows
    ]
    latencies = [row["response_latency_ticks"] for row in rows]
    latency_values = [latency for latency in latencies if latency is not None]
    improved = sum(1 for delta in time_deltas if delta < 0)
    matched = sum(1 for delta in time_deltas if delta == 0)
    worse = sum(1 for delta in time_deltas if delta > 0)
    reference_excess = [
        row["reference"]["time_above_ceiling"]["delta"] for row in rows
    ]
    fault_excess = [
        row["fault"]["time_above_ceiling"]["delta"] for row in rows
    ]
    spare_families = sum(
        1
        for row in rows
        if row["governed_action_ticks"].get("degraded_spare_release", 0) > 0
    )
    frozen_families = sum(
        1
        for row in rows
        if row["governed_action_ticks"].get("frozen_hold", 0) > 0
    )
    return {
        "time_above_ceiling": {
            "mean_delta_ticks": _mean(time_deltas),
            "improved": improved,
            "matched": matched,
            "worse": worse,
        },
        "causality_margin": {
            "margin_ticks": 1,
            "fault_max_excess": max(fault_excess) if fault_excess else 0,
            "fault_families_exceeding_margin": sum(
                1 for delta in fault_excess if delta > 1
            ),
            "healthy_max_excess": max(reference_excess) if reference_excess else 0,
            "healthy_families_exceeding_margin": sum(
                1 for delta in reference_excess if delta > 1
            ),
        },
        "max_excursion": {
            "mean_delta": _mean(excursion_deltas),
        },
        "energy": {
            "mean_overhead_fraction": _mean(energy_overheads),
            "median_overhead_fraction": _median(energy_overheads),
        },
        "invariant_violations": {
            "baseline_total": sum(
                row["fault"]["invariant_violations"]["baseline"] for row in rows
            ),
            "governed_total": sum(
                row["fault"]["invariant_violations"]["governed"] for row in rows
            ),
        },
        "response_latency": {
            "median_ticks": _median(latency_values),
            "families_with_response": len(latency_values),
        },
        "governed_actions": {
            "families_with_spare_release": spare_families,
            "families_with_frozen_hold": frozen_families,
        },
    }


def _conclusion(rows: Sequence[dict[str, Any]]) -> str:
    aggregate = _aggregate(rows)
    total = len(rows)
    matched = aggregate["time_above_ceiling"]["matched"]
    margin = aggregate["causality_margin"]
    within_fault = total - margin["fault_families_exceeding_margin"]
    within_healthy = total - margin["healthy_families_exceeding_margin"]
    violations = aggregate["invariant_violations"]
    median_latency = aggregate["response_latency"]["median_ticks"]
    median_overhead = aggregate["energy"]["median_overhead_fraction"]
    spare = aggregate["governed_actions"]["families_with_spare_release"]
    return (
        f"bounded response ran at exact baseline parity on {matched}/{total} fault "
        f"families and within its {margin['margin_ticks']}-tick causality margin on "
        f"{within_fault}/{total} (max excess {margin['fault_max_excess']} ticks, "
        f"{margin['fault_families_exceeding_margin']} beyond margin); healthy-reference "
        f"runs stayed within the same margin on {within_healthy}/{total} "
        f"(max excess {margin['healthy_max_excess']}, "
        f"{margin['healthy_families_exceeding_margin']} beyond); "
        f"{violations['baseline_total']} baseline / {violations['governed_total']} "
        f"governed invariant violations; median response latency "
        f"{median_latency} ticks; median energy overhead {median_overhead * 100:.2f}% "
        f"with spare-capacity release acting in {spare}/{total} families"
    )


def _fault_onset_tick(config: Any) -> int | None:
    profiles = config.fault_profiles
    if not profiles:
        return None
    onset = getattr(profiles[0], "start_tick", None)
    return onset if onset is not None else None


def _load_scenario(corpus_dir: Path, path: Path) -> Any:
    from aeolus.config import load_scenario

    return load_scenario(corpus_dir / path.name)


def _write_json(path: Path, document: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        receipt = run_response_evidence(argv[0], argv[1])
    except (ValueError, OSError) as exc:
        print(f"cannot run response evidence: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main(sys.argv[1:]))
