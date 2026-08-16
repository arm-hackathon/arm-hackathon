"""Deterministically rebuild the Habitat V2 observability qualification packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import load_scenario_file
from .observability import (
    OPERATIONAL_FEATURE_MANIFEST_SHA256,
    OperationalTrace,
    RawV5Trace,
    project_v5_trace,
)
from .qualification import (
    QualificationCase,
    aggregate_qualification_metrics,
    build_pair_manifest,
    evaluate_hard_negative,
    qualify_pair,
)
from .runner import run_scenario
from .scenario import Scenario

QUALIFICATION_CASES = (
    ("fan_degradation", "air_network"),
    ("branch_resistance", "air_network"),
    ("damper_jam", "air_network"),
    ("primary_sensor_drift", "instrumentation"),
    ("primary_sensor_stuck", "instrumentation"),
    ("ambiguous", None),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_qualification_packet(source_root: Path) -> bytes:
    """Rebuild the canonical packet from tracked scenarios and simulator code."""
    scenario_directory = source_root / "scenarios" / "habitat_v2_observability"

    def load(name: str) -> Scenario:
        return load_scenario_file(scenario_directory / f"{name}.json")

    def operational(name: str) -> OperationalTrace:
        parsed = load(name)
        raw = RawV5Trace.from_trace_bytes(
            run_scenario(parsed).trace_bytes,
            scenario=parsed,
            fixture_id=name,
        )
        return project_v5_trace(raw)

    healthy_scenario = load("healthy_nominal")
    healthy = operational("healthy_nominal")
    records: dict[str, object] = {}
    cases: list[QualificationCase] = []

    for name, expected_subsystem in QUALIFICATION_CASES:
        fault_scenario = load(name)
        fault = operational(name)
        manifest = build_pair_manifest(
            healthy=healthy_scenario,
            fault=fault_scenario,
            treatment_fault_ids=tuple(
                profile["id"] for profile in fault_scenario.data["fault_profiles"]
            ),
        )
        report = qualify_pair(healthy, fault, pair_manifest=manifest)
        records[name] = {
            "operational_provenance": {
                "fault_fixture_id": fault.fixture_id,
                "fault_run_id": fault.run_id,
                "fault_scenario_sha256": fault.scenario_sha256,
                "fault_source_trace_sha256": fault.source_trace_sha256,
                "healthy_fixture_id": healthy.fixture_id,
                "healthy_run_id": healthy.run_id,
                "healthy_scenario_sha256": healthy.scenario_sha256,
                "healthy_source_trace_sha256": healthy.source_trace_sha256,
            },
            "pair_manifest": manifest.as_canonical_mapping(),
            "report": report.as_canonical_mapping(),
            "report_bytes_sha256": hashlib.sha256(report.canonical_bytes()).hexdigest(),
        }
        cases.append(
            QualificationCase(
                report=report,
                expected_concern=True,
                expected_subsystem=expected_subsystem,
                localisation_eligible=expected_subsystem is not None,
            )
        )

    hard_negative = evaluate_hard_negative(operational("healthy_elevated"))
    aggregate = aggregate_qualification_metrics(
        tuple(cases), hard_negatives=(hard_negative,)
    )
    return _canonical_bytes(
        {
            "aggregate": aggregate.as_canonical_mapping(),
            "feature_manifest_sha256": OPERATIONAL_FEATURE_MANIFEST_SHA256,
            "hard_negative": hard_negative.as_canonical_mapping(),
            "healthy_source_trace_sha256": healthy.source_trace_sha256,
            "records": records,
        }
    )
