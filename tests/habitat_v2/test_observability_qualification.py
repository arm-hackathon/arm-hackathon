from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from aeolus.habitat_v2.config import load_scenario_file
from aeolus.habitat_v2.observability import (
    OPERATIONAL_FEATURE_MANIFEST,
    OPERATIONAL_FEATURE_MANIFEST_ID,
    OPERATIONAL_FEATURE_MANIFEST_SHA256,
    OperationalProjectionError,
    OperationalTrace,
    RawV5Trace,
    project_v5_trace,
)
from aeolus.habitat_v2.qualification import (
    OBSERVABILITY_QUALIFICATION_ID,
    AggregateMetrics,
    PairManifest,
    PairValidationError,
    QualificationCase,
    QualificationError,
    aggregate_qualification_metrics,
    build_pair_manifest,
    evaluate_hard_negative,
    qualify_pair,
)
from aeolus.habitat_v2.runner import run_scenario

SCENARIOS = Path(__file__).parents[2] / "scenarios" / "habitat_v2_observability"


def _scenario(name: str):
    return load_scenario_file(SCENARIOS / f"{name}.json")


def _operational(name: str) -> OperationalTrace:
    scenario = _scenario(name)
    raw = RawV5Trace.from_trace_bytes(
        run_scenario(scenario).trace_bytes, scenario=scenario, fixture_id=name
    )
    return project_v5_trace(raw)


def _manifest(name: str) -> PairManifest:
    fault = _scenario(name)
    return build_pair_manifest(
        healthy=_scenario("healthy_nominal"),
        fault=fault,
        treatment_fault_ids=tuple(profile["id"] for profile in fault.data["fault_profiles"]),
    )


def _report(name: str):
    return qualify_pair(_operational("healthy_nominal"), _operational(name), pair_manifest=_manifest(name))


def test_operational_feature_manifest_is_ordered_descriptors_with_identity_hash() -> None:
    assert OPERATIONAL_FEATURE_MANIFEST_ID
    assert len(OPERATIONAL_FEATURE_MANIFEST_SHA256) == 64
    assert isinstance(OPERATIONAL_FEATURE_MANIFEST, tuple)
    assert [descriptor["ordinal"] for descriptor in OPERATIONAL_FEATURE_MANIFEST] == list(range(len(OPERATIONAL_FEATURE_MANIFEST)))
    for descriptor in OPERATIONAL_FEATURE_MANIFEST:
        assert set(descriptor) == {"ordinal", "path", "source", "units", "completed_timing", "decision_treatment"}
        assert descriptor["completed_timing"] == "completed_step"
        assert descriptor["decision_treatment"] in {"compared", "asserted_equal", "deliberately_unscored"}


def test_projection_is_typed_closed_and_excludes_evaluator_truth() -> None:
    scenario = _scenario("healthy_nominal")
    with pytest.raises(TypeError, match="fixture_id"):
        RawV5Trace.from_trace_bytes(run_scenario(scenario).trace_bytes, scenario=scenario)  # type: ignore[call-arg]
    raw = RawV5Trace.from_trace_bytes(
        run_scenario(scenario).trace_bytes, scenario=scenario, fixture_id="explicit-nominal-fixture"
    )
    operational = project_v5_trace(raw)

    assert operational.fixture_id == "explicit-nominal-fixture"
    assert operational.feature_manifest_id == OPERATIONAL_FEATURE_MANIFEST_ID
    assert operational.rows[1].step == 1
    assert set(operational.rows[1].as_canonical_mapping()) == {
        "step", "time_s", "mode", "primary_telemetry", "secondary_telemetry",
        "primary_minus_secondary", "commanded_action", "actual_action", "operational_feedback",
    }
    encoded = json.dumps(operational.as_canonical_mapping(), sort_keys=True)
    for forbidden in (
        "fault_receipt", "active_fault", "truth_telemetry", "residual",
        "effective", "realised_load", "random_seed", "timeline", "family",
        "actuator_receipt", "air_network_receipt", "resource_state",
    ):
        assert forbidden not in encoded


def test_projection_rejects_mapping_impostors_and_unknown_operational_fields() -> None:
    with pytest.raises(TypeError):
        RawV5Trace((), "forged", "0" * 64, "0" * 64, "0" * 64)  # type: ignore[call-arg]

    scenario = _scenario("healthy_nominal")
    raw = RawV5Trace.from_trace_bytes(
        run_scenario(scenario).trace_bytes, scenario=scenario, fixture_id="immutable-raw"
    )
    with pytest.raises(TypeError):
        raw._rows[0]["step"] = 99  # type: ignore[index]

    with pytest.raises(OperationalProjectionError, match="RawV5Trace"):
        project_v5_trace({})  # type: ignore[arg-type]
    with pytest.raises(QualificationError, match="OperationalTrace"):
        qualify_pair({}, {}, pair_manifest=None)  # type: ignore[arg-type]

    row = _operational("healthy_nominal").rows[0].as_canonical_mapping()
    row["truth_telemetry"] = {}
    with pytest.raises(OperationalProjectionError, match="unknown"):
        OperationalTrace.from_canonical_rows([row])


def test_pair_manifest_removes_only_declared_treatment_fault_ids_and_keeps_name_structural() -> None:
    healthy = _scenario("healthy_nominal")
    fault = _scenario("fan_degradation")
    manifest = build_pair_manifest(
        healthy=healthy, fault=fault, treatment_fault_ids=("qualification-fan-degradation",)
    )
    assert manifest.contract_id == OBSERVABILITY_QUALIFICATION_ID
    assert manifest.healthy_run_id == healthy.run_id
    assert manifest.fault_run_id == fault.run_id
    assert manifest.treatment_fault_ids == ("qualification-fan-degradation",)
    assert manifest.treatment_start_step == 2
    assert manifest.operational_feature_manifest_sha256 == OPERATIONAL_FEATURE_MANIFEST_SHA256
    assert manifest.decision_tolerance_contract_sha256
    assert manifest.actuator_feedback_contract_revision
    assert manifest.actuator_feedback_config_sha256
    assert len(manifest.pair_manifest_sha256) == 64

    changed_name = deepcopy(fault.data)
    changed_name["name"] = "different-name-is-a-structural-difference"
    with pytest.raises(PairValidationError, match="structural"):
        build_pair_manifest(healthy=healthy, fault_mapping=changed_name, treatment_fault_ids=("qualification-fan-degradation",))

    extra_fault = deepcopy(fault.data)
    extra_fault["fault_profiles"].append({
        "id": "undeclared-preserved-fault",
        "type": "fan_speed_degradation",
        "start_step": 2,
        "end_step": 6,
        "start_multiplier": 0.9,
        "end_multiplier": 0.9,
    })
    with pytest.raises(PairValidationError, match="structural"):
        build_pair_manifest(healthy=healthy, fault_mapping=extra_fault, treatment_fault_ids=("qualification-fan-degradation",))


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("sensor_model", "random_seed"), 9),
        (("timeline", 0, "loads", "laboratory", "sensible_heat_w"), 777.0),
        (("timeline", 0, "command", "fan_speed_fraction"), 0.11),
        (("zones", 0, "volume_m3"), 999.0),
        (("actuator_feedback", "cooling_slew_w_per_s"), 6.0),
        (("dt_seconds",), 10.0),
    ),
)
def test_pair_manifest_rejects_all_non_treatment_structural_differences(path, value) -> None:
    healthy = _scenario("healthy_nominal")
    mapping = deepcopy(_scenario("fan_degradation").data)
    target = mapping
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(PairValidationError, match="structural"):
        build_pair_manifest(healthy=healthy, fault_mapping=mapping, treatment_fault_ids=("qualification-fan-degradation",))


def test_tolerances_match_zone_and_component_intermediate_paths() -> None:
    from aeolus.habitat_v2.qualification import tolerance_for_path

    assert tolerance_for_path("primary_telemetry.laboratory.co2_ppm") == float("inf")
    assert tolerance_for_path("primary_minus_secondary.laboratory.co2_ppm") == 10.0
    assert tolerance_for_path("operational_feedback.damper_position_by_id.laboratory_supply_damper") == 0.025
    assert tolerance_for_path("operational_feedback.branch_airflow_m3_s.laboratory") == 0.001


def test_half_open_treatment_uses_decision_time_latency() -> None:
    healthy = _operational("healthy_nominal")
    fault = _operational("fan_degradation")
    assert healthy.rows[0] == fault.rows[0]
    assert healthy.rows[1] == fault.rows[1]
    report = qualify_pair(healthy, fault, pair_manifest=_manifest("fan_degradation"))
    assert report.earliest_divergence_step == 2
    assert report.treatment_start_step == 2
    assert report.first_divergence_step == 2
    assert report.decision_step == 3
    assert report.detection_latency_steps == 1
    assert report.detection_latency_seconds == 60.0
    assert report.window_steps == 2


@pytest.mark.parametrize(
    ("fixture", "expected"),
    (
        ("fan_degradation", "SUBSYSTEM_LOCALISED"),
        ("branch_resistance", "SUBSYSTEM_LOCALISED"),
        ("damper_jam", "SUBSYSTEM_LOCALISED"),
        ("primary_sensor_drift", "SUBSYSTEM_LOCALISED"),
        ("primary_sensor_stuck", "SUBSYSTEM_LOCALISED"),
    ),
)
def test_representative_fault_families_produce_honest_reports(fixture: str, expected: str) -> None:
    report = _report(fixture)
    assert report.outcome == expected
    assert report.detection_latency_steps is not None
    assert report.detection_latency_seconds is not None


def test_hard_negative_evaluates_elevated_fixture_against_operational_boundary() -> None:
    result = evaluate_hard_negative(_operational("healthy_elevated"))
    assert result.fixture_id == "healthy_elevated"
    assert result.outcome == "NO_OBSERVABLE_CONCERN"
    assert result.false_concern is False
    assert result.operational_decision_boundary_sha256


def test_ambiguous_fixture_abstains_and_abnormal_outcome_remains_explicit() -> None:
    report = _report("ambiguous")
    assert report.outcome == "UNKNOWN"
    assert report.subsystem == "UNKNOWN"
    assert report.detection_latency_steps is not None
    assert "ABNORMAL_OPERATION" in __import__("aeolus.habitat_v2.qualification", fromlist=["OUTCOMES"]).OUTCOMES


def test_reports_bind_pair_and_trace_hashes_and_reject_identity_substitution() -> None:
    report = _report("fan_degradation")
    assert report.pair_manifest_sha256 == _manifest("fan_degradation").pair_manifest_sha256
    assert report.healthy_trace_sha256 == _operational("healthy_nominal").source_trace_sha256
    assert report.fault_trace_sha256 == _operational("fan_degradation").source_trace_sha256
    assert report.report_sha256

    for field in ("decision_contract_id", "pair_manifest_sha256", "healthy_trace_sha256", "fault_fixture_id"):
        mutated = report.as_canonical_mapping()
        mutated[field] = "other"
        with pytest.raises(QualificationError):
            type(report).from_canonical_mapping(mutated)

    manifest = _manifest("fan_degradation")
    mutated_manifest = manifest.as_canonical_mapping()
    mutated_manifest["operational_feature_manifest_sha256"] = "other"
    with pytest.raises(PairValidationError, match="feature manifest"):
        PairManifest.from_canonical_mapping(mutated_manifest)


def test_duplicate_qualification_is_byte_identical_and_fixture_identity_must_be_distinct() -> None:
    healthy = _operational("healthy_nominal")
    fault = _operational("fan_degradation")
    manifest = _manifest("fan_degradation")
    assert qualify_pair(healthy, fault, pair_manifest=manifest).canonical_bytes() == qualify_pair(healthy, fault, pair_manifest=manifest).canonical_bytes()
    with pytest.raises(QualificationError, match="distinct"):
        qualify_pair(healthy, healthy, pair_manifest=manifest)


def test_aggregate_metrics_expose_polarity_denominators_and_outcome_eligibility() -> None:
    fan = _report("fan_degradation")
    hard_negative = evaluate_hard_negative(_operational("healthy_elevated"))
    aggregate = aggregate_qualification_metrics((
        QualificationCase(report=fan, expected_concern=True, expected_subsystem="air_network", localisation_eligible=True),
        QualificationCase(report=_report("ambiguous"), expected_concern=True, expected_subsystem=None, localisation_eligible=False),
    ), hard_negatives=(hard_negative,))
    assert isinstance(aggregate, AggregateMetrics)
    assert aggregate.concern_coverage_denominator == 2
    assert aggregate.concern_coverage_numerator == 2
    assert aggregate.eligible_localisation_denominator == 1
    assert aggregate.healthy_false_concern_count == 0
    assert aggregate.latency_null_non_detection_count == 0
    assert aggregate.ambiguous_abstention_denominator == 1
    assert aggregate.ambiguous_abstention_numerator == 1
    assert aggregate.overclaim_count == 0


def test_manifest_is_deeply_immutable_and_hash_cannot_drift() -> None:
    descriptor = OPERATIONAL_FEATURE_MANIFEST[0]
    with pytest.raises(TypeError):
        descriptor["path"] = "forged"  # type: ignore[index]
    assert OPERATIONAL_FEATURE_MANIFEST_SHA256 == "ea9920963a3a3d50533ac4b20912fbc331a6e45a8ef2d84a958316805b60e9e4"


def test_projection_binds_validated_scenario_and_run_lineage() -> None:
    scenario = _scenario("fan_degradation")
    trace_bytes = run_scenario(scenario).trace_bytes
    projected = project_v5_trace(
        RawV5Trace.from_trace_bytes(trace_bytes, scenario=scenario, fixture_id="fan-fixture")
    )
    assert projected.scenario_sha256 == scenario.scenario_sha256
    assert projected.run_id == scenario.run_id
    assert projected.source_trace_sha256
    with pytest.raises(OperationalProjectionError, match="provenance"):
        OperationalTrace.from_canonical_rows(
            [row.as_canonical_mapping() for row in projected.rows], fixture_id="forged"
        )


def test_qualification_rejects_trace_substitution_before_scoring() -> None:
    healthy = _operational("healthy_nominal")
    wrong_treatment = _operational("branch_resistance")
    with pytest.raises(QualificationError, match="scenario"):
        qualify_pair(healthy, wrong_treatment, pair_manifest=_manifest("fan_degradation"))

    treatment = _operational("fan_degradation")
    reconstructed = OperationalTrace.from_canonical_rows(
        [row.as_canonical_mapping() for row in treatment.rows],
        fixture_id=treatment.fixture_id,
        source_trace_sha256=treatment.source_trace_sha256,
        scenario_sha256=treatment.scenario_sha256,
        run_id=treatment.run_id,
    )
    with pytest.raises(QualificationError, match="validated V5 projection"):
        qualify_pair(healthy, reconstructed, pair_manifest=_manifest("fan_degradation"))

    substituted_manifest = _manifest("fan_degradation")
    object.__setattr__(substituted_manifest, "fault_run_id", "0" * 64)
    remapped = substituted_manifest.as_canonical_mapping()
    _rehash(remapped, "pair_manifest_sha256")
    object.__setattr__(
        substituted_manifest,
        "pair_manifest_sha256",
        remapped["pair_manifest_sha256"],
    )
    with pytest.raises(QualificationError, match="run"):
        qualify_pair(healthy, treatment, pair_manifest=substituted_manifest)


def test_pair_manifest_rejects_healthy_control_containing_declared_treatment() -> None:
    fault = _scenario("fan_degradation")
    with pytest.raises(PairValidationError, match="healthy counterpart"):
        build_pair_manifest(
            healthy=fault,
            fault=fault,
            treatment_fault_ids=("qualification-fan-degradation",),
        )


def test_treatment_ids_are_explicit_not_inferred_from_profiles() -> None:
    with pytest.raises(PairValidationError, match="explicit"):
        build_pair_manifest(
            healthy=_scenario("healthy_nominal"), fault=_scenario("fan_degradation")
        )


def test_decision_time_latency_keeps_first_divergence_separate() -> None:
    report = qualify_pair(
        _operational("healthy_nominal"),
        _operational("fan_degradation"),
        pair_manifest=build_pair_manifest(
            healthy=_scenario("healthy_nominal"),
            fault=_scenario("fan_degradation"),
            treatment_fault_ids=("qualification-fan-degradation",),
        ),
    )
    assert report.first_divergence_step == 2
    assert report.decision_step == 3
    assert report.detection_latency_steps == 1


def test_report_parser_rejects_malformed_hashes_and_invalid_outcome_combinations() -> None:
    report = _report("fan_degradation").as_canonical_mapping()
    report["healthy_trace_sha256"] = "not-a-hash"
    with pytest.raises(QualificationError, match="identity"):
        type(_report("fan_degradation")).from_canonical_mapping(report)

    report = _report("fan_degradation").as_canonical_mapping()
    report["outcome"] = "NO_OBSERVABLE_CONCERN"
    report["localisation"] = "air_network"
    report["report_sha256"] = __import__("hashlib").sha256(
        json.dumps({key: value for key, value in report.items() if key != "report_sha256"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    with pytest.raises(QualificationError, match="outcome"):
        type(_report("fan_degradation")).from_canonical_mapping(report)


def test_hard_negative_and_aggregate_artifacts_are_provenance_bound() -> None:
    hard_negative = evaluate_hard_negative(_operational("healthy_elevated"))
    assert hard_negative.source_trace_sha256 == _operational("healthy_elevated").source_trace_sha256
    assert hard_negative.feature_manifest_sha256 == OPERATIONAL_FEATURE_MANIFEST_SHA256
    assert hard_negative.result_sha256
    aggregate = aggregate_qualification_metrics((), hard_negatives=(hard_negative,))
    assert aggregate.aggregate_sha256


def test_fixtures_have_clearance_recovery_and_stable_post_treatment_tail() -> None:
    for fixture in ("fan_degradation", "branch_resistance", "damper_jam", "primary_sensor_drift", "primary_sensor_stuck", "ambiguous"):
        scenario = _scenario(fixture)
        rows = _operational(fixture).rows
        assert scenario.data["steps"] >= 10
        assert len(rows) >= 11
        assert all(profile["end_step"] <= scenario.data["steps"] - 4 for profile in scenario.data["fault_profiles"])
        assert rows[6].step == 6
        assert rows[-1].step >= 10


def test_report_binds_half_open_treatment_end_and_scores_temporal_phases() -> None:
    report = _report("fan_degradation")

    assert report.treatment_start_step == 2
    assert report.treatment_end_step == 6
    assert report.phase_bounds == {
        "baseline": (0, 2),
        "treatment": (2, 6),
        "recovery": (6, 9),
        "post_recovery": (9, 11),
    }
    assert report.phase_persistent_concern == {
        "baseline": False,
        "treatment": True,
        "recovery": False,
        "post_recovery": False,
    }
    assert report.clearance_decision_step == 7
    assert report.post_recovery_stable is True


def test_pair_manifest_rejects_mixed_treatment_intervals() -> None:
    mapping = deepcopy(_scenario("ambiguous").data)
    mapping["fault_profiles"][1]["end_step"] = 7
    with pytest.raises(PairValidationError, match="shared half-open interval"):
        build_pair_manifest(
            healthy=_scenario("healthy_nominal"),
            fault_mapping=mapping,
            treatment_fault_ids=(
                "qualification-fan-degradation",
                "qualification-primary-co2-drift",
            ),
        )


def _rehash(mapping: dict, identity_field: str) -> None:
    mapping[identity_field] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in mapping.items() if key != identity_field},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_report_parser_rejects_inconsistent_temporal_receipts() -> None:
    report = _report("fan_degradation").as_canonical_mapping()
    report["post_recovery_stable"] = False
    _rehash(report, "report_sha256")
    with pytest.raises(QualificationError, match="temporal"):
        type(_report("fan_degradation")).from_canonical_mapping(report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("healthy_fixture_id", "", "fixture"),
        ("detection_latency_seconds", "nan", "latency"),
        ("persistent_channels", ["same", "same"], "persistent"),
        ("post_recovery_stable", 1, "stable-tail"),
    ),
)
def test_report_parser_rejects_malformed_semantic_types(field, value, message) -> None:
    report = _report("fan_degradation").as_canonical_mapping()
    report[field] = value
    _rehash(report, "report_sha256")
    with pytest.raises(QualificationError, match=message):
        type(_report("fan_degradation")).from_canonical_mapping(report)


def test_report_parser_rejects_inconsistent_decision_latency_even_when_rehashed() -> None:
    report = _report("fan_degradation").as_canonical_mapping()
    report["decision_step"] = 4
    _rehash(report, "report_sha256")
    with pytest.raises(QualificationError, match="decision"):
        type(_report("fan_degradation")).from_canonical_mapping(report)


def test_aggregate_parser_rejects_unvalidated_case_and_tracks_overclaim_denominator() -> None:
    report = _report("fan_degradation")
    case = QualificationCase(
        report=report,
        expected_concern=True,
        expected_subsystem="not-a-subsystem",
        localisation_eligible=True,
    )
    with pytest.raises(QualificationError, match="case"):
        aggregate_qualification_metrics((case,), hard_negatives=())

    aggregate = aggregate_qualification_metrics(
        (
            QualificationCase(
                report=report,
                expected_concern=True,
                expected_subsystem="air_network",
                localisation_eligible=True,
            ),
        ),
        hard_negatives=(),
    )
    assert aggregate.overclaim_denominator == 1


def test_hard_negative_parser_rejects_contract_substitution_even_when_rehashed() -> None:
    result = evaluate_hard_negative(_operational("healthy_elevated"))
    mapping = result.as_canonical_mapping()
    mapping["feature_manifest_sha256"] = "0" * 64
    _rehash(mapping, "result_sha256")
    with pytest.raises(QualificationError, match="feature manifest"):
        type(result).from_canonical_mapping(mapping)


def test_aggregate_parser_rejects_impossible_counts_even_when_rehashed() -> None:
    hard_negative = evaluate_hard_negative(_operational("healthy_elevated"))
    aggregate = aggregate_qualification_metrics((), hard_negatives=(hard_negative,))
    mapping = aggregate.as_canonical_mapping()
    mapping["healthy_false_concern_count"] = 2
    _rehash(mapping, "aggregate_sha256")
    with pytest.raises(QualificationError, match="denominator"):
        type(aggregate).from_canonical_mapping(mapping)


def test_report_parser_rejects_fixed_window_contract_substitution_even_when_rehashed() -> None:
    report = _report("fan_degradation").as_canonical_mapping()
    report["window_steps"] = 3
    report["persistence_steps"] = 3
    report["decision_step"] = 4
    report["detection_latency_steps"] = 2
    report["detection_latency_seconds"] = 120.0
    report["clearance_decision_step"] = 8
    _rehash(report, "report_sha256")
    with pytest.raises(QualificationError, match="window"):
        type(_report("fan_degradation")).from_canonical_mapping(report)


def test_pair_manifest_rejects_tolerance_contract_substitution_even_when_rehashed() -> None:
    mapping = _manifest("fan_degradation").as_canonical_mapping()
    mapping["decision_tolerance_contract_sha256"] = "0" * 64
    _rehash(mapping, "pair_manifest_sha256")
    with pytest.raises(PairValidationError, match="tolerance"):
        PairManifest.from_canonical_mapping(mapping)


def test_qualification_rejects_reconstructed_pair_manifest_before_scoring() -> None:
    reconstructed = PairManifest.from_canonical_mapping(
        _manifest("fan_degradation").as_canonical_mapping()
    )
    with pytest.raises(QualificationError, match="validated pair manifest"):
        qualify_pair(
            _operational("healthy_nominal"),
            _operational("fan_degradation"),
            pair_manifest=reconstructed,
        )


def test_hard_negative_parser_rejects_boundary_contract_substitution_even_when_rehashed() -> None:
    result = evaluate_hard_negative(_operational("healthy_elevated"))
    mapping = result.as_canonical_mapping()
    mapping["operational_decision_boundary_sha256"] = "0" * 64
    _rehash(mapping, "result_sha256")
    with pytest.raises(QualificationError, match="boundary"):
        type(result).from_canonical_mapping(mapping)


def test_aggregate_identity_binds_report_and_grading_case_provenance() -> None:
    fan = aggregate_qualification_metrics(
        (
            QualificationCase(
                report=_report("fan_degradation"),
                expected_concern=True,
                expected_subsystem="air_network",
                localisation_eligible=True,
            ),
        ),
        hard_negatives=(),
    )
    branch = aggregate_qualification_metrics(
        (
            QualificationCase(
                report=_report("branch_resistance"),
                expected_concern=True,
                expected_subsystem="air_network",
                localisation_eligible=True,
            ),
        ),
        hard_negatives=(),
    )
    assert fan.aggregate_sha256 != branch.aggregate_sha256
