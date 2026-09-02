from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_diagnostics import (
    ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION,
    V4_PROVENANCE_FIELDS,
    Issue56V4DiagnosticsError,
    V4CandidateObservation,
    V4ExecutedObservation,
    bootstrap_equal_weight_group_mean,
    candidate_screening_metrics,
    equal_weight_group_mean,
    executed_action_metrics,
    observation_manifest_sha256,
    provenance_manifest_sha256,
    validate_v4_protocol,
    validate_condition_groups,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
    Issue56V4CorpusError,
    _v4_feature_histories,
    collect_v4_family_samples,
    load_v4_samples,
    verify_v4_serialized_trace,
    V4RiskSample,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    EPISODE_STEPS,
    ISSUE56_V3_SCHEMA_VERSION,
    V3_HORIZONS,
    V3HorizonMetric,
    V3PolicyLabel,
    V3RiskSample,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_features import (
    V4_TEMPORAL_FEATURE_COUNT,
    observable_operating_mode,
    v4_observable_action_mask,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v2 import FEATURE_COUNT
from aeolus.habitat_v2.forecast.projection import MODE_ORDER
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model_protocol import (
    Issue56V4ModelProtocolError,
    load_v4_model_protocol,
    validate_v4_model_protocol,
)
from aeolus.habitat_v2.forecast_issue56_action_risk_v4_model import (
    Issue56V4ModelError,
    V4_ACTION_IDS,
    V4_HORIZON_KEYS,
    V4ModelSample,
    V4RiskModel,
    _select_event_thresholds,
)

_DIAGNOSTICS_TEST_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(script_name: str):
    spec_path = _DIAGNOSTICS_TEST_REPO_ROOT / "scripts" / f"{script_name}.py"
    module_spec = importlib.util.spec_from_file_location(
        f"{script_name}_under_test", spec_path
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


_run_action_risk_v4_model_script = _load_script_module("run_action_risk_v4_model")
_evaluation_gate_status = _run_action_risk_v4_model_script._evaluation_gate_status
_group_bootstrap = _run_action_risk_v4_model_script._group_bootstrap
from aeolus.habitat_v2.forecast_issue55_race import (
    build_family_scenario,
    deterministic_family_ids,
)
from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _candidate(
    group: str,
    family: str,
    step: int,
    action: str,
    *,
    rejected: bool,
    dangerous: bool,
) -> V4CandidateObservation:
    body = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.candidate",
        "condition_group_id": group,
        "family_id": family,
        "decision_step": step,
        "action_id": action,
        "model_rejected": rejected,
        "dangerous": dangerous,
    }
    return V4CandidateObservation(
        group,
        family,
        step,
        action,
        rejected,
        dangerous,
        _digest(body),
    )


def _executed(
    group: str,
    family: str,
    arm: str,
    step: int,
    *,
    selected: str | None,
    dangerous: bool,
    requested: str | None,
    final: str,
    disposition: str,
) -> V4ExecutedObservation:
    body = {
        "schema_version": f"{ISSUE56_V4_DIAGNOSTICS_SCHEMA_VERSION}.executed",
        "condition_group_id": group,
        "family_id": family,
        "arm": arm,
        "decision_step": step,
        "selected_action_id": selected,
        "actual_dangerous": dangerous,
        "requested_command_sha256": requested,
        "final_command_sha256": final,
        "executed_command_sha256": final,
        "disposition": disposition,
    }
    return V4ExecutedObservation(
        group,
        family,
        arm,
        step,
        selected,
        dangerous,
        requested,
        final,
        final,
        disposition,
        _digest(body),
    )


def test_candidate_metrics_are_separate_from_executed_metrics() -> None:
    candidates = (
        _candidate("group-a", "family-a0", 16, "action-0", rejected=True, dangerous=True),
        _candidate("group-a", "family-a0", 16, "action-1", rejected=False, dangerous=True),
        _candidate("group-a", "family-a1", 16, "action-0", rejected=False, dangerous=False),
    )
    metrics = candidate_screening_metrics(candidates)

    assert metrics["sample_count"] == 3
    assert metrics["dangerous_event_recall"] == 0.5
    assert metrics["false_safe_rate"] == 0.5

    executed = (
        _executed(
            "group-a",
            "family-a0",
            "risk-v4",
            16,
            selected="action-0",
            dangerous=True,
            requested="a" * 64,
            final="b" * 64,
            disposition="PROPOSED_MODIFIED",
        ),
        _executed(
            "group-a",
            "family-a1",
            "risk-v4",
            16,
            selected=None,
            dangerous=True,
            requested=None,
            final="c" * 64,
            disposition="ABSTAINED_TO_HOLD",
        ),
    )
    executed_metrics = executed_action_metrics(executed)

    assert executed_metrics["decision_count"] == 2
    assert executed_metrics["proposal_count"] == 1
    assert executed_metrics["abstention_count"] == 1
    assert executed_metrics["selected_action_false_safe_rate"] == 1.0
    assert executed_metrics["hmc_mismatch_count"] == 1
    assert executed_metrics["disposition_counts"]["PROPOSED_MODIFIED"] == 1
    assert executed_metrics["disposition_counts"]["ABSTAINED_TO_HOLD"] == 1


def test_observations_round_trip_and_manifest_is_order_independent() -> None:
    candidate = _candidate(
        "group-a", "family-a0", 16, "action-0", rejected=True, dangerous=True
    )
    executed = _executed(
        "group-a",
        "family-a0",
        "risk-v4",
        16,
        selected="action-0",
        dangerous=True,
        requested="a" * 64,
        final="b" * 64,
        disposition="PROPOSED_MODIFIED",
    )

    assert V4CandidateObservation.from_mapping(candidate.to_mapping()) == candidate
    assert V4ExecutedObservation.from_mapping(executed.to_mapping()) == executed
    assert observation_manifest_sha256((candidate, executed)) == observation_manifest_sha256(
        (executed, candidate)
    )

    tampered = dict(candidate.to_mapping())
    tampered["dangerous"] = False
    with pytest.raises(Issue56V4DiagnosticsError, match="digest"):
        V4CandidateObservation.from_mapping(tampered)


def test_condition_groups_require_two_families_and_group_means_are_equal_weighted() -> None:
    observations = (
        _candidate("group-a", "family-a0", 16, "action-0", rejected=False, dangerous=False),
        _candidate("group-a", "family-a1", 16, "action-0", rejected=False, dangerous=False),
        _candidate("group-b", "family-b0", 16, "action-0", rejected=False, dangerous=False),
        _candidate("group-b", "family-b1", 16, "action-0", rejected=False, dangerous=False),
    )

    assert validate_condition_groups(observations) == {
        "group-a": ("family-a0", "family-a1"),
        "group-b": ("family-b0", "family-b1"),
    }
    aggregate = equal_weight_group_mean({"group-a": (1.0, 3.0), "group-b": (10.0,)})
    assert aggregate["group_means"] == {"group-a": 2.0, "group-b": 10.0}
    assert aggregate["equal_weight_mean"] == 6.0

    with pytest.raises(Issue56V4DiagnosticsError, match="paired families"):
        validate_condition_groups(observations[:1])
    split_family = (
        observations[0],
        _candidate("group-b", "family-a0", 20, "action-0", rejected=False, dangerous=False),
    )
    with pytest.raises(Issue56V4DiagnosticsError, match="multiple condition groups"):
        validate_condition_groups(split_family)


def test_group_bootstrap_is_deterministic_and_validates_support() -> None:
    values = {"group-a": 1.0, "group-b": 3.0, "group-c": 8.0}
    first = bootstrap_equal_weight_group_mean(values, resamples=100)
    second = bootstrap_equal_weight_group_mean(values, resamples=100)

    assert first == second
    assert first["point_estimate"] == 4.0
    assert first["group_ids"] == ["group-a", "group-b", "group-c"]
    with pytest.raises(Issue56V4DiagnosticsError, match="at least two"):
        bootstrap_equal_weight_group_mean({"group-a": 1.0}, resamples=10)


def test_provenance_manifest_requires_every_bound_identity() -> None:
    manifest = {field: "a" * 64 for field in V4_PROVENANCE_FIELDS}
    digest = provenance_manifest_sha256(manifest)

    assert len(digest) == 64
    missing = dict(manifest)
    del missing["hmc_binding_sha256"]
    with pytest.raises(Issue56V4DiagnosticsError, match="provenance fields"):
        provenance_manifest_sha256(missing)


def test_v4_protocol_contract_is_explicitly_pre_model_and_fail_closed() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "habitat_v2_forecast_issue_56_v4_diagnostics_preregistration_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert validate_v4_protocol(protocol) == protocol
    tampered = dict(protocol)
    tampered["scope"] = dict(protocol["scope"])
    tampered["scope"]["training_authorized"] = True
    with pytest.raises(Issue56V4DiagnosticsError, match="authorizes learned-model work"):
        validate_v4_protocol(tampered)


def test_v4_authorized_model_protocol_binds_temporal_corpus() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol, digest = load_v4_model_protocol(root)

    assert validate_v4_model_protocol(protocol) == protocol
    assert len(digest) == 64
    assert protocol["data_contract"]["corpus_schema_version"].endswith("_v4")
    assert protocol["feature_variants"][1]["feature_count"] == V4_TEMPORAL_FEATURE_COUNT
    assert protocol["policy"]["hmc_compatibility_mask"] == "validated_catalogue_actions"

    tampered = dict(protocol)
    tampered["data_contract"] = dict(protocol["data_contract"])
    tampered["data_contract"]["hold_trace_bytes_required"] = False
    with pytest.raises(Issue56V4ModelProtocolError, match="data boundary"):
        validate_v4_model_protocol(tampered)


def test_v4_threshold_selection_fails_closed_when_recall_is_unattainable() -> None:
    probabilities = np.asarray(
        [
            [0.10, 0.90, 0.90, 0.90],
            [0.10, 0.90, 0.90, 0.90],
            [0.10, 0.10, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(Issue56V4ModelError, match="cannot meet"):
        _select_event_thresholds(probabilities, labels)


def test_v4_smoke_family_selection_preserves_all_three_splits() -> None:
    _select_families = _load_script_module("build_action_risk_v4_corpus")._select_families

    roster = deterministic_family_ids(32)
    split = __import__(
        "aeolus.habitat_v2.forecast_issue56_action_risk_v3",
        fromlist=["v3_family_split"],
    ).v3_family_split(roster)
    selected = _select_families(roster, split, 6)

    assert len(selected) == 6
    assert {split[family_id] for family_id in selected} == {
        "TRAIN",
        "VALIDATION",
        "EVALUATION",
    }
    assert all(
        selected[index] == roster[roster.index(selected[index])]
        for index in range(len(selected))
    )


def test_v4_resume_keeps_only_complete_ordered_family_prefix() -> None:
    _build_corpus = _load_script_module("build_action_risk_v4_corpus")
    V4CorpusRunError = _build_corpus.V4CorpusRunError
    _resume_family_groups = _build_corpus._resume_family_groups

    family_ids = ("family-a", "family-b", "family-c")
    rows = [
        {"base_sample": {"family_id": "family-a"}}
        for _ in range(4)
    ] + [
        {"base_sample": {"family_id": "family-b"}}
        for _ in range(2)
    ]

    groups, retained = _resume_family_groups(rows, family_ids, 4)

    assert [len(group) for group in groups] == [4]
    assert retained == 4
    with pytest.raises(V4CorpusRunError, match="rows after an incomplete family"):
        _resume_family_groups(
            rows + [{"base_sample": {"family_id": "family-c"}}] * 4,
            family_ids,
            4,
        )


def test_v4_offline_gate_status_does_not_fabricate_hmc_metrics() -> None:
    metrics = {
        "metrics_finite_verified": True,
        "authority_violation_count": 0,
        "replay_failure_count": 0,
        "provenance_violation_count": 0,
        "non_finite_metric_count": 0,
        "proposal_admission_failure_count": 0,
        "useful_action_count": 20,
        "distinct_selected_action_count": 2,
        "abstention_rate": 0.1,
        "inference_latency_p99_ms": 10.0,
        "dangerous_event_recall": 1.0,
    }

    status = _evaluation_gate_status(metrics)

    assert status["gates"]["maximum_hmc_mismatch_rate"]["status"] == "UNEVALUATED"
    assert status["gates"]["maximum_hmc_mismatch_rate"]["value"] is None
    assert status["all_evaluated_gates_passed"] is True
    assert status["all_preregistered_gates_passed"] is False
    assert status["overall_status"] == "UNEVALUATED_OFFLINE_HMC"


def test_v4_group_bootstrap_reports_insufficient_smoke_support() -> None:
    result = _group_bootstrap([0.25])

    assert result["status"] == "INSUFFICIENT_CONDITION_GROUP_SUPPORT"
    assert result["point"] == 0.25
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None


def test_v4_candidate_models_fit_calibrate_and_round_trip() -> None:
    def sample(index: int, split: str) -> V4RiskSample:
        family = (
            f"{split.lower()}-safe"
            if split == "VALIDATION" and index < 2
            else f"{split.lower()}-{index % 2}"
        )
        value = 0.0 if split == "VALIDATION" and index < 2 else float(index)
        event_pattern = (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        )[index]
        metrics = tuple(
            V3HorizonMetric(
                horizon,
                event_pattern[horizon_index],
                0.0,
                0.0,
            )
            for horizon_index, horizon in enumerate(V3_HORIZONS)
        )
        remaining = V3HorizonMetric(
            EPISODE_STEPS - 16,
            event_pattern[3],
            0.0,
            0.0,
        )
        action_id = V4_ACTION_IDS[index % 4]
        action_bytes = f"action-trace-{split}-{index}".encode("ascii")
        hold_bytes = f"hold-trace-{split}-{index}".encode("ascii")
        action_sha = hashlib.sha256(action_bytes).hexdigest()
        hold_sha = hashlib.sha256(hold_bytes).hexdigest()
        state_digests = tuple("a" * 64 for _ in range(EPISODE_STEPS - 16))
        label_body = {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.label",
            "track": "hmc_persistent_remaining",
            "action_id": action_id,
            "decision_step": 16,
            "current_command_sha256": "b" * 64,
            "requested_command_sha256": "c" * 64,
            "final_command_sha256": "d" * 64,
            "executed_command_sha256": "d" * 64,
            "disposition": "PROPOSED_ACCEPTED",
            "horizon_metrics": [metric.to_mapping() for metric in metrics],
            "remaining_steps": EPISODE_STEPS - 16,
            "remaining_metric": remaining.to_mapping(),
            "state_digests": list(state_digests),
            "trace_sha256": action_sha,
        }
        label = V3PolicyLabel(
            action_id,
            16,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "d" * 64,
            "PROPOSED_ACCEPTED",
            metrics,
            EPISODE_STEPS - 16,
            remaining,
            state_digests,
            action_sha,
            _digest(label_body),
        )
        features = np.full(FEATURE_COUNT, value, dtype=np.float32)
        base_body = {
            "schema_version": f"{ISSUE56_V3_SCHEMA_VERSION}.sample",
            "family_id": family,
            "decision_step": 16,
            "split": split,
                "action_id": action_id,
            "scenario_sha256": "f" * 64,
            "features_f32_hex": features.tobytes().hex(),
            "label": label.to_mapping(),
        }
        base_sample = V3RiskSample(
            family,
            16,
            split,
            action_id,
            "f" * 64,
            features,
            label,
            _digest(base_body),
        )
        mapping = {
            "schema_version": "aeolus_habitat_v2_risk_issue_56_v4_corpus_v4.sample",
            "base_sample": base_sample.to_mapping(),
            "counterfactual_trace_relative_path": f"counterfactual-traces/{action_sha}.json",
            "counterfactual_trace_sha256": action_sha,
            "hold_trace_relative_path": f"counterfactual-traces/{hold_sha}.json",
            "hold_trace_sha256": hold_sha,
            "temporal_features_f32_hex": np.resize(
                features, (V4_TEMPORAL_FEATURE_COUNT,)
            ).tobytes().hex(),
            "observable_action_mask": [True, True, True, True],
            "trajectory_metrics": {
                "safety_exposure": 0.0,
                "safety_violation_steps": 0,
                "comfort_deviation": 0.0,
                "resource_composite": 0.0,
            },
            "hold_trajectory_metrics": {
                "safety_exposure": 0.0,
                "safety_violation_steps": 0,
                "comfort_deviation": 0.0,
                "resource_composite": 0.0,
            },
            "relative_action_targets": {
                "safety_exposure_delta_vs_hold": 0.0,
                "comfort_deviation_delta_vs_hold": 0.0,
                "resource_composite_delta_vs_hold": 0.0,
            },
        }
        mapping["sample_sha256"] = _digest(mapping)
        return V4RiskSample.from_mapping(
            mapping,
            action_bytes,
            hold_bytes,
        )

    train = tuple(sample(index, "TRAIN") for index in range(10))
    validation = tuple(sample(index, "VALIDATION") for index in range(10))
    model_train = tuple(V4ModelSample.from_verified(item) for item in train)
    model_validation = tuple(V4ModelSample.from_verified(item) for item in validation)
    assert not hasattr(model_train[0], "counterfactual_trace_bytes")
    for candidate in ("c0_v3_refit", "c1_shared_hazard_ridge"):
        model = V4RiskModel.fit(model_train, candidate_id=candidate)
        calibrated = model.calibrate(model_validation)
        assert tuple(item.horizon_steps for item in calibrated.predict_features(
            model_train[0].features_f32
            if calibrated.feature_variant == "v3_708_past_only"
            else model_train[0].temporal_features_f32
        ).horizons) == V4_HORIZON_KEYS
        assert V4RiskModel.from_mapping(calibrated.to_mapping()).to_mapping() == calibrated.to_mapping()

    with pytest.raises(Issue56V4ModelError, match="TRAIN"):
        V4RiskModel.fit(validation, candidate_id="c0_v3_refit")
    with pytest.raises(Issue56V4ModelError, match="semantically verified"):
        V4RiskModel.fit(
            tuple(item.base_sample for item in train),
            candidate_id="c0_v3_refit",
        )


@pytest.fixture(scope="module")
def v4_corpus_fixture() -> tuple[object, str, object, tuple[V4RiskSample, ...]]:
    root = Path(__file__).resolve().parents[2]
    bundle = load_forecast_contracts(root)
    family_id = deterministic_family_ids(32)[0]
    scenario = build_family_scenario(bundle.development_scenario, 0)
    samples = collect_v4_family_samples(bundle, scenario, family_id, split="TRAIN")
    return bundle, family_id, scenario, samples


def test_v4_observable_action_mask_admits_every_valid_catalogue_action(
    v4_corpus_fixture: tuple[object, str, object, tuple[V4RiskSample, ...]],
) -> None:
    bundle, family_id, scenario, samples = v4_corpus_fixture

    histories = _v4_feature_histories(bundle, scenario, family_id)
    assert histories
    history = next(iter(histories.values()))
    for mode_index, mode in enumerate(MODE_ORDER):
        mode_values = np.zeros_like(history.mode_f32)
        mode_values[:, mode_index] = 1.0
        mode_history = replace(history, mode_f32=mode_values)
        assert observable_operating_mode(mode_history) == mode
        assert v4_observable_action_mask(bundle, mode_history) == (
            True,
            True,
            True,
            True,
        )
    assert {sample.observable_action_mask for sample in samples} == {
        (True, True, True, True)
    }


def _write_v4_trace_files(root: Path, samples: tuple[V4RiskSample, ...]) -> None:
    for sample in samples:
        trace_path = root / sample.counterfactual_trace_relative_path
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(sample.counterfactual_trace_bytes)
        hold_trace_path = root / sample.hold_trace_relative_path
        hold_trace_path.parent.mkdir(parents=True, exist_ok=True)
        hold_trace_path.write_bytes(sample.hold_trace_bytes)


def _rehash_v4_mapping(mapping: dict[str, object]) -> dict[str, object]:
    base_sample = mapping["base_sample"]
    assert isinstance(base_sample, dict)
    label = base_sample["label"]
    assert isinstance(label, dict)
    label_body = dict(label)
    label_body.pop("label_sha256")
    label["label_sha256"] = _digest(label_body)
    base_body = dict(base_sample)
    base_body.pop("sample_sha256")
    base_sample["sample_sha256"] = _digest(base_body)
    sample_body = dict(mapping)
    sample_body.pop("sample_sha256")
    mapping["sample_sha256"] = _digest(sample_body)
    return mapping


def test_v4_corpus_retains_and_verifies_counterfactual_trace_bytes(
    tmp_path: Path,
    v4_corpus_fixture: tuple[object, str, object, tuple[V4RiskSample, ...]],
) -> None:
    bundle, family_id, scenario, samples = v4_corpus_fixture

    assert len(samples) == 13 * 4
    sample = samples[0]
    assert isinstance(sample, V4RiskSample)
    assert sample.counterfactual_trace_bytes
    assert sample.temporal_features_f32.shape == (V4_TEMPORAL_FEATURE_COUNT,)
    assert len(sample.observable_action_mask) == 4
    assert verify_v4_serialized_trace(
        sample.counterfactual_trace_bytes,
        bundle,
        scenario,
    )["trace_sha256"] == sample.counterfactual_trace_sha256
    restored = V4RiskSample.from_mapping(
        sample.to_mapping(),
        sample.counterfactual_trace_bytes,
        sample.hold_trace_bytes,
    )
    assert restored.to_mapping() == sample.to_mapping()
    _write_v4_trace_files(tmp_path, (sample,))
    loaded = load_v4_samples(
        [sample.to_mapping()],
        tmp_path,
        bundle,
        {family_id: scenario},
    )
    assert loaded[0].to_mapping() == sample.to_mapping()

    with pytest.raises(Issue56V4CorpusError, match="strict replay"):
        verify_v4_serialized_trace(
            sample.counterfactual_trace_bytes[:-1] + b"0",
            bundle,
            scenario,
        )
    with pytest.raises(Issue56V4CorpusError, match="path"):
        replace(
            sample,
            counterfactual_trace_relative_path=(
                "counterfactual-traces/../escape.json"
            ),
        )


def test_v4_semantic_verifier_rejects_swapped_counterfactual_trace(
    v4_corpus_fixture: tuple[object, str, object, tuple[V4RiskSample, ...]],
) -> None:
    bundle, _, scenario, samples = v4_corpus_fixture
    first, second = samples[:2]
    tampered = first.to_mapping()
    tampered["counterfactual_trace_relative_path"] = second.counterfactual_trace_relative_path
    tampered["counterfactual_trace_sha256"] = second.counterfactual_trace_sha256
    base_sample = tampered["base_sample"]
    assert isinstance(base_sample, dict)
    label = base_sample["label"]
    assert isinstance(label, dict)
    label["trace_sha256"] = second.counterfactual_trace_sha256
    _rehash_v4_mapping(tampered)
    swapped = V4RiskSample.from_mapping(
        tampered,
        second.counterfactual_trace_bytes,
        first.hold_trace_bytes,
    )

    with pytest.raises(Issue56V4CorpusError, match="proposal action identity"):
        from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
            verify_v4_sample_against_trace,
        )

        verify_v4_sample_against_trace(swapped, bundle, scenario)


def test_v4_semantic_verifier_rejects_mutated_label(
    v4_corpus_fixture: tuple[object, str, object, tuple[V4RiskSample, ...]],
) -> None:
    bundle, _, scenario, samples = v4_corpus_fixture
    sample = samples[0]
    tampered = sample.to_mapping()
    base_sample = tampered["base_sample"]
    assert isinstance(base_sample, dict)
    label = base_sample["label"]
    assert isinstance(label, dict)
    state_digests = label["state_digests"]
    assert isinstance(state_digests, list)
    state_digests[0] = "0" * 64
    _rehash_v4_mapping(tampered)
    mutated = V4RiskSample.from_mapping(
        tampered,
        sample.counterfactual_trace_bytes,
        sample.hold_trace_bytes,
    )

    with pytest.raises(Issue56V4CorpusError, match="state provenance"):
        from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
            verify_v4_sample_against_trace,
        )

        verify_v4_sample_against_trace(mutated, bundle, scenario)


def test_v4_semantic_verifier_rejects_altered_command_identity(
    v4_corpus_fixture: tuple[object, str, object, tuple[V4RiskSample, ...]],
) -> None:
    bundle, _, scenario, samples = v4_corpus_fixture
    sample = samples[0]
    tampered = sample.to_mapping()
    base_sample = tampered["base_sample"]
    assert isinstance(base_sample, dict)
    label = base_sample["label"]
    assert isinstance(label, dict)
    label["final_command_sha256"] = "0" * 64
    label["executed_command_sha256"] = "0" * 64
    _rehash_v4_mapping(tampered)
    mutated = V4RiskSample.from_mapping(
        tampered,
        sample.counterfactual_trace_bytes,
        sample.hold_trace_bytes,
    )

    with pytest.raises(Issue56V4CorpusError, match="label final command identity"):
        from aeolus.habitat_v2.forecast_issue56_action_risk_v4_corpus import (
            verify_v4_sample_against_trace,
        )

        verify_v4_sample_against_trace(mutated, bundle, scenario)


def test_v4_loader_rejects_symlinked_trace_artifact(
    tmp_path: Path,
    v4_corpus_fixture: tuple[object, str, object, tuple[V4RiskSample, ...]],
) -> None:
    bundle, family_id, scenario, samples = v4_corpus_fixture
    sample = samples[0]
    _write_v4_trace_files(tmp_path, (sample,))
    trace_path = tmp_path / sample.counterfactual_trace_relative_path
    outside = tmp_path / "outside-trace.json"
    outside.write_bytes(sample.counterfactual_trace_bytes)
    trace_path.unlink()
    try:
        trace_path.symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(Issue56V4CorpusError, match="symlink"):
        load_v4_samples(
            [sample.to_mapping()],
            tmp_path,
            bundle,
            {family_id: scenario},
        )
