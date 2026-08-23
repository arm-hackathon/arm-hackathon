from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.evaluation import (
    CandidateOutput,
    CandidateQuery,
    CorpusEvidenceEntry,
    EvaluationError,
    EvaluationRequest,
    HarmEnvelope,
    TrainingReference,
    build_evaluation_evidence_manifest,
    compare_action_aware_and_blinded,
    evaluate,
    target_truth_sha256,
)

INPUT = "29d743472712dff68759477debd25aadba8a0584ad89d164bc5c583260356971"
TARGET = "26e480ca4f07d2092fc6e96fcf2f006948e9e2872ad2b0fd4ae3ac8e947c74db"
SPLIT_TABLE = hashlib.sha256(b"canonical-split-table").hexdigest()


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def split_identity(cluster: str, label: str) -> tuple[str, str]:
    return (
        digest(f"split-assignment:{cluster}:{label}"),
        digest(f"split-assignment-record:{cluster}:{label}"),
    )


def request(
    sample_id: str = "s1",
    cluster: str = "eval-c1",
    *,
    split_label: str = "VALIDATION",
    target: np.ndarray | None = None,
) -> EvaluationRequest:
    split_id, split_record = split_identity(cluster, split_label)
    target_array = np.zeros((2, 51), dtype=np.float32) if target is None else target
    return EvaluationRequest(
        sample_id=sample_id,
        family_cluster_id=cluster,
        split_label=split_label,
        sample_record_sha256=digest(f"sample-record:{sample_id}"),
        split_assignment_id=split_id,
        split_assignment_record_sha256=split_record,
        input_manifest_sha256=INPUT,
        target_manifest_sha256=TARGET,
        targets_f32=target_array,
    )


def training_references() -> tuple[TrainingReference, ...]:
    rows: list[TrainingReference] = []
    for sample_id, cluster, value in (
        ("train-1", "train-c1", 0.0),
        ("train-2", "train-c2", 2.0),
    ):
        split_id, split_record = split_identity(cluster, "TRAIN")
        rows.append(
            TrainingReference(
                sample_id=sample_id,
                family_cluster_id=cluster,
                split_label="TRAIN",
                sample_record_sha256=digest(f"sample-record:{sample_id}"),
                split_assignment_id=split_id,
                split_assignment_record_sha256=split_record,
                target_manifest_sha256=TARGET,
                targets_f32=np.full((2, 51), value, dtype=np.float32),
            )
        )
    return tuple(rows)


def evidence_entry(row: EvaluationRequest | TrainingReference) -> CorpusEvidenceEntry:
    return CorpusEvidenceEntry(
        sample_id=row.sample_id,
        sample_record_sha256=row.sample_record_sha256,
        family_cluster_id=row.family_cluster_id,
        split_assignment_id=row.split_assignment_id,
        split_assignment_record_sha256=row.split_assignment_record_sha256,
        split_label=row.split_label,
        target_manifest_sha256=row.target_manifest_sha256,
        target_truth_sha256=target_truth_sha256(row.targets_f32),
    )


def bind(
    requests: tuple[EvaluationRequest, ...],
    references: tuple[TrainingReference, ...],
):
    manifest = build_evaluation_evidence_manifest(
        SPLIT_TABLE,
        tuple(evidence_entry(row) for row in (*requests, *references)),
    )
    return (
        tuple(
            replace(row, evidence_manifest_sha256=manifest.manifest_sha256)
            for row in requests
        ),
        tuple(
            replace(row, evidence_manifest_sha256=manifest.manifest_sha256)
            for row in references
        ),
        manifest,
    )


def evaluate_bound(
    requests: tuple[EvaluationRequest, ...],
    adapter,
    *,
    references: tuple[TrainingReference, ...] | None = None,
    domain_validator=None,
):
    bound_requests, bound_references, manifest = bind(
        requests,
        training_references() if references is None else references,
    )
    return evaluate(
        bound_requests,
        adapter,
        envelope=envelope(),
        training_references=bound_references,
        evidence_manifest=manifest,
        expected_evidence_manifest_sha256=manifest.manifest_sha256,
        domain_validator=domain_validator,
    )


def envelope() -> HarmEnvelope:
    return HarmEnvelope(
        np.full((2, 51), -1, dtype=np.float32),
        np.full((2, 51), 1, dtype=np.float32),
        np.zeros(51, dtype=np.float32),
    )


def good(req: CandidateQuery) -> CandidateOutput:
    return CandidateOutput(
        sample_id=req.sample_id,
        status="PREDICTION",
        input_manifest_sha256=INPUT,
        target_manifest_sha256=TARGET,
        prediction_f32=np.zeros((2, 51), dtype=np.float32),
    )


def abstain(req: CandidateQuery) -> CandidateOutput:
    return CandidateOutput(req.sample_id, "ABSTAIN", INPUT, TARGET, None)


def bad_identity(req: CandidateQuery) -> CandidateOutput:
    return CandidateOutput(
        "wrong",
        "PREDICTION",
        INPUT,
        TARGET,
        np.zeros((2, 51), dtype=np.float32),
    )


def test_prediction_abstain_and_invalid_are_distinct() -> None:
    def target_free(query: CandidateQuery) -> CandidateOutput:
        assert not hasattr(query, "targets_f32")
        return good(query)

    for adapter, expected in (
        (good, "PREDICTION"),
        (target_free, "PREDICTION"),
        (abstain, "ABSTAIN"),
        (bad_identity, "INVALID_OUTPUT"),
    ):
        result = evaluate_bound((request(),), adapter)
        assert result.outputs[0].status == expected


def test_adapter_cannot_mutate_verified_evaluator_truth() -> None:
    target = np.zeros((2, 51), dtype=np.float32)
    req = request(target=target)

    def mutate_caller_array(query: CandidateQuery) -> CandidateOutput:
        target.fill(9)
        return good(query)

    result = evaluate_bound((req,), mutate_caller_array)
    assert np.all(result.metrics.mae_by_horizon_target == 0)


def test_metrics_positive_harm_existing_harm_and_unsupported_ratios() -> None:
    truth = np.zeros((2, 51), dtype=np.float32)
    truth[0, 0] = 2  # new harmful positive
    req = request("s", "eval-cluster", target=truth)
    prediction = np.zeros((2, 51), dtype=np.float32)
    prediction[0, 0] = 2
    result = evaluate_bound(
        (req,),
        lambda item: CandidateOutput(
            item.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            prediction,
        ),
    )
    assert result.metrics.confusion.true_positive == 1
    assert result.metrics.confusion.existing_harm_count == 0
    assert result.metrics.polarity == "harmful_crossing_positive"
    assert result.metrics.precision.supported is True
    assert result.metrics.false_positive_rate.supported is True


def test_nonfinite_shape_domain_and_timeout_are_invalid_not_abstain() -> None:
    def bad_shape(req: CandidateQuery) -> CandidateOutput:
        return CandidateOutput(
            req.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            np.zeros((1, 51), dtype=np.float32),
        )

    def bad_domain(req: CandidateQuery) -> CandidateOutput:
        return CandidateOutput(
            req.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            np.full((2, 51), 9, dtype=np.float32),
        )

    def timeout(_: CandidateQuery) -> CandidateOutput:
        raise TimeoutError

    for adapter in (bad_shape, bad_domain, timeout):
        result = evaluate_bound(
            (request(),),
            adapter,
            domain_validator=lambda array: bool(np.all(np.abs(array) <= 2)),
        )
        assert result.metrics.invalid_output_count == 1
        assert result.outputs[0].status == "INVALID_OUTPUT"
    timeout_result = evaluate_bound((request(),), timeout)
    assert timeout_result.outputs[0].reason == "adapter_timeout"


def test_training_scale_rejects_nontrain_and_whole_cluster_leakage() -> None:
    with pytest.raises(EvaluationError, match="evaluation split"):
        evaluate_bound(
            (request(split_label="TRAIN"),),
            good,
        )

    nontrain = list(training_references())
    nontrain[0] = replace(nontrain[0], split_label="VALIDATION")
    with pytest.raises(EvaluationError, match="TRAIN references only"):
        evaluate_bound(
            (request(),),
            good,
            references=tuple(nontrain),
        )

    leaking = list(training_references())
    cluster = "eval-c1"
    split_id, split_record = split_identity(cluster, "TRAIN")
    leaking[0] = replace(
        leaking[0],
        family_cluster_id=cluster,
        split_assignment_id=split_id,
        split_assignment_record_sha256=split_record,
    )
    with pytest.raises(EvaluationError, match="corpus evidence family"):
        evaluate_bound(
            (request(),),
            good,
            references=tuple(leaking),
        )


def test_relabelled_train_sample_cannot_enter_evaluation() -> None:
    legitimate_request = request()
    references = training_references()
    bound_requests, bound_references, manifest = bind(
        (legitimate_request,),
        references,
    )
    train = bound_references[0]
    relabelled = EvaluationRequest(
        sample_id=train.sample_id,
        family_cluster_id=train.family_cluster_id,
        split_label="FINAL",
        sample_record_sha256=train.sample_record_sha256,
        split_assignment_id=train.split_assignment_id,
        split_assignment_record_sha256=train.split_assignment_record_sha256,
        input_manifest_sha256=INPUT,
        target_manifest_sha256=TARGET,
        targets_f32=train.targets_f32,
        evidence_manifest_sha256=manifest.manifest_sha256,
    )
    called = False

    def must_not_run(_: CandidateQuery) -> CandidateOutput:
        nonlocal called
        called = True
        raise AssertionError("adapter must not receive relabelled corpus evidence")

    with pytest.raises(EvaluationError, match="frozen corpus evidence"):
        evaluate(
            (relabelled,),
            must_not_run,
            envelope=envelope(),
            training_references=bound_references,
            evidence_manifest=manifest,
            expected_evidence_manifest_sha256=manifest.manifest_sha256,
        )
    assert called is False
    assert bound_requests[0].sample_id == "s1"


def test_target_swap_and_manifest_drift_fail_before_adapter() -> None:
    bound_requests, bound_references, manifest = bind(
        (request(),),
        training_references(),
    )
    swapped = replace(
        bound_requests[0],
        targets_f32=np.ones((2, 51), dtype=np.float32),
    )
    with pytest.raises(EvaluationError, match="frozen corpus evidence"):
        evaluate(
            (swapped,),
            good,
            envelope=envelope(),
            training_references=bound_references,
            evidence_manifest=manifest,
            expected_evidence_manifest_sha256=manifest.manifest_sha256,
        )
    with pytest.raises(EvaluationError, match="expected frozen manifest"):
        evaluate(
            bound_requests,
            good,
            envelope=envelope(),
            training_references=bound_references,
            evidence_manifest=manifest,
            expected_evidence_manifest_sha256="0" * 64,
        )


def test_action_aware_vs_blinded_comparison_has_explicit_polarity() -> None:
    req = request()
    aware = evaluate_bound((req,), good)
    blinded = evaluate_bound(
        (req,),
        lambda item: CandidateOutput(
            item.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            np.full((2, 51), 0.5, dtype=np.float32),
        ),
    )

    comparison = compare_action_aware_and_blinded(aware, blinded)
    assert comparison.supported is True
    assert comparison.blinded_minus_aware > 0
    assert (
        comparison.polarity == "positive_blinded_minus_aware_means_action_information"
    )
