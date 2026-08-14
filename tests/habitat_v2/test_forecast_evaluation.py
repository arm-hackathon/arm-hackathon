from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.evaluation import (
    CandidateOutput,
    EvaluationError,
    EvaluationRequest,
    HarmEnvelope,
    TrainingReference,
    compare_action_aware_and_blinded,
    evaluate,
)

INPUT = "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
TARGET = "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"


def request(sample_id: str = "s1", cluster: str = "eval-c1") -> EvaluationRequest:
    target = np.zeros((2, 51), dtype=np.float32)
    return EvaluationRequest(sample_id, cluster, "VALIDATION", INPUT, TARGET, target)


def training_references() -> tuple[TrainingReference, ...]:
    return (
        TrainingReference(
            "train-1",
            "train-c1",
            "TRAIN",
            TARGET,
            np.zeros((2, 51), dtype=np.float32),
        ),
        TrainingReference(
            "train-2",
            "train-c2",
            "TRAIN",
            TARGET,
            np.full((2, 51), 2, dtype=np.float32),
        ),
    )


def envelope() -> HarmEnvelope:
    return HarmEnvelope(
        np.full((2, 51), -1, dtype=np.float32),
        np.full((2, 51), 1, dtype=np.float32),
        np.zeros(51, dtype=np.float32),
    )


def good(req: EvaluationRequest) -> CandidateOutput:
    return CandidateOutput(
        req.sample_id,
        "PREDICTION",
        INPUT,
        TARGET,
        np.zeros((2, 51), dtype=np.float32),
    )


def abstain(req: EvaluationRequest) -> CandidateOutput:
    return CandidateOutput(req.sample_id, "ABSTAIN", INPUT, TARGET, None)


def bad_identity(req: EvaluationRequest) -> CandidateOutput:
    return CandidateOutput(
        "wrong",
        "PREDICTION",
        INPUT,
        TARGET,
        np.zeros((2, 51), dtype=np.float32),
    )


def test_prediction_abstain_and_invalid_are_distinct() -> None:
    for adapter, expected in (
        (good, "PREDICTION"),
        (abstain, "ABSTAIN"),
        (bad_identity, "INVALID_OUTPUT"),
    ):
        result = evaluate(
            (request(),),
            adapter,
            envelope=envelope(),
            training_references=training_references(),
        )
        assert result.outputs[0].status == expected


def test_metrics_positive_harm_existing_harm_and_unsupported_ratios() -> None:
    truth = np.zeros((2, 51), dtype=np.float32)
    truth[0, 0] = 2  # new harmful positive
    req = EvaluationRequest("s", "eval-cluster", "VALIDATION", INPUT, TARGET, truth)
    prediction = np.zeros((2, 51), dtype=np.float32)
    prediction[0, 0] = 2
    result = evaluate(
        (req,),
        lambda x: CandidateOutput(
            x.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            prediction,
        ),
        envelope=envelope(),
        training_references=training_references(),
    )
    assert result.metrics.confusion.true_positive == 1
    assert result.metrics.confusion.existing_harm_count == 0
    assert result.metrics.polarity == "harmful_crossing_positive"
    assert result.metrics.precision.supported is True
    assert result.metrics.false_positive_rate.supported is True


def test_nonfinite_shape_domain_and_timeout_are_invalid_not_abstain() -> None:
    def bad_shape(req: EvaluationRequest) -> CandidateOutput:
        return CandidateOutput(
            req.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            np.zeros((1, 51), dtype=np.float32),
        )

    def bad_domain(req: EvaluationRequest) -> CandidateOutput:
        return CandidateOutput(
            req.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            np.full((2, 51), 9, dtype=np.float32),
        )

    def timeout(_: EvaluationRequest) -> CandidateOutput:
        raise TimeoutError

    for adapter in (bad_shape, bad_domain, timeout):
        result = evaluate(
            (request(),),
            adapter,
            envelope=envelope(),
            training_references=training_references(),
            domain_validator=lambda array: bool(np.all(np.abs(array) <= 2)),
        )
        assert result.metrics.invalid_output_count == 1
        assert result.outputs[0].status == "INVALID_OUTPUT"
    timeout_result = evaluate(
        (request(),),
        timeout,
        envelope=envelope(),
        training_references=training_references(),
    )
    assert timeout_result.outputs[0].reason == "adapter_timeout"


def test_training_scale_rejects_nontrain_and_whole_cluster_leakage() -> None:
    with pytest.raises(EvaluationError, match="evaluation split"):
        evaluate(
            (replace(request(), split_label="TRAIN"),),
            good,
            envelope=envelope(),
            training_references=training_references(),
        )

    nontrain = list(training_references())
    nontrain[0] = replace(nontrain[0], split_label="VALIDATION")
    with pytest.raises(EvaluationError, match="TRAIN"):
        evaluate(
            (request(),),
            good,
            envelope=envelope(),
            training_references=tuple(nontrain),
        )

    leaked = list(training_references())
    leaked[0] = replace(leaked[0], family_cluster_id="eval-c1")
    with pytest.raises(EvaluationError, match="cluster"):
        evaluate(
            (request(),),
            good,
            envelope=envelope(),
            training_references=tuple(leaked),
        )


def test_action_aware_vs_blinded_comparison_has_explicit_polarity() -> None:
    req = request()
    aware = evaluate(
        (req,),
        good,
        envelope=envelope(),
        training_references=training_references(),
    )
    blinded = evaluate(
        (req,),
        lambda item: CandidateOutput(
            item.sample_id,
            "PREDICTION",
            INPUT,
            TARGET,
            np.full((2, 51), 0.5, dtype=np.float32),
        ),
        envelope=envelope(),
        training_references=training_references(),
    )

    comparison = compare_action_aware_and_blinded(aware, blinded)
    assert comparison.supported is True
    assert comparison.blinded_minus_aware > 0
    assert (
        comparison.polarity == "positive_blinded_minus_aware_means_action_information"
    )
