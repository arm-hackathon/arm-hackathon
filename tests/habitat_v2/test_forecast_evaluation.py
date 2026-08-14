from __future__ import annotations

import numpy as np

from aeolus.habitat_v2.forecast.evaluation import (
    CandidateOutput,
    EvaluationRequest,
    HarmEnvelope,
    evaluate,
)

INPUT = "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
TARGET = "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"


def request(sample_id: str = "s1", cluster: str = "c1") -> EvaluationRequest:
    target = np.zeros((2, 51), dtype=np.float32)
    return EvaluationRequest(sample_id, cluster, INPUT, TARGET, target)


def envelope() -> HarmEnvelope:
    return HarmEnvelope(
        np.full((2, 51), -1, dtype=np.float32),
        np.full((2, 51), 1, dtype=np.float32),
        np.zeros(51, dtype=np.float32),
    )


def good(req: EvaluationRequest) -> CandidateOutput:
    return CandidateOutput(req.sample_id, "PREDICTION", INPUT, TARGET, np.zeros((2, 51), dtype=np.float32))


def abstain(req: EvaluationRequest) -> CandidateOutput:
    return CandidateOutput(req.sample_id, "ABSTAIN", INPUT, TARGET, None)


def bad_identity(req: EvaluationRequest) -> CandidateOutput:
    return CandidateOutput("wrong", "PREDICTION", INPUT, TARGET, np.zeros((2, 51), dtype=np.float32))


def test_prediction_abstain_and_invalid_are_distinct() -> None:
    for adapter, expected in ((good, "PREDICTION"), (abstain, "ABSTAIN"), (bad_identity, "INVALID_OUTPUT")):
        result = evaluate((request(),), adapter, envelope=envelope(), training_targets_f32=np.ones((2, 2, 51), dtype=np.float32))
        assert result.outputs[0].status == expected


def test_metrics_positive_harm_existing_harm_and_unsupported_ratios() -> None:
    truth = np.zeros((2, 51), dtype=np.float32)
    truth[0, 0] = 2  # new harmful positive
    req = EvaluationRequest("s", "cluster", INPUT, TARGET, truth)
    prediction = np.zeros((2, 51), dtype=np.float32)
    prediction[0, 0] = 2
    result = evaluate((req,), lambda x: CandidateOutput(x.sample_id, "PREDICTION", INPUT, TARGET, prediction), envelope=envelope(), training_targets_f32=np.ones((3, 2, 51), dtype=np.float32))
    assert result.metrics.confusion.true_positive == 1
    assert result.metrics.confusion.existing_harm_count == 0
    assert result.metrics.polarity == "harmful_crossing_positive"
    assert result.metrics.precision.supported is True
    assert result.metrics.false_positive_rate.supported is True


def test_nonfinite_shape_and_domain_invalid_are_invalid_not_abstain() -> None:
    def bad_shape(req: EvaluationRequest) -> CandidateOutput:
        return CandidateOutput(req.sample_id, "PREDICTION", INPUT, TARGET, np.zeros((1, 51), dtype=np.float32))

    def bad_domain(req: EvaluationRequest) -> CandidateOutput:
        return CandidateOutput(req.sample_id, "PREDICTION", INPUT, TARGET, np.full((2, 51), 9, dtype=np.float32))

    for adapter in (bad_shape, bad_domain):
        result = evaluate((request(),), adapter, envelope=envelope(), training_targets_f32=np.ones((2, 2, 51), dtype=np.float32), domain_validator=lambda a: bool(np.all(np.abs(a) <= 2)))
        assert result.metrics.invalid_output_count == 1
