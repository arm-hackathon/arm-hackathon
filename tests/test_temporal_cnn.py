"""Compact deployment-compatible temporal Conv1D detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from aeolus.config import load_scenario
from aeolus.detector import CLASS_NAMES, FEATURE_WIDTH, WINDOW_TICKS
from aeolus.model_input import build_model_input_contract, model_artifact_metadata
from aeolus.temporal_cnn import (
    CNN_DILATIONS,
    CNN_ONNX_OPERATORS,
    CNN_RECEPTIVE_FIELD_TICKS,
    TemporalCNNDetector,
    _causal_conv_backward,
    _causal_conv_forward,
    export_temporal_cnn_onnx,
    load_temporal_cnn,
    save_temporal_cnn,
    temporal_cnn_parameter_count,
    train_temporal_cnn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, str]:
    config = load_scenario(REPO_ROOT / "scenarios" / "standard_habitat.json")
    return model_artifact_metadata(build_model_input_contract(config))


def _window(class_index: int, sample_index: int) -> list[list[float]]:
    ticks = np.arange(WINDOW_TICKS, dtype=np.float32)
    window = np.zeros((WINDOW_TICKS, FEATURE_WIDTH), dtype=np.float32)
    window[:, class_index] = (class_index + 1.0) * (ticks + 1.0) / WINDOW_TICKS
    window[:, 8 + class_index] = sample_index / 100.0
    return window.tolist()


def _rows(samples_per_class: int = 8) -> list[dict]:
    return [
        {
            "label": label,
            "features": _window(class_index, sample_index),
            "family_id": f"{label}-{sample_index}",
            "scenario_role": "reference" if class_index == 0 else "fault",
            "end_tick": 20 + sample_index,
            "observable_onset_tick": 10,
        }
        for class_index, label in enumerate(CLASS_NAMES)
        for sample_index in range(samples_per_class)
    ]


def test_temporal_cnn_predicts_finite_probability_vectors():
    detector, _ = train_temporal_cnn(
        _rows(),
        _rows(4),
        contract_metadata=_contract(),
        epochs=10,
        weighting_mode="balanced",
    )

    probabilities = detector.predict_probabilities(
        [_window(0, 1), _window(2, 2)]
    )

    assert isinstance(detector, TemporalCNNDetector)
    assert probabilities.shape == (2, len(CLASS_NAMES))
    assert np.isfinite(probabilities).all()
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(2))


def test_temporal_cnn_training_and_json_round_trip_are_deterministic(tmp_path: Path):
    first, first_receipt = train_temporal_cnn(
        _rows(),
        _rows(4),
        contract_metadata=_contract(),
        epochs=10,
        weighting_mode="sqrt_inverse",
    )
    second, second_receipt = train_temporal_cnn(
        _rows(),
        _rows(4),
        contract_metadata=_contract(),
        epochs=10,
        weighting_mode="sqrt_inverse",
    )

    assert first == second
    assert first_receipt == second_receipt
    assert first_receipt["weighting_mode"] == "sqrt_inverse"
    assert first_receipt["parameter_count"] == temporal_cnn_parameter_count(first)
    assert first_receipt["parameter_count"] == 416
    assert first_receipt["architecture"]["dilations"] == list(CNN_DILATIONS)
    assert (
        first_receipt["architecture"]["receptive_field_ticks"]
        == CNN_RECEPTIVE_FIELD_TICKS
    )
    path = save_temporal_cnn(first, tmp_path / "cnn.json")
    assert load_temporal_cnn(path, expected_contract=_contract()) == first


def test_temporal_cnn_rejects_unknown_weighting_and_malformed_windows():
    with pytest.raises(ValueError, match="weighting_mode"):
        train_temporal_cnn(
            _rows(),
            _rows(4),
            contract_metadata=_contract(),
            epochs=10,
            weighting_mode="invented",
        )

    detector, _ = train_temporal_cnn(
        _rows(), _rows(4), contract_metadata=_contract(), epochs=10
    )
    with pytest.raises(ValueError, match="exactly 10 ticks"):
        detector.predict_probabilities([[[0.0] * FEATURE_WIDTH] * 9])


def test_temporal_cnn_onnx_uses_declared_standard_ops_and_matches_python(
    tmp_path: Path,
):
    import onnxruntime as ort

    detector, _ = train_temporal_cnn(
        _rows(), _rows(4), contract_metadata=_contract(), epochs=10
    )
    path = export_temporal_cnn_onnx(detector, tmp_path / "cnn.onnx")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    operators = tuple(node.op_type for node in model.graph.node)

    assert operators == CNN_ONNX_OPERATORS
    assert model.metadata_props
    convolution_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    dilations = [
        onnx.helper.get_attribute_value(
            next(attribute for attribute in node.attribute if attribute.name == "dilations")
        )
        for node in convolution_nodes
    ]
    pads = [
        onnx.helper.get_attribute_value(
            next(attribute for attribute in node.attribute if attribute.name == "pads")
        )
        for node in convolution_nodes
    ]
    assert dilations == [[1], [2], [4], [1]]
    assert pads == [[2, 0], [4, 0], [8, 0], [0, 0]]

    windows = np.asarray(
        [_window(class_index, 3) for class_index in range(len(CLASS_NAMES))],
        dtype=np.float32,
    )
    expected = detector.predict_probabilities(windows)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["probabilities"], {"window": windows})[0]
    assert np.max(np.abs(expected - actual)) <= 1e-5


def test_causal_convolution_weight_gradient_matches_finite_difference():
    rng = np.random.default_rng(9)
    tensor = rng.normal(size=(1, 5, 2))
    weights = rng.normal(size=(2, 2, 3))
    biases = rng.normal(size=2)
    upstream = rng.normal(size=(1, 5, 2))
    output, cache = _causal_conv_forward(tensor, weights, biases, dilation=2)
    _, analytic, _ = _causal_conv_backward(upstream, cache, weights)
    index = (1, 0, 2)
    epsilon = 1e-6
    positive = weights.copy()
    negative = weights.copy()
    positive[index] += epsilon
    negative[index] -= epsilon
    positive_output, _ = _causal_conv_forward(tensor, positive, biases, dilation=2)
    negative_output, _ = _causal_conv_forward(tensor, negative, biases, dilation=2)
    numerical = (
        float(np.sum(positive_output * upstream))
        - float(np.sum(negative_output * upstream))
    ) / (2.0 * epsilon)

    assert output.shape == upstream.shape
    assert analytic[index] == pytest.approx(numerical, rel=1e-6, abs=1e-7)
