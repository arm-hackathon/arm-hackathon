"""Deterministic compact causal TCN detector and FP32 ONNX export."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from aeolus.corpus import EXCLUDED_TRANSITION_LABEL
from aeolus.detector import (
    CLASS_INDEX,
    CLASS_NAMES,
    FEATURE_WIDTH,
    WINDOW_TICKS,
    Prediction,
    classification_metrics,
)


CNN_MODEL_FORMAT = "aeolus_causal_tcn_detector_v1"
CNN_CHANNELS = 4
CNN_KERNEL_TICKS = 3
CNN_DILATIONS = (1, 2, 4)
CNN_RECEPTIVE_FIELD_TICKS = 15
CNN_ONNX_OPERATORS = (
    "Sub",
    "Div",
    "Transpose",
    "Conv",
    "Relu",
    "Conv",
    "Relu",
    "Conv",
    "Relu",
    "Conv",
    "Gather",
    "Softmax",
)
_WEIGHTING_MODES = frozenset({"balanced", "sqrt_inverse"})
_CONTRACT_KEYS = frozenset(
    {"model_input_version", "selector_sha256", "topology_sha256"}
)
_MODEL_KEYS = frozenset(
    {
        "format",
        "window_ticks",
        "feature_width",
        "class_names",
        "contract_metadata",
        "weighting_mode",
        "means",
        "scales",
        "conv1_weights",
        "conv1_biases",
        "conv2_weights",
        "conv2_biases",
        "conv3_weights",
        "conv3_biases",
        "output_weights",
        "output_biases",
    }
)


Kernel = tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(frozen=True)
class TemporalCNNDetector:
    """Width-four causal dilated TCN over exact observable telemetry windows."""

    window_ticks: int
    feature_width: int
    class_names: tuple[str, ...]
    contract_metadata: dict[str, str]
    weighting_mode: str
    means: tuple[float, ...]
    scales: tuple[float, ...]
    conv1_weights: Kernel
    conv1_biases: tuple[float, ...]
    conv2_weights: Kernel
    conv2_biases: tuple[float, ...]
    conv3_weights: Kernel
    conv3_biases: tuple[float, ...]
    output_weights: Kernel
    output_biases: tuple[float, ...]

    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
        """Return one four-class probability vector per exact telemetry window."""
        _validate_detector(self)
        tensor = _window_tensor(windows)
        normalised = (
            tensor - np.asarray(self.means, dtype=np.float64)[None, None, :]
        ) / np.asarray(self.scales, dtype=np.float64)[None, None, :]
        probabilities, _ = _forward_tcn(normalised, _parameter_arrays(self))
        return probabilities

    def predict_window(self, features: Sequence[Sequence[float]]) -> Prediction:
        """Predict one exact ten-tick telemetry window."""
        probabilities = self.predict_probabilities([features])[0]
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self.class_names[index],
            confidence=float(probabilities[index]),
            probabilities={
                name: float(probabilities[class_index])
                for class_index, name in enumerate(self.class_names)
            },
        )

    def label_window(self, features: list[object]) -> str:
        """Implement the evaluator's window-labeller protocol."""
        return self.predict_window(features).label

    def reset(self) -> None:
        """A compatibility no-op: gating state is deliberately external."""


def train_temporal_cnn(
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    contract_metadata: Mapping[str, str],
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2_penalty: float = 1e-4,
    weighting_mode: str = "balanced",
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    initialization_seed: int = 0,
) -> tuple[TemporalCNNDetector, dict[str, Any]]:
    """Train the fixed width-four causal TCN candidate deterministically."""
    if epochs < 5 or epochs % 5:
        raise ValueError("CNN epochs must be a positive multiple of five")
    if weighting_mode not in _WEIGHTING_MODES:
        raise ValueError("CNN weighting_mode must be balanced or sqrt_inverse")
    hyperparameters = (learning_rate, l2_penalty, beta1, beta2, epsilon)
    if any(not math.isfinite(value) for value in hyperparameters):
        raise ValueError("CNN hyperparameters must be finite")
    if learning_rate <= 0.0 or l2_penalty < 0.0 or epsilon <= 0.0:
        raise ValueError("CNN learning rate/epsilon must be positive and L2 non-negative")
    if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
        raise ValueError("CNN Adam beta values must be between zero and one")
    contract = _validated_contract(contract_metadata)
    training_tensor, training_labels = _scored_tensor(training_rows)
    validation_tensor, validation_labels = _scored_tensor(validation_rows)
    _require_all_classes(training_labels, "training")
    _require_all_classes(validation_labels, "validation")

    means = training_tensor.mean(axis=(0, 1))
    scales = training_tensor.std(axis=(0, 1))
    scales[scales < 1e-6] = 1.0
    training_normalised = (
        training_tensor - means[None, None, :]
    ) / scales[None, None, :]
    validation_normalised = (
        validation_tensor - means[None, None, :]
    ) / scales[None, None, :]

    parameters = _initial_parameters(initialization_seed)
    first_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    sample_weights, counts = _sample_weights(training_labels, weighting_mode)

    best_score = -1.0
    best_loss = math.inf
    best_epoch = 0
    best_parameters = {name: value.copy() for name, value in parameters.items()}
    for epoch in range(1, epochs + 1):
        probabilities, cache = _forward_tcn(training_normalised, parameters)
        errors = probabilities.copy()
        errors[np.arange(len(training_labels)), training_labels] -= 1.0
        errors *= sample_weights[:, None]
        gradients = _backward_tcn(errors, cache, parameters, l2_penalty)
        for name, parameter in parameters.items():
            gradient = gradients[name]
            first_moments[name] = beta1 * first_moments[name] + (1.0 - beta1) * gradient
            second_moments[name] = beta2 * second_moments[name] + (1.0 - beta2) * (
                gradient * gradient
            )
            corrected_first = first_moments[name] / (1.0 - beta1**epoch)
            corrected_second = second_moments[name] / (1.0 - beta2**epoch)
            parameter -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )

        if epoch % 5 == 0:
            validation_probabilities, _ = _forward_tcn(
                validation_normalised, parameters
            )
            metrics = classification_metrics(
                validation_labels,
                np.argmax(validation_probabilities, axis=1),
                validation_probabilities,
            )
            validation_loss = _cross_entropy(
                validation_labels, validation_probabilities
            )
            score = float(metrics["macro_f1"])
            if score > best_score + 1e-12 or (
                math.isclose(score, best_score, abs_tol=1e-12)
                and validation_loss < best_loss
            ):
                best_score = score
                best_loss = validation_loss
                best_epoch = epoch
                best_parameters = {
                    name: value.copy() for name, value in parameters.items()
                }

    detector = _detector_from_arrays(
        best_parameters,
        contract=contract,
        weighting_mode=weighting_mode,
        means=means,
        scales=scales,
    )
    class_weight_totals = {
        name: float(sample_weights[training_labels == index].sum())
        for index, name in enumerate(CLASS_NAMES)
    }
    return detector, {
        "architecture": {
            "name": "causal_tcn_detector_v1",
            "input_shape": [WINDOW_TICKS, FEATURE_WIDTH],
            "channels": CNN_CHANNELS,
            "kernel_ticks": CNN_KERNEL_TICKS,
            "dilations": list(CNN_DILATIONS),
            "receptive_field_ticks": CNN_RECEPTIVE_FIELD_TICKS,
            "classifier": "one_by_one_conv_final_timestep",
            "class_count": len(CLASS_NAMES),
        },
        "epochs_requested": epochs,
        "selected_epoch": best_epoch,
        "learning_rate": learning_rate,
        "l2_penalty": l2_penalty,
        "initialization_seed": initialization_seed,
        "weighting_mode": weighting_mode,
        "training_rows": int(len(training_labels)),
        "validation_rows": int(len(validation_labels)),
        "training_class_counts": {
            name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "class_weight_total_by_class": class_weight_totals,
        "validation_macro_f1": best_score,
        "validation_cross_entropy": best_loss,
        "parameter_count": temporal_cnn_parameter_count(detector),
    }


def temporal_cnn_parameter_count(detector: TemporalCNNDetector) -> int:
    """Return the exact number of trainable scalar parameters."""
    _validate_detector(detector)
    return sum(
        int(value.size) for value in _parameter_arrays(detector).values()
    )


def save_temporal_cnn(detector: TemporalCNNDetector, path: str | Path) -> Path:
    """Persist one strict, finite and byte-stable causal TCN artifact."""
    _validate_detector(detector)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _detector_document(detector),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_temporal_cnn(
    path: str | Path, *, expected_contract: Mapping[str, str] | None = None
) -> TemporalCNNDetector:
    """Load a strict causal TCN artifact and fail closed on contract drift."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"CNN detector artifact not found: {source}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"CNN detector artifact is unreadable: {exc}") from None
    if not isinstance(document, dict) or set(document) != _MODEL_KEYS:
        raise ValueError("CNN detector artifact schema is incompatible")
    if document.get("format") != CNN_MODEL_FORMAT:
        raise ValueError("CNN detector artifact format is incompatible")
    try:
        detector = TemporalCNNDetector(
            window_ticks=document["window_ticks"],
            feature_width=document["feature_width"],
            class_names=tuple(document["class_names"]),
            contract_metadata=dict(document["contract_metadata"]),
            weighting_mode=document["weighting_mode"],
            means=tuple(document["means"]),
            scales=tuple(document["scales"]),
            conv1_weights=_tuple3(document["conv1_weights"]),
            conv1_biases=tuple(document["conv1_biases"]),
            conv2_weights=_tuple3(document["conv2_weights"]),
            conv2_biases=tuple(document["conv2_biases"]),
            conv3_weights=_tuple3(document["conv3_weights"]),
            conv3_biases=tuple(document["conv3_biases"]),
            output_weights=_tuple3(document["output_weights"]),
            output_biases=tuple(document["output_biases"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("CNN detector artifact fields are malformed") from exc
    _validate_detector(detector)
    if expected_contract is not None:
        if detector.contract_metadata != _validated_contract(expected_contract):
            raise ValueError("CNN detector contract does not match this inference setup")
    return detector


def export_temporal_cnn_onnx(
    detector: TemporalCNNDetector, path: str | Path
) -> Path:
    """Export the causal TCN through standard, static-quantisation-friendly ops."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("CNN ONNX export requires the 'ml' project extra") from exc
    _validate_detector(detector)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _parameter_arrays(detector)
    initializers = [
        numpy_helper.from_array(np.asarray(detector.means, dtype=np.float32), name="means"),
        numpy_helper.from_array(np.asarray(detector.scales, dtype=np.float32), name="scales"),
        numpy_helper.from_array(np.asarray(-1, dtype=np.int64), name="last_index"),
    ]
    initializers.extend(
        numpy_helper.from_array(value.astype(np.float32), name=name)
        for name, value in arrays.items()
    )
    nodes = [
        helper.make_node("Sub", ["window", "means"], ["centred"]),
        helper.make_node("Div", ["centred", "scales"], ["normalised"]),
        helper.make_node(
            "Transpose", ["normalised"], ["channels_first"], perm=[0, 2, 1]
        ),
        _conv_node("channels_first", "conv1", dilation=1, output="conv1_raw"),
        helper.make_node("Relu", ["conv1_raw"], ["conv1_active"]),
        _conv_node("conv1_active", "conv2", dilation=2, output="conv2_raw"),
        helper.make_node("Relu", ["conv2_raw"], ["conv2_active"]),
        _conv_node("conv2_active", "conv3", dilation=4, output="conv3_raw"),
        helper.make_node("Relu", ["conv3_raw"], ["conv3_active"]),
        helper.make_node(
            "Conv",
            ["conv3_active", "output_weights", "output_biases"],
            ["logit_sequence"],
            kernel_shape=[1],
            dilations=[1],
            strides=[1],
            pads=[0, 0],
        ),
        helper.make_node(
            "Gather", ["logit_sequence", "last_index"], ["logits"], axis=2
        ),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "aeolus_causal_tcn_fault_detector",
        [
            helper.make_tensor_value_info(
                "window", TensorProto.FLOAT, [None, WINDOW_TICKS, FEATURE_WIDTH]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, len(CLASS_NAMES)]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="aeolus",
        producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    metadata = {
        "format": CNN_MODEL_FORMAT,
        "window_ticks": str(detector.window_ticks),
        "feature_width": str(detector.feature_width),
        "class_names": json.dumps(list(detector.class_names), separators=(",", ":")),
        "weighting_mode": detector.weighting_mode,
        "dilations": json.dumps(list(CNN_DILATIONS), separators=(",", ":")),
        "receptive_field_ticks": str(CNN_RECEPTIVE_FIELD_TICKS),
        **detector.contract_metadata,
    }
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return destination


def _conv_node(input_name: str, prefix: str, *, dilation: int, output: str):
    from onnx import helper

    return helper.make_node(
        "Conv",
        [input_name, f"{prefix}_weights", f"{prefix}_biases"],
        [output],
        kernel_shape=[CNN_KERNEL_TICKS],
        dilations=[dilation],
        strides=[1],
        pads=[dilation * (CNN_KERNEL_TICKS - 1), 0],
    )


def _detector_document(detector: TemporalCNNDetector) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": CNN_MODEL_FORMAT,
        "window_ticks": detector.window_ticks,
        "feature_width": detector.feature_width,
        "class_names": list(detector.class_names),
        "contract_metadata": dict(detector.contract_metadata),
        "weighting_mode": detector.weighting_mode,
        "means": list(detector.means),
        "scales": list(detector.scales),
    }
    for name in (
        "conv1_weights",
        "conv1_biases",
        "conv2_weights",
        "conv2_biases",
        "conv3_weights",
        "conv3_biases",
        "output_weights",
        "output_biases",
    ):
        document[name] = np.asarray(getattr(detector, name)).tolist()
    return document


def _detector_from_arrays(
    parameters: Mapping[str, NDArray[np.float64]],
    *,
    contract: dict[str, str],
    weighting_mode: str,
    means: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> TemporalCNNDetector:
    detector = TemporalCNNDetector(
        window_ticks=WINDOW_TICKS,
        feature_width=FEATURE_WIDTH,
        class_names=CLASS_NAMES,
        contract_metadata=contract,
        weighting_mode=weighting_mode,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        conv1_weights=_array_to_kernel(parameters["conv1_weights"]),
        conv1_biases=tuple(float(value) for value in parameters["conv1_biases"]),
        conv2_weights=_array_to_kernel(parameters["conv2_weights"]),
        conv2_biases=tuple(float(value) for value in parameters["conv2_biases"]),
        conv3_weights=_array_to_kernel(parameters["conv3_weights"]),
        conv3_biases=tuple(float(value) for value in parameters["conv3_biases"]),
        output_weights=_array_to_kernel(parameters["output_weights"]),
        output_biases=tuple(float(value) for value in parameters["output_biases"]),
    )
    _validate_detector(detector)
    return detector


def _validate_detector(detector: TemporalCNNDetector) -> None:
    if detector.window_ticks != WINDOW_TICKS or detector.feature_width != FEATURE_WIDTH:
        raise ValueError("CNN detector input shape is incompatible")
    if detector.class_names != CLASS_NAMES:
        raise ValueError("CNN detector class vocabulary is incompatible")
    if detector.weighting_mode not in _WEIGHTING_MODES:
        raise ValueError("CNN detector weighting_mode is incompatible")
    _validated_contract(detector.contract_metadata)
    arrays = {
        "means": (detector.means, (FEATURE_WIDTH,)),
        "scales": (detector.scales, (FEATURE_WIDTH,)),
        "conv1_weights": (
            detector.conv1_weights,
            (CNN_CHANNELS, FEATURE_WIDTH, CNN_KERNEL_TICKS),
        ),
        "conv1_biases": (detector.conv1_biases, (CNN_CHANNELS,)),
        "conv2_weights": (
            detector.conv2_weights,
            (CNN_CHANNELS, CNN_CHANNELS, CNN_KERNEL_TICKS),
        ),
        "conv2_biases": (detector.conv2_biases, (CNN_CHANNELS,)),
        "conv3_weights": (
            detector.conv3_weights,
            (CNN_CHANNELS, CNN_CHANNELS, CNN_KERNEL_TICKS),
        ),
        "conv3_biases": (detector.conv3_biases, (CNN_CHANNELS,)),
        "output_weights": (
            detector.output_weights,
            (len(CLASS_NAMES), CNN_CHANNELS, 1),
        ),
        "output_biases": (detector.output_biases, (len(CLASS_NAMES),)),
    }
    for name, (value, expected_shape) in arrays.items():
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CNN detector {name} must be numeric") from exc
        if array.shape != expected_shape:
            raise ValueError(f"CNN detector {name} shape is incompatible")
        if not np.isfinite(array).all():
            raise ValueError(f"CNN detector {name} must be finite")
    if np.any(np.asarray(detector.scales, dtype=np.float64) <= 0.0):
        raise ValueError("CNN detector scales must be positive")


def _validated_contract(metadata: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metadata, Mapping) or set(metadata) != _CONTRACT_KEYS:
        raise ValueError("CNN detector contract metadata is malformed")
    contract = dict(metadata)
    if any(not isinstance(value, str) for value in contract.values()):
        raise ValueError("CNN detector contract metadata is malformed")
    if contract["model_input_version"] != "model_input_v1":
        raise ValueError("CNN detector model-input version is incompatible")
    for key in ("selector_sha256", "topology_sha256"):
        value = contract[key]
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("CNN detector contract hash is malformed")
    return contract


def _window_tensor(
    windows: Sequence[Sequence[Sequence[float]]],
) -> NDArray[np.float64]:
    try:
        tensor = np.asarray(windows)
    except (TypeError, ValueError) as exc:
        raise ValueError("CNN detector windows must be a rectangular numeric array") from exc
    if tensor.ndim != 3 or tensor.shape[0] == 0:
        raise ValueError("CNN detector requires at least one feature window")
    if tensor.shape[1] != WINDOW_TICKS:
        raise ValueError(f"CNN detector window must contain exactly {WINDOW_TICKS} ticks")
    if tensor.shape[2] != FEATURE_WIDTH:
        raise ValueError(f"CNN detector window must contain exactly {FEATURE_WIDTH} features")
    if tensor.dtype == np.bool_ or not np.issubdtype(tensor.dtype, np.number):
        raise ValueError("CNN detector features must be numeric")
    result = tensor.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("CNN detector features must be finite")
    return result


def _scored_tensor(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    scored = [row for row in rows if row.get("label") != EXCLUDED_TRANSITION_LABEL]
    if not scored:
        raise ValueError("CNN detector dataset contains no scored rows")
    unsupported = sorted({row.get("label") for row in scored} - set(CLASS_NAMES))
    if unsupported:
        raise ValueError(f"CNN detector dataset contains unsupported label {unsupported[0]!r}")
    tensor = _window_tensor([row["features"] for row in scored])
    labels = np.asarray([CLASS_INDEX[row["label"]] for row in scored], dtype=np.int64)
    return tensor, labels


def _initial_parameters(seed: int) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return {
        "conv1_weights": rng.normal(
            0.0,
            math.sqrt(2.0 / (FEATURE_WIDTH * CNN_KERNEL_TICKS)),
            size=(CNN_CHANNELS, FEATURE_WIDTH, CNN_KERNEL_TICKS),
        ),
        "conv1_biases": np.zeros(CNN_CHANNELS, dtype=np.float64),
        "conv2_weights": rng.normal(
            0.0,
            math.sqrt(2.0 / (CNN_CHANNELS * CNN_KERNEL_TICKS)),
            size=(CNN_CHANNELS, CNN_CHANNELS, CNN_KERNEL_TICKS),
        ),
        "conv2_biases": np.zeros(CNN_CHANNELS, dtype=np.float64),
        "conv3_weights": rng.normal(
            0.0,
            math.sqrt(2.0 / (CNN_CHANNELS * CNN_KERNEL_TICKS)),
            size=(CNN_CHANNELS, CNN_CHANNELS, CNN_KERNEL_TICKS),
        ),
        "conv3_biases": np.zeros(CNN_CHANNELS, dtype=np.float64),
        "output_weights": rng.normal(
            0.0,
            math.sqrt(2.0 / CNN_CHANNELS),
            size=(len(CLASS_NAMES), CNN_CHANNELS, 1),
        ),
        "output_biases": np.zeros(len(CLASS_NAMES), dtype=np.float64),
    }


def _parameter_arrays(
    detector: TemporalCNNDetector,
) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(getattr(detector, name), dtype=np.float64)
        for name in (
            "conv1_weights",
            "conv1_biases",
            "conv2_weights",
            "conv2_biases",
            "conv3_weights",
            "conv3_biases",
            "output_weights",
            "output_biases",
        )
    }


def _forward_tcn(
    tensor: NDArray[np.float64],
    parameters: Mapping[str, NDArray[np.float64]],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    raw1, cache1 = _causal_conv_forward(
        tensor, parameters["conv1_weights"], parameters["conv1_biases"], 1
    )
    active1 = np.maximum(raw1, 0.0)
    raw2, cache2 = _causal_conv_forward(
        active1, parameters["conv2_weights"], parameters["conv2_biases"], 2
    )
    active2 = np.maximum(raw2, 0.0)
    raw3, cache3 = _causal_conv_forward(
        active2, parameters["conv3_weights"], parameters["conv3_biases"], 4
    )
    active3 = np.maximum(raw3, 0.0)
    logit_sequence, output_cache = _causal_conv_forward(
        active3, parameters["output_weights"], parameters["output_biases"], 1
    )
    logits = logit_sequence[:, -1, :]
    probabilities = _softmax(logits)
    return probabilities, {
        "raw1": raw1,
        "raw2": raw2,
        "raw3": raw3,
        "cache1": cache1,
        "cache2": cache2,
        "cache3": cache3,
        "output_cache": output_cache,
        "logit_sequence_shape": logit_sequence.shape,
    }


def _backward_tcn(
    errors: NDArray[np.float64],
    cache: Mapping[str, Any],
    parameters: Mapping[str, NDArray[np.float64]],
    l2_penalty: float,
) -> dict[str, NDArray[np.float64]]:
    output_gradient = np.zeros(cache["logit_sequence_shape"], dtype=np.float64)
    output_gradient[:, -1, :] = errors
    active3_gradient, output_weight_gradient, output_bias_gradient = (
        _causal_conv_backward(
            output_gradient,
            cache["output_cache"],
            parameters["output_weights"],
        )
    )
    raw3_gradient = np.where(cache["raw3"] > 0.0, active3_gradient, 0.0)
    active2_gradient, conv3_weight_gradient, conv3_bias_gradient = (
        _causal_conv_backward(
            raw3_gradient, cache["cache3"], parameters["conv3_weights"]
        )
    )
    raw2_gradient = np.where(cache["raw2"] > 0.0, active2_gradient, 0.0)
    active1_gradient, conv2_weight_gradient, conv2_bias_gradient = (
        _causal_conv_backward(
            raw2_gradient, cache["cache2"], parameters["conv2_weights"]
        )
    )
    raw1_gradient = np.where(cache["raw1"] > 0.0, active1_gradient, 0.0)
    _, conv1_weight_gradient, conv1_bias_gradient = _causal_conv_backward(
        raw1_gradient, cache["cache1"], parameters["conv1_weights"]
    )
    return {
        "conv1_weights": conv1_weight_gradient
        + l2_penalty * parameters["conv1_weights"],
        "conv1_biases": conv1_bias_gradient,
        "conv2_weights": conv2_weight_gradient
        + l2_penalty * parameters["conv2_weights"],
        "conv2_biases": conv2_bias_gradient,
        "conv3_weights": conv3_weight_gradient
        + l2_penalty * parameters["conv3_weights"],
        "conv3_biases": conv3_bias_gradient,
        "output_weights": output_weight_gradient
        + l2_penalty * parameters["output_weights"],
        "output_biases": output_bias_gradient,
    }


def _causal_conv_forward(
    tensor: NDArray[np.float64],
    weights: NDArray[np.float64],
    biases: NDArray[np.float64],
    dilation: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    batch, temporal_steps, input_channels = tensor.shape
    output_channels, weight_inputs, kernel_ticks = weights.shape
    if weight_inputs != input_channels or kernel_ticks < 1 or dilation < 1:
        raise ValueError("causal convolution shapes are incompatible")
    left_padding = dilation * (kernel_ticks - 1)
    padded = np.pad(tensor, ((0, 0), (left_padding, 0), (0, 0)))
    patches = np.stack(
        [
            padded[:, index * dilation : index * dilation + temporal_steps, :]
            for index in range(kernel_ticks)
        ],
        axis=3,
    )
    patch_matrix = patches.reshape((batch * temporal_steps, -1))
    weight_matrix = weights.reshape((output_channels, -1))
    output = (patch_matrix @ weight_matrix.T).reshape(
        (batch, temporal_steps, output_channels)
    ) + biases[None, None, :]
    return output, {
        "patches": patches,
        "input_shape": tensor.shape,
        "dilation": dilation,
        "left_padding": left_padding,
    }


def _causal_conv_backward(
    output_gradient: NDArray[np.float64],
    cache: Mapping[str, Any],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    patches = cache["patches"]
    batch, temporal_steps, input_channels, kernel_ticks = patches.shape
    output_channels = weights.shape[0]
    gradient_matrix = output_gradient.reshape((batch * temporal_steps, output_channels))
    patch_matrix = patches.reshape((batch * temporal_steps, -1))
    weight_gradient = (gradient_matrix.T @ patch_matrix).reshape(weights.shape)
    bias_gradient = output_gradient.sum(axis=(0, 1))
    patch_gradient = (gradient_matrix @ weights.reshape((output_channels, -1))).reshape(
        patches.shape
    )
    left_padding = int(cache["left_padding"])
    dilation = int(cache["dilation"])
    padded_gradient = np.zeros(
        (batch, temporal_steps + left_padding, input_channels), dtype=np.float64
    )
    for index in range(kernel_ticks):
        start = index * dilation
        padded_gradient[:, start : start + temporal_steps, :] += patch_gradient[
            :, :, :, index
        ]
    input_gradient = padded_gradient[:, left_padding:, :]
    if input_gradient.shape != cache["input_shape"]:
        raise ValueError("causal convolution backward shape drifted")
    return input_gradient, weight_gradient, bias_gradient


def _sample_weights(
    labels: NDArray[np.int64], weighting_mode: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if weighting_mode == "balanced":
        weights = np.asarray(
            [1.0 / counts[label] for label in labels], dtype=np.float64
        )
    else:
        weights = np.asarray(
            [1.0 / math.sqrt(counts[label]) for label in labels], dtype=np.float64
        )
    weights /= weights.sum()
    return weights, counts


def _require_all_classes(labels: NDArray[np.int64], split: str) -> None:
    present = set(int(value) for value in labels)
    if present != set(range(len(CLASS_NAMES))):
        missing = [
            CLASS_NAMES[index]
            for index in range(len(CLASS_NAMES))
            if index not in present
        ]
        raise ValueError(f"CNN {split} rows are missing class {missing[0]!r}")


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _cross_entropy(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def _array_to_kernel(value: NDArray[np.float64]) -> Kernel:
    return tuple(
        tuple(tuple(float(item) for item in kernel) for kernel in channel)
        for channel in value
    )


def _tuple3(value: Sequence[Sequence[Sequence[float]]]) -> Kernel:
    return tuple(
        tuple(tuple(kernel) for kernel in channel)
        for channel in value
    )
