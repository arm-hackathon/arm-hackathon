"""Four-class training, artifacts, inference and ONNX evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from aeolus.corpus import generate_corpus_v2
from aeolus.detector import (
    CLASS_NAMES,
    FEATURE_WIDTH,
    ONNX_MAX_ABSOLUTE_PROBABILITY_ERROR,
    WINDOW_TICKS,
    SoftmaxDetector,
    TemporalMLPDetector,
    enforce_onnx_parity,
    evidence_conclusion,
    load_detector,
    main,
    predict_scenario,
    save_detector,
    train_and_export,
    train_softmax_detector,
    train_temporal_mlp_detector,
    temporal_summary_v1,
)
from aeolus.evaluate import evaluate_v2
from aeolus.families import build_family_evidence, load_family_manifest
from aeolus.model_input import build_model_input_contract, model_artifact_metadata
from aeolus.config import load_scenario
from aeolus.sweep import generate_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "scenarios"


def _contract() -> dict[str, str]:
    config = load_scenario(SCENARIOS / "standard_habitat.json")
    return model_artifact_metadata(build_model_input_contract(config))


def _window(class_index: int, sample_index: int) -> list[list[float]]:
    window = np.zeros((WINDOW_TICKS, FEATURE_WIDTH), dtype=np.float32)
    window[:, class_index] = class_index + 1.0
    window[-1, 8 + class_index] = sample_index / 100.0
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


@pytest.fixture
def synthetic_detector() -> SoftmaxDetector:
    detector, _ = train_softmax_detector(
        _rows(), _rows(4), contract_metadata=_contract(), epochs=100
    )
    return detector


def test_softmax_detector_predicts_probabilities_for_exact_windows(
    synthetic_detector,
):
    prediction = synthetic_detector.predict_window(_window(2, 99))

    assert prediction.label in CLASS_NAMES
    assert prediction.confidence == max(prediction.probabilities.values())
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert set(prediction.probabilities) == set(CLASS_NAMES)
    assert {
        synthetic_detector.predict_window(_window(index, 3)).label
        for index in range(len(CLASS_NAMES))
    } == set(CLASS_NAMES)


def test_training_balances_unequal_class_counts():
    rows = _rows()
    rows.extend(row for row in _rows(4) if row["label"] == "nominal")

    _, receipt = train_softmax_detector(
        rows, _rows(4), contract_metadata=_contract(), epochs=20
    )

    assert receipt["training_class_counts"]["nominal"] == 12
    assert receipt["training_class_counts"]["blocked_path"] == 8
    assert receipt["class_weight_total_per_class"] == 0.25


def test_temporal_summary_has_exact_shape_and_safe_zero_request_ratios():
    window = np.zeros((WINDOW_TICKS, FEATURE_WIDTH), dtype=np.float32)
    window[:, (17, 20, 23)] = 2.0
    summary = temporal_summary_v1([window.tolist()])

    assert summary.shape == (1, 135)
    assert np.isfinite(summary).all()
    assert np.array_equal(summary[0, 120:], np.zeros(15))


def test_temporal_mlp_training_and_json_round_trip_are_deterministic(tmp_path):
    first, first_receipt = train_temporal_mlp_detector(
        _rows(), _rows(4), contract_metadata=_contract(), epochs=10
    )
    second, second_receipt = train_temporal_mlp_detector(
        _rows(), _rows(4), contract_metadata=_contract(), epochs=10
    )

    assert isinstance(first, TemporalMLPDetector)
    assert first == second
    assert first_receipt == second_receipt
    path = save_detector(first, tmp_path / "mlp.json")
    assert load_detector(path, expected_contract=_contract()) == first
    assert sum(first.predict_window(_window(2, 3)).probabilities.values()) == pytest.approx(1.0)


def _advantage_metrics(macro_f1, false_alarm, latency, recall=0.9):
    return {
        "macro_f1": macro_f1,
        "nominal_false_alarm_rate": false_alarm,
        "detection_latency_ticks": {"overall_median": latency},
        "per_class": {
            name: {"recall": recall}
            for name in CLASS_NAMES[1:]
        },
    }


def test_attainable_advantage_policy_accepts_quality_and_latency_wins():
    rule = _advantage_metrics(0.8, 0.02, 10.0)
    quality = evidence_conclusion(
        _advantage_metrics(0.9, 0.025, 10.0), rule
    )
    latency = evidence_conclusion(
        _advantage_metrics(0.795, 0.025, 8.0), rule
    )

    assert quality["ai_advantage_demonstrated"] is True
    assert quality["macro_f1_error_reduction_fraction"] >= 0.25
    assert latency["ai_advantage_demonstrated"] is True
    assert latency["latency_reduction_fraction"] >= 0.20


@pytest.mark.parametrize(
    "features, message",
    [
        ([[0.0] * FEATURE_WIDTH] * (WINDOW_TICKS - 1), "exactly 10 ticks"),
        ([[0.0] * (FEATURE_WIDTH - 1)] * WINDOW_TICKS, "exactly 24 features"),
        (
            [[[True] + [0.0] * (FEATURE_WIDTH - 1)][0]] * WINDOW_TICKS,
            "must be numeric",
        ),
        (
            [[[float("inf")] + [0.0] * (FEATURE_WIDTH - 1)][0]] * WINDOW_TICKS,
            "must be finite",
        ),
    ],
)
def test_detector_rejects_malformed_windows(synthetic_detector, features, message):
    with pytest.raises(ValueError, match=message):
        synthetic_detector.predict_window(features)


def test_detector_json_round_trip_and_contract_check(synthetic_detector, tmp_path):
    path = save_detector(synthetic_detector, tmp_path / "detector.json")

    restored = load_detector(path, expected_contract=_contract())
    assert restored == synthetic_detector
    assert restored.predict_window(_window(1, 3)) == synthetic_detector.predict_window(
        _window(1, 3)
    )

    incompatible = dict(_contract())
    incompatible["selector_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        load_detector(path, expected_contract=incompatible)


def test_detector_loader_rejects_non_finite_or_wrong_shape(synthetic_detector, tmp_path):
    path = save_detector(synthetic_detector, tmp_path / "detector.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scales"] = document["scales"][:-1]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization shape"):
        load_detector(path)

    document["scales"].append(1.0)
    document["class_names"][0] = "healthy"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="class vocabulary"):
        load_detector(path)

    document["class_names"] = list(CLASS_NAMES)
    document["contract_metadata"] = list(document["contract_metadata"].items())
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fields are malformed"):
        load_detector(path)


def _integration_spec(tmp_path: Path) -> Path:
    base_name = "standard_habitat.json"
    (tmp_path / base_name).write_bytes((SCENARIOS / base_name).read_bytes())
    telemetry = {
        "airflow_noise_fraction": 0.01,
        "airflow_bias_fraction": 0.005,
        "actuator_position_noise_fraction": 0.01,
    }
    splits = {}
    for split, seed, start, profile_id in (
        ("train", 101, 25, "train"),
        ("validation", 501, 35, "validation"),
        ("test", 1001, 45, "test"),
    ):
        splits[split] = {
            "seeds": [seed],
            "fault_start_ticks": [start],
            "operating_profiles": [
                {
                    "id": profile_id,
                    "source_multiplier": 1.0,
                    "shared_airflow_capacity": 30.0,
                    "telemetry": telemetry,
                }
            ],
            "gradual_end_effectiveness": [0.5],
            "blocked_effectiveness": [0.15],
        }
    spec = {
        "schema_version": "aeolus_sweep_v1",
        "base_scenario": base_name,
        "targets": ["cabin_a"],
        "splits": splits,
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_onnx_parity_over_acceptance_bound_is_rejected():
    with pytest.raises(ValueError, match="ONNX parity exceeds"):
        enforce_onnx_parity(
            {
                "max_absolute_probability_error": (
                    ONNX_MAX_ABSOLUTE_PROBABILITY_ERROR + 1e-8
                )
            }
        )


@pytest.mark.parametrize("error", (float("nan"), float("inf"), -float("inf")))
def test_onnx_parity_nonfinite_error_is_rejected(error):
    with pytest.raises(ValueError, match="ONNX parity must be finite"):
        enforce_onnx_parity({"max_absolute_probability_error": error})


def test_train_export_onnx_and_predict_are_reproducible(tmp_path):
    sweep_dir = tmp_path / "sweep"
    receipt = generate_sweep(_integration_spec(tmp_path), sweep_dir)
    corpus_dir = tmp_path / "corpus"
    generate_corpus_v2(sweep_dir / "families.json", corpus_dir)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = train_and_export(
        corpus_dir / "corpus.jsonl",
        sweep_dir / "families.json",
        receipt["family_manifest_sha256"],
        first_dir / "detector.json",
        first_dir / "detector.onnx",
        first_dir / "metrics.json",
    )
    second = train_and_export(
        corpus_dir / "corpus.jsonl",
        sweep_dir / "families.json",
        receipt["family_manifest_sha256"],
        second_dir / "detector.json",
        second_dir / "detector.onnx",
        second_dir / "metrics.json",
    )

    assert first == second
    assert first["families_by_split"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert first["split_evidence"]["test"]["families"] == 3
    assert first["onnx_parity"]["max_absolute_probability_error"] < 1e-5
    assert first["rule_calibration"]["selection_split"] == "validation"
    assert first["candidate_selection"]["selected_candidate"] in {
        "softmax_detector",
        "temporal_mlp_detector",
    }
    assert first["model"]["detection_latency_ticks"]["causal_stride_ticks"] == 1
    for name in ("detector.json", "detector.onnx", "metrics.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    assert first["artifact_sizes_bytes"] == {
        "model_json": (first_dir / "detector.json").stat().st_size,
        "model_onnx": (first_dir / "detector.onnx").stat().st_size,
        "metrics_json": (first_dir / "metrics.json").stat().st_size,
    }

    import onnx

    graph = onnx.load(first_dir / "detector.onnx")
    onnx.checker.check_model(graph)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    assert metadata["format"] in {
        "aeolus_softmax_detector_v1",
        "aeolus_temporal_mlp_detector_v1",
    }
    if metadata["format"] == "aeolus_temporal_mlp_detector_v1":
        assert metadata["transform_version"] == "temporal_summary_v1"
    assert metadata["model_input_version"] == "model_input_v1"
    assert metadata["class_names"] == json.dumps(
        list(CLASS_NAMES), separators=(",", ":")
    )

    manifest = load_family_manifest(sweep_dir / "families.json")
    corpus_lines = (corpus_dir / "corpus.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rows = [json.loads(line) for line in corpus_lines]
    evaluator_result = evaluate_v2(
        rows,
        load_detector(first_dir / "detector.json"),
        expected_contract=manifest.contract_metadata,
        expected_families=build_family_evidence(manifest),
        target_split="test",
    )
    assert evaluator_result["scored_total"] > 0

    predictions = predict_scenario(
        first_dir / "detector.json", SCENARIOS / "standard_habitat.json"
    )
    assert len(predictions) == 120 - WINDOW_TICKS + 1
    assert predictions[0]["end_tick"] == WINDOW_TICKS
    assert set(predictions[0]["probabilities"]) == set(CLASS_NAMES)


def test_predict_cli_emits_structured_json(synthetic_detector, tmp_path, capsys):
    detector_path = save_detector(synthetic_detector, tmp_path / "detector.json")

    assert main(["predict", str(detector_path), str(SCENARIOS / "standard_habitat.json")]) == 0
    first = json.loads(capsys.readouterr().out.splitlines()[0])

    assert first["end_tick"] == WINDOW_TICKS
    assert first["label"] in CLASS_NAMES
    assert first["confidence"] == max(first["probabilities"].values())
