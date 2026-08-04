"""Target-explicit ONNX benchmark evidence boundaries."""

from __future__ import annotations

import platform
import hashlib
from pathlib import Path

import numpy as np

from aeolus.config import load_scenario
from aeolus.baseline import RuleBaseline
from aeolus.edge_benchmark import benchmark_onnx, benchmark_rule
from aeolus.model_input import build_model_input_contract, model_artifact_metadata
from aeolus.temporal_cnn import export_temporal_cnn_onnx, train_temporal_cnn
from aeolus.detector import CLASS_NAMES, FEATURE_WIDTH, WINDOW_TICKS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict]:
    rows = []
    for class_index, label in enumerate(CLASS_NAMES):
        for sample in range(4):
            window = np.zeros((WINDOW_TICKS, FEATURE_WIDTH), dtype=np.float32)
            window[:, class_index] = class_index + 1.0
            window[-1, 8 + class_index] = sample / 10.0
            rows.append({"label": label, "features": window.tolist()})
    return rows


def test_onnx_benchmark_records_target_and_refuses_non_arm_claim(tmp_path: Path):
    config = load_scenario(REPO_ROOT / "scenarios" / "standard_habitat.json")
    contract = model_artifact_metadata(build_model_input_contract(config))
    detector, _ = train_temporal_cnn(
        _rows(), _rows(), contract_metadata=contract, epochs=10
    )
    model = export_temporal_cnn_onnx(detector, tmp_path / "cnn.onnx")
    windows = np.asarray([row["features"] for row in _rows()[:2]], dtype=np.float32)

    receipt = benchmark_onnx(
        model,
        windows,
        warmup_iterations=2,
        measured_iterations=10,
        batch_size=1,
        intra_op_threads=1,
    )

    assert receipt["schema_version"] == "aeolus_onnx_benchmark_v1"
    assert len(receipt["model_sha256"]) == 64
    assert receipt["model_bytes"] == model.stat().st_size
    assert receipt["warmup_iterations"] == 2
    assert receipt["measured_iterations"] == 10
    assert receipt["batch_size"] == 1
    assert receipt["intra_op_threads"] == 1
    assert receipt["latency_ms"]["median"] > 0.0
    assert receipt["latency_ms"]["p95"] > 0.0
    assert receipt["throughput_windows_per_second"] > 0.0
    assert receipt["process_max_rss_kib"] > 0
    assert receipt["machine"] == platform.machine()
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        assert receipt["arm_performance_claim"] is False
        assert receipt["claim_scope"] == "local_readiness_only"


def test_rule_benchmark_is_comparable_and_cannot_claim_arm_on_x86():
    scenario = REPO_ROOT / "scenarios" / "standard_habitat.json"
    config = load_scenario(scenario)
    windows = np.asarray([row["features"] for row in _rows()[:3]], dtype=np.float32)

    receipt = benchmark_rule(
        RuleBaseline(config),
        windows,
        scenario_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest(),
        warmup_iterations=2,
        measured_iterations=10,
    )

    assert receipt["schema_version"] == "aeolus_rule_benchmark_v1"
    assert receipt["method"] == "calibrated_rules"
    assert receipt["batch_size"] == 1
    assert receipt["scenario_sha256"] == hashlib.sha256(
        scenario.read_bytes()
    ).hexdigest()
    assert receipt["latency_ms"]["median"] > 0.0
    assert receipt["throughput_windows_per_second"] > 0.0
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        assert receipt["arm_performance_claim"] is False
        assert receipt["claim_scope"] == "local_readiness_only"
