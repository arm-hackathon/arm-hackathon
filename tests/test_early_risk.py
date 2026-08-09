"""Leakage-safe temporal early-risk prediction contracts."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from aeolus.early_risk import (
    EARLY_RISK_CLASS_NAMES,
    FORECAST_HORIZON_TICKS,
    NO_EARLY_RISK,
    WINDOW_TICKS,
    build_early_risk_rows_from_traces,
    calibrate_early_risk_abstention,
    future_risk_label,
    load_early_risk_artifact,
    save_early_risk_artifact,
    train_early_risk_predictor,
)

FEATURE_WIDTH = 24
CONTRACT = {
    "model_input_version": "model_input_v1",
    "selector_sha256": "1" * 64,
    "topology_sha256": "2" * 64,
}
ZONE_IDS = ("cabin_a", "cabin_b")


def _features(ticks: int = 30, *, class_index: int = 0) -> list[list[float]]:
    values = np.zeros((ticks, FEATURE_WIDTH), dtype=np.float32)
    if class_index:
        values[:, 14 + class_index] = np.arange(ticks, dtype=np.float32) / 100.0
        values[:, 16 + class_index] = np.arange(ticks, dtype=np.float32) / 50.0
    return values.tolist()


def _safe_trace(ticks: int = 30) -> dict[str, list[float]]:
    return {zone_id: [0.20] * ticks for zone_id in ZONE_IDS}


def test_future_label_predicts_one_unique_physical_ceiling_crossing():
    physical = _safe_trace()
    physical["cabin_a"][20] = 0.301

    assert (
        future_risk_label(
            physical,
            end_index=9,
            zone_ids=ZONE_IDS,
            ceiling=0.30,
            horizon_ticks=FORECAST_HORIZON_TICKS,
        )
        == "risk:cabin_a"
    )
    assert (
        future_risk_label(
            physical,
            end_index=7,
            zone_ids=ZONE_IDS,
            ceiling=0.30,
            horizon_ticks=FORECAST_HORIZON_TICKS,
        )
        == NO_EARLY_RISK
    )


def test_future_label_excludes_already_unsafe_or_ambiguous_targets():
    already_unsafe = _safe_trace()
    already_unsafe["cabin_a"][9] = 0.31
    assert (
        future_risk_label(
            already_unsafe,
            end_index=9,
            zone_ids=ZONE_IDS,
            ceiling=0.30,
            horizon_ticks=FORECAST_HORIZON_TICKS,
        )
        is None
    )

    ambiguous = _safe_trace()
    ambiguous["cabin_a"][15] = 0.31
    ambiguous["cabin_b"][16] = 0.32
    assert (
        future_risk_label(
            ambiguous,
            end_index=9,
            zone_ids=ZONE_IDS,
            ceiling=0.30,
            horizon_ticks=FORECAST_HORIZON_TICKS,
        )
        is None
    )


def test_corpus_rows_contain_observable_windows_not_future_physical_truth():
    physical = _safe_trace()
    physical["cabin_b"][18] = 0.301
    rows = build_early_risk_rows_from_traces(
        family_id="fresh-train-family",
        split="train",
        scenario_role="fault",
        feature_trace=_features(),
        physical_co2_trace=physical,
        zone_ids=ZONE_IDS,
        ceiling=0.30,
    )

    assert rows
    assert set(rows[0]) == {
        "family_id",
        "split",
        "scenario_role",
        "start_tick",
        "end_tick",
        "label",
        "features",
    }
    assert all(len(row["features"]) == WINDOW_TICKS for row in rows)
    assert any(row["label"] == "risk:cabin_b" for row in rows)
    assert all(
        "physical" not in key and "future" not in key for row in rows for key in row
    )


def test_non_eligible_lookalike_never_receives_a_positive_label():
    physical = _safe_trace()
    physical["cabin_a"][18] = 0.301
    rows = build_early_risk_rows_from_traces(
        family_id="transient-lookalike",
        split="train",
        scenario_role="fault",
        feature_trace=_features(),
        physical_co2_trace=physical,
        zone_ids=ZONE_IDS,
        ceiling=0.30,
        positive_eligible=False,
    )

    assert rows
    assert {row["label"] for row in rows} == {NO_EARLY_RISK}


def test_corpus_builder_rejects_final_split_and_malformed_trace():
    kwargs = {
        "family_id": "forbidden-final-family",
        "split": "final",
        "scenario_role": "fault",
        "feature_trace": _features(),
        "physical_co2_trace": _safe_trace(),
        "zone_ids": ZONE_IDS,
        "ceiling": 0.30,
    }
    with pytest.raises(ValueError, match="final"):
        build_early_risk_rows_from_traces(**kwargs)

    kwargs["split"] = "train"
    kwargs["feature_trace"][0] = [math.nan] * FEATURE_WIDTH
    with pytest.raises(ValueError, match="finite"):
        build_early_risk_rows_from_traces(**kwargs)


def _training_rows(samples_per_class: int = 6) -> list[dict]:
    rows: list[dict] = []
    for class_index, label in enumerate(EARLY_RISK_CLASS_NAMES):
        for sample_index in range(samples_per_class):
            rows.append(
                {
                    "family_id": f"family-{class_index}-{sample_index}",
                    "split": "train",
                    "scenario_role": "reference" if class_index == 0 else "fault",
                    "start_tick": 1,
                    "end_tick": WINDOW_TICKS,
                    "label": label,
                    "features": _features(WINDOW_TICKS, class_index=class_index),
                }
            )
    return rows


def test_smallest_predictor_training_is_deterministic_and_probabilistic():
    training = _training_rows()
    validation = _training_rows(3)

    first, first_receipt = train_early_risk_predictor(
        training,
        validation,
        contract_metadata=CONTRACT,
        epochs=40,
    )
    second, second_receipt = train_early_risk_predictor(
        training,
        validation,
        contract_metadata=CONTRACT,
        epochs=40,
    )

    assert first == second
    assert first_receipt == second_receipt
    prediction = first.predict_window(_features(WINDOW_TICKS, class_index=2))
    assert prediction.label in EARLY_RISK_CLASS_NAMES
    assert prediction.probability == max(prediction.probabilities.values())
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.margin >= 0.0


def test_abstention_calibration_and_artifact_round_trip_are_deterministic(tmp_path):
    training = _training_rows(8)
    validation = _training_rows(4)
    for row in validation:
        row["split"] = "validation"
    predictor, training_receipt = train_early_risk_predictor(
        training,
        validation,
        contract_metadata=CONTRACT,
        epochs=80,
    )

    calibrated, calibration_receipt = calibrate_early_risk_abstention(
        predictor,
        validation,
        max_reference_warning_fraction=0.0,
        max_negative_fault_warning_fraction=0.0,
    )

    assert calibration_receipt["eligible"] is True
    assert 0.0 <= calibrated.min_probability <= 1.0
    assert 0.0 <= calibrated.min_margin <= 1.0
    assert calibration_receipt["reference_warning_fraction"] == 0.0

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_document = save_early_risk_artifact(
        first_path,
        calibrated,
        training_receipt=training_receipt,
        calibration_receipt=calibration_receipt,
    )
    second_document = save_early_risk_artifact(
        second_path,
        calibrated,
        training_receipt=training_receipt,
        calibration_receipt=calibration_receipt,
    )

    assert first_document == second_document
    assert first_path.read_bytes() == second_path.read_bytes()
    loaded, loaded_training, loaded_calibration = load_early_risk_artifact(first_path)
    assert loaded.artifact_sha256 == first_document["artifact_sha256"]
    assert loaded.artifact_model_sha256 is not None
    assert replace(
        loaded,
        artifact_sha256=None,
        artifact_model_sha256=None,
    ) == calibrated
    assert loaded_training == training_receipt
    assert loaded_calibration == calibration_receipt
