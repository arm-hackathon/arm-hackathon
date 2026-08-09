"""Leakage-safe temporal early-risk corpus rows and compact prediction.

The predictor consumes only fixed ``model_input_v1`` telemetry windows. Future
physical CO2 is used solely to derive development labels and is never included
in model-facing rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aeolus.config import load_scenario
from aeolus.detector import temporal_summary_v1
from aeolus.families import load_family_manifest, validate_manifest_disjointness
from aeolus.model_input import (
    MODEL_INPUT_SHAPE,
    build_model_input_contract,
    model_artifact_metadata,
    model_input_v1,
)
from aeolus.scenario import RECOVERY_RUN, RunSpec, run_recovery_scenario

WINDOW_TICKS = 10
FEATURE_WIDTH = MODEL_INPUT_SHAPE[0]
FORECAST_HORIZON_TICKS = 12
NO_EARLY_RISK = "no_early_risk"
EARLY_RISK_CLASS_NAMES = (
    NO_EARLY_RISK,
    "risk:cabin_a",
    "risk:cabin_b",
)
_CLASS_INDEX = {name: index for index, name in enumerate(EARLY_RISK_CLASS_NAMES)}
TEMPORAL_FEATURES = 135


@dataclass(frozen=True)
class EarlyRiskPrediction:
    """One probabilistic early-risk prediction before policy abstention."""

    label: str
    probability: float
    margin: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EarlyRiskPredictor:
    """Deterministic softmax regression over frozen temporal summaries."""

    window_ticks: int
    feature_width: int
    class_names: tuple[str, ...]
    contract_metadata: dict[str, str]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    biases: tuple[float, ...]
    min_probability: float = 0.0
    min_margin: float = 0.0
    artifact_sha256: str | None = None
    artifact_model_sha256: str | None = None

    def assert_artifact_identity(self, artifact_sha256: str) -> None:
        """Fail closed unless live model bytes match the loaded artifact."""
        _validate_predictor(self)
        if self.artifact_sha256 != artifact_sha256:
            raise ValueError("early-risk predictor artifact identity does not match")
        if self.artifact_model_sha256 is None:
            raise ValueError("early-risk predictor model payload is not artifact-bound")

    def predict_probabilities(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
        """Return one probability vector per exact observable window."""
        _validate_predictor(self)
        matrix = temporal_summary_v1(windows)
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        biases = np.asarray(self.biases, dtype=np.float64)
        return _softmax(((matrix - means) / scales) @ weights + biases)

    def predict_window(
        self, features: Sequence[Sequence[float]]
    ) -> EarlyRiskPrediction:
        """Predict one ten-tick telemetry window."""
        probabilities = self.predict_probabilities([features])[0]
        order = np.argsort(probabilities)
        top_index = int(order[-1])
        second = float(probabilities[int(order[-2])])
        top = float(probabilities[top_index])
        margin = top - second
        raw_label = self.class_names[top_index]
        label = raw_label
        if raw_label != NO_EARLY_RISK and (
            top < self.min_probability or margin < self.min_margin
        ):
            label = NO_EARLY_RISK
        return EarlyRiskPrediction(
            label=label,
            probability=top,
            margin=margin,
            probabilities={
                name: float(probabilities[index])
                for index, name in enumerate(self.class_names)
            },
        )


def future_risk_label(
    physical_co2_trace: Mapping[str, Sequence[float]],
    *,
    end_index: int,
    zone_ids: Sequence[str],
    ceiling: float,
    horizon_ticks: int = FORECAST_HORIZON_TICKS,
) -> str | None:
    """Label one end tick from future physical CO2 outcomes.

    ``None`` means the row is unscorable because it is already unsafe or more
    than one target becomes unsafe. ``no_early_risk`` is a valid negative.
    """
    zones, trace_length = _validate_physical_trace(physical_co2_trace, zone_ids)
    if not isinstance(end_index, int) or isinstance(end_index, bool):
        raise ValueError("early-risk end_index must be an integer")
    if end_index < 0 or end_index >= trace_length:
        raise ValueError("early-risk end_index is out of range")
    if not _finite_number(ceiling) or float(ceiling) <= 0.0:
        raise ValueError("early-risk CO2 ceiling must be positive and finite")
    if (
        not isinstance(horizon_ticks, int)
        or isinstance(horizon_ticks, bool)
        or horizon_ticks < 1
    ):
        raise ValueError("early-risk horizon must be a positive integer")

    threshold = float(ceiling)
    if any(float(physical_co2_trace[zone][end_index]) >= threshold for zone in zones):
        return None
    stop = min(trace_length, end_index + horizon_ticks + 1)
    unsafe_targets = [
        zone
        for zone in zones
        if any(
            float(value) >= threshold
            for value in physical_co2_trace[zone][end_index + 1 : stop]
        )
    ]
    if not unsafe_targets:
        return NO_EARLY_RISK
    if len(unsafe_targets) != 1:
        return None
    return f"risk:{unsafe_targets[0]}"


def build_early_risk_rows_from_traces(
    *,
    family_id: str,
    split: str,
    scenario_role: str,
    feature_trace: Sequence[Sequence[float]],
    physical_co2_trace: Mapping[str, Sequence[float]],
    zone_ids: Sequence[str],
    ceiling: float,
    window_ticks: int = WINDOW_TICKS,
    horizon_ticks: int = FORECAST_HORIZON_TICKS,
    stride_ticks: int = 1,
    positive_eligible: bool = True,
) -> list[dict[str, Any]]:
    """Build causal model rows while keeping future physical truth out of rows."""
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("early-risk family_id must be non-empty")
    if split == "final":
        raise ValueError("final families are prohibited from early-risk development")
    if split not in {"train", "validation"}:
        raise ValueError("early-risk split must be train or validation")
    if scenario_role not in {"reference", "fault"}:
        raise ValueError("early-risk scenario_role must be reference or fault")
    if (
        not isinstance(window_ticks, int)
        or isinstance(window_ticks, bool)
        or window_ticks != WINDOW_TICKS
    ):
        raise ValueError(
            f"early-risk windows must contain exactly {WINDOW_TICKS} ticks"
        )
    if (
        not isinstance(stride_ticks, int)
        or isinstance(stride_ticks, bool)
        or stride_ticks < 1
    ):
        raise ValueError("early-risk stride must be a positive integer")
    if not isinstance(positive_eligible, bool):
        raise ValueError("early-risk positive_eligible flag must be boolean")

    zones, trace_length = _validate_physical_trace(physical_co2_trace, zone_ids)
    if len(feature_trace) != trace_length:
        raise ValueError(
            "early-risk observable and physical traces must have equal lengths"
        )
    observable: list[list[float]] = []
    for vector in feature_trace:
        if not isinstance(vector, Sequence) or len(vector) != FEATURE_WIDTH:
            raise ValueError(
                f"early-risk model input must contain exactly {FEATURE_WIDTH} features"
            )
        if any(not _finite_number(value) for value in vector):
            raise ValueError("early-risk model input must be finite")
        observable.append([float(value) for value in vector])

    rows: list[dict[str, Any]] = []
    for end_index in range(window_ticks - 1, trace_length, stride_ticks):
        label = future_risk_label(
            physical_co2_trace,
            end_index=end_index,
            zone_ids=zones,
            ceiling=ceiling,
            horizon_ticks=horizon_ticks,
        )
        if label is None:
            continue
        if not positive_eligible:
            label = NO_EARLY_RISK
        start_index = end_index - window_ticks + 1
        rows.append(
            {
                "family_id": family_id,
                "split": split,
                "scenario_role": scenario_role,
                "start_tick": start_index + 1,
                "end_tick": end_index + 1,
                "label": label,
                "features": [
                    list(vector) for vector in observable[start_index : end_index + 1]
                ],
            }
        )
    return rows


def generate_early_risk_corpus(
    family_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    forbidden_manifest_paths: Sequence[str | Path],
    run: RunSpec = RECOVERY_RUN,
    window_ticks: int = WINDOW_TICKS,
    horizon_ticks: int = FORECAST_HORIZON_TICKS,
    stride_ticks: int = 3,
) -> dict[str, Any]:
    """Generate a byte-stable development corpus and prove final isolation.

    Reused reference scenarios are emitted once rather than once per fault
    family. Scenario runs are cached only for the duration of generation and
    rows are streamed to JSONL to keep memory bounded.
    """
    development = load_family_manifest(Path(family_manifest_path))
    present_splits = {family.split for family in development.families}
    if present_splits != {"train", "validation"}:
        raise ValueError(
            "early-risk development manifest must contain train and validation only"
        )
    if not forbidden_manifest_paths:
        raise ValueError(
            "early-risk corpus requires at least one forbidden final manifest"
        )
    forbidden_hashes: list[str] = []
    for forbidden_path in forbidden_manifest_paths:
        forbidden = load_family_manifest(Path(forbidden_path))
        if {family.split for family in forbidden.families} != {"final"}:
            raise ValueError(
                "early-risk forbidden manifest must contain final families only"
            )
        validate_manifest_disjointness(development, forbidden)
        forbidden_hashes.append(forbidden.manifest_sha256)

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"early-risk output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    corpus_path = destination / "corpus.jsonl"

    scenario_entries: dict[Path, tuple[str, str, str, bool]] = {}
    family_metadata: dict[str, dict[str, str]] = {}
    for family in development.families:
        family_metadata[family.family_id] = {
            "split": family.split,
            "fault_class": family.fault_class,
            "reference_scenario_sha256": hashlib.sha256(
                family.reference_path.read_bytes()
            ).hexdigest(),
            "fault_scenario_sha256": hashlib.sha256(
                family.fault_path.read_bytes()
            ).hexdigest(),
        }
        scenario_entries.setdefault(
            family.reference_path,
            (
                f"reference:{family.reference_path.stem}",
                family.split,
                "reference",
                False,
            ),
        )
        scenario_entries[family.fault_path] = (
            family.family_id,
            family.split,
            "fault",
            family.fault_class == "gradual_primary_fan_degradation",
        )

    total_rows = 0
    label_counts = {
        split: {label: 0 for label in EARLY_RISK_CLASS_NAMES}
        for split in ("train", "validation")
    }
    scenario_counts = {
        split: {"reference": 0, "fault": 0} for split in ("train", "validation")
    }
    contract_metadata: dict[str, str] | None = None
    with corpus_path.open("w", encoding="utf-8") as handle:
        for path, (row_family_id, split, role, positive_eligible) in sorted(
            scenario_entries.items(), key=lambda item: item[0].name
        ):
            config = load_scenario(path)
            contract = build_model_input_contract(config)
            metadata = model_artifact_metadata(contract)
            if contract_metadata is None:
                contract_metadata = metadata
            elif metadata != contract_metadata:
                raise ValueError(
                    "early-risk scenarios do not share one model input contract"
                )
            result = run_recovery_scenario(
                config,
                run_id=f"early-risk-{path.stem}",
                governed=False,
                run=run,
            )
            feature_trace = [
                model_input_v1(record, contract).tolist() for record in result.records
            ]
            crew_cabins = tuple(
                zone
                for zone in config.non_processing_zones()
                if zone.preset == "crew_cabin"
            )
            physical_trace = {
                zone.id: [
                    float(state.zone_co2_mass[zone.id] / zone.air_volume)
                    for state in result.states
                ]
                for zone in crew_cabins
            }
            rows = build_early_risk_rows_from_traces(
                family_id=row_family_id,
                split=split,
                scenario_role=role,
                feature_trace=feature_trace,
                physical_co2_trace=physical_trace,
                zone_ids=tuple(zone.id for zone in crew_cabins),
                ceiling=run.crew_cabin_co2_concentration_ceiling,
                window_ticks=window_ticks,
                horizon_ticks=horizon_ticks,
                stride_ticks=stride_ticks,
                positive_eligible=positive_eligible,
            )
            scenario_counts[split][role] += 1
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                label_counts[split][row["label"]] += 1
                total_rows += 1

    assert contract_metadata is not None
    if any(
        label_counts[split][label] == 0
        for split in ("train", "validation")
        for label in EARLY_RISK_CLASS_NAMES
    ):
        raise ValueError("early-risk generated corpus is missing a required class")
    family_counts = {
        split: sum(1 for family in development.families if family.split == split)
        for split in ("train", "validation")
    }
    manifest: dict[str, Any] = {
        "schema_version": "aeolus_early_risk_corpus_v1",
        "source_family_manifest_sha256": development.manifest_sha256,
        "forbidden_final_manifest_sha256": sorted(forbidden_hashes),
        "contract_metadata": contract_metadata,
        "window_ticks": window_ticks,
        "forecast_horizon_ticks": horizon_ticks,
        "stride_ticks": stride_ticks,
        "run": {
            "total_ticks": run.total_ticks,
            "warmup_ticks": run.warmup_ticks,
            "crew_cabin_co2_concentration_ceiling": (
                run.crew_cabin_co2_concentration_ceiling
            ),
        },
        "family_counts_by_split": family_counts,
        "scenario_counts_by_split": scenario_counts,
        "label_counts_by_split": label_counts,
        "total_rows": total_rows,
        "families": family_metadata,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_early_risk_corpus(
    corpus_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load finite rows and partition them by train/validation split."""
    rows: list[dict[str, Any]] = []
    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"early-risk corpus row {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"early-risk corpus row {line_number} is malformed")
            rows.append(row)
    split_rows = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("train", "validation")
    }
    if len(split_rows["train"]) + len(split_rows["validation"]) != len(rows):
        raise ValueError("early-risk corpus contains an unsupported split")
    return rows, split_rows


def train_early_risk_predictor(
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    contract_metadata: Mapping[str, str],
    epochs: int = 400,
    learning_rate: float = 0.25,
    l2_penalty: float = 1e-4,
) -> tuple[EarlyRiskPredictor, dict[str, Any]]:
    """Train deterministic class-balanced temporal softmax regression."""
    if (
        not isinstance(epochs, int)
        or isinstance(epochs, bool)
        or epochs < 10
        or epochs % 10
    ):
        raise ValueError("early-risk epochs must be a positive multiple of ten")
    if not _finite_number(learning_rate) or float(learning_rate) <= 0.0:
        raise ValueError("early-risk learning rate must be positive and finite")
    if not _finite_number(l2_penalty) or float(l2_penalty) < 0.0:
        raise ValueError("early-risk L2 penalty must be non-negative and finite")
    contract = _validate_contract_metadata(contract_metadata)
    train_x, train_y = _row_matrix(training_rows)
    validation_x, validation_y = _row_matrix(validation_rows)
    _require_all_classes(train_y, "training")
    _require_all_classes(validation_y, "validation")

    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    normalized = (train_x - means) / scales
    validation_normalized = (validation_x - means) / scales
    class_count = len(EARLY_RISK_CLASS_NAMES)
    weights = np.zeros((TEMPORAL_FEATURES, class_count), dtype=np.float64)
    biases = np.zeros(class_count, dtype=np.float64)
    counts = np.bincount(train_y, minlength=class_count)
    sample_weights = np.asarray(
        [1.0 / (class_count * counts[label]) for label in train_y],
        dtype=np.float64,
    )

    best_score = -1.0
    best_loss = math.inf
    best_epoch = 0
    best_weights = weights.copy()
    best_biases = biases.copy()
    for epoch in range(1, epochs + 1):
        probabilities = _softmax(normalized @ weights + biases)
        errors = probabilities.copy()
        errors[np.arange(len(train_y)), train_y] -= 1.0
        errors *= sample_weights[:, None]
        weights -= float(learning_rate) * (
            normalized.T @ errors + float(l2_penalty) * weights
        )
        biases -= float(learning_rate) * errors.sum(axis=0)
        if epoch % 10:
            continue
        validation_probabilities = _softmax(validation_normalized @ weights + biases)
        score = _macro_f1(validation_y, np.argmax(validation_probabilities, axis=1))
        loss = _cross_entropy(validation_y, validation_probabilities)
        if score > best_score or (score == best_score and loss < best_loss):
            best_score = score
            best_loss = loss
            best_epoch = epoch
            best_weights = weights.copy()
            best_biases = biases.copy()

    predictor = EarlyRiskPredictor(
        window_ticks=WINDOW_TICKS,
        feature_width=FEATURE_WIDTH,
        class_names=EARLY_RISK_CLASS_NAMES,
        contract_metadata=contract,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        weights=tuple(tuple(float(value) for value in row) for row in best_weights),
        biases=tuple(float(value) for value in best_biases),
    )
    _validate_predictor(predictor)
    receipt = {
        "format": "aeolus_early_risk_training_v1",
        "model_format": "aeolus_early_risk_softmax_v1",
        "window_ticks": WINDOW_TICKS,
        "forecast_horizon_ticks": FORECAST_HORIZON_TICKS,
        "temporal_features": TEMPORAL_FEATURES,
        "class_names": list(EARLY_RISK_CLASS_NAMES),
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "training_class_counts": {
            name: int(counts[index])
            for index, name in enumerate(EARLY_RISK_CLASS_NAMES)
        },
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "validation_cross_entropy": best_loss,
        "contract_metadata": dict(contract),
    }
    return predictor, receipt


def calibrate_early_risk_abstention(
    predictor: EarlyRiskPredictor,
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    max_reference_warning_fraction: float = 0.05,
    max_negative_fault_warning_fraction: float = 0.05,
    min_target_recall: float = 0.50,
) -> tuple[EarlyRiskPredictor, dict[str, Any]]:
    """Calibrate probability/margin thresholds on validation rows only.

    Eligibility is safety-first: thresholds must remain within both negative
    warning budgets. Among eligible settings, the minimum target recall is
    maximized before macro recall, which prevents the larger cabin-B class
    from hiding failure on cabin A.
    """
    _validate_predictor(predictor)
    for name, value in (
        ("reference", max_reference_warning_fraction),
        ("negative fault", max_negative_fault_warning_fraction),
        ("target recall", min_target_recall),
    ):
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"early-risk {name} warning budget must be in [0, 1]")
    matrix, labels = _row_matrix(validation_rows)
    means = np.asarray(predictor.means, dtype=np.float64)
    scales = np.asarray(predictor.scales, dtype=np.float64)
    weights = np.asarray(predictor.weights, dtype=np.float64)
    biases = np.asarray(predictor.biases, dtype=np.float64)
    probabilities = _softmax(((matrix - means) / scales) @ weights + biases)
    raw_predictions = np.argmax(probabilities, axis=1)
    ordered = np.sort(probabilities, axis=1)
    maximum = ordered[:, -1]
    margins = ordered[:, -1] - ordered[:, -2]
    roles = np.asarray([row.get("scenario_role") for row in validation_rows])
    if any(role not in {"reference", "fault"} for role in roles):
        raise ValueError("early-risk validation row role is malformed")

    probability_grid = tuple(float(value) for value in np.linspace(0.30, 0.99, 29))
    margin_grid = tuple(float(value) for value in np.linspace(0.0, 0.40, 17))
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for min_probability in probability_grid:
        for min_margin in margin_grid:
            warnings = (
                (raw_predictions != _CLASS_INDEX[NO_EARLY_RISK])
                & (maximum >= min_probability)
                & (margins >= min_margin)
            )
            reference_mask = (labels == 0) & (roles == "reference")
            negative_fault_mask = (labels == 0) & (roles == "fault")
            reference_fraction = _mask_fraction(warnings, reference_mask)
            negative_fault_fraction = _mask_fraction(warnings, negative_fault_mask)
            recalls: dict[str, float] = {}
            wrong_target = 0
            for index, class_name in enumerate(EARLY_RISK_CLASS_NAMES[1:], start=1):
                class_mask = labels == index
                recalls[class_name] = _mask_fraction(
                    warnings & (raw_predictions == index), class_mask
                )
                wrong_target += int(
                    np.sum(warnings & class_mask & (raw_predictions != index))
                )
            recall_values = tuple(recalls.values())
            minimum_recall = min(recall_values)
            macro_recall = sum(recall_values) / len(recall_values)
            eligible = (
                reference_fraction <= float(max_reference_warning_fraction)
                and negative_fault_fraction
                <= float(max_negative_fault_warning_fraction)
                and minimum_recall >= float(min_target_recall)
            )
            metrics: dict[str, Any] = {
                "min_probability": min_probability,
                "min_margin": min_margin,
                "eligible": eligible,
                "reference_warnings": int(np.sum(warnings & reference_mask)),
                "reference_rows": int(np.sum(reference_mask)),
                "reference_warning_fraction": reference_fraction,
                "negative_fault_warnings": int(np.sum(warnings & negative_fault_mask)),
                "negative_fault_rows": int(np.sum(negative_fault_mask)),
                "negative_fault_warning_fraction": negative_fault_fraction,
                "wrong_target_warnings": wrong_target,
                "per_target_recall": recalls,
                "minimum_target_recall": minimum_recall,
                "macro_target_recall": macro_recall,
                "warnings": int(np.sum(warnings)),
            }
            score = (
                1.0 if eligible else 0.0,
                minimum_recall if eligible else -reference_fraction,
                macro_recall if eligible else -negative_fault_fraction,
                -float(wrong_target),
                -negative_fault_fraction,
                -reference_fraction,
                min_probability,
                min_margin,
            )
            candidates.append((score, metrics))
    _, selected = max(candidates, key=lambda item: item[0])
    calibrated = replace(
        predictor,
        min_probability=float(selected["min_probability"]),
        min_margin=float(selected["min_margin"]),
    )
    _validate_predictor(calibrated)
    receipt: dict[str, Any] = {
        "format": "aeolus_early_risk_calibration_v1",
        "validation_rows": len(validation_rows),
        "max_reference_warning_fraction": float(max_reference_warning_fraction),
        "max_negative_fault_warning_fraction": float(
            max_negative_fault_warning_fraction
        ),
        "required_minimum_target_recall": float(min_target_recall),
        **selected,
    }
    return calibrated, receipt


def save_early_risk_artifact(
    path: str | Path,
    predictor: EarlyRiskPredictor,
    *,
    training_receipt: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Save one canonical, self-hashed learned artifact."""
    _validate_predictor(predictor)
    payload: dict[str, Any] = {
        "schema_version": "aeolus_early_risk_artifact_v1",
        "model_format": "aeolus_early_risk_softmax_v1",
        "window_ticks": predictor.window_ticks,
        "feature_width": predictor.feature_width,
        "class_names": list(predictor.class_names),
        "contract_metadata": dict(predictor.contract_metadata),
        "means": list(predictor.means),
        "scales": list(predictor.scales),
        "weights": [list(row) for row in predictor.weights],
        "biases": list(predictor.biases),
        "min_probability": predictor.min_probability,
        "min_margin": predictor.min_margin,
        "training_receipt": dict(training_receipt),
        "calibration_receipt": dict(calibration_receipt),
    }
    document = dict(payload)
    document["artifact_sha256"] = _canonical_json_sha256(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return document


def load_early_risk_artifact(
    path: str | Path,
) -> tuple[EarlyRiskPredictor, dict[str, Any], dict[str, Any]]:
    """Load and verify a self-hashed learned artifact."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("early-risk artifact must be a JSON object")
    expected_hash = document.get("artifact_sha256")
    payload = {
        key: value for key, value in document.items() if key != "artifact_sha256"
    }
    if expected_hash != _canonical_json_sha256(payload):
        raise ValueError("early-risk artifact hash mismatch")
    if payload.get("schema_version") != "aeolus_early_risk_artifact_v1":
        raise ValueError("early-risk artifact schema is unsupported")
    if payload.get("model_format") != "aeolus_early_risk_softmax_v1":
        raise ValueError("early-risk model format is unsupported")
    predictor = EarlyRiskPredictor(
        window_ticks=payload["window_ticks"],
        feature_width=payload["feature_width"],
        class_names=tuple(payload["class_names"]),
        contract_metadata=dict(payload["contract_metadata"]),
        means=tuple(payload["means"]),
        scales=tuple(payload["scales"]),
        weights=tuple(tuple(row) for row in payload["weights"]),
        biases=tuple(payload["biases"]),
        min_probability=payload["min_probability"],
        min_margin=payload["min_margin"],
        artifact_sha256=str(expected_hash),
    )
    predictor = replace(
        predictor,
        artifact_model_sha256=_canonical_json_sha256(
            _predictor_model_document(predictor)
        ),
    )
    _validate_predictor(predictor)
    training_receipt = payload.get("training_receipt")
    calibration_receipt = payload.get("calibration_receipt")
    if not isinstance(training_receipt, dict) or not isinstance(
        calibration_receipt, dict
    ):
        raise ValueError("early-risk artifact receipts are malformed")
    return predictor, training_receipt, calibration_receipt


def _mask_fraction(selected: NDArray[np.bool_], population: NDArray[np.bool_]) -> float:
    population_count = int(np.sum(population))
    if population_count == 0:
        return 0.0
    return float(np.sum(selected & population) / population_count)


def _validate_physical_trace(
    trace: Mapping[str, Sequence[float]], zone_ids: Sequence[str]
) -> tuple[tuple[str, ...], int]:
    if not isinstance(trace, Mapping):
        raise ValueError("early-risk physical trace must be a mapping")
    zones = tuple(zone_ids)
    if not zones or len(set(zones)) != len(zones) or set(trace) != set(zones):
        raise ValueError("early-risk physical trace zone topology is invalid")
    lengths = {len(trace[zone]) for zone in zones}
    if len(lengths) != 1:
        raise ValueError("early-risk physical traces must have equal lengths")
    trace_length = lengths.pop()
    if trace_length < WINDOW_TICKS:
        raise ValueError("early-risk physical trace is shorter than one window")
    if any(not _finite_number(value) for zone in zones for value in trace[zone]):
        raise ValueError("early-risk physical trace must be finite")
    return zones, trace_length


def _row_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if not rows:
        raise ValueError("early-risk training rows must not be empty")
    windows: list[Sequence[Sequence[float]]] = []
    labels: list[int] = []
    for row in rows:
        label = row.get("label")
        if label not in _CLASS_INDEX:
            raise ValueError("early-risk row has an unsupported label")
        features = row.get("features")
        if not isinstance(features, Sequence):
            raise ValueError("early-risk row features are malformed")
        windows.append(features)
        labels.append(_CLASS_INDEX[str(label)])
    return temporal_summary_v1(windows), np.asarray(labels, dtype=np.int64)


def _validate_contract_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    keys = {"model_input_version", "selector_sha256", "topology_sha256"}
    if not isinstance(metadata, Mapping) or set(metadata) != keys:
        raise ValueError("early-risk contract metadata fields are malformed")
    result = {key: metadata[key] for key in sorted(keys)}
    if result["model_input_version"] != "model_input_v1":
        raise ValueError("early-risk model input version is unsupported")
    for key in ("selector_sha256", "topology_sha256"):
        value = result[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("early-risk contract hash is malformed")
    return result


def _validate_predictor(predictor: EarlyRiskPredictor) -> None:
    if (
        predictor.window_ticks != WINDOW_TICKS
        or predictor.feature_width != FEATURE_WIDTH
    ):
        raise ValueError("early-risk predictor input shape is invalid")
    if predictor.class_names != EARLY_RISK_CLASS_NAMES:
        raise ValueError("early-risk predictor class vocabulary is invalid")
    _validate_contract_metadata(predictor.contract_metadata)
    if (
        len(predictor.means) != TEMPORAL_FEATURES
        or len(predictor.scales) != TEMPORAL_FEATURES
    ):
        raise ValueError("early-risk predictor normalization shape is invalid")
    if len(predictor.weights) != TEMPORAL_FEATURES or any(
        len(row) != len(EARLY_RISK_CLASS_NAMES) for row in predictor.weights
    ):
        raise ValueError("early-risk predictor weight shape is invalid")
    if len(predictor.biases) != len(EARLY_RISK_CLASS_NAMES):
        raise ValueError("early-risk predictor bias shape is invalid")
    values = [
        *predictor.means,
        *predictor.scales,
        *predictor.biases,
        *(value for row in predictor.weights for value in row),
    ]
    if any(not _finite_number(value) for value in values):
        raise ValueError("early-risk predictor contains non-finite values")
    if any(float(value) <= 0.0 for value in predictor.scales):
        raise ValueError("early-risk predictor scales must be positive")
    for name, value in (
        ("probability", predictor.min_probability),
        ("margin", predictor.min_margin),
    ):
        if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"early-risk minimum {name} must be in [0, 1]")
    for name, digest in (
        ("artifact identity", predictor.artifact_sha256),
        ("model payload identity", predictor.artifact_model_sha256),
    ):
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"early-risk predictor {name} is malformed")
    if (predictor.artifact_sha256 is None) != (
        predictor.artifact_model_sha256 is None
    ):
        raise ValueError("early-risk predictor artifact binding is incomplete")
    if predictor.artifact_model_sha256 is not None and (
        predictor.artifact_model_sha256
        != _canonical_json_sha256(_predictor_model_document(predictor))
    ):
        raise ValueError("early-risk predictor model payload does not match artifact")


def _predictor_model_document(predictor: EarlyRiskPredictor) -> dict[str, Any]:
    return {
        "window_ticks": predictor.window_ticks,
        "feature_width": predictor.feature_width,
        "class_names": list(predictor.class_names),
        "contract_metadata": dict(predictor.contract_metadata),
        "means": list(predictor.means),
        "scales": list(predictor.scales),
        "weights": [list(row) for row in predictor.weights],
        "biases": list(predictor.biases),
        "min_probability": predictor.min_probability,
        "min_margin": predictor.min_margin,
    }


def _require_all_classes(labels: NDArray[np.int64], split: str) -> None:
    present = set(int(value) for value in labels)
    expected = set(range(len(EARLY_RISK_CLASS_NAMES)))
    if present != expected:
        missing = [
            EARLY_RISK_CLASS_NAMES[index] for index in sorted(expected - present)
        ]
        raise ValueError(f"early-risk {split} rows are missing classes: {missing}")


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _cross_entropy(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean())


def _macro_f1(labels: NDArray[np.int64], predictions: NDArray[np.int64]) -> float:
    scores: list[float] = []
    for index in range(len(EARLY_RISK_CLASS_NAMES)):
        true_positive = int(np.sum((labels == index) & (predictions == index)))
        false_positive = int(np.sum((labels != index) & (predictions == index)))
        false_negative = int(np.sum((labels == index) & (predictions != index)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(sum(scores) / len(scores))


def _canonical_json_sha256(document: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )
