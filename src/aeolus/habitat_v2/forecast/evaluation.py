"""Model-neutral, fail-closed evaluation for Forecast D1 candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Protocol

import numpy as np

INPUT_MANIFEST_SHA256 = (
    "379c8607c929b716f0bffb7343fefdab384bdfb35a8a9ccfcdd55c8dc60f377f"
)
TARGET_MANIFEST_SHA256 = (
    "93f064cabd78758c9b0dd665510acfa101f03da6f717764d506bc3624eec283e"
)
TARGET_COUNT = 51
EVIDENCE_SCHEMA_VERSION = "aeolus_habitat_v2_forecast_evaluation_evidence_v1"
_SPLIT_LABELS = frozenset(("DEVELOPMENT", "TRAIN", "VALIDATION", "FINAL"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvaluationError(ValueError):
    """Caller-supplied evaluation evidence cannot be safely evaluated."""


@dataclass(frozen=True, slots=True)
class CorpusEvidenceEntry:
    """Commitments copied from one validated sample and its family split record."""

    sample_id: str
    sample_record_sha256: str
    family_cluster_id: str
    split_assignment_id: str
    split_assignment_record_sha256: str
    split_label: str
    target_manifest_sha256: str
    target_truth_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationEvidenceManifest:
    schema_version: str
    source_split_table_sha256: str
    entries: tuple[CorpusEvidenceEntry, ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    sample_id: str
    family_cluster_id: str
    split_label: str
    sample_record_sha256: str
    split_assignment_id: str
    split_assignment_record_sha256: str
    input_manifest_sha256: str
    target_manifest_sha256: str
    targets_f32: np.ndarray
    evidence_manifest_sha256: str = ""


@dataclass(frozen=True, slots=True)
class TrainingReference:
    """One exact TRAIN-only target row used to derive normalization scales."""

    sample_id: str
    family_cluster_id: str
    split_label: str
    sample_record_sha256: str
    split_assignment_id: str
    split_assignment_record_sha256: str
    target_manifest_sha256: str
    targets_f32: np.ndarray
    evidence_manifest_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CandidateQuery:
    """Target-free identity passed across the candidate adapter boundary."""

    sample_id: str
    input_manifest_sha256: str
    target_manifest_sha256: str
    evidence_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateOutput:
    sample_id: str
    status: str
    input_manifest_sha256: str
    target_manifest_sha256: str
    prediction_f32: np.ndarray | None


class CandidateAdapter(Protocol):
    def __call__(self, query: CandidateQuery) -> CandidateOutput: ...


@dataclass(frozen=True, slots=True)
class HarmEnvelope:
    """Caller-supplied strict bounds; lower/upper crossings are harmful-positive."""

    lower_f32: np.ndarray
    upper_f32: np.ndarray
    anchor_f32: np.ndarray


@dataclass(frozen=True, slots=True)
class RatioMetric:
    value: float | None
    supported: bool


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    existing_harm_count: int


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Error polarity is lower-is-better; harm classification polarity is positive crossing."""

    polarity: str
    mae_by_horizon_target: np.ndarray
    rmse_by_horizon_target: np.ndarray
    p95_p5_scale: np.ndarray
    scale_supported: np.ndarray
    cluster_macro_normalized_mae: RatioMetric
    precision: RatioMetric
    recall: RatioMetric
    false_positive_rate: RatioMetric
    false_negative_rate: RatioMetric
    coverage: RatioMetric
    selective_error: RatioMetric
    invalid_output_count: int
    confusion: ConfusionCounts


@dataclass(frozen=True, slots=True)
class EvaluatedOutput:
    sample_id: str
    status: str
    reason: str | None
    prediction_f32: np.ndarray | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    outputs: tuple[EvaluatedOutput, ...]
    metrics: EvaluationMetrics


@dataclass(frozen=True, slots=True)
class ActionInformationComparison:
    """Positive delta means action-aware ridge has lower normalized error."""

    supported: bool
    polarity: str
    action_aware_cluster_macro_normalized_mae: float | None
    action_blinded_cluster_macro_normalized_mae: float | None
    blinded_minus_aware: float | None
    reason: str | None


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _ratio(numerator: float, denominator: float) -> RatioMetric:
    return (
        RatioMetric(None, False)
        if denominator == 0
        else RatioMetric(float(numerator / denominator), True)
    )


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise EvaluationError(f"{label} must be finite float32{shape}")
    return array


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvaluationError("corpus evidence is not canonical finite JSON") from error


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise EvaluationError(f"{label} must be one lowercase SHA-256")
    return value


def target_truth_sha256(value: object) -> str:
    """Hash an exact finite float32 target tensor with shape and endian binding."""
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[0] not in (2, 4, 8)
        or array.shape[1:] != (TARGET_COUNT,)
    ):
        raise EvaluationError("target truth shape is unsupported")
    checked = _array(array, tuple(array.shape), "target truth")
    shape = _canonical({"dtype": "float32-le", "shape": list(checked.shape)})
    little_endian = checked.astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(
        b"aeolus-forecast-target-truth-v1\0" + shape + little_endian
    ).hexdigest()


def _entry_mapping(entry: CorpusEvidenceEntry) -> dict[str, str]:
    return {
        "sample_id": entry.sample_id,
        "sample_record_sha256": entry.sample_record_sha256,
        "family_cluster_id": entry.family_cluster_id,
        "split_assignment_id": entry.split_assignment_id,
        "split_assignment_record_sha256": entry.split_assignment_record_sha256,
        "split_label": entry.split_label,
        "target_manifest_sha256": entry.target_manifest_sha256,
        "target_truth_sha256": entry.target_truth_sha256,
    }


def _validate_evidence_entries(
    entries: tuple[CorpusEvidenceEntry, ...],
) -> dict[str, CorpusEvidenceEntry]:
    if not entries or any(type(entry) is not CorpusEvidenceEntry for entry in entries):
        raise EvaluationError("corpus evidence entries are malformed")
    by_sample: dict[str, CorpusEvidenceEntry] = {}
    by_assignment: dict[str, tuple[str, str, str]] = {}
    by_family: dict[str, tuple[str, str, str]] = {}
    sample_record_hashes: set[str] = set()
    for entry in entries:
        for label, value in (
            ("sample ID", entry.sample_id),
            ("family cluster ID", entry.family_cluster_id),
            ("split assignment ID", entry.split_assignment_id),
        ):
            if type(value) is not str or not value:
                raise EvaluationError(f"corpus evidence {label} is malformed")
        for label, value in (
            ("sample record", entry.sample_record_sha256),
            ("split assignment record", entry.split_assignment_record_sha256),
            ("target truth", entry.target_truth_sha256),
        ):
            _require_sha256(value, f"corpus evidence {label}")
        if entry.target_manifest_sha256 != TARGET_MANIFEST_SHA256:
            raise EvaluationError("corpus evidence target manifest drifts")
        if entry.split_label not in _SPLIT_LABELS:
            raise EvaluationError("corpus evidence split label is unsupported")
        if (
            entry.sample_id in by_sample
            or entry.sample_record_sha256 in sample_record_hashes
        ):
            raise EvaluationError("corpus evidence sample identity is duplicated")
        by_sample[entry.sample_id] = entry
        sample_record_hashes.add(entry.sample_record_sha256)
        assignment = (
            entry.family_cluster_id,
            entry.split_assignment_record_sha256,
            entry.split_label,
        )
        if (
            entry.split_assignment_id in by_assignment
            and by_assignment[entry.split_assignment_id] != assignment
        ):
            raise EvaluationError("corpus evidence split assignment is inconsistent")
        by_assignment[entry.split_assignment_id] = assignment
        family = (
            entry.split_assignment_id,
            entry.split_assignment_record_sha256,
            entry.split_label,
        )
        if (
            entry.family_cluster_id in by_family
            and by_family[entry.family_cluster_id] != family
        ):
            raise EvaluationError(
                "corpus evidence family has multiple split assignments"
            )
        by_family[entry.family_cluster_id] = family
    return by_sample


def build_evaluation_evidence_manifest(
    source_split_table_sha256: str,
    entries: Sequence[CorpusEvidenceEntry],
) -> EvaluationEvidenceManifest:
    """Freeze commitments copied from contract-validated corpus/split records."""
    _require_sha256(source_split_table_sha256, "source split table")
    if isinstance(entries, (str, bytes)):
        raise EvaluationError("corpus evidence entries are malformed")
    ordered = tuple(sorted(tuple(entries), key=lambda entry: entry.sample_id))
    _validate_evidence_entries(ordered)
    body = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_split_table_sha256": source_split_table_sha256,
        "entries": [_entry_mapping(entry) for entry in ordered],
    }
    return EvaluationEvidenceManifest(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        source_split_table_sha256=source_split_table_sha256,
        entries=ordered,
        manifest_sha256=hashlib.sha256(_canonical(body)).hexdigest(),
    )


def _validate_evidence_manifest(
    manifest: EvaluationEvidenceManifest,
    expected_manifest_sha256: str,
) -> dict[str, CorpusEvidenceEntry]:
    _require_sha256(expected_manifest_sha256, "expected corpus evidence manifest")
    if type(manifest) is not EvaluationEvidenceManifest:
        raise EvaluationError("corpus evidence manifest is malformed")
    if manifest.schema_version != EVIDENCE_SCHEMA_VERSION:
        raise EvaluationError("corpus evidence manifest schema drifts")
    _require_sha256(manifest.source_split_table_sha256, "source split table")
    _require_sha256(manifest.manifest_sha256, "corpus evidence manifest")
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise EvaluationError(
            "corpus evidence manifest is not the expected frozen manifest"
        )
    if (
        type(manifest.entries) is not tuple
        or tuple(sorted(manifest.entries, key=lambda entry: entry.sample_id))
        != manifest.entries
    ):
        raise EvaluationError("corpus evidence entries are not canonically ordered")
    by_sample = _validate_evidence_entries(manifest.entries)
    body = {
        "schema_version": manifest.schema_version,
        "source_split_table_sha256": manifest.source_split_table_sha256,
        "entries": [_entry_mapping(entry) for entry in manifest.entries],
    }
    if hashlib.sha256(_canonical(body)).hexdigest() != manifest.manifest_sha256:
        raise EvaluationError("corpus evidence manifest self-hash mismatch")
    return by_sample


def _validate_request(
    request: EvaluationRequest,
    evidence_by_sample: dict[str, CorpusEvidenceEntry],
    evidence_manifest_sha256: str,
) -> None:
    if (
        type(request) is not EvaluationRequest
        or not request.sample_id
        or not request.family_cluster_id
    ):
        raise EvaluationError("expected sample identity is malformed")
    if request.split_label not in {"VALIDATION", "FINAL"}:
        raise EvaluationError("evaluation split must be VALIDATION or FINAL")
    target = np.asarray(request.targets_f32)
    if (
        target.ndim != 2
        or target.shape[1:] != (TARGET_COUNT,)
        or target.shape[0] not in (2, 4, 8)
    ):
        raise EvaluationError("expected target shape is unsupported")
    _array(target, target.shape, "expected target")
    if (request.input_manifest_sha256, request.target_manifest_sha256) != (
        INPUT_MANIFEST_SHA256,
        TARGET_MANIFEST_SHA256,
    ):
        raise EvaluationError("expected manifest identity drift")
    evidence = evidence_by_sample.get(request.sample_id)
    if evidence is None or (
        request.sample_record_sha256,
        request.family_cluster_id,
        request.split_assignment_id,
        request.split_assignment_record_sha256,
        request.split_label,
        request.target_manifest_sha256,
        target_truth_sha256(request.targets_f32),
        request.evidence_manifest_sha256,
    ) != (
        evidence.sample_record_sha256,
        evidence.family_cluster_id,
        evidence.split_assignment_id,
        evidence.split_assignment_record_sha256,
        evidence.split_label,
        evidence.target_manifest_sha256,
        evidence.target_truth_sha256,
        evidence_manifest_sha256,
    ):
        raise EvaluationError(
            "evaluation request does not match frozen corpus evidence"
        )


def _training_targets(
    references: Sequence[TrainingReference],
    *,
    shape: tuple[int, int],
    evaluation_clusters: set[str],
    evaluation_sample_ids: set[str],
    evidence_by_sample: dict[str, CorpusEvidenceEntry],
    evidence_manifest_sha256: str,
) -> np.ndarray:
    if isinstance(references, (str, bytes)) or not references:
        raise EvaluationError("normalization requires TRAIN reference rows")
    items = tuple(references)
    for item in items:
        if (
            type(item) is not TrainingReference
            or not item.sample_id
            or not item.family_cluster_id
        ):
            raise EvaluationError("TRAIN reference identity is malformed")
        if item.split_label != "TRAIN":
            raise EvaluationError("normalization accepts TRAIN references only")
        if item.target_manifest_sha256 != TARGET_MANIFEST_SHA256:
            raise EvaluationError("TRAIN reference target manifest drifts")
        _array(item.targets_f32, shape, "TRAIN reference target")
        evidence = evidence_by_sample.get(item.sample_id)
        if evidence is None or (
            item.sample_record_sha256,
            item.family_cluster_id,
            item.split_assignment_id,
            item.split_assignment_record_sha256,
            item.split_label,
            item.target_manifest_sha256,
            target_truth_sha256(item.targets_f32),
            item.evidence_manifest_sha256,
        ) != (
            evidence.sample_record_sha256,
            evidence.family_cluster_id,
            evidence.split_assignment_id,
            evidence.split_assignment_record_sha256,
            evidence.split_label,
            evidence.target_manifest_sha256,
            evidence.target_truth_sha256,
            evidence_manifest_sha256,
        ):
            raise EvaluationError(
                "TRAIN reference does not match frozen corpus evidence"
            )
    training_sample_ids = {item.sample_id for item in items}
    if len(training_sample_ids) != len(items):
        raise EvaluationError("TRAIN reference sample IDs must be unique")
    if training_sample_ids & evaluation_sample_ids:
        raise EvaluationError("TRAIN and evaluation sample identities overlap")
    training_clusters = {item.family_cluster_id for item in items}
    if len(training_clusters) < 2:
        raise EvaluationError("normalization requires at least two TRAIN clusters")
    if training_clusters & evaluation_clusters:
        raise EvaluationError("TRAIN and evaluation family clusters overlap")
    return np.stack([item.targets_f32 for item in items])


def _validate_envelope(
    envelope: HarmEnvelope, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(envelope) is not HarmEnvelope:
        raise EvaluationError("harm envelope must be supplied explicitly")
    lower, upper = (
        _array(envelope.lower_f32, shape, "lower envelope"),
        _array(envelope.upper_f32, shape, "upper envelope"),
    )
    anchor = _array(envelope.anchor_f32, (TARGET_COUNT,), "anchor")
    if (lower >= upper).any():
        raise EvaluationError("harm envelope must have strict lower/upper bounds")
    return lower, upper, anchor


def _harmful(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (values <= lower) | (values >= upper)


def evaluate(
    requests: Sequence[EvaluationRequest],
    adapter: CandidateAdapter,
    *,
    envelope: HarmEnvelope,
    training_references: Sequence[TrainingReference],
    evidence_manifest: EvaluationEvidenceManifest,
    expected_evidence_manifest_sha256: str,
    domain_validator: Callable[[np.ndarray], bool] | None = None,
) -> EvaluationResult:
    """Evaluate an adapter through one model-neutral callable boundary.

    Output identity and declared status are checked before its prediction field
    is read.  Any adapter failure is an INVALID_OUTPUT rather than abstention.
    """
    if isinstance(requests, (str, bytes)) or not callable(adapter) or not requests:
        raise EvaluationError(
            "evaluation requires non-empty requests and one callable adapter"
        )
    items = tuple(requests)
    evidence_by_sample = _validate_evidence_manifest(
        evidence_manifest,
        expected_evidence_manifest_sha256,
    )
    for request in items:
        _validate_request(
            request,
            evidence_by_sample,
            expected_evidence_manifest_sha256,
        )
    if len({request.sample_id for request in items}) != len(items):
        raise EvaluationError("expected sample IDs must be unique")
    shape = tuple(items[0].targets_f32.shape)
    if any(tuple(request.targets_f32.shape) != shape for request in items):
        raise EvaluationError("evaluation cannot mix target horizons")
    truth_by_sample = {
        request.sample_id: _readonly(np.array(request.targets_f32, copy=True))
        for request in items
    }
    lower, upper, anchor = _validate_envelope(envelope, shape)
    training = _training_targets(
        training_references,
        shape=shape,
        evaluation_clusters={request.family_cluster_id for request in items},
        evaluation_sample_ids={request.sample_id for request in items},
        evidence_by_sample=evidence_by_sample,
        evidence_manifest_sha256=expected_evidence_manifest_sha256,
    )
    scale = np.percentile(training, 95, axis=0) - np.percentile(training, 5, axis=0)
    supported_scale = scale > 0.0
    outputs: list[EvaluatedOutput] = []
    for request in items:
        query = CandidateQuery(
            sample_id=request.sample_id,
            input_manifest_sha256=request.input_manifest_sha256,
            target_manifest_sha256=request.target_manifest_sha256,
            evidence_manifest_sha256=request.evidence_manifest_sha256,
        )
        try:
            candidate = adapter(query)
        except TimeoutError:
            outputs.append(
                EvaluatedOutput(
                    request.sample_id, "INVALID_OUTPUT", "adapter_timeout", None
                )
            )
            continue
        except Exception:
            outputs.append(
                EvaluatedOutput(
                    request.sample_id, "INVALID_OUTPUT", "adapter_exception", None
                )
            )
            continue
        # Identity and status checks precede *any* prediction access.
        if type(candidate) is not CandidateOutput or (
            candidate.sample_id,
            candidate.input_manifest_sha256,
            candidate.target_manifest_sha256,
        ) != (
            request.sample_id,
            request.input_manifest_sha256,
            request.target_manifest_sha256,
        ):
            outputs.append(
                EvaluatedOutput(
                    request.sample_id, "INVALID_OUTPUT", "identity_or_manifest", None
                )
            )
            continue
        if candidate.status not in {"PREDICTION", "ABSTAIN"}:
            outputs.append(
                EvaluatedOutput(
                    request.sample_id, "INVALID_OUTPUT", "unsupported_status", None
                )
            )
            continue
        if candidate.status == "ABSTAIN":
            if candidate.prediction_f32 is not None:
                outputs.append(
                    EvaluatedOutput(
                        request.sample_id,
                        "INVALID_OUTPUT",
                        "abstain_has_prediction",
                        None,
                    )
                )
            else:
                outputs.append(
                    EvaluatedOutput(request.sample_id, "ABSTAIN", None, None)
                )
            continue
        try:
            prediction = _array(candidate.prediction_f32, shape, "prediction")
            if domain_validator is not None and not bool(domain_validator(prediction)):
                raise EvaluationError("domain invalid")
        except Exception:
            outputs.append(
                EvaluatedOutput(
                    request.sample_id, "INVALID_OUTPUT", "prediction_invalid", None
                )
            )
            continue
        outputs.append(
            EvaluatedOutput(
                request.sample_id,
                "PREDICTION",
                None,
                _readonly(np.array(prediction, copy=True)),
            )
        )
    predicted = [
        (request, output)
        for request, output in zip(items, outputs, strict=True)
        if output.status == "PREDICTION"
    ]
    if predicted:
        errors = np.stack(
            [
                output.prediction_f32 - truth_by_sample[request.sample_id]
                for request, output in predicted
            ]
        )
        mae, rmse = np.mean(np.abs(errors), axis=0), np.sqrt(np.mean(errors**2, axis=0))
        normalized_by_cluster: list[float] = []
        for cluster in sorted({request.family_cluster_id for request, _ in predicted}):
            cluster_errors = np.stack(
                [
                    np.abs(output.prediction_f32 - truth_by_sample[request.sample_id])
                    for request, output in predicted
                    if request.family_cluster_id == cluster
                ]
            )
            if supported_scale.any():
                normalized_by_cluster.append(
                    float(
                        np.mean(
                            cluster_errors[:, supported_scale] / scale[supported_scale]
                        )
                    )
                )
        normalized = (
            RatioMetric(float(np.mean(normalized_by_cluster)), True)
            if normalized_by_cluster
            else RatioMetric(None, False)
        )
        selective = RatioMetric(float(np.mean(np.abs(errors))), True)
    else:
        mae, rmse = np.full(shape, np.nan), np.full(shape, np.nan)
        normalized, selective = RatioMetric(None, False), RatioMetric(None, False)
    tp = fp = tn = fn = existing = 0
    anchor_harm = _harmful(anchor, lower[0], upper[0])
    for request, output in zip(items, outputs, strict=True):
        if output.status not in {"PREDICTION", "ABSTAIN"}:
            continue
        truth_harm = _harmful(truth_by_sample[request.sample_id], lower, upper)
        pred_harm = (
            _harmful(output.prediction_f32, lower, upper)
            if output.status == "PREDICTION"
            else np.zeros(shape, dtype=bool)
        )
        valid = np.broadcast_to(~anchor_harm, shape)
        existing += int(np.broadcast_to(anchor_harm, shape).sum())
        tp += int(np.logical_and(pred_harm & truth_harm, valid).sum())
        fp += int(np.logical_and(pred_harm & ~truth_harm, valid).sum())
        tn += int(np.logical_and(~pred_harm & ~truth_harm, valid).sum())
        fn += int(np.logical_and(~pred_harm & truth_harm, valid).sum())
    invalid = sum(output.status == "INVALID_OUTPUT" for output in outputs)
    metrics = EvaluationMetrics(
        "harmful_crossing_positive",
        _readonly(mae.astype(np.float64)),
        _readonly(rmse.astype(np.float64)),
        _readonly(scale.astype(np.float64)),
        _readonly(supported_scale.astype(bool)),
        normalized,
        _ratio(tp, tp + fp),
        _ratio(tp, tp + fn),
        _ratio(fp, fp + tn),
        _ratio(fn, fn + tp),
        _ratio(len(predicted), len(items)),
        selective,
        invalid,
        ConfusionCounts(tp, fp, tn, fn, existing),
    )
    return EvaluationResult(tuple(outputs), metrics)


def compare_action_aware_and_blinded(
    action_aware: EvaluationResult,
    action_blinded: EvaluationResult,
) -> ActionInformationComparison:
    """Compare paired ridge results without inventing an admission threshold."""

    if (
        type(action_aware) is not EvaluationResult
        or type(action_blinded) is not EvaluationResult
    ):
        raise EvaluationError(
            "action-information comparison requires evaluation results"
        )
    aware_ids = tuple(output.sample_id for output in action_aware.outputs)
    blind_ids = tuple(output.sample_id for output in action_blinded.outputs)
    aware_status = tuple(output.status for output in action_aware.outputs)
    blind_status = tuple(output.status for output in action_blinded.outputs)
    if aware_ids != blind_ids or aware_status != blind_status:
        raise EvaluationError("action-aware and blinded evaluations are not paired")
    polarity = "positive_blinded_minus_aware_means_action_information"
    aware_metric = action_aware.metrics.cluster_macro_normalized_mae
    blind_metric = action_blinded.metrics.cluster_macro_normalized_mae
    if (
        action_aware.metrics.invalid_output_count
        or action_blinded.metrics.invalid_output_count
        or not aware_metric.supported
        or not blind_metric.supported
        or aware_metric.value is None
        or blind_metric.value is None
    ):
        return ActionInformationComparison(
            False,
            polarity,
            aware_metric.value,
            blind_metric.value,
            None,
            "paired normalized comparison is unsupported",
        )
    return ActionInformationComparison(
        True,
        polarity,
        aware_metric.value,
        blind_metric.value,
        float(blind_metric.value - aware_metric.value),
        None,
    )
