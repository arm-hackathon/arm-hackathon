"""Knowledge distillation study for Issue #54: how small is safe?

This module trains progressively smaller student models that copy the frozen
MLP and ridge teachers' predictions and evaluates three behavioural
properties: prediction accuracy (NMAE), action-ranking agreement, and
safety-margin closeness.

The module is deliberately outside the HMC authority core. Students are
``DEVELOPMENT_EVIDENCE_ONLY`` with ``actuator_authority=False``. HMC remains
the sole actuator authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .forecast.baselines import (
    ACTION_COUNT,
    TARGET_COUNT,
)
from .forecast.contracts import canonical_json_bytes
from .forecast_issue52 import TargetManifest


ISSUE54_SCHEMA_VERSION = "aeolus_habitat_v2_forecast_issue_54_v1"
HORIZON_STEPS = 8
OUTPUT_DIM = HORIZON_STEPS * TARGET_COUNT  # 408
MLP_WINDOW_STEPS = 16
MLP_FEATURE_COUNT = MLP_WINDOW_STEPS * 194 + ACTION_COUNT + 1  # 3132
RIDGE_WINDOW_STEPS = 4
RIDGE_FEATURE_COUNT = (
    RIDGE_WINDOW_STEPS * (194 + 167 * 5 + 4 + 4 + 287 * 4) + ACTION_COUNT
)  # 8767
_TEACHER_LABELS = ("mlp", "ridge")
_STUDENT_LABELS = ("sanity-2.1m", "medium-500k", "small-100k", "tiny-25k", "linear")
_STUDENT_HIDDEN = {
    "mlp": {
        "sanity-2.1m": (512, 512, 256),
        "medium-500k": (140,),
        "small-100k": (28,),
        "tiny-25k": (7,),
    },
    "ridge": {
        "sanity-2.1m": (200,),
        "medium-500k": (50,),
        "small-100k": (10,),
        "tiny-25k": (2,),
    },
}


class Issue54DistillationError(ValueError):
    """Distillation evidence is outside the frozen Issue #54 contract."""


@dataclass(frozen=True, slots=True)
class DistillationSample:
    """One (input, teacher_prediction, ground_truth) triple with provenance."""

    family_id: str
    split: str
    candidate_id: str
    teacher_label: str
    input_f32: np.ndarray
    teacher_prediction_f32: np.ndarray
    ground_truth_f32: np.ndarray
    sample_sha256: str


@dataclass(frozen=True, slots=True)
class StudentMlpModel:
    """Frozen pure-NumPy regression MLP with GELU hidden activations."""

    teacher_label: str
    student_id: str
    input_width: int
    hidden_widths: tuple[int, ...]
    output_width: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    seed: int
    selected_epoch: int
    train_mse: float
    validation_mse: float
    actuator_authority: bool = False

    def predict_flat(self, input_f32: np.ndarray) -> np.ndarray:
        features = np.asarray(input_f32, dtype=np.float32)
        if features.shape != (self.input_width,):
            raise Issue54DistillationError("student input shape mismatch")
        normalised = (features - self.feature_mean) / self.feature_scale
        hidden = normalised
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            pre = hidden @ weight + bias
            if index < len(self.weights) - 1:
                hidden = _gelu_exact(pre.astype(np.float64)).astype(np.float32)
            else:
                hidden = pre.astype(np.float32)
        result = np.asarray(
            hidden * self.target_std + self.target_mean, dtype=np.float32
        )
        result.setflags(write=False)
        return result

    def predict(self, input_f32: np.ndarray) -> np.ndarray:
        flat = self.predict_flat(input_f32)
        return flat.reshape(HORIZON_STEPS, TARGET_COUNT)


@dataclass(frozen=True, slots=True)
class StudentRidgeModel:
    """Frozen ridge regression student with teacher predictions as targets."""

    teacher_label: str
    student_id: str
    input_width: int
    output_width: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coef: np.ndarray
    actuator_authority: bool = False

    def predict_flat(self, input_f32: np.ndarray) -> np.ndarray:
        features = np.asarray(input_f32, dtype=np.float32)
        if features.shape != (self.input_width,):
            raise Issue54DistillationError("ridge student input shape mismatch")
        model_dtype = self.coef.dtype
        feature64 = features.astype(model_dtype, copy=False)
        value = (
            (feature64 - self.feature_mean) / self.feature_scale
        ) @ self.coef + self.target_mean
        result = np.asarray(value, dtype=np.float32)
        if not np.isfinite(result).all():
            raise Issue54DistillationError("ridge student prediction is non-finite")
        result.setflags(write=False)
        return result

    def predict(self, input_f32: np.ndarray) -> np.ndarray:
        flat = self.predict_flat(input_f32)
        return flat.reshape(HORIZON_STEPS, TARGET_COUNT)


StudentModel = StudentMlpModel | StudentRidgeModel


@dataclass(frozen=True, slots=True)
class TargetScaleInfo:
    """Per-target descriptor information needed for evaluation metrics."""

    descriptor_id: str
    nominal: float
    scale: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class DistillationCorpusManifest:
    schema_version: str
    teacher_label: str
    corpus_id: str
    family_split: dict[str, str]
    family_ids: tuple[str, ...]
    samples_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """Evaluation result for one student on FINAL samples."""

    teacher_label: str
    student_id: str
    corpus_id: str
    student_param_count: int
    nmae_teacher: float
    nmae_student: float
    nmae_ratio: float
    nmae_ratio_lower_ci: float
    nmae_ratio_upper_ci: float
    top1_agreement: float
    kendall_tau: float
    safety_exposure_difference: float
    family_count: int


@dataclass(frozen=True, slots=True)
class DistillationCurvePoint:
    teacher_label: str
    student_id: str
    corpus_id: str
    param_count: int
    nmae_ratio: float
    nmae_ratio_ci: tuple[float, float]
    top1_agreement: float
    kendall_tau: float
    safety_exposure_difference: float


def _gelu_exact(value: np.ndarray) -> np.ndarray:
    """Exact-error-function GELU matching torch.nn.GELU."""
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * value * (1.0 + erf(value / math.sqrt(2.0)))


def _gelu_derivative(value: np.ndarray) -> np.ndarray:
    """Derivative of the exact-erf GELU: Phi(x) + x*phi(x)."""
    cdf = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(
        value / math.sqrt(2.0)
    ))
    pdf = np.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
    return cdf + value * pdf


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _as_f32(value: np.ndarray, label: str) -> np.ndarray:
    array64 = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array64).all():
        raise Issue54DistillationError(f"{label} is non-finite")
    result = array64.astype(np.float32)
    if not np.isfinite(result).all():
        raise Issue54DistillationError(f"{label} does not remain finite float32")
    return _readonly(result)


def _sample_sha256(
    family_id: str,
    split: str,
    candidate_id: str,
    teacher_label: str,
    input_f32: np.ndarray,
    teacher_prediction_f32: np.ndarray,
    ground_truth_f32: np.ndarray,
) -> str:
    payload = {
        "family_id": family_id,
        "split": split,
        "candidate_id": candidate_id,
        "teacher_label": teacher_label,
        "input": np.asarray(input_f32, dtype=np.float32).tobytes().hex(),
        "teacher_prediction": np.asarray(teacher_prediction_f32, dtype=np.float32)
        .tobytes()
        .hex(),
        "ground_truth": np.asarray(ground_truth_f32, dtype=np.float32)
        .tobytes()
        .hex(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def make_distillation_sample(
    family_id: str,
    split: str,
    candidate_id: str,
    teacher_label: str,
    input_f32: np.ndarray,
    teacher_prediction_f32: np.ndarray,
    ground_truth_f32: np.ndarray,
) -> DistillationSample:
    if teacher_label not in _TEACHER_LABELS:
        raise Issue54DistillationError(f"unknown teacher label: {teacher_label}")
    input_arr = np.asarray(input_f32, dtype=np.float32)
    pred_arr = np.asarray(teacher_prediction_f32, dtype=np.float32)
    truth_arr = np.asarray(ground_truth_f32, dtype=np.float32)
    expected_input = MLP_FEATURE_COUNT if teacher_label == "mlp" else RIDGE_FEATURE_COUNT
    if input_arr.shape != (expected_input,):
        raise Issue54DistillationError(
            f"input shape {input_arr.shape} does not match teacher {teacher_label}"
        )
    if pred_arr.shape != (OUTPUT_DIM,):
        raise Issue54DistillationError("teacher prediction must be flat float32[408]")
    if truth_arr.shape != (OUTPUT_DIM,):
        raise Issue54DistillationError("ground truth must be flat float32[408]")
    if not np.isfinite(input_arr).all() or not np.isfinite(pred_arr).all():
        raise Issue54DistillationError("sample contains non-finite values")
    sha = _sample_sha256(
        family_id, split, candidate_id, teacher_label, input_arr, pred_arr, truth_arr
    )
    return DistillationSample(
        family_id=family_id,
        split=split,
        candidate_id=candidate_id,
        teacher_label=teacher_label,
        input_f32=_readonly(input_arr.copy()),
        teacher_prediction_f32=_readonly(pred_arr.copy()),
        ground_truth_f32=_readonly(truth_arr.copy()),
        sample_sha256=sha,
    )


def family_split(family_ids: Sequence[str]) -> dict[str, str]:
    """Assign whole families by the preregistered SHA-256 hash split (60/20/20)."""
    if not family_ids:
        return {}
    order = sorted(
        family_ids,
        key=lambda fid: hashlib.sha256(
            f"issue54-split-v1|{fid}".encode("utf-8")
        ).digest(),
    )
    proportions = (0.60, 0.20, 0.20)
    labels = ("TRAIN", "VALIDATION", "FINAL")
    counts = [int(len(order) * p) for p in proportions]
    remaining = len(order) - sum(counts)
    remainders = sorted(
        range(len(labels)),
        key=lambda i: (-(len(order) * proportions[i] - counts[i]), i),
    )
    for i in remainders[:remaining]:
        counts[i] += 1
    result: dict[str, str] = {}
    cursor = 0
    for label, count in zip(labels, counts, strict=True):
        for fid in order[cursor : cursor + count]:
            result[fid] = label
        cursor += count
    return dict(sorted(result.items()))


def deterministic_family_ids(n: int, corpus_id: str) -> list[str]:
    return [f"issue54-{corpus_id}-family-{i:04d}" for i in range(n)]


def build_corpus_manifest(
    teacher_label: str,
    corpus_id: str,
    family_ids: Sequence[str],
    split: Mapping[str, str],
    samples_sha256: str,
) -> DistillationCorpusManifest:
    body = {
        "schema_version": ISSUE54_SCHEMA_VERSION,
        "teacher_label": teacher_label,
        "corpus_id": corpus_id,
        "family_split": dict(split),
        "family_ids": list(family_ids),
        "samples_sha256": samples_sha256,
    }
    manifest_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return DistillationCorpusManifest(
        schema_version=ISSUE54_SCHEMA_VERSION,
        teacher_label=teacher_label,
        corpus_id=corpus_id,
        family_split=dict(split),
        family_ids=tuple(family_ids),
        samples_sha256=samples_sha256,
        manifest_sha256=manifest_sha,
    )


def validate_samples(
    samples: Sequence[DistillationSample],
    manifest: DistillationCorpusManifest,
) -> None:
    if not samples:
        raise Issue54DistillationError("corpus contains no samples")
    sample_digest = hashlib.sha256()
    seen: set[str] = set()
    family_ids = set(manifest.family_ids)
    for sample in samples:
        if sample.teacher_label != manifest.teacher_label:
            raise Issue54DistillationError("sample teacher label drift")
        if sample.family_id not in family_ids:
            raise Issue54DistillationError("sample family not in manifest")
        if sample.split != manifest.family_split[sample.family_id]:
            raise Issue54DistillationError("sample split does not match manifest")
        expected_sha = _sample_sha256(
            sample.family_id,
            sample.split,
            sample.candidate_id,
            sample.teacher_label,
            sample.input_f32,
            sample.teacher_prediction_f32,
            sample.ground_truth_f32,
        )
        if expected_sha != sample.sample_sha256:
            raise Issue54DistillationError("sample digest mismatch")
        if sample.sample_sha256 in seen:
            raise Issue54DistillationError("duplicate sample")
        seen.add(sample.sample_sha256)
        sample_digest.update(
            canonical_json_bytes(
                {
                    "family_id": sample.family_id,
                    "candidate_id": sample.candidate_id,
                    "sha": sample.sample_sha256,
                }
            )
        )
        sample_digest.update(b"\n")
    if sample_digest.hexdigest() != manifest.samples_sha256:
        raise Issue54DistillationError("corpus samples digest mismatch")


def _param_count(
    input_width: int, hidden_widths: Sequence[int], output_width: int
) -> int:
    total = 0
    prev = input_width
    for h in hidden_widths:
        total += prev * h + h
        prev = h
    total += prev * output_width + output_width
    return total


def student_param_count(teacher_label: str, student_id: str) -> int:
    if student_id == "linear":
        input_width = (
            MLP_FEATURE_COUNT if teacher_label == "mlp" else RIDGE_FEATURE_COUNT
        )
        return input_width * OUTPUT_DIM + OUTPUT_DIM
    hidden = _STUDENT_HIDDEN[teacher_label][student_id]
    input_width = (
        MLP_FEATURE_COUNT if teacher_label == "mlp" else RIDGE_FEATURE_COUNT
    )
    return _param_count(input_width, hidden, OUTPUT_DIM)


def _init_weights(
    rng: np.random.Generator, input_width: int, hidden_widths: Sequence[int], output_width: int
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    prev = input_width
    for h in hidden_widths:
        scale = math.sqrt(2.0 / prev)
        weights.append(rng.normal(0.0, scale, size=(prev, h)).astype(np.float64))
        biases.append(np.zeros(h, dtype=np.float64))
        prev = h
    weights.append(
        rng.normal(0.0, math.sqrt(2.0 / prev), size=(prev, output_width)).astype(
            np.float64
        )
    )
    biases.append(np.zeros(output_width, dtype=np.float64))
    return tuple(weights), tuple(biases)


def train_student_mlp(
    train_samples: Sequence[DistillationSample],
    validation_samples: Sequence[DistillationSample],
    teacher_label: str,
    student_id: str,
    *,
    epochs: int = 200,
    learning_rate: float = 0.001,
    l2_penalty: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    seed: int = 540054,
) -> StudentMlpModel:
    """Train a regression MLP student via distillation (GELU + MSE + Adam)."""
    if epochs < 5 or epochs % 5:
        raise Issue54DistillationError("epochs must be a positive multiple of 5")
    if teacher_label not in _TEACHER_LABELS:
        raise Issue54DistillationError(f"unknown teacher: {teacher_label}")
    if student_id not in _STUDENT_HIDDEN[teacher_label]:
        raise Issue54DistillationError(f"unknown student id: {student_id}")
    hidden_widths = _STUDENT_HIDDEN[teacher_label][student_id]
    input_width = (
        MLP_FEATURE_COUNT if teacher_label == "mlp" else RIDGE_FEATURE_COUNT
    )

    def _matrix(samples: Sequence[DistillationSample]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([s.input_f32 for s in samples]).astype(np.float64)
        t = np.stack([s.teacher_prediction_f32 for s in samples]).astype(np.float64)
        return x, t

    x_train, t_train = _matrix(train_samples)
    x_val, t_val = _matrix(validation_samples)
    feature_mean = x_train.mean(axis=0)
    feature_scale = x_train.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    x_train_n = (x_train - feature_mean) / feature_scale
    x_val_n = (x_val - feature_mean) / feature_scale
    target_mean = t_train.mean(axis=0)
    target_std = t_train.std(axis=0)
    target_std[target_std == 0.0] = 1.0
    t_train_n = (t_train - target_mean) / target_std
    t_val_n = (t_val - target_mean) / target_std

    rng = np.random.default_rng(seed)
    init_w, init_b = _init_weights(rng, input_width, hidden_widths, OUTPUT_DIM)
    weights: list[np.ndarray] = list(init_w)
    biases: list[np.ndarray] = list(init_b)
    first_m = [np.zeros_like(w) for w in weights]
    second_m = [np.zeros_like(w) for w in weights]
    fb_first = [np.zeros_like(b) for b in biases]
    fb_second = [np.zeros_like(b) for b in biases]

    best_val_mse = math.inf
    best_epoch = 0
    best_w = tuple(w.copy() for w in weights)
    best_b = tuple(b.copy() for b in biases)
    last_train_mse = math.inf

    for epoch in range(1, epochs + 1):
        # Forward
        activations: list[np.ndarray] = [x_train_n]
        pre_acts: list[np.ndarray] = []
        h = x_train_n
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            z = h @ w + b
            pre_acts.append(z)
            if i < len(weights) - 1:
                h = _gelu_exact(z)
            else:
                h = z
            activations.append(h)
        pred = activations[-1]
        n_elem = pred.shape[0] * pred.shape[1]
        error = (pred - t_train_n) * (2.0 / n_elem)
        last_train_mse = float(np.mean((pred - t_train_n) ** 2))

        # Backward
        deltas: list[np.ndarray | None] = [None] * len(weights)
        deltas[-1] = error
        for i in range(len(weights) - 2, -1, -1):
            d_h = deltas[i + 1] @ weights[i + 1].T
            deltas[i] = d_h * _gelu_derivative(pre_acts[i])

        grads_w: list[np.ndarray] = []
        grads_b: list[np.ndarray] = []
        for i in range(len(weights)):
            gw = activations[i].T @ deltas[i] + l2_penalty * weights[i]
            gb = deltas[i].sum(axis=0)
            grads_w.append(gw)
            grads_b.append(gb)

        # Adam update
        for i in range(len(weights)):
            first_m[i] = beta1 * first_m[i] + (1 - beta1) * grads_w[i]
            second_m[i] = beta2 * second_m[i] + (1 - beta2) * grads_w[i] ** 2
            corrected_m = first_m[i] / (1 - beta1**epoch)
            corrected_s = second_m[i] / (1 - beta2**epoch)
            weights[i] -= learning_rate * corrected_m / (
                np.sqrt(corrected_s) + epsilon
            )
            fb_first[i] = beta1 * fb_first[i] + (1 - beta1) * grads_b[i]
            fb_second[i] = beta2 * fb_second[i] + (1 - beta2) * grads_b[i] ** 2
            corrected_bm = fb_first[i] / (1 - beta1**epoch)
            corrected_bs = fb_second[i] / (1 - beta2**epoch)
            biases[i] -= learning_rate * corrected_bm / (
                np.sqrt(corrected_bs) + epsilon
            )

        # Validation
        if epoch % 5 == 0:
            hv = x_val_n
            for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
                zv = hv @ w + b
                hv = _gelu_exact(zv) if i < len(weights) - 1 else zv
            val_mse = float(np.mean((hv - t_val_n) ** 2))
            if val_mse < best_val_mse - 1e-12:
                best_val_mse = val_mse
                best_epoch = epoch
                best_w = tuple(w.copy() for w in weights)
                best_b = tuple(b.copy() for b in biases)

    return StudentMlpModel(
        teacher_label=teacher_label,
        student_id=student_id,
        input_width=input_width,
        hidden_widths=tuple(hidden_widths),
        output_width=OUTPUT_DIM,
        feature_mean=_readonly(feature_mean.astype(np.float32)),
        feature_scale=_readonly(feature_scale.astype(np.float32)),
        target_mean=_readonly(target_mean.astype(np.float32)),
        target_std=_readonly(target_std.astype(np.float32)),
        weights=tuple(_readonly(w.astype(np.float32)) for w in best_w),
        biases=tuple(_readonly(b.astype(np.float32)) for b in best_b),
        seed=seed,
        selected_epoch=best_epoch,
        train_mse=last_train_mse,
        validation_mse=best_val_mse,
        actuator_authority=False,
    )


def fit_student_ridge(
    train_samples: Sequence[DistillationSample],
    teacher_label: str,
    student_id: str,
    *,
    alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0, 100.0),
) -> StudentRidgeModel:
    """Fit a ridge regression student with teacher predictions as distillation targets."""
    if teacher_label not in _TEACHER_LABELS:
        raise Issue54DistillationError(f"unknown teacher: {teacher_label}")
    input_width = (
        MLP_FEATURE_COUNT if teacher_label == "mlp" else RIDGE_FEATURE_COUNT
    )
    x = np.stack([s.input_f32 for s in train_samples]).astype(np.float64)
    t = np.stack([s.teacher_prediction_f32 for s in train_samples]).astype(np.float64)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    x_n = (x - feature_mean) / feature_scale
    t_centered = t - t.mean(axis=0)

    clusters = sorted({s.family_id for s in train_samples})
    if len(clusters) < 2:
        raise Issue54DistillationError("ridge student needs >=2 families")

    best_alpha = alphas[0]
    best_error = math.inf
    for alpha in alphas:
        fold_errors: list[float] = []
        for cluster in clusters:
            mask = np.array([s.family_id == cluster for s in train_samples])
            training = ~mask
            if not training.any() or not mask.any():
                continue
            gram = x_n[training] @ x_n[training].T
            dual = np.linalg.solve(
                gram + alpha * np.eye(gram.shape[0]), t_centered[training]
            )
            coef = x_n[training].T @ dual
            pred = (x_n[mask] - 0) @ coef + t.mean(axis=0)
            fold_errors.append(float(np.mean((pred - t[mask]) ** 2)))
        if fold_errors:
            mean_error = float(np.mean(fold_errors))
            if mean_error < best_error:
                best_error = mean_error
                best_alpha = alpha

    gram = x_n @ x_n.T
    dual = np.linalg.solve(
        gram + best_alpha * np.eye(gram.shape[0]), t_centered
    )
    coef = x_n.T @ dual
    if not np.isfinite(coef).all():
        raise Issue54DistillationError("ridge student coefficients are non-finite")

    return StudentRidgeModel(
        teacher_label=teacher_label,
        student_id=student_id,
        input_width=input_width,
        output_width=OUTPUT_DIM,
        feature_mean=_readonly(feature_mean.astype(np.float32)),
        feature_scale=_readonly(feature_scale.astype(np.float32)),
        target_mean=_readonly(t.mean(axis=0).astype(np.float32)),
        coef=_readonly(coef.astype(np.float32)),
        actuator_authority=False,
    )


def _nmae_flat(prediction_f32: np.ndarray, truth_f32: np.ndarray, scales: np.ndarray) -> float:
    pred = np.asarray(prediction_f32, dtype=np.float64).reshape(HORIZON_STEPS, TARGET_COUNT)
    truth = np.asarray(truth_f32, dtype=np.float64).reshape(HORIZON_STEPS, TARGET_COUNT)
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        return math.inf
    return float(np.mean(np.abs(pred - truth) / scales[None, :]))


def extract_scales(manifest: TargetManifest) -> np.ndarray:
    return np.asarray(
        [d.scale for d in manifest.descriptors], dtype=np.float64
    )


def extract_nominals(manifest: TargetManifest) -> np.ndarray:
    return np.asarray(
        [d.nominal for d in manifest.descriptors], dtype=np.float64
    )


def extract_safety_bounds(manifest: TargetManifest) -> tuple[np.ndarray, np.ndarray]:
    lowers = np.empty(TARGET_COUNT, dtype=np.float64)
    uppers = np.empty(TARGET_COUNT, dtype=np.float64)
    for i, d in enumerate(manifest.descriptors):
        lowers[i] = d.lower if d.crossing_lower is None else float(d.crossing_lower)
        uppers[i] = d.upper if d.crossing_upper is None else float(d.crossing_upper)
    return lowers, uppers


def compute_nmae_family(
    predictions: Mapping[str, np.ndarray],
    samples: Sequence[DistillationSample],
    scales: np.ndarray,
) -> dict[str, float]:
    """Return per-family mean NMAE over candidates."""
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        pred = predictions.get(sample.sample_sha256)
        if pred is None:
            continue
        value = _nmae_flat(pred, sample.ground_truth_f32, scales)
        grouped.setdefault(sample.family_id, []).append(value)
    return {
        fid: float(np.mean(vals)) if vals else math.inf
        for fid, vals in grouped.items()
    }


def _candidate_score(prediction_f32: np.ndarray, nominals: np.ndarray, scales: np.ndarray) -> float:
    """Simplified tracking score: mean(|pred - nominal| / scale). Lower is better."""
    pred = np.asarray(prediction_f32, dtype=np.float64).reshape(
        HORIZON_STEPS, TARGET_COUNT
    )
    if not np.isfinite(pred).all():
        return math.inf
    return float(np.mean(np.abs(pred - nominals[None, :]) / scales[None, :]))


def _safety_exposure(
    prediction_f32: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Mean normalized safety bound crossing magnitude."""
    pred = np.asarray(prediction_f32, dtype=np.float64).reshape(
        HORIZON_STEPS, TARGET_COUNT
    )
    if not np.isfinite(pred).all():
        return math.inf
    lower_cross = np.maximum(0.0, lowers[None, :] - pred) / scales[None, :]
    upper_cross = np.maximum(0.0, pred - uppers[None, :]) / scales[None, :]
    return float(np.mean(lower_cross + upper_cross))


def _kendall_tau_b(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    n = len(rank_a)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a[i] - rank_a[j]
            db = rank_b[i] - rank_b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def compute_ranking_and_safety(
    teacher_predictions: Mapping[str, np.ndarray],
    student_predictions: Mapping[str, np.ndarray],
    samples: Sequence[DistillationSample],
    nominals: np.ndarray,
    scales: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> tuple[float, float, float]:
    """Return (top1_agreement, kendall_tau, safety_exposure_difference) over FINAL decisions."""
    by_family: dict[str, list[DistillationSample]] = {}
    for sample in samples:
        by_family.setdefault(sample.family_id, []).append(sample)

    top1_matches = 0
    total_decisions = 0
    tau_values: list[float] = []
    safety_diffs: list[float] = []

    for family_id, family_samples in sorted(by_family.items()):
        candidates = sorted(family_samples, key=lambda s: s.candidate_id)
        if len(candidates) < 2:
            continue
        teacher_scores = []
        student_scores = []
        for sample in candidates:
            tp = teacher_predictions.get(sample.sample_sha256)
            sp = student_predictions.get(sample.sample_sha256)
            if tp is None or sp is None:
                break
            teacher_scores.append(_candidate_score(tp, nominals, scales))
            student_scores.append(_candidate_score(sp, nominals, scales))
            safety_diffs.append(
                abs(
                    _safety_exposure(tp, lowers, uppers, scales)
                    - _safety_exposure(sp, lowers, uppers, scales)
                )
            )
        if len(teacher_scores) < 2:
            continue
        total_decisions += 1
        teacher_rank = sorted(range(len(teacher_scores)), key=lambda i: teacher_scores[i])
        student_rank = sorted(range(len(student_scores)), key=lambda i: student_scores[i])
        if teacher_rank[0] == student_rank[0]:
            top1_matches += 1
        tau_values.append(_kendall_tau_b(teacher_rank, student_rank))

    top1 = top1_matches / total_decisions if total_decisions else 0.0
    tau = float(np.mean(tau_values)) if tau_values else 0.0
    safety_diff = float(np.mean(safety_diffs)) if safety_diffs else math.inf
    return top1, tau, safety_diff


def _bootstrap_indices(seed: int, repetitions: int, count: int) -> np.ndarray:
    indices = np.empty((repetitions, count), dtype=np.int64)
    for r in range(repetitions):
        for d in range(count):
            indices[r, d] = int.from_bytes(
                hashlib.sha256(
                    f"issue54-bootstrap-v1|{seed}|{r}|{d}".encode("utf-8")
                ).digest()[:8],
                "big",
            ) % count
    return indices


def bootstrap_nmae_ratio(
    student_family_nmae: Mapping[str, float],
    teacher_family_nmae: Mapping[str, float],
    *,
    seed: int = 540054,
    repetitions: int = 10000,
) -> tuple[float, float, float]:
    """Return (point_ratio, lower_ci, upper_ci) for NMAE_student / NMAE_teacher."""
    families = sorted(set(student_family_nmae) & set(teacher_family_nmae))
    if not families:
        return math.inf, math.inf, math.inf
    s_vals = np.array([student_family_nmae[f] for f in families], dtype=np.float64)
    t_vals = np.array([teacher_family_nmae[f] for f in families], dtype=np.float64)
    if not np.isfinite(s_vals).all() or not np.isfinite(t_vals).all():
        return math.inf, math.inf, math.inf
    point = float(np.mean(s_vals) / np.mean(t_vals)) if np.mean(t_vals) != 0 else (
        1.0 if np.mean(s_vals) == 0 else math.inf
    )
    indices = _bootstrap_indices(seed, repetitions, len(families))
    s_boot = np.mean(s_vals[indices], axis=1)
    t_boot = np.mean(t_vals[indices], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            t_boot != 0, s_boot / t_boot, np.where(s_boot == 0, 1.0, np.inf)
        )
    if not np.isfinite(ratios).all():
        return point, math.inf, math.inf
    lower = float(np.quantile(ratios, 0.025))
    upper = float(np.quantile(ratios, 0.975))
    return point, lower, upper


def evaluate_student(
    student: StudentModel,
    final_samples: Sequence[DistillationSample],
    scales: np.ndarray,
    nominals: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> DistillationResult:
    """Evaluate one student on FINAL samples against the teacher."""
    teacher_preds: dict[str, np.ndarray] = {}
    student_preds: dict[str, np.ndarray] = {}
    for sample in final_samples:
        teacher_preds[sample.sample_sha256] = sample.teacher_prediction_f32
        student_preds[sample.sample_sha256] = student.predict_flat(sample.input_f32)

    teacher_nmae = compute_nmae_family(teacher_preds, final_samples, scales)
    student_nmae = compute_nmae_family(student_preds, final_samples, scales)
    point, lower, upper = bootstrap_nmae_ratio(student_nmae, teacher_nmae)

    top1, tau, safety_diff = compute_ranking_and_safety(
        teacher_preds, student_preds, final_samples, nominals, scales, lowers, uppers
    )

    if isinstance(student, StudentMlpModel):
        pc = _param_count(student.input_width, student.hidden_widths, student.output_width)
    else:
        pc = student.input_width * student.output_width + student.output_width

    return DistillationResult(
        teacher_label=student.teacher_label,
        student_id=student.student_id,
        corpus_id="",
        student_param_count=pc,
        nmae_teacher=float(np.mean(list(teacher_nmae.values()))) if teacher_nmae else math.inf,
        nmae_student=float(np.mean(list(student_nmae.values()))) if student_nmae else math.inf,
        nmae_ratio=point,
        nmae_ratio_lower_ci=lower,
        nmae_ratio_upper_ci=upper,
        top1_agreement=top1,
        kendall_tau=tau,
        safety_exposure_difference=safety_diff,
        family_count=len(set(s.family_id for s in final_samples)),
    )


def save_student_mlp(student: StudentMlpModel, path: str | Path) -> Path:
    """Persist a student MLP as a canonical NPZ artifact."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": ISSUE54_SCHEMA_VERSION,
        "release_tier": "DEVELOPMENT_EVIDENCE_ONLY",
        "actuator_authority": False,
        "teacher_label": student.teacher_label,
        "student_id": student.student_id,
        "input_width": student.input_width,
        "hidden_widths": list(student.hidden_widths),
        "output_width": student.output_width,
        "seed": student.seed,
        "selected_epoch": student.selected_epoch,
        "train_mse": student.train_mse,
        "validation_mse": student.validation_mse,
        "param_count": _param_count(
            student.input_width, student.hidden_widths, student.output_width
        ),
    }
    arrays = {
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        "feature_mean": student.feature_mean,
        "feature_scale": student.feature_scale,
        "target_mean": student.target_mean,
        "target_std": student.target_std,
    }
    for i, (w, b) in enumerate(zip(student.weights, student.biases, strict=True)):
        arrays[f"w{i}"] = w
        arrays[f"b{i}"] = b
    np.savez(dest, **arrays)
    return dest


def load_student_mlp(
    source: str | Path | bytes,
    expected_sha256: str | None = None,
) -> StudentMlpModel:
    """Load a student MLP artifact and reject identity or authority drift."""
    import io

    if type(source) is bytes:
        raw = source
    else:
        raw = Path(source).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise Issue54DistillationError("student MLP artifact SHA-256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata_json"].item()))
        if metadata.get("schema_version") != ISSUE54_SCHEMA_VERSION:
            raise Issue54DistillationError("student MLP schema drift")
        if metadata.get("actuator_authority") is not False:
            raise Issue54DistillationError("student MLP claims authority")
        n_layers = len(metadata["hidden_widths"]) + 1
        weights = tuple(
            np.asarray(value[f"w{i}"]).copy() for i in range(n_layers)
        )
        biases = tuple(
            np.asarray(value[f"b{i}"]).copy() for i in range(n_layers)
        )
        feature_mean = np.asarray(value["feature_mean"]).copy()
        feature_scale = np.asarray(value["feature_scale"]).copy()
        target_mean = np.asarray(value["target_mean"]).copy()
        target_std = np.asarray(value["target_std"]).copy()
    for a in (*weights, *biases, feature_mean, feature_scale, target_mean, target_std):
        if not np.isfinite(a).all():
            raise Issue54DistillationError("student MLP artifact is non-finite")
        a.setflags(write=False)
    return StudentMlpModel(
        teacher_label=metadata["teacher_label"],
        student_id=metadata["student_id"],
        input_width=metadata["input_width"],
        hidden_widths=tuple(metadata["hidden_widths"]),
        output_width=metadata["output_width"],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_std=target_std,
        weights=weights,
        biases=biases,
        seed=metadata["seed"],
        selected_epoch=metadata["selected_epoch"],
        train_mse=metadata["train_mse"],
        validation_mse=metadata["validation_mse"],
        actuator_authority=False,
    )
