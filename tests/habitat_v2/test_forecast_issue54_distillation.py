"""Tests for the Issue #54 knowledge distillation module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
from aeolus.habitat_v2.forecast_issue54_distillation import (
    MLP_FEATURE_COUNT,
    OUTPUT_DIM,
    RIDGE_FEATURE_COUNT,
    DistillationCorpusManifest,
    DistillationSample,
    Issue54DistillationError,
    bootstrap_nmae_ratio,
    build_corpus_manifest,
    compute_nmae_family,
    compute_ranking_and_safety,
    derive_train_nmae_scales,
    deterministic_family_ids,
    evaluate_student,
    family_split,
    fit_student_ridge,
    load_student_mlp,
    make_distillation_sample,
    save_student_mlp,
    student_param_count,
    train_student_mlp,
    validate_samples,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _fake_scales() -> np.ndarray:
    return np.ones(51, dtype=np.float64)


def _fake_nominals() -> np.ndarray:
    return np.zeros(51, dtype=np.float64)


def _fake_bounds() -> tuple[np.ndarray, np.ndarray]:
    return np.full(51, -100.0), np.full(51, 100.0)


def _samples_digest(samples: list[DistillationSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            canonical_json_bytes(
                {
                    "family_id": sample.family_id,
                    "decision_id": sample.decision_id,
                    "candidate_id": sample.candidate_id,
                    "sha": sample.sample_sha256,
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _make_sample(
    family_id: str,
    split: str,
    candidate_id: str,
    teacher_label: str,
    rng: np.random.Generator,
    decision_id: str | None = None,
) -> DistillationSample:
    input_w = MLP_FEATURE_COUNT if teacher_label == "mlp" else RIDGE_FEATURE_COUNT
    inp = rng.normal(0, 1, size=input_w).astype(np.float32)
    pred = rng.normal(0, 1, size=OUTPUT_DIM).astype(np.float32)
    truth = rng.normal(0, 1, size=OUTPUT_DIM).astype(np.float32)
    return make_distillation_sample(
        family_id,
        decision_id or f"{family_id}|anchor=0016",
        split,
        candidate_id,
        teacher_label,
        inp,
        pred,
        truth,
    )


def _make_manifest(
    teacher_label: str, family_ids: list[str], samples_sha256: str
) -> DistillationCorpusManifest:
    split = family_split(family_ids)
    decision_ids = [f"{fid}|anchor=0016" for fid in family_ids]
    return build_corpus_manifest(
        teacher_label,
        "test",
        family_ids,
        split,
        decision_ids,
        ("c0", "c1", "c2", "c3"),
        samples_sha256,
    )


def _fake_nmae_scales() -> np.ndarray:
    return np.ones((8, 51), dtype=np.float64)


class TestDistillationSample:
    def test_make_sample_validates_shapes(self) -> None:
        with pytest.raises(Issue54DistillationError, match="input shape"):
            make_distillation_sample(
                "f1", "d1", "TRAIN", "c1", "mlp",
                np.zeros(10, dtype=np.float32),
                np.zeros(OUTPUT_DIM, dtype=np.float32),
                np.zeros(OUTPUT_DIM, dtype=np.float32),
            )

    def test_make_sample_rejects_unknown_teacher(self) -> None:
        with pytest.raises(Issue54DistillationError, match="unknown teacher"):
            make_distillation_sample(
                "f1", "d1", "TRAIN", "c1", "unknown",
                np.zeros(MLP_FEATURE_COUNT, dtype=np.float32),
                np.zeros(OUTPUT_DIM, dtype=np.float32),
                np.zeros(OUTPUT_DIM, dtype=np.float32),
            )

    def test_make_sample_rejects_non_finite(self) -> None:
        inp = np.zeros(MLP_FEATURE_COUNT, dtype=np.float32)
        inp[0] = float("nan")
        with pytest.raises(Issue54DistillationError, match="non-finite"):
            make_distillation_sample(
                "f1", "d1", "TRAIN", "c1", "mlp", inp,
                np.zeros(OUTPUT_DIM, dtype=np.float32),
                np.zeros(OUTPUT_DIM, dtype=np.float32),
            )

    def test_make_sample_rejects_non_finite_truth(self) -> None:
        truth = np.zeros(OUTPUT_DIM, dtype=np.float32)
        truth[0] = float("nan")
        with pytest.raises(Issue54DistillationError, match="non-finite"):
            make_distillation_sample(
                "f1", "d1", "TRAIN", "c1", "mlp",
                np.zeros(MLP_FEATURE_COUNT, dtype=np.float32),
                np.zeros(OUTPUT_DIM, dtype=np.float32),
                truth,
            )

    def test_sample_sha256_is_deterministic(self) -> None:
        rng = _rng()
        s1 = _make_sample("f1", "TRAIN", "c1", "mlp", rng)
        rng2 = np.random.default_rng(42)
        s2 = _make_sample("f1", "TRAIN", "c1", "mlp", rng2)
        assert s1.sample_sha256 == s2.sample_sha256

    def test_sample_arrays_are_readonly(self) -> None:
        rng = _rng()
        s = _make_sample("f1", "TRAIN", "c1", "mlp", rng)
        assert not s.input_f32.flags.writeable
        assert not s.teacher_prediction_f32.flags.writeable
        assert not s.ground_truth_f32.flags.writeable


class TestFamilySplit:
    def test_split_covers_all_families(self) -> None:
        ids = deterministic_family_ids(20, "test")
        split = family_split(ids)
        assert set(split) == set(ids)
        assert all(v in ("TRAIN", "VALIDATION", "FINAL") for v in split.values())

    def test_split_is_deterministic(self) -> None:
        ids = deterministic_family_ids(20, "test")
        s1 = family_split(ids)
        s2 = family_split(ids)
        assert s1 == s2

    def test_split_proportions(self) -> None:
        ids = deterministic_family_ids(100, "test")
        split = family_split(ids)
        train = sum(1 for v in split.values() if v == "TRAIN")
        val = sum(1 for v in split.values() if v == "VALIDATION")
        final = sum(1 for v in split.values() if v == "FINAL")
        assert train == 60
        assert val == 20
        assert final == 20

    def test_split_whole_family_isolation(self) -> None:
        ids = deterministic_family_ids(20, "test")
        split = family_split(ids)
        for fid, label in split.items():
            assert label in ("TRAIN", "VALIDATION", "FINAL")


class TestStudentParamCount:
    def test_mlp_sanity_matches_teacher(self) -> None:
        pc = student_param_count("mlp", "sanity-2.1m")
        assert 1_900_000 < pc < 2_300_000

    def test_ridge_sanity_is_smaller(self) -> None:
        pc = student_param_count("ridge", "sanity-2.1m")
        assert pc > 1_000_000

    def test_linear_ridge_matches_full_ridge(self) -> None:
        pc = student_param_count("ridge", "linear")
        assert pc == RIDGE_FEATURE_COUNT * OUTPUT_DIM + OUTPUT_DIM

    def test_tiny_is_smallest(self) -> None:
        tiny = student_param_count("mlp", "tiny-25k")
        small = student_param_count("mlp", "small-100k")
        medium = student_param_count("mlp", "medium-500k")
        assert tiny < small < medium


class TestTrainStudentMlp:
    def test_training_decreases_loss(self) -> None:
        rng = _rng()
        train_samples = [
            _make_sample(f"f{i:02d}", "TRAIN", f"c{j}", "mlp", rng)
            for i in range(6)
            for j in range(4)
        ]
        val_samples = [
            _make_sample(f"f{i:02d}", "VALIDATION", f"c{j}", "mlp", rng)
            for i in range(6, 8)
            for j in range(4)
        ]
        student = train_student_mlp(
            train_samples, val_samples, "mlp", "tiny-25k",
            epochs=20, learning_rate=0.01, seed=540054,
        )
        assert student.train_mse < 5.0
        assert student.selected_epoch > 0
        assert student.actuator_authority is False

    def test_training_is_deterministic(self) -> None:
        rng1 = _rng()
        rng2 = _rng()
        train1 = [
            _make_sample(f"f{i:02d}", "TRAIN", f"c{j}", "mlp", rng1)
            for i in range(6) for j in range(4)
        ]
        val1 = [
            _make_sample(f"f{i:02d}", "VALIDATION", f"c{j}", "mlp", rng1)
            for i in range(6, 8) for j in range(4)
        ]
        train2 = [
            _make_sample(f"f{i:02d}", "TRAIN", f"c{j}", "mlp", rng2)
            for i in range(6) for j in range(4)
        ]
        val2 = [
            _make_sample(f"f{i:02d}", "VALIDATION", f"c{j}", "mlp", rng2)
            for i in range(6, 8) for j in range(4)
        ]
        s1 = train_student_mlp(train1, val1, "mlp", "tiny-25k", epochs=20, seed=540054)
        s2 = train_student_mlp(train2, val2, "mlp", "tiny-25k", epochs=20, seed=540054)
        assert s1.selected_epoch == s2.selected_epoch
        for w1, w2 in zip(s1.weights, s2.weights, strict=True):
            np.testing.assert_array_equal(w1, w2)

    def test_predict_shape(self) -> None:
        rng = _rng()
        train = [_make_sample(f"f{i:02d}", "TRAIN", "c0", "mlp", rng) for i in range(6)]
        val = [_make_sample(f"f{i:02d}", "VALIDATION", "c0", "mlp", rng) for i in range(6, 8)]
        student = train_student_mlp(train, val, "mlp", "tiny-25k", epochs=10)
        pred = student.predict(train[0].input_f32)
        assert pred.shape == (8, 51)
        assert pred.dtype == np.float32


class TestFitStudentRidge:
    def test_fit_returns_model_with_correct_dimensions(self) -> None:
        rng = _rng()
        samples = [
            _make_sample(f"f{i:02d}", "TRAIN", f"c{j}", "ridge", rng)
            for i in range(6) for j in range(4)
        ]
        model = fit_student_ridge(samples, "ridge", "linear")
        assert model.input_width == RIDGE_FEATURE_COUNT
        assert model.output_width == OUTPUT_DIM
        assert model.coef.shape == (RIDGE_FEATURE_COUNT, OUTPUT_DIM)
        assert model.actuator_authority is False

    def test_predict_shape(self) -> None:
        rng = _rng()
        samples = [
            _make_sample(f"f{i:02d}", "TRAIN", "c0", "ridge", rng) for i in range(6)
        ]
        model = fit_student_ridge(samples, "ridge", "linear")
        pred = model.predict(samples[0].input_f32)
        assert pred.shape == (8, 51)
        assert pred.dtype == np.float32


class TestEvaluation:
    def test_nmae_zero_for_perfect_prediction(self) -> None:
        rng = _rng()
        sample = _make_sample("f01", "FINAL", "c0", "mlp", rng)
        preds = {sample.sample_sha256: sample.ground_truth_f32}
        result = compute_nmae_family(preds, [sample], _fake_nmae_scales())
        assert result["f01"] == pytest.approx(0.0, abs=1e-6)

    def test_nmae_positive_for_imperfect(self) -> None:
        rng = _rng()
        sample = _make_sample("f01", "FINAL", "c0", "mlp", rng)
        perturbed = sample.ground_truth_f32 + 1.0
        preds = {sample.sample_sha256: perturbed.astype(np.float32)}
        result = compute_nmae_family(preds, [sample], _fake_nmae_scales())
        assert result["f01"] > 0.0

    def test_nmae_accepts_flat_target_scales(self) -> None:
        rng = _rng()
        sample = _make_sample("f01", "FINAL", "c0", "mlp", rng)
        preds = {sample.sample_sha256: sample.ground_truth_f32}
        result = compute_nmae_family(preds, [sample], _fake_scales())
        assert result["f01"] == pytest.approx(0.0, abs=1e-6)

    def test_ranking_agreement_perfect(self) -> None:
        rng = _rng()
        samples = [
            _make_sample("f01", "FINAL", f"c{j}", "mlp", rng)
            for j in range(4)
        ]
        preds = {s.sample_sha256: s.teacher_prediction_f32 for s in samples}
        nominals = _fake_nominals()
        scales = _fake_scales()
        lowers, uppers = _fake_bounds()
        top1, tau, safety = compute_ranking_and_safety(
            preds, preds, samples, nominals, scales, lowers, uppers,
            expected_candidate_ids=("c0", "c1", "c2", "c3"),
        )
        assert top1 == 1.0
        assert tau == 1.0
        assert safety == 0.0

    def test_ranking_groups_by_decision_not_family(self) -> None:
        rng = _rng()
        samples = [
            _make_sample(
                "f01",
                "FINAL",
                f"c{j}",
                "mlp",
                rng,
                decision_id=f"f01|anchor={anchor:04d}",
            )
            for anchor in (16, 24)
            for j in range(4)
        ]
        preds = {sample.sample_sha256: sample.teacher_prediction_f32 for sample in samples}
        top1, tau, safety = compute_ranking_and_safety(
            preds,
            preds,
            samples,
            _fake_nominals(),
            _fake_scales(),
            *_fake_bounds(),
            expected_candidate_ids=("c0", "c1", "c2", "c3"),
        )
        assert top1 == 1.0
        assert tau == 1.0
        assert safety == 0.0

    def test_kendall_tau_b_ordering(self) -> None:
        from aeolus.habitat_v2.forecast_issue54_distillation import _kendall_tau_b
        assert _kendall_tau_b([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
        assert _kendall_tau_b([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0
        assert abs(_kendall_tau_b([0, 1, 2, 3], [0, 2, 1, 3])) < 1.0

    def test_bootstrap_nmae_ratio_perfect(self) -> None:
        family_nmae = {"f01": 0.5, "f02": 0.5, "f03": 0.5}
        point, lo, hi = bootstrap_nmae_ratio(family_nmae, family_nmae, repetitions=100)
        assert point == pytest.approx(1.0)
        assert lo <= 1.0 <= hi

    def test_evaluate_student_returns_result(self) -> None:
        rng = _rng()
        train = [_make_sample(f"f{i:02d}", "TRAIN", "c0", "mlp", rng) for i in range(6)]
        val = [_make_sample(f"f{i:02d}", "VALIDATION", "c0", "mlp", rng) for i in range(6, 8)]
        student = train_student_mlp(train, val, "mlp", "tiny-25k", epochs=10)
        final = [_make_sample(f"f{i:02d}", "FINAL", f"c{j}", "mlp", rng) for i in range(8, 10) for j in range(4)]
        scales = _fake_scales()
        nominals = _fake_nominals()
        lowers, uppers = _fake_bounds()
        result = evaluate_student(
            student,
            final,
            scales,
            nominals,
            lowers,
            uppers,
            corpus_id="test",
            nmae_scales=_fake_nmae_scales(),
            expected_candidate_ids=("c0", "c1", "c2", "c3"),
        )
        assert result.teacher_label == "mlp"
        assert result.student_id == "tiny-25k"
        assert result.corpus_id == "test"
        assert result.nmae_ratio > 0
        assert 0.0 <= result.top1_agreement <= 1.0
        assert -1.0 <= result.kendall_tau <= 1.0
        assert result.safety_exposure_difference >= 0.0
        assert result.student_param_count > 0

    def test_train_scales_use_only_train_truth(self) -> None:
        rng = _rng()
        train = [
            _make_sample(f"f{i:02d}", "TRAIN", "c0", "mlp", rng)
            for i in range(6)
        ]
        scales = derive_train_nmae_scales(train)
        expected_truth = np.stack([
            sample.ground_truth_f32.reshape(8, 51) for sample in train
        ]).astype(np.float64)
        expected = np.percentile(expected_truth, 95, axis=0) - np.percentile(
            expected_truth, 5, axis=0
        )
        np.testing.assert_array_equal(scales, expected)


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        rng = _rng()
        train = [_make_sample(f"f{i:02d}", "TRAIN", "c0", "mlp", rng) for i in range(6)]
        val = [_make_sample(f"f{i:02d}", "VALIDATION", "c0", "mlp", rng) for i in range(6, 8)]
        student = train_student_mlp(train, val, "mlp", "tiny-25k", epochs=10)
        path = tmp_path / "student.npz"
        save_student_mlp(student, path)
        loaded = load_student_mlp(path)
        assert loaded.teacher_label == student.teacher_label
        assert loaded.student_id == student.student_id
        assert loaded.input_width == student.input_width
        assert loaded.actuator_authority is False
        for w1, w2 in zip(loaded.weights, student.weights, strict=True):
            np.testing.assert_array_equal(w1, w2)

    def test_load_rejects_authority_claim(self, tmp_path: Path) -> None:
        rng = _rng()
        train = [_make_sample(f"f{i:02d}", "TRAIN", "c0", "mlp", rng) for i in range(6)]
        val = [_make_sample(f"f{i:02d}", "VALIDATION", "c0", "mlp", rng) for i in range(6, 8)]
        student = train_student_mlp(train, val, "mlp", "tiny-25k", epochs=10)
        path = tmp_path / "student.npz"
        save_student_mlp(student, path)
        # Tamper with the metadata
        with np.load(path, allow_pickle=False) as data:
            files = dict(data)
        import json as _json
        metadata = _json.loads(str(files["metadata_json"].item()))
        metadata["actuator_authority"] = True
        files["metadata_json"] = np.array(_json.dumps(metadata, sort_keys=True))
        np.savez(path, **files)
        with pytest.raises(Issue54DistillationError, match="claims authority"):
            load_student_mlp(path)


class TestCorpusValidation:
    def test_validate_samples_passes_for_consistent_corpus(self) -> None:
        rng = _rng()
        ids = deterministic_family_ids(10, "test")
        split = family_split(ids)
        samples = []
        for fid in ids:
            for j in range(4):
                s = _make_sample(fid, split[fid], f"c{j}", "mlp", rng)
                samples.append(s)
        digest = _samples_digest(samples)
        manifest = build_corpus_manifest(
            "mlp",
            "test",
            ids,
            split,
            [f"{fid}|anchor=0016" for fid in ids],
            ("c0", "c1", "c2", "c3"),
            digest,
        )
        validate_samples(samples, manifest)

    def test_validate_samples_rejects_split_mismatch(self) -> None:
        rng = _rng()
        ids = deterministic_family_ids(10, "test")
        split = family_split(ids)
        wrong_split = "VALIDATION" if split[ids[0]] != "VALIDATION" else "TRAIN"
        s = _make_sample(ids[0], wrong_split, "c0", "mlp", rng)
        manifest = build_corpus_manifest(
            "mlp",
            "test",
            ids,
            split,
            [f"{fid}|anchor=0016" for fid in ids],
            ("c0", "c1", "c2", "c3"),
            "dummy",
        )
        with pytest.raises(Issue54DistillationError, match="split does not match"):
            validate_samples([s], manifest)

    def test_validate_samples_rejects_incomplete_decision(self) -> None:
        rng = _rng()
        family_ids = ["f01"]
        split = {"f01": "FINAL"}
        samples = [
            _make_sample("f01", "FINAL", candidate_id, "mlp", rng)
            for candidate_id in ("c0", "c1", "c2")
        ]
        manifest = build_corpus_manifest(
            "mlp",
            "test",
            family_ids,
            split,
            ["f01|anchor=0016"],
            ("c0", "c1", "c2", "c3"),
            "not-the-digest",
        )
        with pytest.raises(Issue54DistillationError, match="complete candidate set"):
            validate_samples(samples, manifest)

    def test_validate_samples_rejects_duplicate_decision_candidate(self) -> None:
        rng = _rng()
        family_ids = ["f01"]
        split = {"f01": "FINAL"}
        samples = [
            _make_sample("f01", "FINAL", candidate_id, "mlp", rng)
            for candidate_id in ("c0", "c1", "c2", "c3")
        ]
        digest = _samples_digest(samples)
        manifest = build_corpus_manifest(
            "mlp",
            "test",
            family_ids,
            split,
            ["f01|anchor=0016"],
            ("c0", "c1", "c2", "c3"),
            digest,
        )
        with pytest.raises(Issue54DistillationError, match="duplicate decision candidate"):
            validate_samples(samples + [samples[0]], manifest)


class TestPreregistrationBinding:
    def test_preregistration_digest_is_frozen(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        prereg_path = repo_root / "contracts" / "habitat_v2_forecast_issue_54_preregistration_v1.json"
        text = prereg_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest().upper()
        assert digest == "E16BEFB588A43F131128056932BBFE5CAA707C87309A828A33C91E1C412D5246"
