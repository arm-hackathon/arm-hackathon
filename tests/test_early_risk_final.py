"""Fail-closed verdict tests for the one-shot early-risk final evaluation."""

from __future__ import annotations

from copy import deepcopy
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest

import aeolus.early_risk_final as final_module
from aeolus.early_risk import load_early_risk_artifact
from aeolus.early_risk_final import (
    FROZEN_ARTIFACT_BYTES_SHA256,
    FROZEN_FINAL_FAMILY_COMPOSITION,
    FROZEN_SWEEP_CANONICAL_SHA256,
    FROZEN_THRESHOLDS,
    FinalGateThresholds,
    _claim_final_suite,
    _load_generation_receipt,
    _require_final_families,
    _summarise,
    _validate_corpus_against_frozen_spec,
    _validate_forbidden_manifests,
    _validate_frozen_artifact,
    _validate_frozen_sweep,
    assess_final_summary,
    run_final_evaluation,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "models" / "early-risk-softmax-v1-candidate.json"


def _passing_summary() -> dict:
    return {
        "final_families": 144,
        "unique_healthy_references": 4,
        "harmful_physical_families": 24,
        "harmful_gradual_families": 10,
        "healthy_reference_interventions": 0,
        "frozen_sensor_interventions": 0,
        "wrong_target_interventions": 0,
        "invariant_violations": 0,
        "missed_harmful_physical_families": 0,
        "worsened_harmful_physical_families": 0,
        "transient_repeated_protection_families": 0,
        "transient_handback_recurrence_families": 0,
        "transient_handback_timeout_families": 0,
        "transient_nonzero_final_reserve_families": 0,
        "harmful_gradual_earlier_protection_families": 4,
        "harmful_gradual_positive_excess_reduction_families": 3,
        "median_excess_reduction_fraction_vs_governor_harmful_gradual": 0.10,
    }


def test_final_gate_accepts_exact_boundary_pass():
    assessment = assess_final_summary(_passing_summary(), FinalGateThresholds())

    assert assessment["safety_gate_pass"] is True
    assert assessment["benefit_gate_pass"] is True
    assert assessment["verdict"] == "PASS"
    assert assessment["required_earlier_protection_families"] == 4
    assert assessment["required_positive_excess_reduction_families"] == 3


@pytest.mark.parametrize(
    "field",
    [
        "healthy_reference_interventions",
        "frozen_sensor_interventions",
        "wrong_target_interventions",
        "invariant_violations",
        "missed_harmful_physical_families",
        "worsened_harmful_physical_families",
        "transient_repeated_protection_families",
        "transient_handback_recurrence_families",
        "transient_handback_timeout_families",
        "transient_nonzero_final_reserve_families",
    ],
)
def test_final_gate_rejects_each_safety_failure(field):
    summary = _passing_summary()
    summary[field] = 1

    assessment = assess_final_summary(summary, FinalGateThresholds())

    assert assessment["safety_gate_pass"] is False
    assert assessment["verdict"] == "REJECT_SAFETY"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("harmful_gradual_families", 7),
        ("harmful_gradual_earlier_protection_families", 3),
        ("harmful_gradual_positive_excess_reduction_families", 2),
        ("median_excess_reduction_fraction_vs_governor_harmful_gradual", 0.099),
    ],
)
def test_final_gate_rejects_each_benefit_failure(field, value):
    summary = deepcopy(_passing_summary())
    summary[field] = value

    assessment = assess_final_summary(summary, FinalGateThresholds())

    assert assessment["safety_gate_pass"] is True
    assert assessment["benefit_gate_pass"] is False
    assert assessment["verdict"] == "REJECT_BENEFIT"


def test_final_gate_rejects_too_few_final_families():
    summary = _passing_summary()
    summary["final_families"] = 119

    assessment = assess_final_summary(summary, FinalGateThresholds())

    assert assessment["admission_gate_pass"] is False
    assert assessment["safety_gate_pass"] is False
    assert assessment["verdict"] == "REJECT_SAFETY"


def test_final_gate_rejects_missing_or_extra_metrics():
    missing = _passing_summary()
    missing.pop("invariant_violations")
    with pytest.raises(ValueError, match="fields"):
        assess_final_summary(missing, FinalGateThresholds())

    extra = _passing_summary()
    extra["unknown"] = 0
    with pytest.raises(ValueError, match="fields"):
        assess_final_summary(extra, FinalGateThresholds())


def test_final_suite_claim_is_exclusive(tmp_path):
    lock = tmp_path / ".early-risk-final-run-lock.json"
    document = {"schema_version": "test-final-lock-v1"}

    _claim_final_suite(lock, document)

    assert lock.exists()
    with pytest.raises(FileExistsError, match="already been claimed"):
        _claim_final_suite(lock, document)


def test_run_final_evaluation_has_no_threshold_override():
    assert "thresholds" not in signature(run_final_evaluation).parameters
    assert FROZEN_THRESHOLDS == FinalGateThresholds(
        minimum_final_families=144,
        minimum_harmful_gradual_families=8,
        minimum_earlier_protection_fraction=0.40,
        minimum_positive_excess_reduction_fraction=0.25,
        minimum_median_excess_reduction_fraction=0.10,
    )


def test_frozen_sweep_rejects_other_valid_final_hash():
    _validate_frozen_sweep(
        SimpleNamespace(
            schema_version="aeolus_sweep_v4",
            suite_role="final",
            sha256=FROZEN_SWEEP_CANONICAL_SHA256,
        )
    )
    with pytest.raises(ValueError, match="canonical hash"):
        _validate_frozen_sweep(
            SimpleNamespace(
                schema_version="aeolus_sweep_v4",
                suite_role="final",
                sha256="0" * 64,
            )
        )


def test_frozen_final_family_count_and_composition_are_exact():
    families = []
    index = 0
    for fault_class, count in FROZEN_FINAL_FAMILY_COMPOSITION.items():
        for _ in range(count):
            families.append(
                SimpleNamespace(
                    split="final",
                    fault_class=fault_class,
                    reference_path=Path(f"reference-{index % 4}.json"),
                )
            )
            index += 1
    manifest = SimpleNamespace(families=tuple(families))

    assert len(_require_final_families(manifest)) == 144

    altered = list(families)
    altered[-1] = SimpleNamespace(
        split="final",
        fault_class="blocked_path",
        reference_path=Path("reference-3.json"),
    )
    with pytest.raises(ValueError, match="composition"):
        _require_final_families(SimpleNamespace(families=tuple(altered)))


def test_frozen_artifact_rejects_reserialized_bytes(tmp_path):
    predictor, _, _ = load_early_risk_artifact(ARTIFACT)
    _validate_frozen_artifact(ARTIFACT, predictor)
    substituted = tmp_path / "candidate.json"
    substituted.write_bytes(ARTIFACT.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact bytes"):
        _validate_frozen_artifact(substituted, predictor)

    assert final_module._sha256_file(ARTIFACT) == FROZEN_ARTIFACT_BYTES_SHA256


def test_forbidden_manifests_must_be_exact_and_distinct(monkeypatch, tmp_path):
    identities = list(final_module.FROZEN_FORBIDDEN_MANIFESTS)
    paths = [tmp_path / "development.json", tmp_path / "prior-final.json"]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    manifests = {
        str(paths[index].resolve()): SimpleNamespace(
            manifest_sha256=identity,
            families=tuple(
                SimpleNamespace(split=split)
                for split in final_module.FROZEN_FORBIDDEN_MANIFESTS[identity][
                    "splits"
                ]
                for _ in range(
                    final_module.FROZEN_FORBIDDEN_MANIFESTS[identity][
                        "family_count"
                    ]
                    // len(
                        final_module.FROZEN_FORBIDDEN_MANIFESTS[identity]["splits"]
                    )
                )
            ),
        )
        for index, identity in enumerate(identities)
    }
    monkeypatch.setattr(
        final_module,
        "load_family_manifest",
        lambda path: manifests[str(Path(path).resolve())],
    )
    monkeypatch.setattr(final_module, "validate_manifest_disjointness", lambda *_: None)
    monkeypatch.setattr(
        final_module,
        "_sha256_file",
        lambda path: final_module.FROZEN_FORBIDDEN_MANIFESTS[
            manifests[str(Path(path).resolve())].manifest_sha256
        ]["bytes_sha256"],
    )

    receipts = _validate_forbidden_manifests(paths, final_manifest=object())

    assert {receipt["canonical_manifest_sha256"] for receipt in receipts} == set(
        identities
    )
    with pytest.raises(ValueError, match="distinct"):
        _validate_forbidden_manifests(
            [paths[0], paths[0]], final_manifest=object()
        )


def test_corpus_preflight_rejects_any_payload_substitution(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    expected_files = {
        "families.json": '{"families": []}\n',
        "scenario.json": '{"value": 1}\n',
        "sweep-receipt.json": '{"schema_version": "aeolus_sweep_v4"}\n',
    }
    for name, content in expected_files.items():
        (corpus / name).write_text(content, encoding="utf-8", newline="\n")

    def fake_generate(_sweep_path, output_dir):
        output = Path(output_dir)
        for name, content in expected_files.items():
            (output / name).write_text(content, encoding="utf-8", newline="\n")

    monkeypatch.setattr(final_module, "generate_sweep", fake_generate)
    proof = _validate_corpus_against_frozen_spec(Path("frozen.json"), corpus)
    assert proof["corpus_file_count"] == 3

    (corpus / "scenario.json").write_text(
        '{"value": 2}\n', encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="payload tree"):
        _validate_corpus_against_frozen_spec(Path("frozen.json"), corpus)


def test_generation_receipt_rejects_stale_or_forged_fields(tmp_path):
    path = tmp_path / "sweep-receipt.json"
    expected = {"schema_version": "aeolus_sweep_v4", "total_families": 144}
    path.write_text('{"schema_version": "development", "total_families": 144}')

    with pytest.raises(ValueError, match="stale or substituted"):
        _load_generation_receipt(path, expected_document=expected)


def test_any_negative_physical_reduction_counts_as_worsening():
    row = {
        "family_id": "tiny-negative",
        "fault_class": "gradual_primary_fan_degradation",
        "target_zone_id": "cabin_a",
        "reserve_off_integrated_excess": 1.0,
        "advisory_excess_reduction_vs_governor": -5e-13,
        "advisory_excess_reduction_fraction_vs_governor": -5e-13,
        "protection_lead_ticks": 1,
        "advisory_governor_first_protect_tick": 1,
        "advisory_governor_protect_targets": ["cabin_a"],
        "advisory_governor_protect_entry_count": 1,
        "advisory_governor_handback_recurrence_count": 0,
        "advisory_governor_handback_timeout_count": 0,
        "advisory_governor_invariant_violation_count": 0,
        "advisory_governor_final_physical_zero": True,
        "model_warning_count": 1,
        "accepted_advisory_observation_count": 1,
    }

    summary, diagnostics = _summarise([row], [])

    assert summary["harmful_physical_families"] == 1
    assert summary["worsened_harmful_physical_families"] == 1
    assert diagnostics["worsened_harmful_family_ids"] == ["tiny-negative"]
