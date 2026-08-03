"""End-to-end fail-closed checks for the v3 frozen protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.corpus import generate_corpus_v2
from aeolus.detector import evidence_conclusion
from aeolus.families import load_family_manifest, parse_family_manifest, validate_manifest_disjointness
from aeolus.protocol import (
    build_final,
    final_evaluate,
    load_frozen_policy,
    select_development,
    sha256_file,
)
from aeolus.sweep import generate_sweep


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_v3_spec(tmp_path: Path, role: str) -> Path:
    base_name = "standard_habitat.json"
    (tmp_path / base_name).write_bytes((REPO_ROOT / "scenarios" / base_name).read_bytes())
    telemetry = {
        "airflow_noise_fraction": 0.01,
        "airflow_bias_fraction": 0.01,
        "airflow_drift_fraction": 0.01,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.01,
        "co2_sensor_bias_fraction": 0.01,
        "co2_sensor_drift_fraction": 0.01,
    }
    names = ("train", "validation") if role == "development" else ("final",)
    document = {
        "schema_version": "aeolus_sweep_v3",
        "suite_role": role,
        "base_scenario": base_name,
        "targets": ["cabin_a"],
        "splits": {
            split: {
                "seeds": [1000 + number + (100 if role == "final" else 0)],
                "fault_start_ticks": [25],
                "operating_profiles": [{"id": split, "source_multiplier": 1.0, "shared_airflow_capacity": 30.0, "telemetry": telemetry}],
                "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
                "blocked_effectiveness": [0.65],
            }
            for number, split in enumerate(names)
        },
    }
    path = tmp_path / f"{role}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _suite(tmp_path: Path, role: str) -> tuple[Path, Path, str]:
    sweep = tmp_path / f"{role}-sweep"
    generate_sweep(_write_v3_spec(tmp_path, role), sweep)
    corpus = tmp_path / f"{role}-corpus"
    generated = generate_corpus_v2(sweep / "families.json", corpus)
    return sweep, corpus, str(generated["family_manifest_sha256"])


def test_build_final_rejects_non_final_suite_and_existing_output(tmp_path: Path):
    development_spec = _write_v3_spec(tmp_path, "development")
    with pytest.raises(ValueError, match="requires an aeolus_sweep_v3 final suite"):
        build_final(development_spec, tmp_path / "output")

    final_spec = _write_v3_spec(tmp_path, "final")
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "sentinel").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="final output directory is not empty"):
        build_final(final_spec, output)


def test_final_manifest_is_accepted_and_canonical_overlap_is_rejected(tmp_path: Path):
    development_sweep, _, _ = _suite(tmp_path, "development")
    final_sweep, _, _ = _suite(tmp_path, "final")
    development = load_family_manifest(development_sweep / "families.json")
    final = load_family_manifest(final_sweep / "families.json")
    validate_manifest_disjointness(development, final)

    copied = json.loads((development_sweep / "families.json").read_text(encoding="utf-8"))
    for family in copied["families"]:
        family["split"] = "final"
    overlapping_final = parse_family_manifest(copied, base_dir=development_sweep)
    assert {family.split for family in overlapping_final.families} == {"final"}
    with pytest.raises(ValueError, match="not disjoint"):
        validate_manifest_disjointness(development, overlapping_final)


def test_selection_and_final_evaluation_freeze_all_final_decisions(tmp_path: Path):
    development_sweep, development_corpus, development_hash = _suite(tmp_path, "development")
    final_sweep, final_corpus, final_hash = _suite(tmp_path, "final")
    detector = tmp_path / "detector.json"
    onnx = tmp_path / "detector.onnx"
    policy_path = tmp_path / "policy.json"
    policy = select_development(
        development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
        detector, onnx, policy_path,
    )
    assert policy["candidate_selection"]["selection_split"] == "validation"
    assert policy["validation_model_rule_comparison"]["onnx_parity"]["samples_checked"] > 0

    with pytest.raises(ValueError, match="stale"):
        load_frozen_policy(policy_path, expected_development_manifest_sha256="0" * 64)
    malformed = dict(policy)
    malformed["unexpected"] = True
    bad_policy = tmp_path / "bad-policy.json"
    bad_policy.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_frozen_policy(bad_policy)

    forged_policy = json.loads(json.dumps(policy))
    forged_policy["frozen_policy_outcome"]["ai_advantage_demonstrated"] = True
    forged_policy["frozen_policy_outcome"]["preferred_method"] = "temporal_mlp_detector"
    forged_path = tmp_path / "forged-policy.json"
    forged_path.write_text(json.dumps(forged_policy), encoding="utf-8")
    with pytest.raises(ValueError, match="not consistent with validation evidence"):
        final_evaluate(
            final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            forged_path, sha256_file(forged_path), detector, sha256_file(detector), onnx, sha256_file(onnx),
            tmp_path / "forged-final-report.json",
        )

    forged_comparison = json.loads(json.dumps(policy))
    forged_comparison["validation_model_rule_comparison"]["model"]["accuracy"] += 0.01
    comparison = forged_comparison["validation_model_rule_comparison"]
    forged_comparison["frozen_policy_outcome"] = evidence_conclusion(
        comparison["model"],
        comparison["rule_baseline"],
        model_name=forged_comparison["candidate_selection"]["selected_candidate"],
    )
    forged_comparison_path = tmp_path / "forged-comparison-policy.json"
    forged_comparison_path.write_text(json.dumps(forged_comparison), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match recomputed validation evidence"):
        final_evaluate(
            final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            forged_comparison_path, sha256_file(forged_comparison_path),
            detector, sha256_file(detector), onnx, sha256_file(onnx), tmp_path / "forged-comparison-report.json",
        )

    forged_selection = json.loads(json.dumps(policy))
    selected = forged_selection["candidate_selection"]["selected_candidate"]
    substituted = next(name for name in ("softmax_detector", "temporal_mlp_detector") if name != selected)
    forged_selection["candidate_selection"]["selected_candidate"] = substituted
    forged_selection["candidate_selection"]["selected_validation_metrics"] = (
        forged_selection["candidate_selection"]["candidates"][substituted]["validation_metrics"]
    )
    comparison = forged_selection["validation_model_rule_comparison"]
    forged_selection["frozen_policy_outcome"] = evidence_conclusion(
        comparison["model"], comparison["rule_baseline"], model_name=substituted
    )
    forged_selection_path = tmp_path / "forged-selection-policy.json"
    forged_selection_path.write_text(json.dumps(forged_selection), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate selection does not match"):
        final_evaluate(
            final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            forged_selection_path, sha256_file(forged_selection_path),
            detector, sha256_file(detector), onnx, sha256_file(onnx),
            tmp_path / "forged-selection-report.json",
        )

    forged_rule = json.loads(json.dumps(policy))
    forged_rule["rule_parameters"]["residual_threshold"] = 1.0
    forged_rule_path = tmp_path / "forged-rule-policy.json"
    forged_rule_path.write_text(json.dumps(forged_rule), encoding="utf-8")
    with pytest.raises(ValueError, match="rule calibration does not match"):
        final_evaluate(
            final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            forged_rule_path, sha256_file(forged_rule_path),
            detector, sha256_file(detector), onnx, sha256_file(onnx),
            tmp_path / "forged-rule-report.json",
        )

    report_path = tmp_path / "final-report.json"
    report = final_evaluate(
        final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
        development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
        policy_path, sha256_file(policy_path), detector, sha256_file(detector), onnx, sha256_file(onnx), report_path,
    )
    assert report["frozen_policy_outcome"] == policy["frozen_policy_outcome"]
    assert "candidate_selection" not in report

    with pytest.raises(ValueError, match="expected SHA-256"):
        final_evaluate(
            final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            policy_path, sha256_file(policy_path), detector, "0" * 64, onnx, sha256_file(onnx), tmp_path / "stale.json",
        )
    with pytest.raises(ValueError, match="must not already exist"):
        final_evaluate(
            final_corpus / "corpus.jsonl", final_sweep / "families.json", final_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            policy_path, sha256_file(policy_path), detector, sha256_file(detector), onnx, sha256_file(onnx), report_path,
        )
    with pytest.raises(ValueError, match="final manifest must contain exactly"):
        final_evaluate(
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            policy_path, sha256_file(policy_path), detector, sha256_file(detector), onnx, sha256_file(onnx), tmp_path / "non-final.json",
        )

    overlapping_document = json.loads((development_sweep / "families.json").read_text(encoding="utf-8"))
    for family in overlapping_document["families"]:
        family["split"] = "final"
    overlapping_manifest_path = development_sweep / "overlapping-families.json"
    overlapping_manifest_path.write_text(json.dumps(overlapping_document), encoding="utf-8")
    overlapping_corpus = tmp_path / "overlapping-corpus"
    overlapping_info = generate_corpus_v2(overlapping_manifest_path, overlapping_corpus)
    with pytest.raises(ValueError, match="not disjoint"):
        final_evaluate(
            overlapping_corpus / "corpus.jsonl", overlapping_manifest_path,
            str(overlapping_info["family_manifest_sha256"]),
            development_corpus / "corpus.jsonl", development_sweep / "families.json", development_hash,
            policy_path, sha256_file(policy_path), detector, sha256_file(detector), onnx, sha256_file(onnx), tmp_path / "overlap.json",
        )


def test_selection_rejects_retired_or_final_manifest_rows(tmp_path: Path):
    sweep, corpus, manifest_hash = _suite(tmp_path, "development")
    document = json.loads((sweep / "families.json").read_text(encoding="utf-8"))
    for family in document["families"]:
        family["split"] = "final"
    (sweep / "families.json").write_text(json.dumps(document), encoding="utf-8")
    replacement_corpus = tmp_path / "replacement-corpus"
    replacement = generate_corpus_v2(sweep / "families.json", replacement_corpus)
    with pytest.raises(ValueError, match="development manifest must contain exactly"):
        select_development(
            replacement_corpus / "corpus.jsonl", sweep / "families.json",
            str(replacement["family_manifest_sha256"]),
            tmp_path / "detector.json", tmp_path / "detector.onnx", tmp_path / "policy.json",
        )
