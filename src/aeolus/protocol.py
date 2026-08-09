"""Frozen v3 development-selection and final-only evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from aeolus.baseline import RuleBaseline, RuleParameters
from aeolus.corpus import generate_corpus_v2
from aeolus.detector import (
    Detector,
    calibrate_rule_baseline,
    detector_serialized_size,
    enforce_onnx_parity,
    evaluate_detector,
    evaluate_rule_baseline,
    evidence_conclusion,
    export_onnx,
    load_detector,
    load_verified_corpus,
    save_detector,
    train_softmax_detector,
    train_temporal_mlp_detector,
    validate_onnx_parity,
)
from aeolus.families import (
    FamilyManifest,
    build_family_evidence,
    validate_manifest_disjointness,
)
from aeolus.config import load_scenario
from aeolus.sweep import SWEEP_V3_VERSION, generate_sweep, load_sweep_spec

POLICY_FORMAT = "aeolus_frozen_policy_v3"
_POLICY_KEYS = frozenset(
    {
        "format",
        "source_development_manifest_sha256",
        "contract_metadata",
        "detector_json_sha256",
        "rule_parameters",
        "candidate_selection",
        "validation_model_rule_comparison",
        "frozen_policy_outcome",
    }
)
_CONTRACT_KEYS = frozenset({"model_input_version", "selector_sha256", "topology_sha256"})
_RULE_PARAMETER_KEYS = frozenset(
    {
        "residual_threshold",
        "isolation_margin",
        "blockage_jump",
        "frozen_normalized_range",
        "persistence_ticks",
    }
)
_CANDIDATE_NAMES = frozenset({"softmax_detector", "temporal_mlp_detector"})
_SELECTION_ORDER = [
    "macro_f1_descending",
    "cross_entropy_ascending",
    "serialized_json_size_ascending",
    "candidate_name_lexicographic",
]


def sha256_file(path: str | Path) -> str:
    """Return the exact-byte SHA-256 of a persisted protocol artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_finite_json(value: object, description: str) -> None:
    """Reject non-finite numeric evidence at every nested policy boundary."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite number")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _require_finite_json(nested, f"{description}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json(nested, f"{description}[{index}]")


def _select_candidate(
    train_rows: list[dict], validation_rows: list[dict], contract_metadata: Mapping[str, str]
) -> tuple[str, Detector, dict[str, Any]]:
    """Rebuild the deterministic validation-selected candidate and its receipt."""
    softmax, softmax_receipt = train_softmax_detector(
        train_rows, validation_rows, contract_metadata=dict(contract_metadata)
    )
    temporal, temporal_receipt = train_temporal_mlp_detector(
        train_rows, validation_rows, contract_metadata=dict(contract_metadata)
    )
    candidates: list[tuple[str, Detector, dict[str, Any], dict[str, Any], int]] = []
    for name, detector, receipt in (
        ("softmax_detector", softmax, softmax_receipt),
        ("temporal_mlp_detector", temporal, temporal_receipt),
    ):
        metrics = evaluate_detector(detector, validation_rows)
        candidates.append((name, detector, receipt, metrics, detector_serialized_size(detector)))
    selected_name, detector, _, selected_validation, _ = min(
        candidates,
        key=lambda item: (
            -float(item[3]["macro_f1"]),
            float(item[3]["cross_entropy"]),
            item[4],
            item[0],
        ),
    )
    return selected_name, detector, {
        "selection_split": "validation",
        "selection_order": _SELECTION_ORDER,
        "selected_candidate": selected_name,
        "selected_validation_metrics": selected_validation,
        "candidates": {
            name: {
                "training": receipt,
                "validation_metrics": metrics,
                "serialized_json_size_bytes": size,
            }
            for name, _, receipt, metrics, size in candidates
        },
    }


def select_development(
    corpus_path: str | Path,
    family_manifest_path: str | Path,
    expected_family_manifest_sha256: str,
    detector_json_path: str | Path,
    detector_onnx_path: str | Path,
    policy_path: str | Path,
) -> dict[str, Any]:
    """Select and freeze a detector/rule policy using train and validation only."""
    rows, manifest = load_verified_corpus(
        corpus_path, family_manifest_path, expected_family_manifest_sha256
    )
    _require_exact_manifest_splits(manifest, {"train", "validation"}, "development")
    _require_new_outputs(detector_json_path, detector_onnx_path, policy_path)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation")}
    if not split_rows["train"] or not split_rows["validation"]:
        raise ValueError("development selection requires non-empty train and validation rows")

    selected_name, detector, candidate_selection = _select_candidate(
        split_rows["train"], split_rows["validation"], manifest.contract_metadata
    )
    evidence = build_family_evidence(manifest)
    validation_evidence = {
        family_id: item for family_id, item in evidence.items() if item.split == "validation"
    }
    baseline_config = load_scenario(manifest.families[0].reference_path)
    rule_parameters, rule_calibration = calibrate_rule_baseline(
        split_rows["validation"], baseline_config, validation_evidence
    )
    save_detector(detector, detector_json_path)
    export_onnx(detector, detector_onnx_path)
    parity = validate_onnx_parity(detector, detector_onnx_path, split_rows["validation"])
    enforce_onnx_parity(parity)
    model_validation = evaluate_detector(
        detector, split_rows["validation"], family_evidence=validation_evidence
    )
    rule_validation = evaluate_rule_baseline(
        split_rows["validation"],
        RuleBaseline(baseline_config, rule_parameters),
        family_evidence=validation_evidence,
    )
    frozen_outcome = evidence_conclusion(
        model_validation, rule_validation, model_name=selected_name
    )
    policy: dict[str, Any] = {
        "format": POLICY_FORMAT,
        "source_development_manifest_sha256": manifest.manifest_sha256,
        "contract_metadata": dict(manifest.contract_metadata),
        "detector_json_sha256": sha256_file(detector_json_path),
        "rule_parameters": rule_parameters.as_dict(),
        "candidate_selection": candidate_selection,
        "validation_model_rule_comparison": {
            "model": model_validation,
            "rule_baseline": rule_validation,
            "onnx_parity": parity,
            "rule_calibration": rule_calibration,
        },
        "frozen_policy_outcome": frozen_outcome,
    }
    _validate_policy(policy)
    _write_json_new(policy_path, policy)
    return policy


def build_final(sweep_spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Generate the independent final suite and its final-only corpus."""
    spec = load_sweep_spec(sweep_spec_path)
    if spec.schema_version != SWEEP_V3_VERSION or spec.suite_role != "final":
        raise ValueError("build-final requires an aeolus_sweep_v3 final suite")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"final output directory is not empty: {output}")
    sweep_dir = output / "sweep"
    corpus_dir = output / "corpus"
    receipt = generate_sweep(sweep_spec_path, sweep_dir)
    corpus = generate_corpus_v2(sweep_dir / "families.json", corpus_dir)
    return {"sweep": receipt, "corpus_manifest_sha256": corpus["manifest_sha256"]}


def final_evaluate(
    final_corpus_path: str | Path,
    final_family_manifest_path: str | Path,
    expected_final_manifest_sha256: str,
    development_corpus_path: str | Path,
    development_family_manifest_path: str | Path,
    expected_development_manifest_sha256: str,
    policy_path: str | Path,
    expected_policy_sha256: str,
    detector_json_path: str | Path,
    expected_detector_json_sha256: str,
    detector_onnx_path: str | Path,
    expected_detector_onnx_sha256: str,
    report_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one frozen policy once on an isolated final-only corpus."""
    _require_sha256(expected_policy_sha256, "expected policy hash")
    _require_sha256(expected_detector_json_sha256, "expected detector hash")
    _require_sha256(expected_detector_onnx_sha256, "expected ONNX hash")
    _require_sha256(expected_development_manifest_sha256, "expected development manifest hash")
    if sha256_file(policy_path) != expected_policy_sha256:
        raise ValueError("policy artifact does not match the expected SHA-256")
    if sha256_file(detector_json_path) != expected_detector_json_sha256:
        raise ValueError("detector artifact does not match the expected SHA-256")
    if sha256_file(detector_onnx_path) != expected_detector_onnx_sha256:
        raise ValueError("ONNX artifact does not match the expected SHA-256")
    report = Path(report_path)
    if report.exists():
        raise ValueError("final report output must not already exist")

    rows, final_manifest = load_verified_corpus(
        final_corpus_path, final_family_manifest_path, expected_final_manifest_sha256
    )
    _require_exact_manifest_splits(final_manifest, {"final"}, "final")
    development_rows, development_manifest = load_verified_corpus(
        development_corpus_path,
        development_family_manifest_path,
        expected_development_manifest_sha256,
    )
    _require_exact_manifest_splits(
        development_manifest, {"train", "validation"}, "development"
    )
    validate_manifest_disjointness(development_manifest, final_manifest)
    policy = load_frozen_policy(
        policy_path,
        expected_policy_sha256=expected_policy_sha256,
        expected_development_manifest_sha256=expected_development_manifest_sha256,
        expected_detector_json_sha256=expected_detector_json_sha256,
        expected_contract=final_manifest.contract_metadata,
    )
    detector = load_detector(detector_json_path, expected_contract=final_manifest.contract_metadata)
    split_rows = {
        split: [row for row in development_rows if row["split"] == split]
        for split in ("train", "validation")
    }
    _selected_name, selected_detector, expected_selection = _select_candidate(
        split_rows["train"], split_rows["validation"], development_manifest.contract_metadata
    )
    if policy["candidate_selection"] != expected_selection:
        raise ValueError("policy candidate selection does not match recomputed development evidence")
    if detector != selected_detector:
        raise ValueError("detector artifact does not match the recomputed selected candidate")

    development_evidence = build_family_evidence(development_manifest)
    validation_evidence = {
        family_id: item
        for family_id, item in development_evidence.items()
        if item.split == "validation"
    }
    validation_rows = split_rows["validation"]
    recomputed_parity = validate_onnx_parity(detector, detector_onnx_path, validation_rows)
    enforce_onnx_parity(recomputed_parity)
    saved_comparison = policy["validation_model_rule_comparison"]
    if saved_comparison["onnx_parity"] != recomputed_parity:
        raise ValueError("policy ONNX parity does not match recomputed validation evidence")
    validation_config = load_scenario(development_manifest.families[0].reference_path)
    recomputed_parameters, recomputed_calibration = calibrate_rule_baseline(
        validation_rows, validation_config, validation_evidence
    )
    saved_comparison = policy["validation_model_rule_comparison"]
    if (
        policy["rule_parameters"] != recomputed_parameters.as_dict()
        or saved_comparison["rule_calibration"] != recomputed_calibration
    ):
        raise ValueError("policy rule calibration does not match recomputed validation evidence")
    recomputed_validation_model = evaluate_detector(
        detector, validation_rows, family_evidence=validation_evidence
    )
    recomputed_validation_rule = evaluate_rule_baseline(
        validation_rows,
        RuleBaseline(validation_config, recomputed_parameters),
        family_evidence=validation_evidence,
    )
    saved_comparison = policy["validation_model_rule_comparison"]
    if (
        saved_comparison["model"] != recomputed_validation_model
        or saved_comparison["rule_baseline"] != recomputed_validation_rule
    ):
        raise ValueError("policy does not match recomputed validation evidence")

    evidence = build_family_evidence(final_manifest)
    config = load_scenario(final_manifest.families[0].reference_path)
    model_metrics = evaluate_detector(detector, rows, family_evidence=evidence)
    rule_metrics = evaluate_rule_baseline(
        rows, RuleBaseline(config, recomputed_parameters), family_evidence=evidence
    )
    result: dict[str, Any] = {
        "format": "aeolus_final_evaluation_v3",
        "final_family_manifest_sha256": final_manifest.manifest_sha256,
        "development_family_manifest_sha256": development_manifest.manifest_sha256,
        "policy_sha256": expected_policy_sha256,
        "detector_json_sha256": expected_detector_json_sha256,
        "detector_onnx_sha256": expected_detector_onnx_sha256,
        "contract_metadata": dict(final_manifest.contract_metadata),
        "model": model_metrics,
        "rule_baseline": rule_metrics,
        "frozen_policy_outcome": policy["frozen_policy_outcome"],
    }
    _write_json_new(report, result)
    return result


def load_frozen_policy(
    path: str | Path,
    *,
    expected_policy_sha256: str | None = None,
    expected_development_manifest_sha256: str | None = None,
    expected_detector_json_sha256: str | None = None,
    expected_contract: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load a fail-closed strict policy and bind it to supplied artifacts."""
    source = Path(path)
    if expected_policy_sha256 is not None:
        _require_sha256(expected_policy_sha256, "expected policy hash")
        if sha256_file(source) != expected_policy_sha256:
            raise ValueError("policy artifact does not match the expected SHA-256")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"policy artifact not found: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"policy artifact is not valid JSON: {exc}") from None
    _validate_policy(document)
    if expected_development_manifest_sha256 is not None and document["source_development_manifest_sha256"] != expected_development_manifest_sha256:
        raise ValueError("policy development manifest hash is stale")
    if expected_detector_json_sha256 is not None and document["detector_json_sha256"] != expected_detector_json_sha256:
        raise ValueError("policy detector hash is stale")
    if expected_contract is not None and document["contract_metadata"] != dict(expected_contract):
        raise ValueError("policy contract does not match the final suite")
    return document


def _require_exact_manifest_splits(manifest: FamilyManifest, expected: set[str], role: str) -> None:
    actual = {family.split for family in manifest.families}
    if actual != expected:
        raise ValueError(f"{role} manifest must contain exactly splits {sorted(expected)!r}")


def _require_new_outputs(*paths: str | Path) -> None:
    for raw in paths:
        if Path(raw).exists():
            raise ValueError(f"protocol output already exists: {raw}")


def _write_json_new(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError(f"protocol output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _require_sha256(value: object, description: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{description} must be lowercase SHA-256")


def _validate_policy(document: object) -> None:
    if not isinstance(document, dict) or set(document) != _POLICY_KEYS:
        raise ValueError("policy artifact schema is incompatible")
    if document["format"] != POLICY_FORMAT:
        raise ValueError("policy artifact format is incompatible")
    _require_sha256(document["source_development_manifest_sha256"], "policy development manifest hash")
    _require_sha256(document["detector_json_sha256"], "policy detector hash")
    contract = document["contract_metadata"]
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS or any(not isinstance(value, str) for value in contract.values()):
        raise ValueError("policy contract metadata is malformed")
    _require_sha256(contract["selector_sha256"], "policy selector hash")
    _require_sha256(contract["topology_sha256"], "policy topology hash")
    if contract["model_input_version"] != "model_input_v1":
        raise ValueError("policy model input version is incompatible")
    parameters = document["rule_parameters"]
    if not isinstance(parameters, dict) or set(parameters) != _RULE_PARAMETER_KEYS:
        raise ValueError("policy rule parameters are malformed")
    try:
        RuleParameters(**parameters)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy rule parameters are malformed") from exc
    selection = document["candidate_selection"]
    comparison = document["validation_model_rule_comparison"]
    outcome = document["frozen_policy_outcome"]
    selection_keys = {
        "selection_split", "selection_order", "selected_candidate",
        "selected_validation_metrics", "candidates",
    }
    if (
        not isinstance(selection, dict)
        or set(selection) != selection_keys
        or selection["selection_split"] != "validation"
        or selection["selection_order"] != _SELECTION_ORDER
        or selection["selected_candidate"] not in _CANDIDATE_NAMES
        or not isinstance(selection["selected_validation_metrics"], dict)
        or not isinstance(selection["candidates"], dict)
        or set(selection["candidates"]) != _CANDIDATE_NAMES
    ):
        raise ValueError("policy candidate selection is malformed")
    selected = selection["selected_candidate"]
    for candidate in selection["candidates"].values():
        if (
            not isinstance(candidate, dict)
            or set(candidate)
            != {"training", "validation_metrics", "serialized_json_size_bytes"}
            or not isinstance(candidate["training"], dict)
            or not isinstance(candidate["validation_metrics"], dict)
            or isinstance(candidate["serialized_json_size_bytes"], bool)
            or not isinstance(candidate["serialized_json_size_bytes"], int)
            or candidate["serialized_json_size_bytes"] < 0
        ):
            raise ValueError("policy candidate selection is malformed")
    if (
        selection["selected_validation_metrics"]
        != selection["candidates"][selected]["validation_metrics"]
    ):
        raise ValueError("policy selected metrics do not match the selected candidate")
    if (
        not isinstance(comparison, dict)
        or set(comparison)
        != {"model", "rule_baseline", "onnx_parity", "rule_calibration"}
        or any(not isinstance(comparison[key], dict) for key in comparison)
    ):
        raise ValueError("policy validation comparison is malformed")
    expected_outcome = evidence_conclusion(
        comparison["model"], comparison["rule_baseline"], model_name=selected
    )
    if outcome != expected_outcome:
        raise ValueError("policy frozen outcome is not consistent with validation evidence")
    _require_finite_json(selection, "policy candidate selection")
    _require_finite_json(comparison, "policy validation comparison")
    _require_finite_json(outcome, "policy frozen outcome")
