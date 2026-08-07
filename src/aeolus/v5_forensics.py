"""Historical-only replay of recorded V5 healthy-reference alert policies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aeolus.alert_forensics import (
    FORENSIC_EVIDENCE_ROLE,
    build_alert_forensics_report,
    canonical_alert_forensics_sha256,
    summarize_forensic_window,
)
from aeolus.baseline import RuleBaseline, RuleParameters
from aeolus.config import HabitatConfig, load_scenario
from aeolus.detector import load_detector
from aeolus.families import FamilyManifest, ScenarioFamily, load_family_manifest
from aeolus.model_cycle_v4 import AlertGate, AlertGateConfig, WINDOW_TICKS
from aeolus.model_input import build_model_input_contract, model_input_v1
from aeolus.scenario import run_scenario
from aeolus.temporal_cnn import load_temporal_cnn
from aeolus.trace import TickRecord, model_feature_row

V5_FORENSIC_BUNDLE_FORMAT = "aeolus_v5_healthy_alert_forensics_v1"
V5_DEVELOPMENT_REPORT_FORMAT = "aeolus_v5_development_evidence_v1"
V5_DEVELOPMENT_REPORT_NAME = "v5-development-report.json"
_V5_METHOD_NAMES = (
    "rule_baseline",
    "temporal_mlp_balanced_raw",
    "temporal_mlp_balanced_gated",
    "temporal_cnn_balanced_gated",
    "temporal_cnn_sqrt_gated",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class _ReplayMethod:
    name: str
    method_sha256: str
    label_stream: Callable[[HabitatConfig, Sequence[TickRecord]], list[str]]


def run_v5_historical_forensics(
    v5_output_dir: str | Path,
    output_dir: str | Path,
    *,
    source_commit: str,
) -> dict[str, object]:
    """Replay exact V5 methods over deduplicated healthy references only.

    The generated evidence is historical forensic material. It deliberately
    cannot be passed to V6 fitting or selection APIs under the V6 boundary.
    """
    _require_commit(source_commit, "source commit")
    source_root = Path(v5_output_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"forensic output directory is not empty: {destination}")

    report_path = source_root / V5_DEVELOPMENT_REPORT_NAME
    report = _load_json_object(report_path, "V5 development report")
    _validate_v5_report(report, source_commit)
    source_provenance = report["source_provenance"]
    assert isinstance(source_provenance, Mapping)
    source_manifest_sha256 = _required_sha(
        source_provenance.get("source_manifest_sha256"), "V5 source manifest SHA-256"
    )

    family_manifest_path = source_root / "sweep" / "families.json"
    families = load_family_manifest(family_manifest_path)
    if families.manifest_sha256 != report["family_manifest_sha256"]:
        raise ValueError("V5 family manifest digest does not match development report")
    if families.contract_metadata != report["contract_metadata"]:
        raise ValueError("V5 family manifest contract does not match development report")

    corpus_manifest = _load_json_object(source_root / "corpus" / "manifest.json", "V5 corpus manifest")
    if corpus_manifest.get("family_manifest_sha256") != families.manifest_sha256:
        raise ValueError("V5 corpus manifest does not bind the recorded family manifest")
    if corpus_manifest.get("manifest_sha256") != report["corpus_manifest_sha256"]:
        raise ValueError("V5 corpus manifest digest does not match development report")

    methods = _build_methods(source_root, report, families.contract_metadata)
    streams = _reference_streams(families)
    if not streams:
        raise ValueError("V5 family manifest contains no healthy reference streams")

    destination.mkdir(parents=True, exist_ok=True)
    methods_dir = destination / "methods"
    methods_dir.mkdir()
    method_receipts: list[dict[str, object]] = []
    for method in methods:
        rows, predictions = _replay_healthy_reference_streams(method, streams)
        forensic_report = build_alert_forensics_report(
            rows,
            predictions,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
            family_manifest_sha256=families.manifest_sha256,
            method_name=method.name,
            method_sha256=method.method_sha256,
        )
        report_filename = f"{method.name}.json"
        _write_json(methods_dir / report_filename, forensic_report)
        method_receipts.append(
            {
                "method_name": method.name,
                "method_sha256": method.method_sha256,
                "report_file": f"methods/{report_filename}",
                "report_sha256": canonical_alert_forensics_sha256(forensic_report),
                "input_row_count": forensic_report["input_row_count"],
                "healthy_alert_episode_count": forensic_report[
                    "healthy_alert_episode_count"
                ],
            }
        )

    bundle_without_digest: dict[str, object] = {
        "format": V5_FORENSIC_BUNDLE_FORMAT,
        "evidence_role": FORENSIC_EVIDENCE_ROLE,
        "source_commit": source_commit,
        "source_manifest_sha256": source_manifest_sha256,
        "family_manifest_sha256": families.manifest_sha256,
        "v5_development_report_sha256": _sha256_file(report_path),
        "v5_corpus_manifest_sha256": _sha256_file(source_root / "corpus" / "manifest.json"),
        "reference_stream_count": len(streams),
        "window_ticks": WINDOW_TICKS,
        "window_stride_ticks": 1,
        "methods": method_receipts,
    }
    bundle = {
        **bundle_without_digest,
        "bundle_sha256": _canonical_sha256(bundle_without_digest),
    }
    _write_json(destination / "v5-healthy-alert-forensics.json", bundle)
    return bundle


def _build_methods(
    source_root: Path,
    report: Mapping[str, Any],
    contract_metadata: Mapping[str, str],
) -> tuple[_ReplayMethod, ...]:
    rule_baseline = report.get("rule_baseline")
    candidates = report.get("candidates")
    models = report.get("models")
    if not isinstance(rule_baseline, Mapping):
        raise ValueError("V5 report rule baseline is malformed")
    if not isinstance(candidates, Mapping) or not isinstance(models, Mapping):
        raise ValueError("V5 report model receipts are malformed")

    parameters = rule_baseline.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("V5 report rule parameters are malformed")
    try:
        rule_parameters = RuleParameters(**dict(parameters))
    except (TypeError, ValueError) as exc:
        raise ValueError("V5 report rule parameters are incompatible") from exc
    rule_receipt = {"method": "rule_baseline", "parameters": rule_parameters.as_dict()}

    def label_rules(config: HabitatConfig, records: Sequence[TickRecord]) -> list[str]:
        detector = RuleBaseline(config, rule_parameters)
        return [
            detector.label_window([model_feature_row(record) for record in window])
            for window in _rolling_windows(records)
        ]

    methods = [
        _ReplayMethod(
            name="rule_baseline",
            method_sha256=_canonical_sha256(rule_receipt),
            label_stream=label_rules,
        )
    ]
    for candidate_name in _V5_METHOD_NAMES[1:]:
        candidate = candidates.get(candidate_name)
        if not isinstance(candidate, Mapping):
            raise ValueError(f"V5 candidate receipt missing: {candidate_name}")
        base_model = candidate.get("base_model")
        gate_document = candidate.get("gate")
        if not isinstance(base_model, str) or not isinstance(models.get(base_model), Mapping):
            raise ValueError(f"V5 candidate model receipt is malformed: {candidate_name}")
        model_receipt = models[base_model]
        assert isinstance(model_receipt, Mapping)
        expected_sha = _required_sha(model_receipt.get("json_sha256"), "V5 model JSON SHA-256")
        model_path = source_root / "models" / f"{base_model}.json"
        if _sha256_file(model_path) != expected_sha:
            raise ValueError(f"V5 model artifact digest drifted: {base_model}")
        detector = (
            load_temporal_cnn(model_path, expected_contract=contract_metadata)
            if base_model.startswith("temporal_cnn_")
            else load_detector(model_path, expected_contract=contract_metadata)
        )
        gate = _parse_gate(gate_document, candidate_name)
        receipt = {
            "method": candidate_name,
            "model_name": base_model,
            "model_json_sha256": expected_sha,
            "gate": None if gate is None else gate.as_dict(),
        }
        methods.append(
            _ReplayMethod(
                name=candidate_name,
                method_sha256=_canonical_sha256(receipt),
                label_stream=_model_labeller(detector, gate),
            )
        )
    return tuple(methods)


def _model_labeller(detector: Any, gate: AlertGateConfig | None) -> Callable[[HabitatConfig, Sequence[TickRecord]], list[str]]:
    def label(config: HabitatConfig, records: Sequence[TickRecord]) -> list[str]:
        contract = build_model_input_contract(config)
        windows = [
            [model_input_v1(record, contract).tolist() for record in window]
            for window in _rolling_windows(records)
        ]
        if gate is None:
            return [detector.label_window(window) for window in windows]
        policy = AlertGate(detector, gate)
        return [policy.label_window(window) for window in windows]

    return label


def _parse_gate(document: object, candidate_name: str) -> AlertGateConfig | None:
    if candidate_name.endswith("_raw"):
        if document is not None:
            raise ValueError(f"raw V5 candidate unexpectedly has a gate: {candidate_name}")
        return None
    if not isinstance(document, Mapping):
        raise ValueError(f"gated V5 candidate gate is malformed: {candidate_name}")
    try:
        return AlertGateConfig(
            fault_threshold=document["fault_threshold"],
            persistence_windows=document["persistence_windows"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gated V5 candidate gate is malformed: {candidate_name}") from exc


def _reference_streams(manifest: FamilyManifest) -> tuple[tuple[str, ScenarioFamily, tuple[str, ...]], ...]:
    grouped: dict[str, list[ScenarioFamily]] = {}
    for family in manifest.families:
        reference_sha = _canonical_scenario_sha256(family.reference_path)
        grouped.setdefault(reference_sha, []).append(family)
    return tuple(
        (
            reference_sha,
            min(families, key=lambda family: family.family_id),
            tuple(family.family_id for family in sorted(families, key=lambda family: family.family_id)),
        )
        for reference_sha, families in sorted(grouped.items())
    )


def _replay_healthy_reference_streams(
    method: _ReplayMethod,
    streams: Sequence[tuple[str, ScenarioFamily, tuple[str, ...]]],
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    predictions: list[str] = []
    for reference_sha, representative, related_family_ids in streams:
        config = load_scenario(representative.reference_path)
        records = run_scenario(config)
        windows = _rolling_windows(records)
        labels = method.label_stream(config, records)
        if len(windows) != len(labels):
            raise ValueError(f"V5 replay method did not label every healthy window: {method.name}")
        stream_id = f"reference:{reference_sha}"
        for window, label in zip(windows, labels, strict=True):
            rows.append(
                {
                    "family_id": representative.family_id,
                    "scenario_role": "reference",
                    "stream_id": stream_id,
                    "operating_profile_id": representative.reference_path.stem,
                    "start_tick": window[0].tick,
                    "end_tick": window[-1].tick,
                    "context": {
                        **summarize_forensic_window(window),
                        "related_family_ids": list(related_family_ids),
                    },
                }
            )
            predictions.append(label)
    return rows, predictions


def _rolling_windows(records: Sequence[TickRecord]) -> list[Sequence[TickRecord]]:
    if len(records) < WINDOW_TICKS:
        raise ValueError("V5 reference trace is shorter than the recorded window width")
    return [
        records[start : start + WINDOW_TICKS]
        for start in range(0, len(records) - WINDOW_TICKS + 1)
    ]


def _validate_v5_report(report: Mapping[str, Any], source_commit: str) -> None:
    if report.get("schema_version") != V5_DEVELOPMENT_REPORT_FORMAT:
        raise ValueError("forensics requires a V5 development evidence report")
    if report.get("evidence_role") != "development_only":
        raise ValueError("forensics requires V5 development-only evidence")
    if report.get("development_gate_passed") is not False:
        raise ValueError("forensics refuses a report that claims the V5 gate passed")
    provenance = report.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V5 development report source provenance is malformed")
    if provenance.get("head_commit") != source_commit:
        raise ValueError("requested forensic source commit does not match V5 evidence")
    if provenance.get("worktree_dirty") is not False:
        raise ValueError("V5 forensic replay requires a clean recorded worktree")
    _required_sha(provenance.get("source_manifest_sha256"), "V5 source manifest SHA-256")
    _required_sha(report.get("family_manifest_sha256"), "V5 family manifest SHA-256")
    _required_sha(report.get("corpus_manifest_sha256"), "V5 corpus manifest SHA-256")
    if not isinstance(report.get("contract_metadata"), Mapping):
        raise ValueError("V5 development report contract metadata is malformed")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {constant}")
            ),
        )
    except FileNotFoundError:
        raise ValueError(f"{description} is missing: {path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{description} is unreadable: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_scenario_sha256(path: Path) -> str:
    document = _load_json_object(path, "reference scenario")
    return _canonical_sha256(document)


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        raise ValueError(f"required V5 artifact is missing: {path}") from None
    except OSError as exc:
        raise ValueError(f"cannot read V5 artifact {path}: {exc}") from None


def _required_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: object, description: str) -> None:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase git commit")
