from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest


def _record_hash(record: dict[str, object], field: str = "record_sha256") -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    body = dict(record)
    body.pop(field, None)
    record[field] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes

    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_jsonl_bytes

    path.write_bytes(canonical_jsonl_bytes(rows))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _refresh_manifest_artifact(packet: Path, relative_path: str) -> None:
    manifest_path = packet / "development-fixture-only" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = (packet / relative_path).read_bytes()
    for artifact in manifest["table_artifacts"]:
        if artifact["relative_path"] == relative_path:
            artifact["byte_length"] = len(raw)
            artifact["sha256"] = hashlib.sha256(raw).hexdigest()
            break
    else:
        raise AssertionError(f"missing manifest artifact {relative_path}")
    _record_hash(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)


@pytest.fixture(scope="module")
def generated_packets(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    from aeolus.habitat_v2.forecast.pipeline import generate_development_fixture

    repo_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path_factory.mktemp("forecast-packets")
    first = generate_development_fixture(repo_root, output_root, "packet-a")
    second = generate_development_fixture(repo_root, output_root, "packet-b")
    return repo_root, output_root, first, second


def test_generates_and_replays_atomic_development_fixture(
    generated_packets: tuple[Path, Path, dict[str, object], dict[str, object]],
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pipeline import (
        ForecastPipelineError,
        generate_development_fixture,
        validate_development_packet,
    )

    root, output_root, first, second = generated_packets
    bundle = load_forecast_contracts(root)
    packet = output_root / "packet-a"
    sample = json.loads(
        (packet / "development-fixture-only" / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert first["release_tier"] == "DEVELOPMENT_FIXTURE_ONLY"
    assert first["sample_count"] == 4
    assert first["shadow_receipt_matches"] == 96
    assert first["file_sha256"] == second["file_sha256"]
    assert validate_development_packet(packet, bundle)["strict_trace_replays"] == 4
    assert set(sample["input_tensors"]) == {
        "history_numeric",
        "history_availability",
        "history_mode_one_hot",
        "history_health_one_hot",
        "history_alarm_lifecycle_one_hot",
        "history_final_command",
        "proposed_action",
    }
    assert len(sample["input_tensors"]["history_numeric"]) == 4
    assert len(sample["input_tensors"]["history_numeric"][0]) == 194
    assert len(sample["target_truth"]) == 2
    assert len(sample["target_truth"][0]) == 51
    with pytest.raises(ForecastPipelineError, match="destination"):
        generate_development_fixture(root, output_root, "packet-a")


def test_packet_rejects_self_consistent_manifest_and_nested_numeric_drift(
    generated_packets: tuple[Path, Path, dict[str, object], dict[str, object]],
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.pipeline import (
        ForecastPipelineError,
        validate_development_packet,
    )

    root, output_root, _, _ = generated_packets
    bundle = load_forecast_contracts(root)

    manifest_packet = output_root / "tampered-manifest"
    shutil.copytree(output_root / "packet-a", manifest_packet)
    manifest_path = manifest_packet / "development-fixture-only" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hmc_binding_sha256"] = "0" * 64
    _record_hash(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    with pytest.raises(ForecastPipelineError, match="manifest"):
        validate_development_packet(manifest_packet, bundle)

    numeric_packet = output_root / "tampered-numeric"
    shutil.copytree(output_root / "packet-a", numeric_packet)
    samples_path = numeric_packet / "development-fixture-only" / "samples.jsonl"
    samples = _load_jsonl(samples_path)
    samples[0]["target_truth"][0][0] = True
    _record_hash(samples[0])
    _write_jsonl(samples_path, samples)
    _refresh_manifest_artifact(numeric_packet, "development-fixture-only/samples.jsonl")
    with pytest.raises(ForecastPipelineError, match="numeric|boolean"):
        validate_development_packet(numeric_packet, bundle)


def test_packet_rejects_nested_unknown_and_causally_forged_witness(
    generated_packets: tuple[Path, Path, dict[str, object], dict[str, object]],
) -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes, record_identity
    from aeolus.habitat_v2.forecast.pipeline import (
        ForecastPipelineError,
        validate_development_packet,
    )

    root, output_root, _, _ = generated_packets
    bundle = load_forecast_contracts(root)
    witness_spec = bundle.record_contract["records"]["replay_witnesses"]

    nested_packet = output_root / "tampered-nested"
    shutil.copytree(output_root / "packet-a", nested_packet)
    witness_path = nested_packet / "development-fixture-only" / "replay_witnesses.jsonl"
    witnesses = _load_jsonl(witness_path)
    witnesses[0]["step_witnesses"][0]["unexpected"] = "self-consistent-extra"
    _record_hash(witnesses[0])
    _write_jsonl(witness_path, witnesses)
    _refresh_manifest_artifact(
        nested_packet, "development-fixture-only/replay_witnesses.jsonl"
    )
    with pytest.raises(ForecastPipelineError, match="nested|unknown|witness"):
        validate_development_packet(nested_packet, bundle)

    causal_packet = output_root / "tampered-causal"
    shutil.copytree(output_root / "packet-a", causal_packet)
    witness_path = causal_packet / "development-fixture-only" / "replay_witnesses.jsonl"
    witnesses = _load_jsonl(witness_path)
    old_witness_id = witnesses[0]["replay_witness_id"]
    witnesses[0]["step_witnesses"][0]["shadow_plant_receipt_digest"] = "0" * 64
    witnesses[0]["step_witnesses_sha256"] = hashlib.sha256(
        canonical_json_bytes(witnesses[0]["step_witnesses"])
    ).hexdigest()
    witnesses[0]["replay_witness_id"] = record_identity(witnesses[0], witness_spec)
    new_witness_id = witnesses[0]["replay_witness_id"]
    _record_hash(witnesses[0])
    _write_jsonl(witness_path, witnesses)
    _refresh_manifest_artifact(
        causal_packet, "development-fixture-only/replay_witnesses.jsonl"
    )

    for relative in (
        "development-fixture-only/control_runs.jsonl",
        "development-fixture-only/samples.jsonl",
    ):
        path = causal_packet / relative
        rows = _load_jsonl(path)
        for row in rows:
            if row.get("replay_witness_id") == old_witness_id:
                row["replay_witness_id"] = new_witness_id
                _record_hash(row)
        _write_jsonl(path, rows)
        _refresh_manifest_artifact(causal_packet, relative)

    with pytest.raises(ForecastPipelineError, match="causal|witness|replay"):
        validate_development_packet(causal_packet, bundle)


def test_generation_rejects_unsafe_destination_name(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pipeline import (
        ForecastPipelineError,
        generate_development_fixture,
    )

    with pytest.raises(ForecastPipelineError, match="destination"):
        generate_development_fixture(
            Path(__file__).resolve().parents[2], tmp_path, "../escape"
        )
