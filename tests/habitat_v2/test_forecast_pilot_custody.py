from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _design():
    from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design

    return load_approved_pilot_design(ROOT)


def _pair_continuations(design):
    from aeolus.habitat_v2.forecast.pilot import iter_pilot_continuations

    items = iter_pilot_continuations(design)
    control = next(items)
    actions = tuple(
        next(items) for _ in range(len(design.action_ids))
    )
    assert control.variant == "MATCHED_CONTROL"
    assert all(item.pair_id == control.pair_id for item in actions)
    return control, actions


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _resign(record: dict) -> None:
    from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes

    body = dict(record)
    body.pop("record_sha256", None)
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _control_bundle(continuation):
    from aeolus.habitat_v2.forecast.pilot_execution import PilotControlRunBundle

    return PilotControlRunBundle(
        continuation_id=continuation.continuation_id,
        pair_id=continuation.pair_id,
        matched_control_id=continuation.matched_control_id,
        cluster_id=continuation.cluster_id,
        repetition_id=continuation.repetition_id,
        member_id=continuation.member_id,
        anchor_completed_step=continuation.anchor_completed_step,
        noise_seed=continuation.noise_seed,
        hmc_reset_nonce_hex=continuation.hmc_reset_nonce_hex,
        scenario_sha256=_sha("scenario"),
        control_run_id=_sha("control-run"),
        authority_epoch=_sha("epoch"),
        trace_canonical_bytes=b"control-trace-bytes",
        trace_sha256=_sha("control-trace"),
        trace_final_state_sha256=_sha("control-final"),
        replay_final_state_sha256=_sha("control-final"),
        committed_step_count=72,
        witnesses=(
            {"application_step": step, "final_command_sha256": _sha(f"cmd-{step}")}
            for step in range(72)
        ),
        snapshots={},
        states={},
    )


def _action_bundle(continuation):
    from aeolus.habitat_v2.forecast.pilot_execution import PilotActionRunBundle

    return PilotActionRunBundle(
        continuation_id=continuation.continuation_id,
        pair_id=continuation.pair_id,
        matched_control_id=continuation.matched_control_id,
        cluster_id=continuation.cluster_id,
        repetition_id=continuation.repetition_id,
        member_id=continuation.member_id,
        anchor_completed_step=continuation.anchor_completed_step,
        noise_seed=continuation.noise_seed,
        hmc_reset_nonce_hex=continuation.hmc_reset_nonce_hex,
        scenario_sha256=_sha("scenario"),
        control_run_id=_sha("control-run"),
        authority_epoch=_sha("epoch"),
        trace_canonical_bytes=b"action-trace-bytes",
        trace_sha256=_sha(f"action-trace-{continuation.action_id}"),
        trace_final_state_sha256=_sha(f"action-final-{continuation.action_id}"),
        replay_final_state_sha256=_sha(f"action-final-{continuation.action_id}"),
        committed_step_count=72,
        witnesses=(
            {"application_step": step, "final_command_sha256": _sha(f"cmd-{step}")}
            for step in range(72)
        ),
        snapshots={},
        states={},
        action_id=continuation.action_id,
        requested_command_sha256=_sha(f"command-{continuation.action_id}"),
        anchor={
            "application_step": continuation.anchor_completed_step,
            "proposal_receipt": {
                "attempt_class": "CANONICAL_PROPOSAL",
                "validation_outcome": "VALID",
            },
            "final_command_sha256": _sha(f"final-{continuation.action_id}"),
        },
    )


def _pair_records(design):
    from aeolus.habitat_v2.forecast.pilot_custody import build_run_record

    control, actions = _pair_continuations(design)
    records = [build_run_record(design, control, _control_bundle(control))]
    records += [
        build_run_record(design, continuation, _action_bundle(continuation))
        for continuation in actions
    ]
    return records


def test_run_record_is_canonical_self_hashed_and_plan_bound() -> None:
    from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes
    from aeolus.habitat_v2.forecast.pilot_custody import (
        build_run_record,
        validate_run_record,
    )

    design = _design()
    control, _ = _pair_continuations(design)
    record = build_run_record(design, control, _control_bundle(control))

    body = dict(record)
    declared = body.pop("record_sha256")
    assert declared == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert record["schema_version"] == "aeolus_habitat_v2_forecast_pilot_run_record_v1"
    assert record["variant"] == "MATCHED_CONTROL"
    assert record["matched_control_id"] == record["continuation_id"]
    assert record["anchor"] is None
    assert len(record["witnesses"]) == 72
    validate_run_record(record)
    again = build_run_record(design, control, _control_bundle(control))
    assert canonical_json_bytes(record) == canonical_json_bytes(again)


def test_run_record_refuses_bundle_continuation_mismatch() -> None:
    from dataclasses import replace

    from aeolus.habitat_v2.forecast.pilot_custody import (
        PilotCustodyError,
        build_run_record,
    )

    design = _design()
    control, _ = _pair_continuations(design)
    bundle = replace(_control_bundle(control), scenario_sha256=_sha("other-scenario"))
    # scenario is not plan-derived, so agreement holds; mutate a plan field instead
    bad_bundle = replace(
        _control_bundle(control), noise_seed=control.noise_seed + 1
    )
    with pytest.raises(PilotCustodyError, match="continuation"):
        build_run_record(design, control, bad_bundle)
    record = build_run_record(design, control, bundle)
    assert record["scenario_sha256"] == _sha("other-scenario")


def test_pair_validation_accepts_one_control_and_four_actions() -> None:
    from aeolus.habitat_v2.forecast.pilot_custody import validate_pilot_pair

    design = _design()
    records = _pair_records(design)
    validate_pilot_pair(design, records)
    # Order independence: shuffled records still validate.
    validate_pilot_pair(design, list(reversed(records)))


def test_pair_validation_rejects_incomplete_or_divergent_sets() -> None:
    from aeolus.habitat_v2.forecast.pilot_custody import (
        PilotCustodyError,
        validate_pilot_pair,
    )

    design = _design()
    records = _pair_records(design)

    with pytest.raises(PilotCustodyError, match="action"):
        validate_pilot_pair(design, records[:1])
    with pytest.raises(PilotCustodyError, match="action"):
        validate_pilot_pair(design, records[:4])
    with pytest.raises(PilotCustodyError, match="duplicate run identities"):
        validate_pilot_pair(design, records + [records[0]])

    from aeolus.habitat_v2.forecast.pilot import iter_pilot_continuations
    from aeolus.habitat_v2.forecast.pilot_custody import build_run_record

    second_pair_control = next(
        item
        for index, item in enumerate(iter_pilot_continuations(design))
        if index > 0 and item.variant == "MATCHED_CONTROL"
    )
    foreign_control = build_run_record(
        design, second_pair_control, _control_bundle(second_pair_control)
    )
    with pytest.raises(PilotCustodyError, match="exactly one control"):
        validate_pilot_pair(design, records + [foreign_control])

    divergent = [dict(record) for record in records]
    divergent[1]["hmc_reset_nonce_hex"] = _sha("wrong-nonce")
    _resign(divergent[1])
    with pytest.raises(PilotCustodyError, match="matched"):
        validate_pilot_pair(design, divergent)

    wrong_link = [dict(record) for record in records]
    wrong_link[2]["matched_control_id"] = _sha("not-the-pair-control")
    _resign(wrong_link[2])
    with pytest.raises(PilotCustodyError, match="matched control"):
        validate_pilot_pair(design, wrong_link)


def test_pair_validation_rejects_tampered_self_hash() -> None:
    from aeolus.habitat_v2.forecast.pilot_custody import (
        PilotCustodyError,
        validate_pilot_pair,
    )

    design = _design()
    records = [dict(record) for record in _pair_records(design)]
    records[3]["trace_sha256"] = _sha("tampered-trace")
    with pytest.raises(PilotCustodyError, match="self-hash"):
        validate_pilot_pair(design, records)


def test_stage_pair_packet_writes_canonical_artifacts_once(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pilot_custody import (
        PilotCustodyError,
        stage_pair_packet,
    )

    design = _design()
    records = _pair_records(design)
    destination = tmp_path / "pair-packet"
    manifest = stage_pair_packet(destination, design, records)

    records_file = destination / "records.jsonl"
    manifest_file = destination / "manifest.json"
    assert records_file.exists() and manifest_file.exists()
    lines = records_file.read_bytes().splitlines()
    assert len(lines) == 5
    for line, record in zip(lines, records):
        assert json.loads(line) == json.loads(json.dumps(record))
    written_manifest = json.loads(manifest_file.read_bytes())
    assert written_manifest["records_sha256"] == hashlib.sha256(
        records_file.read_bytes()
    ).hexdigest()
    assert written_manifest["record_count"] == 5
    assert manifest["manifest_sha256"] == written_manifest["manifest_sha256"]
    with pytest.raises(PilotCustodyError, match="already exists"):
        stage_pair_packet(destination, design, records)



def test_real_pair_executes_records_stages_and_revalidates(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.pilot_custody import (
        build_run_record,
        stage_pair_packet,
        validate_pilot_pair,
    )
    from aeolus.habitat_v2.forecast.pilot_execution import (
        run_pilot_action_continuation,
        run_pilot_control_continuation,
    )

    design = _design()
    control, actions = _pair_continuations(design)
    records = [
        build_run_record(
            design, control, run_pilot_control_continuation(ROOT, design, control)
        )
    ]
    records += [
        build_run_record(
            design,
            continuation,
            run_pilot_action_continuation(ROOT, design, continuation),
        )
        for continuation in actions
    ]

    manifest = stage_pair_packet(tmp_path / "packet", design, records)

    raw = (tmp_path / "packet" / "records.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["records_sha256"]
    reparsed = [json.loads(line) for line in raw.splitlines()]
    validate_pilot_pair(design, reparsed)
    assert {
        record["trace_sha256"] for record in reparsed
    } == {record["trace_sha256"] for record in records}
