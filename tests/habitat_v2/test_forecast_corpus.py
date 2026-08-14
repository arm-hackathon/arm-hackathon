from __future__ import annotations

import hashlib

import pytest


def _specification() -> dict[str, object]:
    return {
        "id_field": "family_cluster_id",
        "identity_domain": "aeolus-forecast-d1-family-cluster-v1",
        "identity_fields": ["stratum", "generator_contract_sha256", "development_profile_sha256"],
        "required_fields": ["schema_version", "release_tier", "family_cluster_id", "stratum", "generator_contract_sha256", "development_profile_sha256", "record_sha256"],
        "schema_version": "aeolus_habitat_v2_forecast_family_cluster_v1",
    }


def _record() -> dict[str, object]:
    from aeolus.habitat_v2.forecast.corpus import canonical_json_bytes, record_identity

    specification = _specification()
    body: dict[str, object] = {
        "schema_version": specification["schema_version"],
        "release_tier": "DEVELOPMENT_FIXTURE_ONLY",
        "stratum": "constant-occupied",
        "generator_contract_sha256": "a" * 64,
        "development_profile_sha256": "b" * 64,
    }
    record = {**body, "family_cluster_id": record_identity(body, specification)}
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    return record


def test_canonical_jsonl_identity_and_self_hash_are_closed() -> None:
    from aeolus.habitat_v2.forecast.corpus import canonical_jsonl_bytes, validate_jsonl_records

    record = _record()
    data = canonical_jsonl_bytes([record])

    assert validate_jsonl_records(data, _specification()) == [record]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda data: data.replace(b"{", b"{ ", 1),
        lambda data: data.replace(b"\"stratum\"", b"\"stratum\":true,\"stratum\"", 1),
        lambda data: data.replace(b"\"stratum\":\"constant-occupied\"", b"\"stratum\":\"constant-occupied\",\"unknown\":1", 1),
        lambda data: data.rstrip(b"\n"),
    ),
)
def test_strict_jsonl_parser_rejects_noncanonical_duplicate_unknown_and_nonlf(mutation: object) -> None:
    from aeolus.habitat_v2.forecast.corpus import CorpusValidationError, canonical_jsonl_bytes, validate_jsonl_records

    with pytest.raises(CorpusValidationError):
        validate_jsonl_records(mutation(canonical_jsonl_bytes([_record()])), _specification())


def test_record_rejects_bad_identity_self_hash_and_final_custody_field() -> None:
    from aeolus.habitat_v2.forecast.corpus import CorpusValidationError, validate_record

    for field, value in (("family_cluster_id", "0" * 64), ("record_sha256", "0" * 64)):
        tampered = _record()
        tampered[field] = value
        with pytest.raises(CorpusValidationError):
            validate_record(tampered, _specification())
    final_field = _record()
    final_field["custodian_id"] = "not-permitted"
    with pytest.raises(CorpusValidationError):
        validate_record(final_field, _specification())


def test_cluster_ranking_is_deterministic_and_stable_identity_excludes_split() -> None:
    from aeolus.habitat_v2.forecast.corpus import assign_cluster_splits, record_identity

    body = _record()
    before = record_identity(body, _specification())
    body["split_label"] = "TRAIN"
    assert record_identity(body, _specification()) == before
    clusters = [{"family_cluster_id": "1" * 64, "stratum": "x"}, {"family_cluster_id": "2" * 64, "stratum": "x"}]
    first = assign_cluster_splits(clusters, split_key=b"key", split_policy_sha256="3" * 64, split_key_id="test")
    second = assign_cluster_splits(list(reversed(clusters)), split_key=b"key", split_policy_sha256="3" * 64, split_key_id="test")
    assert sorted(first, key=lambda row: row["family_cluster_id"]) == sorted(second, key=lambda row: row["family_cluster_id"])


def test_lineage_rejects_sample_family_alias_and_fit_boundary() -> None:
    from aeolus.habitat_v2.forecast.corpus import CorpusValidationError, iter_training_samples, validate_lineage

    cluster = "a" * 64
    tables = {
        "family_clusters": [{"family_cluster_id": cluster}],
        "split_assignments": [{"family_cluster_id": cluster, "split_assignment_id": "split", "split_label": "TRAIN"}],
        "families": [{"family_id": "family", "family_cluster_id": cluster}],
        "scenario_members": [{"scenario_member_id": "member", "family_id": "family", "plant_run_id": "plant"}],
        "control_runs": [{"control_run_record_id": "run", "scenario_member_id": "member", "control_run_id": "hmc", "replay_witness_id": "witness"}],
        "control_traces": [{"control_trace_record_id": "trace", "control_run_id": "hmc"}],
        "replay_witnesses": [{"replay_witness_id": "witness", "control_run_id": "hmc", "control_trace_record_id": "trace"}],
        "samples": [{"sample_id": "sample", "family_cluster_id": cluster, "family_id": "family", "scenario_member_id": "member", "control_run_record_id": "run", "split_assignment_id": "split", "split_label": "TRAIN", "replay_witness_id": "witness"}],
    }
    validate_lineage(tables)
    assert iter_training_samples(tables["samples"], tables["split_assignments"]) == tuple(tables["samples"])
    tables["samples"][0]["family_id"] = "derived-alias"
    with pytest.raises(CorpusValidationError, match="lineage"):
        validate_lineage(tables)
