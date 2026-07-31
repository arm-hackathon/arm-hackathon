"""Fail-closed corpus-v2 evaluator and manifest regressions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from aeolus.corpus import generate_corpus_v2
from aeolus.evaluate import evaluate_v2
from aeolus.families import FamilyEvidence, build_family_evidence, load_family_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "scenarios"
FAMILY_MANIFEST_PATH = SCENARIOS / "families.json"


def _generated_rows(tmp_path: Path) -> tuple[list[dict], dict[str, str], dict[str, FamilyEvidence]]:
    generate_corpus_v2(FAMILY_MANIFEST_PATH, tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = load_family_manifest(FAMILY_MANIFEST_PATH)
    return rows, manifest.contract_metadata, build_family_evidence(manifest)


def _evaluate(rows: list[dict], contract: dict[str, str], families: dict[str, FamilyEvidence]) -> dict:
    return evaluate_v2(
        rows,
        lambda _: "nominal",
        expected_contract=contract,
        expected_families=families,
        target_split="test",
    )


def test_evaluate_v2_rejects_row_split_outside_validated_family_manifest(tmp_path: Path):
    rows, contract, families = _generated_rows(tmp_path)
    for row in rows:
        if row["family_id"] == "blocked-path-v1":
            row["split"] = "train"

    with pytest.raises(ValueError, match="split does not match family evidence"):
        _evaluate(rows, contract, families)


def test_evaluate_v2_rejects_late_feature_value_that_overflows_float32(tmp_path: Path):
    rows, contract, families = _generated_rows(tmp_path)
    rows[-1]["features"][-1][-1] = 1e100

    with pytest.raises(ValueError, match="overflows float32"):
        _evaluate(rows, contract, families)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda rows: rows[0].__setitem__("label", "invented_fault_class"),
            "label does not match family evidence",
        ),
        (
            lambda rows: rows[0].__setitem__("scenario_role", "invented_role"),
            "scenario_role is unsupported",
        ),
        (
            lambda rows: rows[0].__setitem__("scenario_role", []),
            "scenario_role is unsupported",
        ),
        (
            lambda rows: rows[0].__setitem__("observable_onset_tick", True),
            "observable onset must be an integer",
        ),
        (
            lambda rows: rows[0].__setitem__("features", []),
            "features must contain at least one tick",
        ),
        (
            lambda rows: rows[0].__setitem__("start_tick", 11),
            "start_tick must not exceed end_tick",
        ),
    ),
)
def test_evaluate_v2_rejects_malformed_row_values(tmp_path: Path, mutation, message: str):
    rows, contract, families = _generated_rows(tmp_path)
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        _evaluate(rows, contract, families)


def test_evaluate_v2_rejects_incomplete_or_duplicate_row_identity(tmp_path: Path):
    rows, contract, families = _generated_rows(tmp_path)
    del rows[0]["window_index"]

    with pytest.raises(ValueError, match="schema mismatch"):
        _evaluate(rows, contract, families)

    duplicate_rows = deepcopy(rows)
    duplicate_rows[0]["window_index"] = 0
    duplicate_rows.append(deepcopy(duplicate_rows[0]))
    with pytest.raises(ValueError, match="duplicate row identity"):
        _evaluate(duplicate_rows, contract, families)


def test_generate_corpus_v2_manifest_records_integrity_and_split_counts(tmp_path: Path):
    generated = generate_corpus_v2(FAMILY_MANIFEST_PATH, tmp_path)
    manifest_without_hash = {
        key: value for key, value in generated.items() if key != "manifest_sha256"
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            manifest_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    assert generated["manifest_sha256"] == expected_hash
    assert generated["family_counts_by_split"] == {
        "train": 0,
        "validation": 0,
        "test": 3,
    }


def test_evaluate_v2_rejects_missing_expected_family_stream(tmp_path: Path):
    rows, contract, families = _generated_rows(tmp_path)
    rows = [row for row in rows if row["family_id"] != "blocked-path-v1"]

    with pytest.raises(ValueError, match="family evidence is missing streams"):
        _evaluate(rows, contract, families)


def test_evaluate_v2_rejects_missing_final_window_from_expected_stream(tmp_path: Path):
    rows, contract, families = _generated_rows(tmp_path)
    removed = max(
        (
            row
            for row in rows
            if row["family_id"] == "blocked-path-v1"
            and row["scenario_role"] == "fault"
        ),
        key=lambda row: row["window_index"],
    )
    rows.remove(removed)

    with pytest.raises(ValueError, match="window sequence is incomplete"):
        _evaluate(rows, contract, families)
