from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from aeolus.habitat_v2.qualification import (
    AggregateMetrics,
    HardNegativeResult,
    ObservabilityReport,
    PairManifest,
)
from aeolus.habitat_v2.qualification_packet import build_qualification_packet


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRACKED_PACKET = (
    REPOSITORY_ROOT
    / "docs"
    / "evidence"
    / "habitat-v2-operational-observability-qualification-packet.json"
)
EXPECTED_PACKET_SHA256 = (
    "1afed658237fd62404094eac2d50a78b8db9ad19f9b612add9ff37d1b0e3866b"
)


def test_builder_reproduces_the_tracked_qualification_packet() -> None:
    packet_bytes = build_qualification_packet(REPOSITORY_ROOT)

    assert hashlib.sha256(packet_bytes).hexdigest() == EXPECTED_PACKET_SHA256
    assert packet_bytes == TRACKED_PACKET.read_bytes()

    packet = json.loads(packet_bytes)
    aggregate = AggregateMetrics.from_canonical_mapping(packet["aggregate"])
    hard_negative = HardNegativeResult.from_canonical_mapping(packet["hard_negative"])
    reports = []
    for record in packet["records"].values():
        PairManifest.from_canonical_mapping(record["pair_manifest"])
        reports.append(ObservabilityReport.from_canonical_mapping(record["report"]))

    assert aggregate.concern_coverage_numerator == 6
    assert aggregate.concern_coverage_denominator == 6
    assert aggregate.healthy_false_concern_count == 0
    assert aggregate.healthy_hard_negative_denominator == 1
    assert aggregate.eligible_localisation_numerator == 5
    assert aggregate.eligible_localisation_denominator == 5
    assert aggregate.ambiguous_abstention_numerator == 1
    assert aggregate.ambiguous_abstention_denominator == 1
    assert aggregate.overclaim_count == 0
    assert aggregate.overclaim_denominator == 6
    assert hard_negative.false_concern is False
    assert len(reports) == 6


def test_repository_command_verifies_and_writes_the_packet(tmp_path: Path) -> None:
    output = tmp_path / "qualification-packet.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT / "scripts" / "build_habitat_v2_observability_packet.py"
            ),
            "--source-root",
            str(REPOSITORY_ROOT),
            "--output",
            str(output),
            "--expected-sha256",
            EXPECTED_PACKET_SHA256,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == TRACKED_PACKET.read_bytes()
    assert f"packet_sha256={EXPECTED_PACKET_SHA256}" in completed.stdout
    assert "verified_expected_sha256=true" in completed.stdout


def test_repository_command_rejects_an_unexpected_packet_digest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification-packet.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT / "scripts" / "build_habitat_v2_observability_packet.py"
            ),
            "--source-root",
            str(REPOSITORY_ROOT),
            "--output",
            str(output),
            "--expected-sha256",
            "0" * 64,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "qualification packet SHA-256 mismatch" in completed.stderr
    assert not output.exists()
