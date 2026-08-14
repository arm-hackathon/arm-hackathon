from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest


def test_canonical_gzip_is_byte_deterministic_and_binds_both_identities() -> None:
    from aeolus.habitat_v2.forecast.artifacts import compress_canonical_bytes

    raw = b'{"schema_version":"test-v1","value":1}\n'
    first = compress_canonical_bytes(raw)
    second = compress_canonical_bytes(raw)

    assert first.payload == second.payload
    assert first.uncompressed_sha256 == hashlib.sha256(raw).hexdigest()
    assert first.compressed_sha256 == hashlib.sha256(first.payload).hexdigest()
    assert first.uncompressed_byte_length == len(raw)
    assert first.compressed_byte_length == len(first.payload)
    assert first.payload[:3] == b"\x1f\x8b\x08"
    assert first.payload[3] & 0x08 == 0  # no original filename header
    assert first.payload[4:8] == b"\x00\x00\x00\x00"  # mtime=0
    assert gzip.decompress(first.payload) == raw


def test_expected_identities_reject_self_consistent_compressed_substitution() -> None:
    from aeolus.habitat_v2.forecast.artifacts import (
        ForecastArtifactError,
        compress_canonical_bytes,
        verify_canonical_gzip,
    )

    expected = compress_canonical_bytes(b'{"value":"expected"}\n')
    substitute = compress_canonical_bytes(b'{"value":"substitute"}\n')

    assert (
        verify_canonical_gzip(
            expected,
            expected_uncompressed_sha256=expected.uncompressed_sha256,
            expected_compressed_sha256=expected.compressed_sha256,
        )
        == b'{"value":"expected"}\n'
    )
    with pytest.raises(ForecastArtifactError, match="expected"):
        verify_canonical_gzip(
            substitute,
            expected_uncompressed_sha256=expected.uncompressed_sha256,
            expected_compressed_sha256=expected.compressed_sha256,
        )


def test_streaming_writer_matches_reference_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.artifacts import (
        ForecastArtifactError,
        compress_canonical_bytes,
        verify_canonical_gzip_file,
        write_canonical_gzip,
    )

    chunks = (b'{"row":1}\n', b'{"row":2}\n', b'{"row":3}\n')
    expected = compress_canonical_bytes(b"".join(chunks))
    destination = tmp_path / "records.jsonl.gz"

    receipt = write_canonical_gzip(chunks, destination)

    assert destination.read_bytes() == expected.payload
    assert receipt.uncompressed_sha256 == expected.uncompressed_sha256
    assert receipt.compressed_sha256 == expected.compressed_sha256
    assert receipt.uncompressed_byte_length == expected.uncompressed_byte_length
    assert receipt.compressed_byte_length == expected.compressed_byte_length
    assert (
        verify_canonical_gzip_file(
            destination,
            expected_uncompressed_sha256=receipt.uncompressed_sha256,
            expected_compressed_sha256=receipt.compressed_sha256,
            expected_uncompressed_byte_length=receipt.uncompressed_byte_length,
            expected_compressed_byte_length=receipt.compressed_byte_length,
        )
        == receipt
    )
    with pytest.raises(ForecastArtifactError, match="already exists"):
        write_canonical_gzip(chunks, destination)


def test_streaming_verifier_rejects_pinned_concatenated_gzip_members(
    tmp_path: Path,
) -> None:
    from aeolus.habitat_v2.forecast.artifacts import (
        ForecastArtifactError,
        compress_canonical_bytes,
        verify_canonical_gzip_file,
    )

    first_raw = b'{"row":1}\n'
    second_raw = b'{"row":2}\n'
    payload = (
        compress_canonical_bytes(first_raw).payload
        + compress_canonical_bytes(second_raw).payload
    )
    destination = tmp_path / "concatenated.jsonl.gz"
    destination.write_bytes(payload)
    combined_raw = first_raw + second_raw

    with pytest.raises(ForecastArtifactError, match="canonical gzip member"):
        verify_canonical_gzip_file(
            destination,
            expected_uncompressed_sha256=hashlib.sha256(combined_raw).hexdigest(),
            expected_compressed_sha256=hashlib.sha256(payload).hexdigest(),
            expected_uncompressed_byte_length=len(combined_raw),
            expected_compressed_byte_length=len(payload),
        )
