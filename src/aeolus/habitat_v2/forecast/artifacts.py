"""Deterministic compressed storage for canonical forecast evidence bytes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import gzip
import hashlib
import io
from pathlib import Path
from typing import BinaryIO


class ForecastArtifactError(ValueError):
    """Compressed evidence bytes violate the frozen artifact contract."""


@dataclass(frozen=True, slots=True)
class CanonicalGzipArtifact:
    payload: bytes
    uncompressed_sha256: str
    compressed_sha256: str
    uncompressed_byte_length: int
    compressed_byte_length: int


@dataclass(frozen=True, slots=True)
class CanonicalGzipReceipt:
    uncompressed_sha256: str
    compressed_sha256: str
    uncompressed_byte_length: int
    compressed_byte_length: int


class _HashingWriter:
    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.byte_length = 0

    def write(self, value: bytes) -> int:
        data = bytes(value)
        written = self.raw.write(data)
        if written != len(data):
            raise ForecastArtifactError("compressed artifact write was incomplete")
        self.hasher.update(data)
        self.byte_length += written
        return written

    def flush(self) -> None:
        self.raw.flush()

    def tell(self) -> int:
        return self.raw.tell()


class _HashingSink:
    def __init__(self) -> None:
        self.hasher = hashlib.sha256()
        self.byte_length = 0

    def write(self, value: bytes) -> int:
        data = bytes(value)
        self.hasher.update(data)
        self.byte_length += len(data)
        return len(data)

    def flush(self) -> None:
        return None

    def tell(self) -> int:
        return self.byte_length


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ForecastArtifactError(f"{label} must be lowercase SHA-256")
    return value


def compress_canonical_bytes(raw: bytes) -> CanonicalGzipArtifact:
    """Compress exact bytes without timestamps, filenames or platform metadata."""

    if type(raw) is not bytes:
        raise ForecastArtifactError("canonical artifact input must be exact bytes")
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output,
        mtime=0,
    ) as stream:
        stream.write(raw)
    payload = output.getvalue()
    return CanonicalGzipArtifact(
        payload=payload,
        uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_sha256=hashlib.sha256(payload).hexdigest(),
        uncompressed_byte_length=len(raw),
        compressed_byte_length=len(payload),
    )


def write_canonical_gzip(
    chunks: Iterable[bytes], destination: str | Path
) -> CanonicalGzipReceipt:
    """Stream one deterministic gzip member into an unpublished write-once path."""

    path = Path(destination)
    created = False
    raw_hasher = hashlib.sha256()
    raw_length = 0
    try:
        with path.open("xb") as file:
            created = True
            writer = _HashingWriter(file)
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=writer,
                mtime=0,
            ) as stream:
                for chunk in chunks:
                    if type(chunk) is not bytes:
                        raise ForecastArtifactError(
                            "canonical artifact chunks must be exact bytes"
                        )
                    stream.write(chunk)
                    raw_hasher.update(chunk)
                    raw_length += len(chunk)
            writer.flush()
    except FileExistsError as error:
        raise ForecastArtifactError("compressed artifact already exists") from error
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise

    return CanonicalGzipReceipt(
        uncompressed_sha256=raw_hasher.hexdigest(),
        compressed_sha256=writer.hasher.hexdigest(),
        uncompressed_byte_length=raw_length,
        compressed_byte_length=writer.byte_length,
    )


def verify_canonical_gzip_file(
    source: str | Path,
    *,
    expected_uncompressed_sha256: str,
    expected_compressed_sha256: str,
    expected_uncompressed_byte_length: int,
    expected_compressed_byte_length: int,
) -> CanonicalGzipReceipt:
    """Verify compressed and raw identities without loading the artifact at once."""

    expected_raw = _sha256(
        expected_uncompressed_sha256,
        label="expected uncompressed identity",
    )
    expected_compressed = _sha256(
        expected_compressed_sha256,
        label="expected compressed identity",
    )
    if (
        type(expected_uncompressed_byte_length) is not int
        or type(expected_compressed_byte_length) is not int
        or expected_uncompressed_byte_length < 0
        or expected_compressed_byte_length < 0
    ):
        raise ForecastArtifactError("expected artifact byte lengths are malformed")

    path = Path(source)
    compressed_hasher = hashlib.sha256()
    compressed_length = 0
    raw_hasher = hashlib.sha256()
    raw_length = 0
    canonical_sink = _HashingSink()
    try:
        with path.open("rb") as file:
            header = file.read(10)
            if header != b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff":
                raise ForecastArtifactError(
                    "compressed artifact header is not canonical"
                )
            compressed_hasher.update(header)
            compressed_length += len(header)
            while chunk := file.read(1024 * 1024):
                compressed_hasher.update(chunk)
                compressed_length += len(chunk)

            if (
                compressed_length != expected_compressed_byte_length
                or compressed_hasher.hexdigest() != expected_compressed
            ):
                raise ForecastArtifactError(
                    "compressed artifact differs from expected identity"
                )

            file.seek(0)
            with gzip.GzipFile(fileobj=file, mode="rb") as stream:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=9,
                    fileobj=canonical_sink,
                    mtime=0,
                ) as canonical_stream:
                    while chunk := stream.read(1024 * 1024):
                        raw_hasher.update(chunk)
                        raw_length += len(chunk)
                        canonical_stream.write(chunk)
    except ForecastArtifactError:
        raise
    except (EOFError, OSError) as error:
        raise ForecastArtifactError("compressed artifact cannot be verified") from error

    if (
        raw_length != expected_uncompressed_byte_length
        or raw_hasher.hexdigest() != expected_raw
    ):
        raise ForecastArtifactError(
            "uncompressed artifact differs from expected identity"
        )
    if (
        canonical_sink.byte_length != compressed_length
        or canonical_sink.hasher.hexdigest() != compressed_hasher.hexdigest()
    ):
        raise ForecastArtifactError(
            "compressed artifact is not one canonical gzip member"
        )

    return CanonicalGzipReceipt(
        uncompressed_sha256=raw_hasher.hexdigest(),
        compressed_sha256=compressed_hasher.hexdigest(),
        uncompressed_byte_length=raw_length,
        compressed_byte_length=compressed_length,
    )


def verify_canonical_gzip(
    artifact: CanonicalGzipArtifact,
    *,
    expected_uncompressed_sha256: str,
    expected_compressed_sha256: str,
) -> bytes:
    """Verify one exact canonical gzip member and return its bound raw bytes."""

    if type(artifact) is not CanonicalGzipArtifact:
        raise ForecastArtifactError("compressed artifact has the wrong type")
    expected_raw = _sha256(
        expected_uncompressed_sha256,
        label="expected uncompressed identity",
    )
    expected_compressed = _sha256(
        expected_compressed_sha256,
        label="expected compressed identity",
    )
    if (
        artifact.uncompressed_sha256 != expected_raw
        or artifact.compressed_sha256 != expected_compressed
    ):
        raise ForecastArtifactError(
            "compressed artifact differs from expected identity"
        )
    if type(artifact.payload) is not bytes:
        raise ForecastArtifactError("compressed artifact payload must be exact bytes")
    if (
        type(artifact.uncompressed_byte_length) is not int
        or type(artifact.compressed_byte_length) is not int
        or artifact.uncompressed_byte_length < 0
        or artifact.compressed_byte_length < 0
        or artifact.compressed_byte_length != len(artifact.payload)
        or hashlib.sha256(artifact.payload).hexdigest() != artifact.compressed_sha256
    ):
        raise ForecastArtifactError("compressed artifact receipt is malformed")
    if artifact.payload[:10] != b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff":
        raise ForecastArtifactError("compressed artifact header is not canonical")
    try:
        raw = gzip.decompress(artifact.payload)
    except (EOFError, OSError) as error:
        raise ForecastArtifactError(
            "compressed artifact cannot be decompressed"
        ) from error
    if (
        len(raw) != artifact.uncompressed_byte_length
        or hashlib.sha256(raw).hexdigest() != artifact.uncompressed_sha256
    ):
        raise ForecastArtifactError("uncompressed artifact receipt is malformed")
    if compress_canonical_bytes(raw).payload != artifact.payload:
        raise ForecastArtifactError(
            "compressed artifact is not one canonical gzip member"
        )
    return raw
