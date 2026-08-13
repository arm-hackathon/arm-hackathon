#!/usr/bin/env python3
"""Build and optionally verify the Habitat V2 observability packet."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from aeolus.habitat_v2.qualification_packet import build_qualification_packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the canonical Habitat V2 operational-observability "
            "qualification packet from tracked scenarios and simulator code."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="repository root containing scenarios/habitat_v2_observability",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        help="fail before writing unless the rebuilt packet has this SHA-256",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet_bytes = build_qualification_packet(args.source_root.resolve())
    packet_sha256 = hashlib.sha256(packet_bytes).hexdigest()

    if args.expected_sha256 is not None and packet_sha256 != args.expected_sha256:
        print(
            "qualification packet SHA-256 mismatch: "
            f"expected {args.expected_sha256}, rebuilt {packet_sha256}",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(packet_bytes)
    print(f"packet_sha256={packet_sha256}")
    print(
        "verified_expected_sha256="
        f"{'true' if args.expected_sha256 is not None else 'not-requested'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
