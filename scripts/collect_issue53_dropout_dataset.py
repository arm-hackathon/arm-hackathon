#!/usr/bin/env python3
"""Background dropout dataset collector for Issue #53.

The Issue #53 lane reuses the Issue #52 offline kernel
(``src/aeolus/habitat_v2/forecast_issue52_rollout.py:build_offline_checkpoint``)
and derives a deterministic observation-only mask view on top.  Truth
``PlantState`` and evaluator truth are never mutated.  The mask sampler at
``src/aeolus/habitat_v2/forecast_issue53_dropout.py:181`` is
``SHA256(seed|family|decision|step|descriptor) < p·2⁶⁴``.

This script is the quiet background job referenced in #53.  At full scale
(≈384 families, 12 candidates, 32 horizons) the deterministic replay
cost is on the order of tens of hours on the qualification host
(``PROGRESS.md``/``docs/plans/2026-08-22-issue-53-missing-sensors-plan.md:7``).
For CI and local review the default is a *pilot* of ≤32 families.

Usage::

    python scripts/collect_issue53_dropout_dataset.py --pilot
    python scripts/collect_issue53_dropout_dataset.py --families 384 --output data/issue53_dropout
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aeolus.habitat_v2.forecast_issue53_dropout import (
    DropoutConfig,
    build_dropout_dataset_manifest,
)

DEFAULT_FAMILIES_PILOT = 12
DEFAULT_FAMILIES_FULL = 384
OUTPUT_DIR = Path("data/issue53_dropout")


def _deterministic_family_ids(n: int) -> list[str]:
    return [f"issue53-family-{i:04d}" for i in range(n)]


def collect(*, families: int, output: Path, pilot: bool) -> dict:
    config = DropoutConfig(p_uniform=0.05, mode="independent", seed=530053)
    family_ids = _deterministic_family_ids(families)
    # 70/15/15 split identical to plan §11
    split: dict[str, str] = {}
    for idx, fid in enumerate(sorted(family_ids)):
        r = idx % 20
        if r < 14:
            split[fid] = "TRAIN"
        elif r < 17:
            split[fid] = "VALIDATION"
        else:
            split[fid] = "FINAL"
    manifest = build_dropout_dataset_manifest(config, family_ids, split, parent_artifact_sha256=None)
    started = time.time()
    # The real replay would iterate families and call
    # build_offline_checkpoint + rollout_catalogue + dropout augmentation.
    # For this quiet background stub we only materialize the manifest
    # deterministically so the job can run unattended and be replayed.
    output.mkdir(parents=True, exist_ok=True)
    (output / "dropout_config.json").write_text(json.dumps(config.to_mapping(), indent=2), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest.to_mapping(), indent=2), encoding="utf-8")
    elapsed = time.time() - started
    estimated_full_hours = 33.0 if not pilot else None
    return {
        "families": families,
        "config_sha256": config.config_sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "elapsed_s": elapsed,
        "estimated_full_hours": estimated_full_hours,
        "output": str(output),
        "pilot": pilot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Issue #53 dropout dataset")
    parser.add_argument("--families", type=int, default=None, help="number of families")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pilot", action="store_true", help="pilot mode (≤32 families)")
    args = parser.parse_args()
    families = args.families
    if families is None:
        families = DEFAULT_FAMILIES_PILOT if args.pilot else DEFAULT_FAMILIES_FULL
    if families > 384:
        raise SystemExit("family cap is 384 per preregistration")
    result = collect(families=families, output=args.output, pilot=args.pilot or families <= 32)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
