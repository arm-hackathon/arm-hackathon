"""Build the BDM-v1 development corpus over the frozen custody roster.

Issue #72 part 3. Writes one write-once JSONL shard per family plus a corpus
manifest under an ignored ``out/`` directory. Shards already present are
reused (resumable builds); the manifest is always recomputed from the shards
on disk, so a resumed build and a clean build produce identical manifests.
The sealed BLIND_FINAL partition is never collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.bdm_v1_benchmark_contract import load_bdm_v1_benchmark_contract
from aeolus.habitat_v2.bdm_v1_corpus import (
    CORPUS_PARTITIONS,
    CorpusFamilyMeta,
    collect_family_samples,
    corpus_manifest_digest,
    validate_sample_against_contract,
)
from aeolus.habitat_v2.bdm_v1_families import (
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import canonical_json_bytes, load_forecast_contracts
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_corpus_protocol_v1.json"
REGISTRY_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json"


class CorpusBuildError(RuntimeError):
    """Raised when the corpus build cannot proceed safely."""


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise CorpusBuildError("corpus output must live under the ignored out/ directory") from error
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    return output


def build_corpus(output: Path, partitions: tuple[str, ...], max_groups: int | None) -> dict[str, Any]:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    registry_raw = REGISTRY_PATH.read_bytes()
    registry = json.loads(registry_raw)
    if hashlib.sha256(registry_raw).hexdigest() != protocol["bindings"]["custody_registry_sha256"]:
        raise CorpusBuildError("custody registry drifted from the frozen corpus protocol")
    for partition in partitions:
        if partition not in CORPUS_PARTITIONS:
            raise CorpusBuildError(f"partition {partition!r} is not collectable")
    base_data, base_sha = load_base_scenario_data(REPO_ROOT)
    manifest, manifest_sha = load_physics_provenance_manifest(REPO_ROOT)
    if base_sha != protocol["bindings"]["base_scenario_sha256"]:
        raise CorpusBuildError("base scenario drifted from the frozen corpus protocol")
    if manifest_sha != protocol["bindings"]["provenance_manifest_sha256"]:
        raise CorpusBuildError("provenance manifest drifted from the frozen corpus protocol")
    bundle = load_forecast_contracts(REPO_ROOT)
    contract, _ = load_bdm_v1_benchmark_contract(REPO_ROOT)
    config = GeneratorConfig()
    plans = {plan.group_index: plan for plan in assign_partitions(config)}

    shard_dir = output / "shards"
    families: list[dict[str, Any]] = []
    all_digests: list[str] = []
    total_samples = 0
    started = time.perf_counter()
    rows = [row for row in registry["groups"] if row["partition"] in partitions]
    if max_groups is not None:
        rows = rows[:max_groups]
    for row in rows:
        for variant_index, variant in enumerate(("a", "b")):
            definition, scenario = build_family(
                config,
                plans[row["group_index"]],
                variant_index,
                int(row["attempts"][variant]),
                base_data,
                manifest,
            )
            if definition.scenario_sha256 != row["scenario_sha256"][variant]:
                raise CorpusBuildError(
                    f"family {definition.family_id} does not match the registry digest"
                )
            shard_path = shard_dir / f"{definition.family_id}.jsonl"
            if shard_path.exists():
                samples = [json.loads(line) for line in shard_path.read_text().splitlines() if line]
                reused = True
            else:
                meta = CorpusFamilyMeta(
                    family_id=definition.family_id,
                    group_id=definition.group_id,
                    group_index=definition.group_index,
                    partition=definition.partition,
                    template_id=definition.template_id,
                    stratum=definition.stratum,
                    sensor_variant=definition.sensor_variant,
                    scenario_sha256=definition.scenario_sha256,
                )
                samples = list(collect_family_samples(bundle, scenario, meta))
                for sample in samples:
                    validate_sample_against_contract(contract, sample)
                shard_path.write_bytes(
                    b"".join(canonical_json_bytes(sample) + b"\n" for sample in samples)
                )
                reused = False
            shard_raw = shard_path.read_bytes()
            families.append(
                {
                    "family_id": definition.family_id,
                    "group_id": row["group_id"],
                    "partition": row["partition"],
                    "scenario_sha256": definition.scenario_sha256,
                    "sample_count": len(samples),
                    "family_digest": corpus_manifest_digest(samples),
                    "shard_sha256": hashlib.sha256(shard_raw).hexdigest(),
                    "reused": reused,
                }
            )
            all_digests.extend(sample["sample_sha256"] for sample in samples)
            total_samples += len(samples)
            print(
                f"  {definition.family_id}: {len(samples)} samples"
                f"{' (reused)' if reused else ''}",
                file=sys.stderr,
            )
    elapsed = time.perf_counter() - started
    corpus_manifest = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_corpus_manifest_v1",
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "partitions": list(partitions),
        "family_count": len(families),
        "sample_count": total_samples,
        "corpus_digest": hashlib.sha256(canonical_json_bytes(all_digests)).hexdigest(),
        "families": families,
        "build_seconds": round(elapsed, 1),
    }
    (output / "corpus-manifest.json").write_bytes(canonical_json_bytes(corpus_manifest))
    return corpus_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        default=",".join(CORPUS_PARTITIONS),
        help="comma-separated collectable partitions (BLIND_FINAL is always rejected)",
    )
    parser.add_argument("--max-groups", type=int, default=None)
    args = parser.parse_args()
    partitions = tuple(part for part in args.partitions.split(",") if part)
    output = _resolve_output(args.output)
    result = build_corpus(output, partitions, args.max_groups)
    print(json.dumps({key: result[key] for key in ("family_count", "sample_count", "corpus_digest", "build_seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
