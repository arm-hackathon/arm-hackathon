"""Independently verify a built BDM-v1 development corpus.

Issue #72 part 3. Re-parses every shard, re-validates each sample against the
Issue #70 input schema and the corpus invariants, recomputes every digest, and
(optionally) re-derives complete samples from the frozen roster through the
generator and plant-only rollouts, comparing sample digests byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.bdm_v1_benchmark_contract import load_bdm_v1_benchmark_contract
from aeolus.habitat_v2.bdm_v1_corpus import (
    CorpusFamilyMeta,
    collect_family_samples,
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


class CorpusVerifyError(RuntimeError):
    """Raised when any corpus verification boundary fails."""


def verify_corpus(corpus_path: Path, rederive_families: int) -> dict[str, Any]:
    corpus = corpus_path.resolve()
    manifest = json.loads((corpus / "corpus-manifest.json").read_bytes())
    protocol_raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    if manifest["protocol_sha256"] != hashlib.sha256(protocol_raw).hexdigest():
        raise CorpusVerifyError("corpus manifest binds a different protocol digest")
    registry_raw = REGISTRY_PATH.read_bytes()
    registry = json.loads(registry_raw)
    if hashlib.sha256(registry_raw).hexdigest() != protocol["bindings"]["custody_registry_sha256"]:
        raise CorpusVerifyError("registry drifted from the frozen protocol")
    base_data, base_sha = load_base_scenario_data(REPO_ROOT)
    manifest_sha_digest: str
    _manifest_data, manifest_sha_digest = load_physics_provenance_manifest(REPO_ROOT)
    if base_sha != protocol["bindings"]["base_scenario_sha256"]:
        raise CorpusVerifyError("base scenario drifted from the frozen protocol")
    if manifest_sha_digest != protocol["bindings"]["provenance_manifest_sha256"]:
        raise CorpusVerifyError("provenance manifest drifted from the frozen protocol")
    bundle = load_forecast_contracts(REPO_ROOT)
    contract, _ = load_bdm_v1_benchmark_contract(REPO_ROOT)
    registry_by_family: dict[str, dict[str, Any]] = {}
    for row in registry["groups"]:
        for variant_index, variant in enumerate(("a", "b")):
            family_id = f"bdm-v1-f{row['group_index'] * 2 + variant_index:04d}-{variant}"
            registry_by_family[family_id] = {
                "group_id": row["group_id"],
                "partition": row["partition"],
                "scenario_sha256": row["scenario_sha256"][variant],
                "group_index": row["group_index"],
                "variant_index": variant_index,
                "attempt": int(row["attempts"][variant]),
            }

    digest_order: list[str] = []
    checked_samples = 0
    for family_row in manifest["families"]:
        family_id = family_row["family_id"]
        if family_row["partition"] == "BLIND_FINAL":
            raise CorpusVerifyError("corpus manifest contains sealed blind material")
        bound = registry_by_family.get(family_id)
        if bound is None or bound["partition"] != family_row["partition"]:
            raise CorpusVerifyError(f"family {family_id} is not a registered collectable family")
        if bound["scenario_sha256"] != family_row["scenario_sha256"]:
            raise CorpusVerifyError(f"family {family_id} scenario digest mismatches the registry")
        shard_path = corpus / "shards" / f"{family_id}.jsonl"
        shard_raw = shard_path.read_bytes()
        if hashlib.sha256(shard_raw).hexdigest() != family_row["shard_sha256"]:
            raise CorpusVerifyError(f"shard {family_id} digest mismatches the manifest")
        samples = [json.loads(line) for line in shard_raw.decode("utf-8").splitlines() if line]
        if len(samples) != family_row["sample_count"]:
            raise CorpusVerifyError(f"shard {family_id} sample count mismatches the manifest")
        for sample in samples:
            if sample["family_id"] != family_id:
                raise CorpusVerifyError(f"sample inside shard {family_id} is misfiled")
            validate_sample_against_contract(contract, sample)
            payload = {key: value for key, value in sample.items() if key != "sample_sha256"}
            if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != sample["sample_sha256"]:
                raise CorpusVerifyError(f"sample digest mismatch in shard {family_id}")
            digest_order.append(sample["sample_sha256"])
        checked_samples += len(samples)
    if checked_samples != manifest["sample_count"]:
        raise CorpusVerifyError("manifest sample count mismatches the shards")
    if (
        hashlib.sha256(canonical_json_bytes(digest_order)).hexdigest()
        != manifest["corpus_digest"]
    ):
        raise CorpusVerifyError("corpus digest mismatches the recomputed shard order")

    rederived = 0
    if rederive_families:
        config = GeneratorConfig()
        plans = {plan.group_index: plan for plan in assign_partitions(config)}
        for family_row in manifest["families"][:rederive_families]:
            bound = registry_by_family[family_row["family_id"]]
            definition, scenario = build_family(
                config,
                plans[bound["group_index"]],
                bound["variant_index"],
                bound["attempt"],
                base_data,
                _manifest_data,
            )
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
            shard_path = corpus / "shards" / f"{family_row['family_id']}.jsonl"
            stored = [
                json.loads(line)
                for line in shard_path.read_text().splitlines()
                if line
            ]
            decisions = sorted({sample["decision_step"] for sample in stored})
            fresh = collect_family_samples(bundle, scenario, meta, decision_steps=decisions)
            fresh_by_key = {
                (sample["decision_step"], sample["features"]["candidate_action_index"]): sample
                for sample in fresh
            }
            for sample in stored:
                key = (
                    sample["decision_step"],
                    sample["features"]["candidate_action_index"],
                )
                if key not in fresh_by_key:
                    raise CorpusVerifyError(f"re-derivation lost sample {key}")
                if fresh_by_key[key]["sample_sha256"] != sample["sample_sha256"]:
                    raise CorpusVerifyError(
                        f"re-derived sample digest differs for {family_row['family_id']} {key}"
                    )
            rederived += 1
    return {
        "family_count": manifest["family_count"],
        "sample_count": checked_samples,
        "corpus_digest": manifest["corpus_digest"],
        "rederived_families": rederived,
        "status": "VERIFIED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--rederive-families", type=int, default=3)
    args = parser.parse_args()
    result = verify_corpus(args.corpus, args.rederive_families)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
