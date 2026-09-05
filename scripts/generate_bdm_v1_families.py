"""Generate the BDM-v1 development family roster and frozen custody registry.

Issue #72 (part 2/3). Deterministically builds every family of the roster
(default sizing: 80 causal groups / 160 families across TRAIN, DEV,
CALIBRATION, and sealed BLIND_FINAL partitions), validates each scenario
against the closed v5 schema, runs a bounded no-proposal HMC feasibility
smoke replay per family, and emits:

- ``families.jsonl``: one full family-definition record per line;
- ``scenarios.jsonl``: one generated scenario mapping per line;
- ``generation-receipt.json``: run identity, digests, checks, and timing;
- ``family-custody-registry.json``: the committed custody registry (copied
  byte-identically into ``contracts/`` after review).

The output directory is write-once. No blind outcome is ever computed: the
BLIND_FINAL seal records digests only, with its size justification marked
``TBD_FROM_PILOT`` until the Issue #72 part-3 pilot freezes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.bdm_v1_benchmark_contract import (
    BDM_V1_CONTRACT_ID,
    validate_group_disjointness,
)
from aeolus.habitat_v2.bdm_v1_families import (
    BDM_V1_PARTITIONS_ORDERED,
    DECISION_STEPS,
    EPISODE_STEPS,
    GENERATOR_FAULT_MULTIPLIER_BANDS,
    GENERATOR_FAULT_ONSET_CLASSES,
    GENERATOR_FAULT_DURATION_RANGE,
    GENERATOR_INITIAL_O2_BANDS,
    GENERATOR_O2_EXCESS_THRESHOLD,
    GENERATOR_RAMP_SHAPES,
    GENERATOR_REGISTRY_SEED,
    GENERATOR_SENSOR_BIAS_BANDS,
    GENERATOR_SENSOR_BUNDLE_KINDS,
    GENERATOR_VERSION,
    BASE_SCENARIO_RELATIVE_PATH,
    BdmV1FamilyError,
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.hmc import HabitatManagementComputer
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_family_custody_v1"
REGISTRY_ID = "habitat_v2_bdm_v1_family_custody_v1"
FEASIBILITY_SMOKE_STEPS = 8
BLIND_SIZE_JUSTIFICATION_TBD = "TBD_FROM_PILOT"


class GenerationError(RuntimeError):
    """Raised when roster generation or its checks fail."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise GenerationError("generated output must live under the ignored out/ directory") from error
    if output.exists():
        raise GenerationError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def _feasibility_smoke(bundle: Any, scenario: Any, family_id: str) -> None:
    nonce = hashlib.sha256(b"bdm-v1-feasibility|" + family_id.encode("utf-8")).digest()
    hmc = HabitatManagementComputer.reset(scenario, bundle.hmc_contract, nonce)
    for _step in range(FEASIBILITY_SMOKE_STEPS):
        snapshot, verification = hmc.observe()
        handle = hmc.verify_snapshot(snapshot, verification)
        hmc.propose(None, handle)
        hmc.arbitrate()
        hmc.step()


def generate_roster(
    output: Path,
    *,
    config: GeneratorConfig,
    blind_size_justification: str = BLIND_SIZE_JUSTIFICATION_TBD,
) -> dict[str, Any]:
    if config.seed != GENERATOR_REGISTRY_SEED:
        raise GenerationError(
            "registry generation must use the declared registry seed; "
            "a different seed is a new roster and needs a new registry id"
        )
    base_data, base_sha256 = load_base_scenario_data(REPO_ROOT)
    manifest, manifest_sha256 = load_physics_provenance_manifest(REPO_ROOT)
    bundle = load_forecast_contracts(REPO_ROOT)
    plans = assign_partitions(config)

    definitions: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    total_attempts = 0
    for plan in plans:
        family_ids: list[str] = []
        digests: dict[str, str] = {}
        attempts_by_variant: dict[str, int] = {}
        group_definition: dict[str, Any] | None = None
        for variant_index, variant in enumerate(("a", "b")):
            attempt = 0
            while True:
                if attempt >= config.max_feasibility_attempts:
                    raise GenerationError(
                        f"family {plan.group_id}-{variant} exceeded "
                        f"{config.max_feasibility_attempts} feasibility attempts"
                    )
                try:
                    definition, scenario = build_family(
                        config, plan, variant_index, attempt, base_data, manifest
                    )
                    _feasibility_smoke(bundle, scenario, definition.family_id)
                except (BdmV1FamilyError, ValueError) as error:
                    attempt += 1
                    total_attempts += 1
                    print(
                        f"  retry {plan.group_id}-{variant} attempt {attempt}: {error}",
                        file=sys.stderr,
                    )
                    continue
                break
            total_attempts += 1
            family_ids.append(definition.family_id)
            digests[variant] = definition.scenario_sha256
            attempts_by_variant[variant] = attempt
            definitions.append(definition.to_mapping())
            scenario_rows.append(
                {
                    "family_id": definition.family_id,
                    "scenario_sha256": definition.scenario_sha256,
                    "scenario": scenario.data,
                }
            )
            if group_definition is None:
                group_definition = definition.to_mapping()
        assert group_definition is not None
        group_rows.append(
            {
                "group_id": plan.group_id,
                "group_index": plan.group_index,
                "partition": plan.partition,
                "template_id": plan.template.template_id,
                "stratum": plan.template.stratum,
                "group_key": dict(group_definition["group_key"]),
                "family_ids": family_ids,
                "scenario_sha256": digests,
                "attempts": attempts_by_variant,
            }
        )
        print(f"  group {plan.group_id} ({plan.partition}, {plan.template.template_id}) complete", file=sys.stderr)

    partition_groups: dict[str, list[str]] = {partition: [] for partition in BDM_V1_PARTITIONS_ORDERED}
    for row in group_rows:
        partition_groups[row["partition"]].append(row["group_id"])
    validate_group_disjointness(partition_groups)

    # Regeneration check: rebuild every family from the recorded attempt and
    # require byte-identical digests.
    for row in group_rows:
        plan = plans[row["group_index"]]
        for variant_index, variant in enumerate(("a", "b")):
            rebuilt, _ = build_family(
                config, plan, variant_index, int(row["attempts"][variant]), base_data, manifest
            )
            if rebuilt.scenario_sha256 != row["scenario_sha256"][variant]:
                raise GenerationError(f"regeneration digest mismatch for {rebuilt.family_id}")

    strata_counts: dict[str, dict[str, int]] = {
        partition: {} for partition in BDM_V1_PARTITIONS_ORDERED
    }
    template_counts: dict[str, int] = {}
    for row in group_rows:
        strata_counts[row["partition"]][row["stratum"]] = (
            strata_counts[row["partition"]].get(row["stratum"], 0) + 1
        )
        template_counts[row["template_id"]] = template_counts.get(row["template_id"], 0) + 1

    blind_rows = [row for row in group_rows if row["partition"] == "BLIND_FINAL"]
    blind_definitions = [
        definition
        for definition in definitions
        if definition["partition"] == "BLIND_FINAL"
    ]
    blind_definitions_digest = _sha256_bytes(
        canonical_json_bytes(blind_definitions)
    )

    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "status": "FROZEN_FAMILY_CUSTODY_REGISTRY",
        "authorization": {
            "authorized_by": "repository_owner",
            "authorized_via_issue": 72,
            "authorized_at": "2026-09-05",
        },
        "generator": {
            "generator_version": GENERATOR_VERSION,
            "registry_seed": config.seed,
            "base_scenario_path": BASE_SCENARIO_RELATIVE_PATH.as_posix(),
            "base_scenario_sha256": base_sha256,
            "provenance_manifest_sha256": manifest_sha256,
            "benchmark_contract_id": BDM_V1_CONTRACT_ID,
            "episode_steps": EPISODE_STEPS,
            "decision_steps": list(DECISION_STEPS),
            "feasibility_smoke_steps": FEASIBILITY_SMOKE_STEPS,
            "max_feasibility_attempts": config.max_feasibility_attempts,
        },
        "declared_generator_vocabularies": {
            "initial_oxygen_bands": {
                key: list(band) for key, band in GENERATOR_INITIAL_O2_BANDS.items()
            },
            "o2_excess_threshold": GENERATOR_O2_EXCESS_THRESHOLD,
            "fault_multiplier_bands": {
                key: list(band) for key, band in GENERATOR_FAULT_MULTIPLIER_BANDS.items()
            },
            "fault_onset_classes": {
                key: list(band) for key, band in GENERATOR_FAULT_ONSET_CLASSES.items()
            },
            "fault_duration_range": list(GENERATOR_FAULT_DURATION_RANGE),
            "ramp_shapes": list(GENERATOR_RAMP_SHAPES),
            "sensor_bias_bands": {
                key: list(band) for key, band in GENERATOR_SENSOR_BIAS_BANDS.items()
            },
            "sensor_bundle_kinds": list(GENERATOR_SENSOR_BUNDLE_KINDS),
            "excluded_from_drawing": {
                "sensor_noise_amplitudes": "pinned by the reviewed HMC contract reset check",
                "race_operating_condition_bands": "bound to the Issue 55/56 fixtures",
                "per_zone_load_records": "scaled by the group-level load_scale draw and explicit schedule segments",
            },
        },
        "sizing": {
            "groups_per_partition": {
                partition: int(config.groups_per_partition.get(partition, 0))
                for partition in BDM_V1_PARTITIONS_ORDERED
            },
            "families_per_group": 2,
            "total_groups": len(group_rows),
            "total_families": len(definitions),
        },
        "strata_counts_by_partition": strata_counts,
        "template_counts": dict(sorted(template_counts.items())),
        "blind_seal": {
            "group_count": len(blind_rows),
            "family_count": len(blind_definitions),
            "definitions_digest": blind_definitions_digest,
            "outcome_status": "NOT_COMPUTED",
            "size_justification": blind_size_justification,
            "opening_rule": "one preregistered run after candidate freeze per the BDM-v1 benchmark contract",
        },
        "groups": group_rows,
        "checks": {
            "schema_validation": "passed",
            "feasibility_smoke_replay": "passed",
            "group_disjointness": "passed",
            "regeneration_digests": "passed",
        },
    }

    families_raw = b"".join(
        canonical_json_bytes(definition) + b"\n" for definition in definitions
    )
    scenarios_raw = b"".join(
        canonical_json_bytes(row) + b"\n" for row in scenario_rows
    )
    registry_raw = canonical_json_bytes(registry)
    receipt = {
        "schema_version": "aeolus_habitat_v2_bdm_v1_generation_receipt_v1",
        "generator_version": GENERATOR_VERSION,
        "registry_seed": config.seed,
        "base_scenario_sha256": base_sha256,
        "provenance_manifest_sha256": manifest_sha256,
        "family_count": len(definitions),
        "group_count": len(group_rows),
        "feasibility_attempts_total": total_attempts,
        "families_sha256": _sha256_bytes(families_raw),
        "scenarios_sha256": _sha256_bytes(scenarios_raw),
        "registry_sha256": _sha256_bytes(registry_raw),
        "blind_definitions_digest": blind_definitions_digest,
    }
    (output / "families.jsonl").write_bytes(families_raw)
    (output / "scenarios.jsonl").write_bytes(scenarios_raw)
    (output / "family-custody-registry.json").write_bytes(registry_raw)
    (output / "generation-receipt.json").write_bytes(canonical_json_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the BDM-v1 family roster and custody registry"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--blind-size-justification",
        default=BLIND_SIZE_JUSTIFICATION_TBD,
        help="frozen justification text replacing TBD_FROM_PILOT (set by the part-3 pilot)",
    )
    args = parser.parse_args()
    output = _resolve_output(args.output)
    receipt = generate_roster(
        output,
        config=GeneratorConfig(),
        blind_size_justification=str(args.blind_size_justification),
    )
    print(json.dumps(receipt, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
