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
from collections.abc import Mapping
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from aeolus.habitat_v2.forecast_issue52 import (
    CandidateCatalogue,
    HORIZON_STEPS,
    TargetManifest,
    TrainingSample,
    extend_scenario_for_issue52,
)
from aeolus.habitat_v2.forecast_issue52_rollout import (
    build_offline_checkpoint,
    rollout_catalogue,
    training_samples_from_rollouts,
)
from aeolus.habitat_v2.forecast_issue53_dropout import (
    DropoutConfig,
    DropoutDatasetManifest,
    apply_dropout_to_history,
    build_dropout_dataset_manifest,
)
from aeolus.habitat_v2.hmc_contract import canonical_json_bytes, load_hmc_contract
from aeolus.habitat_v2.scenario import Scenario

DEFAULT_FAMILIES_PILOT = 12
DEFAULT_FAMILIES_FULL = 384
OUTPUT_DIR = Path("data/issue53_dropout")
DEFAULT_SCENARIO = Path("scenarios/habitat_v2_actuator_feedback.json")
DEFAULT_CONTRACT = Path("contracts/habitat_v2_hmc_v1.json")
DATASET_SAMPLE_SCHEMA = "aeolus_habitat_v2_forecast_issue_53_sample_v1"
EVALUATED_K = (0, 1, 3, 6)
PARENT_ARTIFACT_SHA256 = (
    "e0a24b2fd9309ed551dcd6d4fb98eff1fdda6b364de2dbe73584ccf1ada7e61f"
)


def _deterministic_family_ids(n: int) -> list[str]:
    return [f"issue53-family-{i:04d}" for i in range(n)]


def _family_split(family_ids: list[str]) -> dict[str, str]:
    """Assign whole families by the preregistered hash order and proportions."""

    if not family_ids:
        return {}
    order = sorted(
        family_ids,
        key=lambda family_id: hashlib.sha256(
            f"issue53-split-v1|{family_id}".encode("utf-8")
        ).digest(),
    )
    proportions = (0.70, 0.15, 0.15)
    labels = ("TRAIN", "VALIDATION", "FINAL")
    counts = [int(len(order) * proportion) for proportion in proportions]
    remaining = len(order) - sum(counts)
    remainders = sorted(
        range(len(labels)),
        key=lambda index: (-(len(order) * proportions[index] - counts[index]), index),
    )
    for index in remainders[:remaining]:
        counts[index] += 1
    result: dict[str, str] = {}
    cursor = 0
    for label, count in zip(labels, counts):
        for family_id in order[cursor : cursor + count]:
            result[family_id] = label
        cursor += count
    return dict(sorted(result.items()))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _array_payload(value: np.ndarray) -> list[list[float | None]]:
    array = np.asarray(value)
    return [
        [None if not np.isfinite(float(item)) else float(item) for item in row]
        for row in array
    ]


def _record_payload(record: object, values: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "snapshot_sha256": record.snapshot_sha256,
        "verification_receipt_sha256": record.verification_receipt_sha256,
        "control_run_id": record.control_run_id,
        "authority_epoch": record.authority_epoch,
        "topology_sha256": record.topology_sha256,
        "hmc_contract_sha256": record.hmc_contract_sha256,
        "snapshot_schema_sha256": record.snapshot_schema_sha256,
        "scenario_sha256": record.scenario_sha256,
        "previous_verification_receipt_sha256": record.previous_verification_receipt_sha256,
        "previous_control_chain_sha256": record.previous_control_chain_sha256,
        "control_chain_sha256": record.control_chain_sha256,
        "sequence": record.sequence,
        "completed_step": record.completed_step,
        "completed_time_s": record.completed_time_s,
        "mode": record.mode,
        "command": _plain(record.command),
        "command_sha256": record.command_sha256,
        "target_values": _array_payload(np.asarray(values)[None, :])[0],
        "available_mask": np.asarray(mask, dtype=bool).tolist(),
    }


def _sample_payload(
    sample: TrainingSample,
    *,
    dropout_config: DropoutConfig,
    dropout_view: str,
    rollout_sha256: str,
) -> dict:
    history = sample.history
    payload = {
        "schema_version": DATASET_SAMPLE_SCHEMA,
        "family_id": sample.family_id,
        "split": sample.split,
        "scenario_sha256": sample.scenario_sha256,
        "manifest_sha256": sample.manifest_sha256,
        "checkpoint_sha256": sample.checkpoint_sha256,
        "schedule_sha256": sample.schedule_sha256,
        "rollout_sha256": rollout_sha256,
        "dropout_config_sha256": dropout_config.config_sha256,
        "dropout_view": dropout_view,
        "latest_missing_count": int(np.sum(~history.available_mask[-1])),
        "history_records": [
            _record_payload(record, history.target_values[index], history.available_mask[index])
            for index, record in enumerate(history.records)
        ],
        "schedule": sample.schedule.to_mapping(),
        "targets": _array_payload(sample.targets),
    }
    payload["sample_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _scenario_and_contract(
    root: Path,
    scenario_path: Path,
    contract_path: Path,
) -> tuple[Scenario, object]:
    scenario_path = root / scenario_path
    contract_path = root / contract_path
    scenario = Scenario.from_mapping(
        json.loads(scenario_path.read_text(encoding="utf-8"))
    )
    scenario = extend_scenario_for_issue52(scenario)
    return scenario, load_hmc_contract(contract_path)


def _family_scenario(base: Scenario, family_id: str) -> Scenario:
    """Make each family a deterministic simulator seed, not a relabeled clone."""

    data = json.loads(json.dumps(base.data, allow_nan=False))
    seed_bytes = hashlib.sha256(f"issue53-family-seed-v1|{family_id}".encode("utf-8")).digest()
    data["sensor_model"]["random_seed"] = int.from_bytes(seed_bytes[:4], "big")
    data["name"] = f"{data['name']}-{family_id}"
    return Scenario.from_mapping(data)


def _collect_family(
    scenario: Scenario,
    contract: object,
    *,
    family_id: str,
    split: str,
    config: DropoutConfig,
) -> list[dict]:
    scenario = _family_scenario(scenario, family_id)
    checkpoint = build_offline_checkpoint(
        scenario,
        contract,
        decision_step=15,
        family_id=family_id,
    )
    manifest = TargetManifest.from_scenario(scenario)
    catalogue = CandidateCatalogue.from_scenario(
        scenario, base_command=checkpoint.last_final_command
    )
    rollouts = rollout_catalogue(checkpoint, catalogue, manifest=manifest)
    samples = training_samples_from_rollouts(
        checkpoint,
        catalogue,
        rollouts,
        family_id=family_id,
        split=split,
    )
    rollouts_by_id = {rollout.candidate_id: rollout for rollout in rollouts}
    serialized: list[dict] = []
    for k_target in EVALUATED_K:
        for sample in samples:
            rollout = rollouts_by_id[sample.schedule.candidate_id]
            masked_history = apply_dropout_to_history(
                sample.history,
                manifest,
                config,
                family_id=family_id,
                decision_step=15,
                latest_missing_count=k_target,
            )
            masked_sample = TrainingSample(
                family_id=sample.family_id,
                split=sample.split,
                scenario_sha256=sample.scenario_sha256,
                manifest_sha256=sample.manifest_sha256,
                checkpoint_sha256=sample.checkpoint_sha256,
                schedule_sha256=sample.schedule_sha256,
                history=masked_history,
                schedule=sample.schedule,
                targets=sample.targets,
            )
            serialized.append(
                _sample_payload(
                    masked_sample,
                    dropout_config=config,
                    dropout_view=f"k{k_target}",
                    rollout_sha256=rollout.rollout_sha256,
                )
            )
    return serialized


def _sample_digest(sample: Mapping[str, object]) -> str:
    payload = dict(sample)
    digest = payload.pop("sample_sha256", None)
    if type(digest) is not str:
        raise ValueError("sample_sha256 is missing")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_dataset(output: Path) -> dict[str, int | str]:
    """Validate the content-addressed files emitted by ``collect``."""

    config_mapping = json.loads((output / "dropout_config.json").read_text(encoding="utf-8"))
    config = DropoutConfig.from_mapping(config_mapping)
    manifest_mapping = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    manifest = DropoutDatasetManifest.from_mapping(manifest_mapping)
    if manifest.dropout_config_sha256 != config.config_sha256:
        raise ValueError("dataset manifest does not bind dropout config")
    if manifest.parent_artifact_sha256 != PARENT_ARTIFACT_SHA256:
        raise ValueError("dataset manifest does not bind frozen Issue #52 parent")
    sample_digest = hashlib.sha256()
    sample_count = 0
    family_ids: set[str] = set()
    family_views: dict[str, dict[str, int]] = {}
    with (output / "samples.jsonl").open("rb") as stream:
        for raw_line in stream:
            if not raw_line.endswith(b"\n"):
                raise ValueError("dataset sample is missing its newline delimiter")
            line = raw_line[:-1]
            sample = json.loads(line)
            if sample.get("schema_version") != DATASET_SAMPLE_SCHEMA:
                raise ValueError("dataset sample schema is invalid")
            if sample.get("dropout_config_sha256") != config.config_sha256:
                raise ValueError("dataset sample does not bind dropout config")
            family_id = sample.get("family_id")
            if type(family_id) is not str or family_id not in manifest.family_split:
                raise ValueError("dataset sample family is not in the manifest")
            if sample.get("split") != manifest.family_split[family_id]:
                raise ValueError("dataset sample split does not bind manifest")
            if sample.get("dropout_view") not in {f"k{k}" for k in EVALUATED_K}:
                raise ValueError("dataset sample dropout view is invalid")
            if sample.get("latest_missing_count") != int(
                str(sample["dropout_view"])[1:]
            ):
                raise ValueError("dataset sample dropout view is inconsistent")
            if sample.get("sample_sha256") != _sample_digest(sample):
                raise ValueError("dataset sample digest is inconsistent")
            history = sample.get("history_records")
            targets = sample.get("targets")
            if not isinstance(history, list) or len(history) != 16:
                raise ValueError("dataset sample history is not a 16-row window")
            if not isinstance(targets, list) or len(targets) != HORIZON_STEPS:
                raise ValueError("dataset sample targets are not a 32-step horizon")
            if any(
                not isinstance(record, Mapping)
                or not isinstance(record.get("target_values"), list)
                or not isinstance(record.get("available_mask"), list)
                or len(record["target_values"]) != len(record["available_mask"])
                for record in history
            ):
                raise ValueError("dataset sample history rows are malformed")
            if sample.get("latest_missing_count") != sum(
                not bool(value) for value in history[-1]["available_mask"]
            ):
                raise ValueError("dataset sample missing count is inconsistent")
            width = len(history[0]["target_values"])
            if width == 0 or any(
                len(record["target_values"]) != width for record in history
            ) or any(
                not isinstance(row, list) or len(row) != width for row in targets
            ):
                raise ValueError("dataset sample tensor widths are inconsistent")
            for record in history:
                for value, available in zip(
                    record["target_values"], record["available_mask"]
                ):
                    if type(available) is not bool:
                        raise ValueError("dataset sample availability is not boolean")
                    if available:
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                        ):
                            raise ValueError("available history value is not finite")
                    elif value is not None:
                        raise ValueError("unavailable history value must be null")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for row in targets
                for value in row
            ):
                raise ValueError("dataset target label is not finite")
            sample_digest.update(line)
            sample_digest.update(b"\n")
            sample_count += 1
            family_ids.add(family_id)
            views = family_views.setdefault(family_id, {})
            view = str(sample["dropout_view"])
            views[view] = views.get(view, 0) + 1
    if sample_count == 0:
        raise ValueError("dataset contains no samples")
    if manifest.samples_sha256 != sample_digest.hexdigest():
        raise ValueError("dataset samples digest is inconsistent")
    if family_ids != set(manifest.family_ids):
        raise ValueError("dataset samples do not cover every family")
    expected_views = {f"k{k}": 12 for k in EVALUATED_K}
    if any(views != expected_views for views in family_views.values()):
        raise ValueError("dataset family sample views are incomplete")
    return {
        "families": len(family_ids),
        "samples": sample_count,
        "dataset_sha256": manifest.dataset_sha256,
        "samples_sha256": sample_digest.hexdigest(),
    }


def collect(
    *,
    families: int,
    output: Path,
    pilot: bool,
    scenario_path: Path = DEFAULT_SCENARIO,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict:
    if isinstance(families, bool) or not isinstance(families, int) or families < 1:
        raise ValueError("families must be a positive integer")
    if families > DEFAULT_FAMILIES_FULL:
        raise ValueError("family cap is 384 per preregistration")
    if pilot and families > 32:
        raise ValueError("pilot family cap is 32 per preregistration")
    config = DropoutConfig(p_uniform=0.05, mode="independent", seed=530053)
    family_ids = _deterministic_family_ids(families)
    split = _family_split(family_ids)
    started = time.time()
    root = Path(__file__).resolve().parents[1]
    scenario, contract = _scenario_and_contract(root, scenario_path, contract_path)
    samples: list[dict] = []
    for family_id in family_ids:
        samples.extend(
            _collect_family(
                scenario,
                contract,
                family_id=family_id,
                split=split[family_id],
                config=config,
            )
        )
    sample_lines = [canonical_json_bytes(sample) for sample in samples]
    sample_digest = hashlib.sha256()
    for line in sample_lines:
        sample_digest.update(line)
        sample_digest.update(b"\n")
    manifest = build_dropout_dataset_manifest(
        config,
        family_ids,
        split,
        parent_artifact_sha256=PARENT_ARTIFACT_SHA256,
        samples_sha256=sample_digest.hexdigest(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "dropout_config.json").write_text(
        json.dumps(config.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "samples.jsonl").open("wb") as stream:
        for line in sample_lines:
            stream.write(line)
            stream.write(b"\n")
    validate_dataset(output)
    elapsed = time.time() - started
    estimated_full_hours = 33.0 if not pilot else None
    return {
        "families": families,
        "samples": len(samples),
        "candidate_transitions": len(samples) * HORIZON_STEPS // len(EVALUATED_K),
        "parent_artifact_sha256": PARENT_ARTIFACT_SHA256,
        "config_sha256": config.config_sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "samples_sha256": manifest.samples_sha256,
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
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    families = args.families
    if families is None:
        families = DEFAULT_FAMILIES_PILOT if args.pilot else DEFAULT_FAMILIES_FULL
    if families < 1 or families > DEFAULT_FAMILIES_FULL:
        raise SystemExit("families must be in the range 1..384")
    pilot = args.pilot or families <= 32
    if pilot and families > 32:
        raise SystemExit("pilot family cap is 32 per preregistration")
    result = collect(
        families=families,
        output=args.output,
        pilot=pilot,
        scenario_path=args.scenario,
        contract_path=args.contract,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
