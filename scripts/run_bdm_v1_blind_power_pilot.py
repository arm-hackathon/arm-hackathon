"""Blind-size power pilot for the BDM-v1 family roster (Issue #72, part 3/3).

Runs the preregistered pilot that replaces the custody registry's
``TBD_FROM_PILOT`` blind-size justification, as required by the BDM-v1
benchmark contract (``split_custody.blind_size_rule`` and
``thresholds.pilot_definition``). The pilot uses DEV-partition data only.

Pilot rule (declared before execution):

- population: the first ``PILOT_GROUP_COUNT`` DEV causal groups in custody
  registry order, both paired variants each (16 families);
- episode: the Issue #55 ``rules_only`` arm (HMC hold, zero proposals) with
  full 96-step shadow-receipt verification, reusing the preregistered
  ``safety_exposure`` / ``comfort_deviation`` / ``resource_composite``
  metrics unchanged;
- primary sizing rule (R1, zero-event binomial): the hold arm shows zero
  safety exposure, so the blind gate is a zero-event demonstration; with
  ``N`` blind groups and zero harmful groups observed, the one-sided
  ``1 - ALPHA`` upper bound on the harmful-group rate is
  ``1 - ALPHA ** (1 / N)``, and a true harmful rate of
  ``HARMFUL_RATE_PROBE`` is rejected at level ``ALPHA`` when
  ``(1 - HARMFUL_RATE_PROBE) ** N <= ALPHA``;
- secondary sizing rule (R2, continuous proxy): required groups for
  ``POWER_TARGET`` power at two-sided ``ALPHA`` against a
  ``COMFORT_MDE_FRACTION`` relative shift in group-mean comfort deviation,
  assuming the conservative independent-arm difference SD
  ``sqrt(2) * SD_group``; diagnostic only, never the primary gate;
- contract floor (R3): the blind population must materially exceed
  ``CONTRACT_FLOOR_GROUPS`` independent evaluation condition groups.

The receipt is written once under ``out/`` and the deterministic
justification string is printed for the roster generator's
``--blind-size-justification`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.bdm_v1_families import (
    GENERATOR_VERSION,
    GeneratorConfig,
    assign_partitions,
    build_family,
    load_base_scenario_data,
)
from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.forecast_issue55_race import run_race_episode
from aeolus.habitat_v2.physics_provenance import load_physics_provenance_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json"
RECEIPT_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_power_pilot_receipt_v1"

PILOT_GROUP_COUNT = 8
ALPHA = 0.05
POWER_TARGET = 0.80
Z_ALPHA_TWO_SIDED = 1.959963984540054
Z_POWER = 0.8416212335729143
COMFORT_MDE_FRACTION = 0.5
HARMFUL_RATE_PROBE = 0.25
CONTRACT_FLOOR_GROUPS = 3


class PowerPilotError(RuntimeError):
    """Raised when the pilot cannot be executed or its rules fail."""


def zero_event_upper_bound(groups: int, alpha: float = ALPHA) -> float:
    """One-sided upper bound on a harmful-group rate after zero events."""

    if groups < 1 or not 0.0 < alpha < 1.0:
        raise PowerPilotError("zero-event bound requires groups >= 1 and 0 < alpha < 1")
    return 1.0 - alpha ** (1.0 / groups)


def rejects_rate_at_alpha(groups: int, rate: float, alpha: float = ALPHA) -> bool:
    """True when observing zero events in ``groups`` rejects ``rate`` at alpha."""

    if not 0.0 < rate < 1.0:
        raise PowerPilotError("probe rate must lie strictly between 0 and 1")
    return (1.0 - rate) ** groups <= alpha


def continuous_required_groups(
    group_sd: float, group_mean: float, mde_fraction: float = COMFORT_MDE_FRACTION
) -> int:
    """Groups needed for the declared continuous proxy power target."""

    if group_sd < 0.0 or not math.isfinite(group_sd):
        raise PowerPilotError("group SD must be finite and non-negative")
    if group_mean <= 0.0 or not 0.0 < mde_fraction < 1.0:
        raise PowerPilotError("continuous rule requires a positive mean and MDE fraction")
    difference_sd_squared = 2.0 * group_sd * group_sd
    mde = mde_fraction * group_mean
    required = (Z_ALPHA_TWO_SIDED + Z_POWER) ** 2 * difference_sd_squared / (mde * mde)
    return max(1, math.ceil(required))


def _resolve_output(output_path: Path) -> Path:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT / "out")
    except ValueError as error:
        raise PowerPilotError("pilot output must live under the ignored out/ directory") from error
    if output.exists():
        raise PowerPilotError(f"refusing to overwrite existing output directory {output}")
    output.mkdir(parents=True)
    return output


def run_pilot(output: Path) -> dict[str, Any]:
    registry_raw = REGISTRY_PATH.read_bytes()
    registry = json.loads(registry_raw)
    registry_sha256 = hashlib.sha256(registry_raw).hexdigest()
    dev_rows = [row for row in registry["groups"] if row["partition"] == "DEV"]
    if len(dev_rows) < PILOT_GROUP_COUNT:
        raise PowerPilotError("registry does not carry enough DEV groups for the pilot")
    pilot_rows = dev_rows[:PILOT_GROUP_COUNT]

    base_data, base_sha256 = load_base_scenario_data(REPO_ROOT)
    manifest, manifest_sha256 = load_physics_provenance_manifest(REPO_ROOT)
    bundle = load_forecast_contracts(REPO_ROOT)
    config = GeneratorConfig()
    plans = {plan.group_index: plan for plan in assign_partitions(config)}

    group_records: list[dict[str, Any]] = []
    comfort_means: list[float] = []
    exposure_values: list[float] = []
    resource_values: list[float] = []
    for row in pilot_rows:
        variant_rows: list[dict[str, Any]] = []
        comforts: list[float] = []
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
                raise PowerPilotError(
                    f"pilot family {definition.family_id} does not match the registry digest"
                )
            record = run_race_episode(
                bundle,
                scenario,
                "rules_only",
                definition.family_id,
                row["group_index"] * 2 + variant_index,
                None,
            )
            if record.proposal_count != 0 or record.replay_committed_steps != 96:
                raise PowerPilotError("pilot episode violated the hold-replay rule")
            exposure_values.append(record.safety_exposure)
            resource_values.append(record.resource_composite)
            comforts.append(record.comfort_deviation)
            variant_rows.append(
                {
                    "family_id": definition.family_id,
                    "variant": variant,
                    "episode_sha256": record.episode_sha256,
                    "safety_exposure": record.safety_exposure,
                    "safety_violation_steps": record.safety_violation_steps,
                    "comfort_deviation": record.comfort_deviation,
                    "resource_composite": record.resource_composite,
                }
            )
            print(
                f"  {definition.family_id}: exposure={record.safety_exposure:.6f} "
                f"comfort={record.comfort_deviation:.6f}",
                file=sys.stderr,
            )
        comfort_means.append(statistics.mean(comforts))
        group_records.append(
            {
                "group_id": row["group_id"],
                "template_id": row["template_id"],
                "stratum": row["stratum"],
                "variants": variant_rows,
                "group_mean_comfort_deviation": statistics.mean(comforts),
                "group_mean_safety_exposure": 0.0,
            }
        )

    if any(value != 0.0 for value in exposure_values):
        raise PowerPilotError(
            "pilot rule R1 assumes a zero-exposure hold arm; observed non-zero "
            "safety exposure invalidates the declared sizing rule and requires "
            "a revised preregistered pilot"
        )
    if any(value != 0.0 for value in resource_values):
        raise PowerPilotError("pilot observed non-zero hold resource depletion; revise the declared rule")

    comfort_mean = statistics.mean(comfort_means)
    comfort_sd = statistics.stdev(comfort_means)
    declared_blind_groups = int(registry["blind_seal"]["group_count"])
    r1_bound = zero_event_upper_bound(declared_blind_groups)
    r1_rejects = rejects_rate_at_alpha(declared_blind_groups, HARMFUL_RATE_PROBE)
    r2_required = continuous_required_groups(comfort_sd, comfort_mean)
    if not r1_rejects:
        raise PowerPilotError(
            f"blind size {declared_blind_groups} cannot reject the declared harmful rate; enlarge the blind partition"
        )
    if r2_required > declared_blind_groups:
        raise PowerPilotError(
            f"continuous proxy requires {r2_required} groups but only {declared_blind_groups} are declared"
        )
    if declared_blind_groups <= CONTRACT_FLOOR_GROUPS:
        raise PowerPilotError("blind size does not materially exceed the contract floor")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "pilot_rule": {
            "population": f"first {PILOT_GROUP_COUNT} DEV causal groups in registry order, both paired variants",
            "arm": "rules_only (HMC hold, zero proposals, full shadow-receipt verification)",
            "episode_steps": 96,
            "primary_metric": "safety_exposure (Issue #55 preregistered definition)",
            "secondary_proxy": "comfort_deviation group means",
        },
        "registry_sha256_at_pilot": registry_sha256,
        "generator_version": GENERATOR_VERSION,
        "base_scenario_sha256": base_sha256,
        "provenance_manifest_sha256": manifest_sha256,
        "declared_constants": {
            "alpha": ALPHA,
            "power_target": POWER_TARGET,
            "comfort_mde_fraction": COMFORT_MDE_FRACTION,
            "harmful_rate_probe": HARMFUL_RATE_PROBE,
            "contract_floor_groups": CONTRACT_FLOOR_GROUPS,
        },
        "groups": group_records,
        "observations": {
            "families_replayed": len(exposure_values),
            "safety_exposure_all_zero": True,
            "resource_composite_all_zero": True,
            "comfort_group_mean": comfort_mean,
            "comfort_group_sd": comfort_sd,
        },
        "power_analysis": {
            "r1_zero_event_upper_bound_on_harmful_group_rate": r1_bound,
            "r1_rejects_declared_harmful_rate": r1_rejects,
            "r2_continuous_proxy_required_groups": r2_required,
            "r3_contract_floor_groups": CONTRACT_FLOOR_GROUPS,
            "declared_blind_groups": declared_blind_groups,
            "required_blind_groups": declared_blind_groups,
        },
    }
    receipt_raw = canonical_json_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    receipt["receipt_sha256"] = receipt_sha256
    (output / "power-pilot-receipt.json").write_bytes(receipt_raw)

    justification = (
        f"pilot_receipt:{receipt_sha256}"
        f":declared_blind_groups={declared_blind_groups}"
        f":harmful_rate_detectable={HARMFUL_RATE_PROBE}"
        f":alpha={ALPHA}"
        f":zero_event_upper_bound={round(r1_bound, 4)}"
        f":comfort_proxy_required_groups={r2_required}"
        f":contract_floor={CONTRACT_FLOOR_GROUPS}"
    )
    receipt_out = dict(receipt)
    receipt_out["justification"] = justification
    (output / "power-pilot-summary.json").write_bytes(canonical_json_bytes(receipt_out))
    print(json.dumps({"justification": justification, "receipt_sha256": receipt_sha256}, indent=1))
    return receipt_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _resolve_output(args.output)
    run_pilot(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
