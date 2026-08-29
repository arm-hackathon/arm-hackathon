"""Read-only oracle-ceiling diagnosis for the Issue #56 V4 useful-action gates.

This diagnostic reads a finalized V4 corpus `samples.jsonl` and computes, from
ground-truth labels only, how many useful and distinct actions any model could
select on a candidate split. It never trains, tunes, or writes evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from aeolus.habitat_v2.forecast_issue55_race import deterministic_family_ids


REPO_ROOT = Path(__file__).resolve().parents[1]


class CeilingDiagnosisError(ValueError):
    """Raised when the ceiling diagnosis cannot proceed."""


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise CeilingDiagnosisError(f"non-finite JSON value {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CeilingDiagnosisError("duplicate JSON key in corpus row")
            result[key] = value
        return result

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=reject_duplicates,
                    parse_constant=reject_constant,
                )
            except (json.JSONDecodeError, CeilingDiagnosisError) as error:
                raise CeilingDiagnosisError(
                    f"corpus row {line_number} is malformed: {error}"
                ) from error
            if type(row) is not dict:
                raise CeilingDiagnosisError(f"corpus row {line_number} is not an object")
            rows.append(row)
    return rows


def _row_identity(row: dict[str, Any]) -> tuple[str, str, int, str]:
    base = row.get("base_sample")
    if type(base) is not dict:
        raise CeilingDiagnosisError("corpus row lacks a base sample")
    family_id = base.get("family_id")
    split = base.get("split")
    decision_step = base.get("decision_step")
    action_id = base.get("action_id")
    if (
        type(family_id) is not str
        or type(split) is not str
        or type(decision_step) is not int
        or type(action_id) is not str
    ):
        raise CeilingDiagnosisError("corpus row identity is malformed")
    return family_id, split, decision_step, action_id


def _horizon_events(label: dict[str, Any]) -> dict[str, float]:
    events: dict[str, float] = {}
    for metric in label.get("horizon_metrics", []):
        horizon = metric.get("horizon_steps")
        event = metric.get("crossing_event")
        if type(horizon) is not int or type(event) not in {int, float}:
            raise CeilingDiagnosisError("horizon metric is malformed")
        events[str(horizon)] = float(event)
    remaining = label.get("remaining_metric")
    if type(remaining) is not dict or type(remaining.get("crossing_event")) not in {int, float}:
        raise CeilingDiagnosisError("remaining metric is malformed")
    events["remaining"] = float(remaining["crossing_event"])
    return events


def _safety_delta(row: dict[str, Any]) -> float:
    targets = row.get("relative_action_targets")
    if type(targets) is not dict:
        raise CeilingDiagnosisError("corpus row lacks relative targets")
    delta = targets.get("safety_exposure_delta_vs_hold")
    if type(delta) not in {int, float}:
        raise CeilingDiagnosisError("relative safety delta is malformed")
    return float(delta)


def _condition_group(family_id: str) -> str:
    roster = deterministic_family_ids(32)
    if family_id not in roster:
        raise CeilingDiagnosisError(f"family {family_id} is outside the frozen roster")
    return f"condition-group-{roster.index(family_id) // 2:04d}"


def oracle_split_metrics(
    rows: list[dict[str, Any]],
    group_split: dict[str, str],
) -> dict[str, Any]:
    """Compute oracle ceilings for one group->split assignment."""

    decisions: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    horizon_support: dict[str, dict[str, dict[str, int]]] = {
        split: {
            horizon: {"positives": 0, "negatives": 0}
            for horizon in ("4", "16", "32", "remaining")
        }
        for split in ("TRAIN", "VALIDATION", "EVALUATION")
    }
    dangerous_by_group: dict[str, int] = defaultdict(int)
    for row in rows:
        family_id, _, decision_step, action_id = _row_identity(row)
        group = _condition_group(family_id)
        split = group_split[group]
        label = row.get("base_sample", {}).get("label")
        if type(label) is not dict:
            raise CeilingDiagnosisError("corpus row lacks a label")
        events = _horizon_events(label)
        for horizon, event in events.items():
            bucket = horizon_support[split][horizon]
            if event >= 0.5:
                bucket["positives"] += 1
            else:
                bucket["negatives"] += 1
        if events["remaining"] >= 0.5:
            dangerous_by_group[group] += 1
        targets = row.get("relative_action_targets")
        if type(targets) is not dict:
            raise CeilingDiagnosisError("corpus row lacks relative targets")
        comfort_delta = targets.get("comfort_deviation_delta_vs_hold")
        resource_delta = targets.get("resource_composite_delta_vs_hold")
        if type(comfort_delta) not in {int, float} or type(resource_delta) not in {int, float}:
            raise CeilingDiagnosisError("relative comfort/resource deltas are malformed")
        decisions[(family_id, decision_step)].append(
            {
                "action_id": action_id,
                "group": group,
                "split": split,
                "dangerous": events["remaining"] >= 0.5,
                "useful": _safety_delta(row) < 0.0,
                "safety_delta": _safety_delta(row),
                "comfort_delta": float(comfort_delta),
                "resource_delta": float(resource_delta),
            }
        )

    def split_ceiling(split: str) -> dict[str, Any]:
        useful_decisions = 0
        useful_and_safe_decisions = 0
        distinct_useful: set[str] = set()
        distinct_useful_and_safe: set[str] = set()
        argmin_policy_actions: set[str] = set()
        composite_policy_actions: set[str] = set()
        composite_policy_useful = 0
        multi_option_decisions = 0
        decision_count = 0
        for _, actions in decisions.items():
            if not actions or actions[0]["split"] != split:
                continue
            decision_count += 1
            useful = [item for item in actions if item["useful"]]
            safe_and_useful = [item for item in useful if not item["dangerous"]]
            if len(safe_and_useful) >= 2:
                multi_option_decisions += 1
            if useful:
                useful_decisions += 1
                distinct_useful.update(item["action_id"] for item in useful)
            if safe_and_useful:
                useful_and_safe_decisions += 1
                distinct_useful_and_safe.update(item["action_id"] for item in safe_and_useful)
                best = min(safe_and_useful, key=lambda item: (item["safety_delta"], item["action_id"]))
                argmin_policy_actions.add(best["action_id"])
                composite_best = min(
                    safe_and_useful,
                    key=lambda item: (
                        item["safety_delta"]
                        + 0.25 * item["comfort_delta"]
                        + 0.10 * item["resource_delta"],
                        item["action_id"],
                    ),
                )
                composite_policy_actions.add(composite_best["action_id"])
                composite_policy_useful += 1
        return {
            "decision_count": decision_count,
            "oracle_useful_decisions": useful_decisions,
            "oracle_useful_and_safe_decisions": useful_and_safe_decisions,
            "oracle_distinct_useful": len(distinct_useful),
            "oracle_distinct_useful_and_safe": len(distinct_useful_and_safe),
            "argmin_policy_distinct": len(argmin_policy_actions),
            "composite_policy_distinct": len(composite_policy_actions),
            "composite_policy_useful": composite_policy_useful,
            "multi_option_decisions": multi_option_decisions,
        }

    constraints: dict[str, Any] = {}
    for split in ("TRAIN", "VALIDATION", "EVALUATION"):
        horizons = horizon_support[split]
        constraints[split] = {
            "positive_horizons": sum(
                horizons[horizon]["positives"] >= 2 for horizon in horizons
            ),
            "all_horizons_have_positives": all(
                horizons[horizon]["positives"] >= 2 for horizon in horizons
            ),
            "all_horizons_have_negatives": all(
                horizons[horizon]["negatives"] >= 1 for horizon in horizons
            ),
            "horizon_support": horizons,
        }
    return {
        "evaluation": split_ceiling("EVALUATION"),
        "constraints": constraints,
        "dangerous_samples_by_group": dict(sorted(dangerous_by_group.items())),
        "group_split": group_split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only oracle-ceiling diagnosis for the V4 useful-action gates"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    corpus = (REPO_ROOT / args.corpus).resolve() if not args.corpus.is_absolute() else args.corpus
    samples_path = corpus / "samples.jsonl"
    if not samples_path.is_file():
        raise CeilingDiagnosisError(f"corpus samples are missing: {samples_path}")
    rows = _strict_jsonl(samples_path)

    groups = [f"condition-group-{index // 2:04d}" for index in range(0, 32, 2)]

    current = {
        **{group: "TRAIN" for group in groups},
        "condition-group-0000": "VALIDATION",
        "condition-group-0004": "VALIDATION",
        "condition-group-0012": "VALIDATION",
        "condition-group-0001": "EVALUATION",
        "condition-group-0005": "EVALUATION",
        "condition-group-0013": "EVALUATION",
    }
    swap_g2_for_g13 = {
        **{group: "TRAIN" for group in groups},
        "condition-group-0000": "VALIDATION",
        "condition-group-0004": "VALIDATION",
        "condition-group-0012": "VALIDATION",
        "condition-group-0001": "EVALUATION",
        "condition-group-0002": "EVALUATION",
        "condition-group-0005": "EVALUATION",
    }
    swap_g3_for_g13 = {
        **{group: "TRAIN" for group in groups},
        "condition-group-0000": "VALIDATION",
        "condition-group-0004": "VALIDATION",
        "condition-group-0012": "VALIDATION",
        "condition-group-0001": "EVALUATION",
        "condition-group-0003": "EVALUATION",
        "condition-group-0005": "EVALUATION",
    }
    report = {
        "corpus": str(corpus),
        "sample_count": len(rows),
        "groups": groups,
        "splits": {
            "current": oracle_split_metrics(rows, current),
            "swap_g2_for_g13": oracle_split_metrics(rows, swap_g2_for_g13),
            "swap_g3_for_g13": oracle_split_metrics(rows, swap_g3_for_g13),
        },
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
