"""Group-level evaluation primitives for the BDM-v1 research line.

Issue #73 owns this module (single home): every downstream study (#74-#77)
imports its independent-condition-group statistics instead of re-implementing
them. The independent statistical unit is the causal group per the Issue #70
contract; paired sensor variants, counterfactual branches, and within-family
steps are never treated as independent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random
from typing import Any

BDM_V1_EVALUATION_SCHEMA_VERSION = "aeolus_habitat_v2_bdm_v1_evaluation_v1"
DEFAULT_RESAMPLES = 10000
DEFAULT_ALPHA = 0.05


class BdmV1EvaluationError(ValueError):
    """Raised when group-level evaluation inputs are inadmissible."""


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise BdmV1EvaluationError(f"{label} is non-finite or boolean")
    return float(value)


def group_means(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    group_key: str = "group_id",
) -> dict[str, float]:
    """Average one metric over the families inside each causal group."""

    totals: dict[str, list[float]] = {}
    for row in rows:
        group = str(row[group_key])
        totals.setdefault(group, []).append(_finite(float(row[metric]), metric))
    if not totals:
        raise BdmV1EvaluationError(f"no rows supplied for metric {metric!r}")
    return {group: sum(values) / len(values) for group, values in totals.items()}


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: str,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean over independent groups."""

    data = [_finite(value, "bootstrap value") for value in values]
    if len(data) < 2:
        raise BdmV1EvaluationError("bootstrap requires at least two independent groups")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 100
    ):
        raise BdmV1EvaluationError("bootstrap requires at least 100 resamples")
    if not 0.0 < alpha < 1.0:
        raise BdmV1EvaluationError("alpha must lie strictly between 0 and 1")
    rng = random.Random(seed)
    means: list[float] = []
    n = len(data)
    for _ in range(resamples):
        total = 0.0
        for _draw in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lower_index = max(0, min(n - 1, int(math.floor((alpha / 2.0) * resamples))))
    upper_index = max(0, min(resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * resamples)) - 1))
    observed = sum(data) / n
    return observed, means[lower_index], means[upper_index]


def paired_group_differences(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    group_key: str = "group_id",
) -> dict[str, float]:
    """Per-group mean(metric_a) - mean(metric_b) over identical group sets."""

    means_a = group_means(rows_a, metric, group_key=group_key)
    means_b = group_means(rows_b, metric, group_key=group_key)
    if set(means_a) != set(means_b):
        raise BdmV1EvaluationError("paired comparison requires identical group sets")
    return {group: means_a[group] - means_b[group] for group in means_a}


def comparison_table(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    metric: str,
    *,
    baseline_arm: str,
    seed: str,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    group_key: str = "group_id",
) -> dict[str, Any]:
    """Group-level means, CIs, and paired differences versus one baseline arm."""

    if baseline_arm not in arms:
        raise BdmV1EvaluationError("baseline arm is missing from the comparison set")
    baseline_rows = arms[baseline_arm]
    table: dict[str, Any] = {"schema_version": BDM_V1_EVALUATION_SCHEMA_VERSION, "metric": metric, "baseline_arm": baseline_arm, "arms": {}}
    baseline_means = group_means(baseline_rows, metric, group_key=group_key)
    base_mean, base_lo, base_hi = bootstrap_ci(
        list(baseline_means.values()), seed=f"{seed}|{baseline_arm}", resamples=resamples, alpha=alpha
    )
    table["arms"][baseline_arm] = {
        "group_count": len(baseline_means),
        "mean": base_mean,
        "ci_lower": base_lo,
        "ci_upper": base_hi,
        "paired_difference_mean": 0.0,
        "paired_difference_ci_lower": 0.0,
        "paired_difference_ci_upper": 0.0,
    }
    for arm, rows in arms.items():
        if arm == baseline_arm:
            continue
        means = group_means(rows, metric, group_key=group_key)
        if set(means) != set(baseline_means):
            raise BdmV1EvaluationError(f"arm {arm!r} does not cover the baseline groups")
        mean, lo, hi = bootstrap_ci(
            list(means.values()), seed=f"{seed}|{arm}", resamples=resamples, alpha=alpha
        )
        diffs = paired_group_differences(rows, baseline_rows, metric, group_key=group_key)
        diff_mean, diff_lo, diff_hi = bootstrap_ci(
            list(diffs.values()), seed=f"{seed}|{arm}|paired", resamples=resamples, alpha=alpha
        )
        table["arms"][arm] = {
            "group_count": len(means),
            "mean": mean,
            "ci_lower": lo,
            "ci_upper": hi,
            "paired_difference_mean": diff_mean,
            "paired_difference_ci_lower": diff_lo,
            "paired_difference_ci_upper": diff_hi,
        }
    return table


__all__ = [
    "BDM_V1_EVALUATION_SCHEMA_VERSION",
    "BdmV1EvaluationError",
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "bootstrap_ci",
    "comparison_table",
    "group_means",
    "paired_group_differences",
]
