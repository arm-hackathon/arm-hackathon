from __future__ import annotations

from pathlib import Path

from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts
from aeolus.habitat_v2.forecast_issue55_race import build_family_scenario
from aeolus.habitat_v2.forecast_issue56_action_risk_v3 import (
    EPISODE_STEPS,
    ISSUE56_V3_SCHEMA_VERSION,
    V3_HORIZONS,
    V3_LABEL_TRACK,
    collect_v3_family_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v3_labels_bind_hmc_persistence_and_all_command_identities() -> None:
    bundle = load_forecast_contracts(REPO_ROOT)
    scenario = build_family_scenario(bundle.development_scenario, 0)

    samples = collect_v3_family_samples(
        bundle,
        scenario,
        "issue55-v3-test-family",
        split="TRAIN",
    )

    assert len(samples) == 13 * 4
    assert {sample.label.track for sample in samples} == {V3_LABEL_TRACK}
    for sample in samples:
        label = sample.label
        assert label.to_mapping()["schema_version"] == f"{ISSUE56_V3_SCHEMA_VERSION}.label"
        assert tuple(metric.horizon_steps for metric in label.horizon_metrics) == V3_HORIZONS
        assert label.remaining_steps == EPISODE_STEPS - label.decision_step
        assert len(label.state_digests) == label.remaining_steps
        assert label.final_command_sha256 == label.executed_command_sha256
        assert label.requested_command_sha256 != label.current_command_sha256
        assert label.trace_sha256 != label.label_sha256
