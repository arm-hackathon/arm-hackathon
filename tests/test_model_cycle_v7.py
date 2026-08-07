"""Focused tests for V7 calibration observability, the eval bound, and parallel execution."""

from __future__ import annotations

import pickle
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus import model_cycle_v7
from aeolus.model_cycle_v7 import _calibrate, _calibrate_combo_task, _eval_exceeds_bound


def _fake_metrics() -> dict[str, object]:
    return {
        "named_fault_macro_f1": 0.5,
        "healthy_alert_episodes_per_1000_ticks": 1.0,
        "post_onset_uncertainty_fraction": 0.1,
        "named_detection_recall": 0.8,
    }


def _patch_calibration_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace runner dependencies with lightweight fakes for grid-shape tests."""
    monkeypatch.setattr(model_cycle_v7, "evaluate_v6", lambda *args, **kwargs: _fake_metrics())
    monkeypatch.setattr(
        model_cycle_v7, "V7EscalatedRulePolicy", lambda config, parameters: object()
    )
    monkeypatch.setattr(
        model_cycle_v7,
        "V7GatedResidualPolicy",
        lambda config, classifier, min_confidence, parameters: object(),
    )


def test_calibrate_writes_progress_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_calibration_deps(monkeypatch)
    progress = tmp_path / "calibration-progress.log"
    report = _calibrate(
        classifier=object(),  # type: ignore[arg-type]  # policy constructors are faked
        reference_config=object(),
        streams=[],
        window_ticks=10,
        progress_path=progress,
    )
    lines = progress.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 216
    assert lines[0].startswith("eval=1/216 role=baseline")
    assert lines[-1].startswith("eval=216/216 role=candidate")
    assert "SKIPPED" not in progress.read_text(encoding="utf-8")
    assert report["candidate_count"] == 180
    assert report["calibration_evals"] == 216
    assert report["calibration_pathological_evals"] == 0
    assert report["calibration_parallel"] is True


def test_calibrate_is_deterministic_across_parallel_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_calibration_deps(monkeypatch)
    first = _calibrate(
        classifier=object(),  # type: ignore[arg-type]
        reference_config=object(),
        streams=[],
        window_ticks=10,
        progress_path=tmp_path / "first.log",
    )
    second = _calibrate(
        classifier=object(),  # type: ignore[arg-type]
        reference_config=object(),
        streams=[],
        window_ticks=10,
        progress_path=tmp_path / "second.log",
    )
    assert first["selected_parameters"] == second["selected_parameters"]
    assert first["selected_min_confidence"] == second["selected_min_confidence"]
    assert first["selected_metrics"] == second["selected_metrics"]


def test_calibrate_combo_task_applies_bound_within_combo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_cycle_v7, "V7_EVAL_BOUND_MIN_SECONDS", 0.01)
    monkeypatch.setattr(model_cycle_v7, "V7_EVAL_BOUND_FACTOR", 5.0)
    calls: dict[str, int] = {"count": 0}

    def slow_fifth_eval(streams, policy, window_ticks: int) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 5:  # baseline + confidences 0.0/0.3/0.5 fast, 0.7 slow
            time.sleep(0.05)
        return _fake_metrics()

    monkeypatch.setattr(model_cycle_v7, "evaluate_v6", slow_fifth_eval)
    monkeypatch.setattr(
        model_cycle_v7, "V7EscalatedRulePolicy", lambda config, parameters: object()
    )
    monkeypatch.setattr(
        model_cycle_v7,
        "V7GatedResidualPolicy",
        lambda config, classifier, min_confidence, parameters: object(),
    )
    model_cycle_v7._init_worker(
        object(), object(), [], 10  # type: ignore[arg-type]  # policy constructors are faked
    )

    combo, candidates, progress, pathological = _calibrate_combo_task(
        0.3, 0.01, 0.15, 0.05
    )
    assert combo == (0.3, 0.01, 0.15, 0.05)
    assert calls["count"] == 5  # confidence 0.9 was never evaluated
    assert pathological == 1
    assert len(candidates) == 4
    assert len(progress) == 5
    assert progress[-1]["skipped"] is True
    assert all(record["skipped"] is False for record in progress[:-1])


def test_calibrate_combo_task_requires_initialized_context() -> None:
    model_cycle_v7._init_worker(None, None, None, 10)
    with pytest.raises(ValueError, match="worker context"):
        _calibrate_combo_task(0.2, 0.0025, 0.1, 0.02)


def test_calibrate_worker_inputs_are_picklable() -> None:
    from aeolus.config import load_scenario
    from aeolus.evaluate_v6 import V6EvaluationStream
    from aeolus.scenario import run_scenario
    from aeolus.v7_centroid import V7ResidualCentroid

    config = load_scenario("scenarios/standard_habitat.json")
    records = tuple(run_scenario(config))
    stream = V6EvaluationStream(
        family_id="family-1",
        room_family_id="room-1",
        split="calibration",
        scenario_role="reference",
        records=records,
        reference_identity="ref-sha",
    )
    classifier = V7ResidualCentroid.fit(
        [[0.0, 0.0, 0.0, 0.0, 0.0]], ["nominal"], feature_width=5
    )
    for value in (config, stream, classifier):
        pickle.loads(pickle.dumps(value))


def test_eval_exceeds_bound_requires_absolute_and_relative_excess() -> None:
    assert not _eval_exceeds_bound(299.0, [100.0])
    assert not _eval_exceeds_bound(301.0, [])  # no completed baseline yet
    assert not _eval_exceeds_bound(301.0, [100.0])  # 3.0x median, below factor
    assert _eval_exceeds_bound(301.0, [60.0])  # 5.0x+ median and over the floor
    assert _eval_exceeds_bound(1000.0, [100.0, 100.0, 100.0, 100.0, 100.0])


def test_model_input_contract_validation_is_memoized() -> None:
    from aeolus.config import load_scenario
    from aeolus.model_input import _validate_contract, build_model_input_contract

    config = load_scenario("scenarios/standard_habitat.json")
    contract = build_model_input_contract(config)
    before = _validate_contract.cache_info()
    topology_a = _validate_contract(contract)
    topology_b = _validate_contract(contract)
    info = _validate_contract.cache_info()
    assert topology_a == topology_b
    assert info.hits > before.hits
    assert info.currsize >= 1


def test_observable_context_contract_validation_is_memoized() -> None:
    from aeolus.config import load_scenario
    from aeolus.observable_context import (
        _validate_contract,
        build_observable_context_contract,
    )

    config = load_scenario("scenarios/standard_habitat.json")
    contract = build_observable_context_contract(config)
    before = _validate_contract.cache_info()
    _validate_contract(contract)
    _validate_contract(contract)
    info = _validate_contract.cache_info()
    assert info.hits > before.hits
    assert info.currsize >= 1


def test_contract_validation_cache_does_not_poison_errors() -> None:
    from aeolus.config import load_scenario
    from aeolus.model_input import _validate_contract, build_model_input_contract

    config = load_scenario("scenarios/standard_habitat.json")
    contract = build_model_input_contract(config)
    _validate_contract(contract)
    broken = replace(contract, selector_hash="deadbeef")
    with pytest.raises(ValueError):
        _validate_contract(broken)
    with pytest.raises(ValueError):
        _validate_contract(broken)  # exceptions are not cached
