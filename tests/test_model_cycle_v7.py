"""Focused tests for V7 calibration observability and the eval bound."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from aeolus import model_cycle_v7
from aeolus.model_cycle_v7 import _calibrate, _eval_exceeds_bound


def _fake_metrics() -> dict[str, object]:
    return {
        "named_fault_macro_f1": 0.5,
        "healthy_alert_episodes_per_1000_ticks": 1.0,
        "post_onset_uncertainty_fraction": 0.1,
        "named_detection_recall": 0.8,
    }


def _patch_calibration_deps(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace runner dependencies with lightweight fakes for grid-shape tests."""
    calls: dict[str, int] = {"count": 0}

    def fake_evaluate(streams, policy, window_ticks: int) -> dict[str, object]:
        calls["count"] += 1
        return _fake_metrics()

    monkeypatch.setattr(model_cycle_v7, "evaluate_v6", fake_evaluate)
    monkeypatch.setattr(
        model_cycle_v7, "V7EscalatedRulePolicy", lambda config, parameters: object()
    )
    monkeypatch.setattr(
        model_cycle_v7,
        "V7GatedResidualPolicy",
        lambda config, classifier, min_confidence, parameters: object(),
    )
    return calls


def test_calibrate_writes_progress_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_calibration_deps(monkeypatch)
    progress = tmp_path / "calibration-progress.log"
    report = _calibrate(
        classifier=None,  # type: ignore[arg-type]  # policy constructors are faked
        reference_config=None,
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


def test_calibrate_skips_pathological_confidence_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_cycle_v7, "V7_EVAL_BOUND_MIN_SECONDS", 0.01)
    monkeypatch.setattr(model_cycle_v7, "V7_EVAL_BOUND_FACTOR", 5.0)
    calls = _patch_calibration_deps(monkeypatch)

    def slow_last_combo_first_confidence(streams, policy, window_ticks: int) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 212:  # last combo's first confidence eval
            time.sleep(0.05)
        return _fake_metrics()

    monkeypatch.setattr(model_cycle_v7, "evaluate_v6", slow_last_combo_first_confidence)
    progress = tmp_path / "calibration-progress.log"
    report = _calibrate(
        classifier=None,  # type: ignore[arg-type]  # policy constructors are faked
        reference_config=None,
        streams=[],
        window_ticks=10,
        progress_path=progress,
    )
    text = progress.read_text(encoding="utf-8")
    assert "SKIPPED" in text
    assert report["calibration_evals"] == 212  # 216 planned, tail of last combo skipped
    assert report["calibration_pathological_evals"] == 1  # one bound trip ended the combo
    assert report["candidate_count"] == 176  # 180 candidates minus the 4 skipped
    assert "eval=216/216" not in text


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
