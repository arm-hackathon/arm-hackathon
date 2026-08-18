"""Tests for the AEOLUS guided tour (scripts/aeolus_tour.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "aeolus_tour", REPO_ROOT / "scripts/aeolus_tour.py"
)
assert SPEC is not None and SPEC.loader is not None
tour = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("aeolus_tour", tour)
SPEC.loader.exec_module(tour)


def test_replay_renders_both_arms_with_events(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(tour, "_pause", lambda seconds=0.0: None)
    tour.replay_paired_experiment()
    output = capsys.readouterr().out
    assert "CONTROL" in output and "ADVISED" in output
    assert "HMC ACCEPTED" in output
    assert "HMC OVERRIDDEN" in output
    assert "19.94" in output  # control integrated exceedance
    assert "0.00" in output  # advised integrated exceedance


def test_replay_refuses_tampered_artifact(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        tour, "REPLAY_ARTIFACT", REPO_ROOT / "pyproject.toml",
    )
    tour.replay_paired_experiment()  # must refuse without raising
    assert "refusing" in capsys.readouterr().out


def test_live_forecast_runs_with_visitor_action_choice(monkeypatch, capsys) -> None:
    answers = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    tour.run_live_forecast()
    output = capsys.readouterr().out
    assert "normal-eva_transition-v1" in output
    assert "<- your choice" in output
    assert "ACCEPTED" in output or "OVERRIDDEN" in output


def test_menu_quit_path(monkeypatch, capsys) -> None:
    answers = iter(["4", "", "5"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert tour.menu() == 0
    output = capsys.readouterr().out
    assert "THE PLANT" in output  # explanation was shown
    assert "Thanks for visiting" in output
