"""Focused sealed tests for the fail-closed V2 FIT+CAL launcher."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from aeolus.habitat_v2.forecast.contracts import (
    canonical_json_bytes,
    load_forecast_contracts,
)
from aeolus.habitat_v2.forecast.pilot import load_approved_pilot_design
from aeolus.habitat_v2.forecast.pilot_campaign import PilotCampaignError
from aeolus.habitat_v2.forecast.qualification_split import (
    build_qualification_split,
    load_qualified_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_v2_fitcal_qualified.py"
spec = importlib.util.spec_from_file_location("fitcal_launcher", SCRIPT)
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


def _manifest(allowed: frozenset[str]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "aeolus_habitat_v2_forecast_pilot_campaign_manifest_v1",
        "pairs_completed": launcher.AUTHORIZED_PACKETS,
        "hmc_runs_executed": launcher.AUTHORIZED_EXAMPLES,
        "allowed_cluster_ids": sorted(allowed),
    }
    body["campaign_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _arrays(n: int = 2) -> launcher.CorpusArrays:
    return launcher.CorpusArrays(
        history=np.ones((n, 16, 194), dtype=np.float32),
        available=np.ones((n, 16, 167), dtype=np.bool_),
        action=np.ones((n, 27), dtype=np.float32),
        action_present=np.asarray([True] * n),
        targets=np.ones((n, 8, 51), dtype=np.float32),
        clusters=tuple("fit" for _ in range(n)),
    )


def test_dry_run_verifies_protocol_and_never_creates_or_accesses_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "_assert_source_provenance", lambda *_: "sealed")
    launcher.run(ROOT, tmp_path / "never-created", dry_run=True)
    assert not (tmp_path / "never-created").exists()


def test_resume_preserves_existing_custody_validated_output_and_uses_exact_allowlist(tmp_path: Path) -> None:
    design = load_approved_pilot_design(ROOT)
    contracts = load_forecast_contracts(ROOT)
    split = build_qualification_split(design, load_qualified_protocol(ROOT))
    preflight = launcher._load_pinned_resource_preflight(ROOT)
    corpus = tmp_path / "corpus"; corpus.mkdir(); sentinel = corpus / "validated-pair"; sentinel.mkdir()
    received: dict[str, object] = {}
    def runner(*args: object, **kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        (corpus / "campaign-manifest.json").write_bytes(canonical_json_bytes(_manifest(split.authorized_cluster_ids)))
        return _manifest(split.authorized_cluster_ids)
    launcher._generate_or_resume_corpus(ROOT, corpus, design=design, contracts=contracts, preflight=preflight, allowed_cluster_ids=split.authorized_cluster_ids, runner=runner)
    assert sentinel.exists()
    assert received["resume"] is True
    assert "allowed_cluster_ids" not in received
    assert received["pair_limit"] is None


def test_corrupt_partial_pair_fails_closed_without_deletion(tmp_path: Path) -> None:
    design = load_approved_pilot_design(ROOT); contracts = load_forecast_contracts(ROOT)
    split = build_qualification_split(design, load_qualified_protocol(ROOT))
    preflight = launcher._load_pinned_resource_preflight(ROOT)
    corpus = tmp_path / "corpus"; bad = corpus / "corrupt-pair"; bad.mkdir(parents=True); (bad / "partial").write_text("do not delete", encoding="utf-8")
    with pytest.raises(PilotCampaignError):
        launcher._generate_or_resume_corpus(ROOT, corpus, design=design, contracts=contracts, preflight=preflight, allowed_cluster_ids=split.authorized_cluster_ids)
    assert (bad / "partial").read_text(encoding="utf-8") == "do not delete"


def test_action_aware_action_blind_and_persistence_baseline_paths() -> None:
    arrays = _arrays()
    blind = launcher._features(arrays, object(), action_aware=False)
    aware = launcher._features(arrays, object(), action_aware=True)
    assert blind.shape == (2, 5776)
    assert aware.shape == (2, 5804)
    original = launcher.compact_target_history
    launcher.compact_target_history = lambda history, _layout, **_: history[:, :51]  # type: ignore[assignment]
    try:
        persistence = launcher._persistence_predictions(arrays, object())
    finally:
        launcher.compact_target_history = original
    assert persistence.shape == (2, 8, 51)
    assert np.all(persistence == 1.0)


def test_cal_metric_and_strict_gate_boundary() -> None:
    targets = np.zeros((1, 8, 51), dtype=np.float32); scale = np.ones(51, dtype=np.float32)
    good = launcher._metrics(np.full_like(targets, 0.5), targets, scale)
    tied = launcher._metrics(np.full_like(targets, 1.0), targets, scale)
    persistence = launcher._metrics(np.full_like(targets, 1.0), targets, scale)
    assert good["aggregate_normalized_mae"] == pytest.approx(0.5)
    assert launcher._cal_gate(good, persistence)
    assert not launcher._cal_gate(tied, persistence)


def test_qualification_custody_cli_has_no_validation_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    monkeypatch.setattr(sys, "argv", ["qual_v2_prepare.py", "custody", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(ROOT / "scripts" / "qual_v2_prepare.py"), run_name="__main__")


def test_runtime_guard_refuses_existing_lock_without_deleting_it(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.qualified_runtime_guard import QualifiedRuntimeGuard, QualifiedRuntimeGuardError, QualifiedRuntimeLimits
    lock = tmp_path / ".aeolus-v2-qualified.lock"; lock.write_text("do not delete", encoding="utf-8")
    guard = QualifiedRuntimeGuard(tmp_path, QualifiedRuntimeLimits(1, 1, 1, 82, 1))
    with pytest.raises(QualifiedRuntimeGuardError, match="exclusive"):
        guard.__enter__()
    assert lock.read_text(encoding="utf-8") == "do not delete"


def test_caller_selected_cluster_roster_is_refused_before_pair_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast import pilot_campaign
    design = load_approved_pilot_design(ROOT); contracts = load_forecast_contracts(ROOT)
    monkeypatch.setattr(pilot_campaign, "run_pilot_pair", lambda *args: pytest.fail("must not execute HMC"))
    with pytest.raises(PilotCampaignError, match="unknown roster IDs"):
        pilot_campaign.run_pilot_campaign(ROOT, design, contracts, preflight=launcher._load_pinned_resource_preflight(ROOT), output_root=tmp_path, allowed_cluster_ids=frozenset({"validation"}), pair_limit=1, worker_count=1, resume=False)
