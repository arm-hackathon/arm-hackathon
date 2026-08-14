from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "raw",
    [
        '{"key":1,"key":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
    ],
)
def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path,
    raw: str,
) -> None:
    from aeolus.habitat_v2.forecast.contracts import (
        ForecastContractError,
        _strict_json,
    )

    candidate = tmp_path / "candidate.json"
    candidate.write_text(raw, encoding="utf-8")

    with pytest.raises(ForecastContractError, match="duplicate|non-finite"):
        _strict_json(candidate)


def test_loader_rejects_frozen_artifact_self_hash_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeolus.habitat_v2.forecast import contracts

    root = Path(__file__).resolve().parents[2]
    original = contracts._strict_json

    def tampered(path: Path) -> object:
        value = original(path)
        if path.name == "habitat_v2_forecast_hmc_binding_v1.json":
            value = dict(value)
            value["final_hmc_tree_sha"] = "0" * 40
        return value

    monkeypatch.setattr(contracts, "_strict_json", tampered)
    with pytest.raises(contracts.ForecastContractError, match="does not bind"):
        contracts.load_forecast_contracts(root)


def test_rejects_current_hmc_source_drift(tmp_path: Path) -> None:
    from aeolus.habitat_v2.forecast.contracts import (
        ForecastContractError,
        _validate_current_source_bytes,
    )

    source = tmp_path / "src" / "authority.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"modified authority bytes")
    manifest_entry = {
        "path": "src/authority.py",
        "sha256": "58d75ce875a7699fba841feb9af7a4f4de9a3b1ea5e1c4f7e39e24d4a20d49bd",
        "git_blob_sha1": "0" * 40,
    }

    with pytest.raises(ForecastContractError, match="current HMC source"):
        _validate_current_source_bytes(tmp_path, manifest_entry)


def test_loads_closed_development_fixture_contract_bundle() -> None:
    from aeolus.habitat_v2.forecast.contracts import load_forecast_contracts

    root = Path(__file__).resolve().parents[2]
    bundle = load_forecast_contracts(root)

    assert bundle.release_tier == "DEVELOPMENT_FIXTURE_ONLY"
    assert len(bundle.alarm_slots) == 287
    assert tuple(action.action_id for action in bundle.actions) == (
        "normal-occupied-v1",
        "normal-eva_transition-v1",
        "normal-contingency-v1",
        "normal-dormant-v1",
    )
    assert bundle.topology.zone_ids == (
        "air_processing_bay",
        "airlock_suitport",
        "common_galley",
        "crew_quarters_a",
        "crew_quarters_b",
        "equipment_power_bay",
        "hygiene_medical",
        "laboratory",
    )
