from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "contracts" / "habitat_v2_bdm_v1_family_custody_v1.json"
PILOT_RECEIPT_SHA256 = (
    "0b654326ebca2b717c4ed155f322d77604f1e538d59812aa36bde5a49cb500bd"
)


def _load_pilot_module():
    spec = importlib.util.spec_from_file_location(
        "run_bdm_v1_blind_power_pilot",
        REPO_ROOT / "scripts" / "run_bdm_v1_blind_power_pilot.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pilot():
    return _load_pilot_module()


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_bytes())


def test_zero_event_bound_math(pilot) -> None:
    assert pilot.zero_event_upper_bound(12) == pytest.approx(
        0.22092219194555585, rel=1e-12
    )
    assert pilot.rejects_rate_at_alpha(10, 0.25) is False
    assert pilot.rejects_rate_at_alpha(11, 0.25) is True
    assert pilot.rejects_rate_at_alpha(12, 0.25) is True
    assert pilot.rejects_rate_at_alpha(3, 0.25) is False
    with pytest.raises(pilot.PowerPilotError):
        pilot.zero_event_upper_bound(0)
    with pytest.raises(pilot.PowerPilotError):
        pilot.rejects_rate_at_alpha(12, 0.0)


def test_continuous_required_groups_math(pilot) -> None:
    required = pilot.continuous_required_groups(
        44.92471247646109, 119.40912938405414
    )
    assert required == 9
    assert (
        pilot.continuous_required_groups(2 * 44.92471247646109, 119.40912938405414)
        > required
    )
    assert (
        pilot.continuous_required_groups(
            44.92471247646109, 119.40912938405414, mde_fraction=0.25
        )
        > required
    )
    with pytest.raises(pilot.PowerPilotError):
        pilot.continuous_required_groups(1.0, 0.0)
    with pytest.raises(pilot.PowerPilotError):
        pilot.continuous_required_groups(-1.0, 1.0)
    hand = math.ceil(
        (pilot.Z_ALPHA_TWO_SIDED + pilot.Z_POWER) ** 2
        * 2.0
        * 44.92471247646109**2
        / (0.5 * 119.40912938405414) ** 2
    )
    assert hand == required


def test_registry_justification_binds_pilot_receipt(pilot, registry: dict) -> None:
    seal = registry["blind_seal"]
    justification = seal["size_justification"]
    assert justification.startswith("pilot_receipt:")
    parts = justification.split(":")
    assert parts[1] == PILOT_RECEIPT_SHA256
    fields = dict(part.split("=") for part in parts[2:])
    assert int(fields["declared_blind_groups"]) == seal["group_count"] == 12
    assert seal["family_count"] == 24
    assert seal["outcome_status"] == "NOT_COMPUTED"
    assert float(fields["harmful_rate_detectable"]) == pilot.HARMFUL_RATE_PROBE
    assert float(fields["alpha"]) == pilot.ALPHA
    assert float(fields["zero_event_upper_bound"]) == pytest.approx(
        pilot.zero_event_upper_bound(12), abs=1e-4
    )
    assert (
        int(fields["comfort_proxy_required_groups"]) <= seal["group_count"]
    )
    assert int(fields["contract_floor"]) == pilot.CONTRACT_FLOOR_GROUPS
    assert pilot.rejects_rate_at_alpha(
        seal["group_count"], pilot.HARMFUL_RATE_PROBE
    )
    assert seal["group_count"] > pilot.CONTRACT_FLOOR_GROUPS
