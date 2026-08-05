"""V5 load-preserving nominal occupancy-shape contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.families import load_family_manifest, validate_manifest_disjointness
from aeolus.sweep import generate_sweep, parse_sweep_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _telemetry() -> dict[str, float]:
    return {
        "airflow_noise_fraction": 0.02,
        "airflow_bias_fraction": 0.01,
        "airflow_drift_fraction": 0.01,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.02,
        "co2_sensor_bias_fraction": 0.01,
        "co2_sensor_drift_fraction": 0.01,
    }


def _occupancy_shape() -> dict[str, object]:
    return {
        "zone_period_multipliers": {
            "cabin_a": [1.2, 0.85, 1.0571428571428572],
            "cabin_b": [1.2, 0.8, 1.1, 1.0166666666666666],
            "lab": [0.8, 1.1, 0.3125],
        }
    }


def _v5_document() -> dict[str, object]:
    profile = {
        "id": "v5-staggered-load",
        "source_multiplier": 1.0,
        "shared_airflow_capacity": 30.0,
        "telemetry": _telemetry(),
        "occupancy_shape": _occupancy_shape(),
    }
    return {
        "schema_version": "aeolus_sweep_v5",
        "suite_role": "development",
        "base_scenario": "standard_habitat.json",
        "targets": ["cabin_a"],
        "splits": {
            "train": {
                "seeds": [1100],
                "fault_start_ticks": [25],
                "operating_profiles": [profile],
                "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
                "blocked_effectiveness": [0.65],
            },
            "validation": {
                "seeds": [1300],
                "fault_start_ticks": [25],
                "operating_profiles": [profile],
                "gradual_profiles": [{"duration_ticks": 30, "end_effectiveness": 0.75}],
                "blocked_effectiveness": [0.65],
            },
        },
    }


def _write_base(path: Path) -> None:
    path.write_bytes((REPO_ROOT / "scenarios" / "standard_habitat.json").read_bytes())


def _declared_load(periods: list[dict[str, float | int]]) -> float:
    return sum(
        (int(period["end_tick"]) - int(period["start_tick"]) + 1)
        * float(period["multiplier"])
        for period in periods
    )


def test_v5_parses_complete_load_preserving_non_processing_shape(tmp_path: Path):
    path = tmp_path / "v5.json"
    path.write_text(json.dumps(_v5_document()), encoding="utf-8")
    _write_base(tmp_path / "standard_habitat.json")

    parsed = parse_sweep_spec(_v5_document(), source_path=path)

    profile = parsed.splits["train"].operating_profiles[0]
    assert parsed.schema_version == "aeolus_sweep_v5"
    assert parsed.suite_role == "development"
    assert profile.profile_id == "v5-staggered-load"
    assert tuple(profile.occupancy_shape) == ("cabin_a", "cabin_b", "lab")


def test_v5_applies_shape_without_changing_base_period_boundaries_or_load(tmp_path: Path):
    spec_path = tmp_path / "v5.json"
    spec_path.write_text(json.dumps(_v5_document()), encoding="utf-8")
    _write_base(tmp_path / "standard_habitat.json")

    generate_sweep(spec_path, tmp_path / "sweep")

    base = json.loads((tmp_path / "standard_habitat.json").read_text())
    reference = json.loads(
        (tmp_path / "sweep" / "train-s1100-v5-staggered-load-reference.json").read_text()
    )
    base_zones = {zone["id"]: zone for zone in base["zones"]}
    reference_zones = {zone["id"]: zone for zone in reference["zones"]}
    for zone_id in ("cabin_a", "cabin_b", "lab"):
        original = base_zones[zone_id]["occupancy_profile"]
        reshaped = reference_zones[zone_id]["occupancy_profile"]
        assert [
            (period["start_tick"], period["end_tick"]) for period in reshaped
        ] == [(period["start_tick"], period["end_tick"]) for period in original]
        assert _declared_load(reshaped) == pytest.approx(_declared_load(original))


def test_v5_keeps_shaped_reference_and_fault_paired(tmp_path: Path):
    spec_path = tmp_path / "v5.json"
    spec_path.write_text(json.dumps(_v5_document()), encoding="utf-8")
    _write_base(tmp_path / "standard_habitat.json")

    generate_sweep(spec_path, tmp_path / "sweep")

    reference = json.loads(
        (tmp_path / "sweep" / "train-s1100-v5-staggered-load-reference.json").read_text()
    )
    fault_path = next(
        path
        for path in (tmp_path / "sweep").glob("train-s1100-v5-staggered-load-*.json")
        if not path.name.endswith("-reference.json")
    )
    fault = json.loads(fault_path.read_text())
    assert fault["fault_profiles"]
    reference["fault_profiles"] = fault["fault_profiles"]
    assert fault == reference


def test_v5_rejects_shape_that_changes_total_declared_load(tmp_path: Path):
    path = tmp_path / "v5.json"
    _write_base(tmp_path / "standard_habitat.json")
    document = _v5_document()
    multipliers = document["splits"]["train"]["operating_profiles"][0]["occupancy_shape"][
        "zone_period_multipliers"
    ]
    multipliers["lab"] = [1.0, 1.0, 1.1]

    with pytest.raises(ValueError, match="preserve declared load"):
        parse_sweep_spec(document, source_path=path)


def test_v5_rejects_incomplete_or_processing_occupancy_shape(tmp_path: Path):
    path = tmp_path / "v5.json"
    _write_base(tmp_path / "standard_habitat.json")

    missing = _v5_document()
    del missing["splits"]["train"]["operating_profiles"][0]["occupancy_shape"][
        "zone_period_multipliers"
    ]["lab"]
    with pytest.raises(ValueError, match="occupancy shape"):
        parse_sweep_spec(missing, source_path=path)

    processing = _v5_document()
    processing["splits"]["train"]["operating_profiles"][0]["occupancy_shape"][
        "zone_period_multipliers"
    ]["processing"] = []
    with pytest.raises(ValueError, match="occupancy shape"):
        parse_sweep_spec(processing, source_path=path)


def test_checked_in_v5_declares_fresh_counterfactual_development_families(
    tmp_path: Path,
):
    v3_development_path = REPO_ROOT / "scenarios" / "sweep-v3-development.json"
    v3_final_path = REPO_ROOT / "scenarios" / "sweep-v3-final.json"
    v4_path = REPO_ROOT / "scenarios" / "sweep-v4-development.json"
    v5_path = REPO_ROOT / "scenarios" / "sweep-v5-development.json"

    parsed = parse_sweep_spec(
        json.loads(v5_path.read_text(encoding="utf-8")), source_path=v5_path
    )

    assert parsed.schema_version == "aeolus_sweep_v5"
    assert parsed.splits["train"].seeds == tuple(range(1100, 1106))
    assert parsed.splits["validation"].seeds == tuple(range(1300, 1306))
    assert [profile.profile_id for profile in parsed.splits["train"].operating_profiles] == [
        "v5-lab-peak-transition",
        "v5-primary-high-baseline",
        "v5-primary-low-baseline",
        "v5-staggered-load",
    ]

    manifests = {}
    for name, spec_path in (
        ("v3-development", v3_development_path),
        ("v3-final", v3_final_path),
        ("v4", v4_path),
        ("v5", v5_path),
    ):
        destination = tmp_path / name
        receipt = generate_sweep(spec_path, destination)
        manifests[name] = load_family_manifest(destination / "families.json")
        if name == "v5":
            assert receipt["families_by_split"] == {"train": 720, "validation": 720}
            assert receipt["total_families"] == 1440

    for historical in ("v3-development", "v3-final", "v4"):
        validate_manifest_disjointness(manifests[historical], manifests["v5"])
