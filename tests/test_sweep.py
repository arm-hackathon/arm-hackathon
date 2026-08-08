"""Deterministic family-sweep generation and validation."""

import json
from pathlib import Path

import pytest

from aeolus.config import load_scenario
from aeolus.families import (
    RECOVERY_COUNTERFACTUAL_ARMS,
    RECOVERY_FAMILY_MANIFEST_VERSION,
    load_family_manifest,
    parse_family_manifest,
)
from aeolus.sweep import generate_sweep, main, parse_sweep_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_spec(tmp_path: Path) -> Path:
    base_name = "standard_habitat.json"
    (tmp_path / base_name).write_bytes(
        (REPO_ROOT / "scenarios" / base_name).read_bytes()
    )
    telemetry = {
        "airflow_noise_fraction": 0.01,
        "airflow_bias_fraction": 0.005,
        "actuator_position_noise_fraction": 0.01,
    }
    document = {
        "schema_version": "aeolus_sweep_v1",
        "base_scenario": base_name,
        "targets": ["cabin_a"],
        "splits": {
            "train": {
                "seeds": [10],
                "fault_start_ticks": [25],
                "operating_profiles": [
                    {
                        "id": "train",
                        "source_multiplier": 0.8,
                        "shared_airflow_capacity": 24.0,
                        "telemetry": telemetry,
                    }
                ],
                "gradual_end_effectiveness": [0.4],
                "blocked_effectiveness": [0.1],
            },
            "validation": {
                "seeds": [20],
                "fault_start_ticks": [35],
                "operating_profiles": [
                    {
                        "id": "validation",
                        "source_multiplier": 1.0,
                        "shared_airflow_capacity": 30.0,
                        "telemetry": telemetry,
                    }
                ],
                "gradual_end_effectiveness": [0.5],
                "blocked_effectiveness": [0.2],
            },
            "test": {
                "seeds": [30],
                "fault_start_ticks": [45],
                "operating_profiles": [
                    {
                        "id": "test",
                        "source_multiplier": 1.2,
                        "shared_airflow_capacity": 36.0,
                        "telemetry": telemetry,
                    }
                ],
                "gradual_end_effectiveness": [0.6],
                "blocked_effectiveness": [0.3],
            },
        },
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _small_recovery_spec(tmp_path: Path) -> Path:
    base_name = "recovery_habitat.json"
    (tmp_path / base_name).write_bytes(
        (REPO_ROOT / "scenarios" / base_name).read_bytes()
    )
    telemetry = {
        "airflow_noise_fraction": 0.01,
        "airflow_bias_fraction": 0.005,
        "airflow_drift_fraction": 0.005,
        "actuator_position_noise_fraction": 0.01,
        "co2_sensor_noise_fraction": 0.01,
        "co2_sensor_bias_fraction": 0.005,
        "co2_sensor_drift_fraction": 0.005,
    }

    def split(seed: int) -> dict:
        return {
            "seeds": [seed],
            "fault_start_ticks": [25],
            "operating_profiles": [
                {
                    "id": "primary-low-reserve",
                    "source_multiplier": 0.8,
                    "shared_airflow_capacity": 24.0,
                    "telemetry": telemetry,
                }
            ],
            "gradual_profiles": [
                {"duration_ticks": 30, "end_effectiveness": 0.75}
            ],
            "blocked_effectiveness": [0.65],
            "transient_blocked_profiles": [
                {"blocked_effectiveness": 0.65, "duration_ticks": 60}
            ],
            "transient_gradual_profiles": [
                {
                    "start_effectiveness": 1.0,
                    "end_effectiveness": 0.75,
                    "duration_ticks": 60,
                }
            ],
        }

    document = {
        "schema_version": "aeolus_sweep_v4",
        "base_scenario": base_name,
        "targets": ["cabin_a"],
        "suite_role": "development",
        "splits": {"train": split(211), "validation": split(601)},
    }
    path = tmp_path / "recovery-sweep.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_generate_sweep_is_byte_stable_and_family_valid(tmp_path):
    spec = _small_spec(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_receipt = generate_sweep(spec, first)
    second_receipt = generate_sweep(spec, second)

    assert first_receipt == second_receipt
    assert first_receipt["families_by_split"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert first_receipt["total_families"] == 9
    assert {
        path.relative_to(first) for path in first.iterdir() if path.is_file()
    } == {
        path.relative_to(second) for path in second.iterdir() if path.is_file()
    }
    for first_path in first.iterdir():
        if first_path.is_file():
            assert first_path.read_bytes() == (second / first_path.name).read_bytes()

    manifest = load_family_manifest(first / "families.json")
    assert len(manifest.families) == 9
    assert {family.fault_class for family in manifest.families} == {
        "nominal",
        "gradual_primary_fan_degradation",
        "blocked_path",
        "frozen_sensor",
    } - {"nominal"}
    for family in manifest.families:
        reference = load_scenario(family.reference_path)
        fault = load_scenario(family.fault_path)
        assert reference.telemetry == fault.telemetry
        assert reference.simulation == fault.simulation
        assert reference.air_system == fault.air_system


def test_sweep_rejects_seed_shared_between_splits(tmp_path):
    spec_path = _small_spec(tmp_path)
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    document["splits"]["validation"]["seeds"] = [10]

    with pytest.raises(ValueError, match="more than one split"):
        parse_sweep_spec(document, source_path=spec_path)


def test_sweep_rejects_unknown_fields(tmp_path):
    spec_path = _small_spec(tmp_path)
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    document["splits"]["train"]["random_split"] = True

    with pytest.raises(ValueError, match="unexpected field 'random_split'"):
        parse_sweep_spec(document, source_path=spec_path)


def test_generate_sweep_rejects_non_empty_destination(tmp_path):
    spec = _small_spec(tmp_path)
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "keep.txt").write_text("owned", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        generate_sweep(spec, destination)


def test_sweep_cli_reports_receipt(tmp_path, capsys):
    spec = _small_spec(tmp_path)
    destination = tmp_path / "generated"

    assert main([str(spec), str(destination)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["total_families"] == 9
    assert (destination / "families.json").is_file()


def test_checked_in_sweep_spec_has_declared_full_counts():
    spec_path = REPO_ROOT / "scenarios" / "sweep-v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    parsed = parse_sweep_spec(spec, source_path=spec_path)

    assert parsed.targets == ("cabin_a", "cabin_b", "lab")
    assert len(parsed.splits["train"].seeds) == 6
    assert len(parsed.splits["validation"].seeds) == 2
    assert len(parsed.splits["test"].seeds) == 3


def test_checked_in_sweep_v2_has_iid_primary_and_declared_stress_counts():
    spec_path = REPO_ROOT / "scenarios" / "sweep-v2.json"
    parsed = parse_sweep_spec(
        json.loads(spec_path.read_text(encoding="utf-8")), source_path=spec_path
    )

    assert tuple(parsed.splits) == ("train", "validation", "test", "stress")
    assert parsed.splits["train"].operating_profiles == parsed.splits[
        "validation"
    ].operating_profiles == parsed.splits["test"].operating_profiles
    assert parsed.splits["train"].gradual_profiles == parsed.splits[
        "validation"
    ].gradual_profiles == parsed.splits["test"].gradual_profiles
    expected = {"train": 360, "validation": 120, "test": 180, "stress": 180}
    for split, split_spec in parsed.splits.items():
        faults_per_target = (
            len(split_spec.gradual_profiles)
            + len(split_spec.blocked_effectiveness)
            + 1
        )
        count = (
            len(split_spec.seeds)
            * len(split_spec.fault_start_ticks)
            * len(split_spec.operating_profiles)
            * len(parsed.targets)
            * faults_per_target
        )
        assert count == expected[split]
    seed_sets = [set(split.seeds) for split in parsed.splits.values()]
    assert all(
        not left & right
        for index, left in enumerate(seed_sets)
        for right in seed_sets[index + 1 :]
    )


def test_generate_recovery_sweep_v4_is_byte_stable_and_complete(tmp_path):
    spec = _small_recovery_spec(tmp_path)
    first = tmp_path / "first-recovery"
    second = tmp_path / "second-recovery"

    first_receipt = generate_sweep(spec, first)
    second_receipt = generate_sweep(spec, second)

    assert first_receipt == second_receipt
    assert first_receipt["families_by_split"] == {"train": 5, "validation": 5}
    assert first_receipt["total_families"] == 10
    for first_path in first.iterdir():
        if first_path.is_file():
            assert first_path.read_bytes() == (second / first_path.name).read_bytes()

    manifest = load_family_manifest(first / "families.json")
    assert manifest.schema_version == RECOVERY_FAMILY_MANIFEST_VERSION
    assert all(
        family.counterfactual_group_id == family.family_id
        and family.base_condition_id
        and family.counterfactual_arms == RECOVERY_COUNTERFACTUAL_ARMS
        for family in manifest.families
    )
    base_conditions: dict[tuple[str, str], int] = {}
    for family in manifest.families:
        key = (family.split, family.base_condition_id)
        base_conditions[key] = base_conditions.get(key, 0) + 1
    assert set(base_conditions.values()) == {5}
    assert {family.fault_class for family in manifest.families} == {
        "blocked_path",
        "frozen_sensor",
        "gradual_primary_fan_degradation",
        "transient_blocked_path",
        "transient_gradual_primary_fan_degradation",
    }
    transient = [
        load_scenario(family.fault_path).fault_profiles[0]
        for family in manifest.families
        if family.fault_class.startswith("transient_")
    ]
    assert {profile.end_tick - profile.start_tick for profile in transient} == {60}


def test_recovery_sweep_rejects_family_id_rounding_collision(tmp_path):
    spec_path = _small_recovery_spec(tmp_path)
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    for split in document["splits"].values():
        split["blocked_effectiveness"] = [0.6501, 0.6504]
    spec_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate generated family_id"):
        generate_sweep(spec_path, tmp_path / "collision")


def test_recovery_manifest_rejects_incomplete_arms_and_cross_split_conditions(
    tmp_path,
):
    spec = _small_recovery_spec(tmp_path)
    output = tmp_path / "strict-recovery-manifest"
    generate_sweep(spec, output)
    source = json.loads((output / "families.json").read_text(encoding="utf-8"))

    incomplete = json.loads(json.dumps(source))
    incomplete["families"][0]["counterfactual_arms"].pop()
    with pytest.raises(ValueError, match="counterfactual arms are incomplete"):
        parse_family_manifest(incomplete, base_dir=output)

    cross_split = json.loads(json.dumps(source))
    train_base = next(
        family["base_condition_id"]
        for family in cross_split["families"]
        if family["split"] == "train"
    )
    validation_family = next(
        family for family in cross_split["families"] if family["split"] == "validation"
    )
    validation_family["base_condition_id"] = train_base
    with pytest.raises(ValueError, match="more than one split"):
        parse_family_manifest(cross_split, base_dir=output)


def test_checked_in_recovery_sweeps_have_frozen_counts_and_disjoint_seeds():
    development_path = REPO_ROOT / "scenarios" / "sweep-recovery-development.json"
    final_path = REPO_ROOT / "scenarios" / "sweep-recovery-final.json"
    development = parse_sweep_spec(
        json.loads(development_path.read_text(encoding="utf-8")),
        source_path=development_path,
    )
    final = parse_sweep_spec(
        json.loads(final_path.read_text(encoding="utf-8")), source_path=final_path
    )

    assert development.targets == final.targets == ("cabin_a", "cabin_b", "lab")
    assert tuple(development.splits) == ("train", "validation")
    assert tuple(final.splits) == ("final",)
    expected = {"train": 504, "validation": 252, "final": 252}
    all_seeds: list[set[int]] = []
    for spec in (development, final):
        for split_name, split_spec in spec.splits.items():
            profile_count = (
                len(split_spec.gradual_profiles)
                + len(split_spec.blocked_effectiveness)
                + len(split_spec.transient_blocked_profiles)
                + len(split_spec.transient_gradual_profiles)
                + 1
            )
            count = (
                len(split_spec.seeds)
                * len(split_spec.fault_start_ticks)
                * len(split_spec.operating_profiles)
                * len(spec.targets)
                * profile_count
            )
            assert count == expected[split_name]
            assert count // (len(spec.targets) * profile_count) >= 12
            all_seeds.append(set(split_spec.seeds))
    assert all(
        not left & right
        for index, left in enumerate(all_seeds)
        for right in all_seeds[index + 1 :]
    )
