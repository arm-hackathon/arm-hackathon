import pytest

from aeolus.habitat_v2.scenario import Scenario, ScenarioValidationError

from ._helpers import reference_scenario_mapping, reversed_object_keys


def test_valid_semantic_mutation_changes_scenario_digest_and_run_id() -> None:
    baseline = Scenario.from_mapping(reference_scenario_mapping())
    mutations = [
        (
            ("timeline", 0, "loads", "crew_cabin", "co2_generation_mol_s"),
            0.00026,
        ),
        (("timeline", 0, "command", "scrubber_duty"), 0.61),
        (("equipment", "fan_power_w_per_m3_s"), 1_001.0),
        (("zones", 0, "initial", "co2_ppm"), 801.0),
    ]

    for path, value in mutations:
        mapping = reference_scenario_mapping()
        target = mapping
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        changed = Scenario.from_mapping(mapping)
        assert changed.scenario_sha256 != baseline.scenario_sha256
        assert changed.run_id != baseline.run_id


def test_object_key_order_does_not_change_scenario_identity() -> None:
    mapping = reference_scenario_mapping()

    first = Scenario.from_mapping(mapping)
    reordered = Scenario.from_mapping(reversed_object_keys(mapping))

    assert reordered.scenario_sha256 == first.scenario_sha256
    assert reordered.run_id == first.run_id
    assert reordered.canonical_bytes == first.canonical_bytes


def test_unknown_top_level_key_is_rejected() -> None:
    mapping = reference_scenario_mapping()
    mapping["unreviewed_override"] = True

    with pytest.raises(ScenarioValidationError, match="unknown top-level fields"):
        Scenario.from_mapping(mapping)


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (("zones", 0), "zone"),
        (("zones", 0, "initial"), "zone initial state"),
        (("equipment",), "equipment"),
        (("initial_utility",), "initial utility"),
        (("timeline", 0), "timeline segment"),
        (("timeline", 0, "loads", "crew_cabin"), "zone load"),
        (("timeline", 0, "command"), "plant command"),
    ],
)
def test_unknown_nested_key_is_rejected(
    path: tuple[str | int, ...], label: str
) -> None:
    mapping = reference_scenario_mapping()
    target: object = mapping
    for part in path:
        target = target[part]  # type: ignore[index]
    assert isinstance(target, dict)
    target["unreviewed_override"] = True

    with pytest.raises(ScenarioValidationError, match=f"unknown {label} fields"):
        Scenario.from_mapping(mapping)


@pytest.mark.parametrize(
    ("path", "field", "label"),
    [
        (("zones", 0), "volume_m3", "zone"),
        (("zones", 0, "initial"), "pressure_pa", "zone initial state"),
        (("equipment",), "battery_capacity_wh", "equipment"),
        (("initial_utility",), "oxygen_store_mol", "initial utility"),
        (("timeline", 0), "generation_w", "timeline segment"),
        (
            ("timeline", 0, "loads", "crew_cabin"),
            "co2_generation_mol_s",
            "zone load",
        ),
        (("timeline", 0, "command"), "scrubber_duty", "plant command"),
    ],
)
def test_missing_nested_key_is_rejected(
    path: tuple[str | int, ...], field: str, label: str
) -> None:
    mapping = reference_scenario_mapping()
    target: object = mapping
    for part in path:
        target = target[part]  # type: ignore[index]
    assert isinstance(target, dict)
    del target[field]

    with pytest.raises(ScenarioValidationError, match=f"missing {label} fields"):
        Scenario.from_mapping(mapping)
