"""Tests for scenario graph loading and validation (src/icarus/config.py)."""

import pytest

from icarus.config import load_scenario, parse_scenario

ZONE_IDS = {"cabin_a", "cabin_b", "lab", "processing"}
CONNECTION_IDS = {
    "cabin_a_to_processing",
    "processing_to_cabin_a",
    "cabin_b_to_processing",
    "processing_to_cabin_b",
    "lab_to_processing",
    "processing_to_lab",
}


def test_standard_habitat_loads_four_zones_and_six_directed_connections(
    standard_scenario_path,
):
    config = load_scenario(standard_scenario_path)

    assert config.version == 1
    assert len(config.zones) == 4
    assert len(config.connections) == 6
    assert {z.id for z in config.zones} == ZONE_IDS
    assert {c.id for c in config.connections} == CONNECTION_IDS


def test_standard_habitat_zone_fields(standard_scenario_path):
    config = load_scenario(standard_scenario_path)
    by_id = {z.id: z for z in config.zones}

    assert by_id["cabin_a"].label == "Crew Cabin A"
    assert by_id["cabin_a"].preset == "crew_cabin"
    assert by_id["cabin_b"].preset == "crew_cabin"
    assert by_id["lab"].preset == "lab"
    assert by_id["processing"].preset == "air_processing"

    # Crew cabins carry a positive CO2 source; lab and processing start at zero.
    assert by_id["cabin_a"].co2_generation_per_second > 0.0
    assert by_id["cabin_b"].co2_generation_per_second > 0.0
    assert by_id["lab"].co2_generation_per_second == 0.0
    assert by_id["processing"].co2_generation_per_second == 0.0
    assert all(z.air_volume > 0.0 for z in config.zones)


def test_standard_habitat_hub_paths_are_healthy_and_paired(standard_scenario_path):
    config = load_scenario(standard_scenario_path)

    assert config.processing_zone().id == "processing"
    assert {z.id for z in config.non_processing_zones()} == {"cabin_a", "cabin_b", "lab"}
    assert all(c.health == 1.0 for c in config.connections)
    for zone in config.non_processing_zones():
        outbound = config.path_to_processing(zone.id)
        inbound = config.path_from_processing(zone.id)
        assert outbound.from_zone == zone.id
        assert outbound.to_zone == "processing"
        assert inbound.from_zone == "processing"
        assert inbound.to_zone == zone.id


def _mutate(doc, fn):
    fn(doc)
    return doc


def _drop_connection(doc, connection_id):
    doc["connections"] = [c for c in doc["connections"] if c["id"] != connection_id]


INVALID_CASES = [
    pytest.param(
        lambda d: d.update(version=99),
        "unsupported version",
        id="unsupported_version",
    ),
    pytest.param(
        lambda d: d.pop("version"),
        "version",
        id="missing_version",
    ),
    pytest.param(
        lambda d: d.update(zones=[]),
        "no zones",
        id="no_zones",
    ),
    pytest.param(
        lambda d: d.pop("zones"),
        "zones",
        id="missing_zones_key",
    ),
    pytest.param(
        lambda d: [
            z.update(preset="storage") for z in d["zones"] if z["id"] == "processing"
        ],
        "no air_processing zone",
        id="no_air_processing_zone",
    ),
    pytest.param(
        lambda d: [
            z.update(preset="air_processing") for z in d["zones"] if z["id"] == "lab"
        ],
        "more than one air_processing zone",
        id="two_air_processing_zones",
    ),
    pytest.param(
        lambda d: d["zones"].append(dict(d["zones"][0])),
        "duplicate zone id",
        id="duplicate_zone_id",
    ),
    pytest.param(
        lambda d: d["connections"].append(dict(d["connections"][0])),
        "duplicate connection id",
        id="duplicate_connection_id",
    ),
    pytest.param(
        lambda d: [
            z.update(preset="hydroponics") for z in d["zones"] if z["id"] == "cabin_a"
        ],
        "unsupported preset",
        id="unsupported_preset",
    ),
    pytest.param(
        lambda d: [z.update(air_volume=0.0) for z in d["zones"] if z["id"] == "cabin_a"],
        "air_volume",
        id="zero_air_volume",
    ),
    pytest.param(
        lambda d: [
            z.update(air_volume=-10.0) for z in d["zones"] if z["id"] == "cabin_a"
        ],
        "air_volume",
        id="negative_air_volume",
    ),
    pytest.param(
        lambda d: [
            z.update(co2_generation_per_second=-0.5)
            for z in d["zones"]
            if z["id"] == "cabin_a"
        ],
        "co2_generation_per_second",
        id="negative_co2_source",
    ),
    pytest.param(
        lambda d: [
            c.update(to="nowhere")
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "unknown zone",
        id="connection_unknown_target_zone",
    ),
    pytest.param(
        lambda d: [
            c.update(**{"from": "nowhere"})
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "unknown zone",
        id="connection_unknown_source_zone",
    ),
    pytest.param(
        lambda d: [
            c.update(max_airflow=0.0)
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "max_airflow",
        id="zero_max_airflow",
    ),
    pytest.param(
        lambda d: [
            c.update(max_airflow=-3.0)
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "max_airflow",
        id="negative_max_airflow",
    ),
    pytest.param(
        lambda d: [
            c.update(health=-0.1)
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "health",
        id="health_below_zero",
    ),
    pytest.param(
        lambda d: [
            c.update(health=1.5)
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "health",
        id="health_above_one",
    ),
    pytest.param(
        lambda d: _drop_connection(d, "cabin_a_to_processing"),
        "path to the air_processing bay",
        id="missing_path_to_processing",
    ),
    pytest.param(
        lambda d: _drop_connection(d, "processing_to_cabin_a"),
        "return path",
        id="missing_return_path",
    ),
    pytest.param(
        lambda d: d["connections"].append(
            {
                "id": "cabin_a_to_cabin_b",
                "from": "cabin_a",
                "to": "cabin_b",
                "max_airflow": 1.0,
                "health": 1.0,
            }
        ),
        "hub layout",
        id="connection_not_touching_processing",
    ),
    pytest.param(
        lambda d: d["connections"].append(
            {
                "id": "cabin_a_to_processing_again",
                "from": "cabin_a",
                "to": "processing",
                "max_airflow": 1.0,
                "health": 1.0,
            }
        ),
        "more than one path",
        id="duplicate_directed_path",
    ),
    pytest.param(
        lambda d: [z.pop("label") for z in d["zones"] if z["id"] == "cabin_a"],
        "label",
        id="missing_zone_field",
    ),
    pytest.param(
        lambda d: [
            c.pop("health") for c in d["connections"] if c["id"] == "cabin_a_to_processing"
        ],
        "health",
        id="missing_connection_field",
    ),
    pytest.param(
        lambda d: [
            z.update(air_volume="lots") for z in d["zones"] if z["id"] == "cabin_a"
        ],
        "air_volume",
        id="non_numeric_air_volume",
    ),
    pytest.param(
        lambda d: [
            c.update(health=True)
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "health",
        id="bool_health_is_not_a_number",
    ),
    # Python's json accepts NaN/Infinity tokens; a validated scenario must not.
    pytest.param(
        lambda d: [
            z.update(air_volume=float("nan"))
            for z in d["zones"]
            if z["id"] == "cabin_a"
        ],
        "finite",
        id="nan_air_volume",
    ),
    pytest.param(
        lambda d: [
            z.update(air_volume=float("inf"))
            for z in d["zones"]
            if z["id"] == "cabin_a"
        ],
        "finite",
        id="infinite_air_volume",
    ),
    pytest.param(
        lambda d: [
            z.update(co2_generation_per_second=float("nan"))
            for z in d["zones"]
            if z["id"] == "cabin_a"
        ],
        "finite",
        id="nan_co2_source",
    ),
    pytest.param(
        lambda d: [
            c.update(max_airflow=float("inf"))
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "finite",
        id="infinite_max_airflow",
    ),
    pytest.param(
        lambda d: [
            c.update(health=float("nan"))
            for c in d["connections"]
            if c["id"] == "cabin_a_to_processing"
        ],
        "finite",
        id="nan_health",
    ),
    # A self-loop touches no other zone; the air_processing variant used to
    # slip past the hub rules and crash the run with a raw KeyError.
    pytest.param(
        lambda d: d["connections"].append(
            {
                "id": "processing_self_loop",
                "from": "processing",
                "to": "processing",
                "max_airflow": 5.0,
                "health": 1.0,
            }
        ),
        "itself",
        id="air_processing_self_loop",
    ),
    pytest.param(
        lambda d: d["connections"].append(
            {
                "id": "cabin_a_self_loop",
                "from": "cabin_a",
                "to": "cabin_a",
                "max_airflow": 5.0,
                "health": 1.0,
            }
        ),
        "itself",
        id="non_processing_self_loop",
    ),
]


@pytest.mark.parametrize("mutate,match", INVALID_CASES)
def test_invalid_scenario_is_rejected_clearly(standard_doc, mutate, match):
    with pytest.raises(ValueError, match=match):
        parse_scenario(_mutate(standard_doc, mutate))


def test_load_scenario_rejects_unparseable_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_scenario(path)


def test_load_scenario_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_scenario(tmp_path / "does_not_exist.json")


def test_load_scenario_rejects_non_object_document(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="object"):
        load_scenario(path)
