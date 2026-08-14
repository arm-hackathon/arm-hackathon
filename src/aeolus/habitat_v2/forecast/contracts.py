"""Closed loaders for the Habitat V2 Forecast D1 development fixture.

The fixture is deliberately not a production authority.  This module only
admits the five frozen D1 artifacts when they still bind the reviewed HMC,
reference scenario and eight-zone topology exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any

from ..hmc_contract import HMCContract
from ..physics import CanonicalExternalCommand, validate_external_command
from ..scenario import Scenario
from ..telemetry import ObservableTopology, derive_observable_topology


RELEASE_TIER = "DEVELOPMENT_FIXTURE_ONLY"
_BINDING_SHA256 = "f1890ca5813a98bb13bc628263c85c152fca28c0464843dc8304773b43a05bcc"
_ALARM_SHA256 = "f27db07c4b7d15a09ec625855d3131bb21734d8725ccb9644a78b982921d9aec"
_CATALOGUE_SHA256 = "476df714510cc9435a4b82ebb23c8ebfab7d6953930c3b0481124a2af45521f9"
_PROFILE_SHA256 = "e6748b21735b3fce668ffccc0b820ebf4df5ab61d204bffb540b3b4e612e3fed"
_RECORDS_SHA256 = "04fa1a8bad2220a6d800fd7ddbeb94646b044ef6f1c7005c45a8cae3f26bd3c7"
_FIXTURE_SCENARIO_SHA256 = (
    "d321f86acddbdc3fb73df47f03367fc7acab0c8cfb6dbd66096d30bef5c0e3e8"
)
_REFERENCE_SCENARIO_SHA256 = (
    "a9ee8eecdb4a952ef95347edcabb7dad614280eb496877cc9cddf8a5c9f77de7"
)


class ForecastContractError(ValueError):
    """A frozen D1 contract is missing, substituted or semantically invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ForecastContractError(
            "contract contains noncanonical JSON data"
        ) from error


def _reject_constant(value: str) -> None:
    raise ForecastContractError(f"non-finite JSON constant {value!r} is forbidden")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForecastContractError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForecastContractError(f"cannot parse frozen contract {label}") from error
    if type(value) is not dict:
        raise ForecastContractError(f"frozen contract {label} must be one JSON object")
    _reject_bad_json(value, label=label)
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ForecastContractError(f"cannot parse frozen contract {path}") from error
    return _strict_json_bytes(raw, label=path.name)


def _reject_bad_json(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ForecastContractError(f"{label} has a non-string object key")
        for key, nested in value.items():
            _reject_bad_json(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_bad_json(nested, label=f"{label}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise ForecastContractError(f"{label} contains a non-finite number")


def _exact(mapping: Mapping[str, Any], fields: set[str], label: str) -> None:
    unknown, missing = sorted(set(mapping) - fields), sorted(fields - set(mapping))
    if unknown or missing:
        raise ForecastContractError(f"{label} has unknown={unknown}, missing={missing}")


def _sha(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ForecastContractError(f"{label} must be lowercase SHA-256")
    return value


def _self_hash(mapping: Mapping[str, Any], field: str, expected: str) -> None:
    _sha(mapping.get(field), label=field)
    body = dict(mapping)
    body.pop(field)
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if mapping[field] != expected or actual != expected:
        raise ForecastContractError(
            f"{field} does not bind the frozen canonical artifact"
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(nested) for key, nested in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _validate_current_source_bytes(
    root: Path,
    manifest_entry: Mapping[str, Any],
) -> None:
    path = root / manifest_entry["path"]
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ForecastContractError(
            f"cannot read current HMC source {manifest_entry['path']}"
        ) from error
    if (
        hashlib.sha256(raw).hexdigest() == manifest_entry["sha256"]
        and _git_blob_sha1(raw) == manifest_entry["git_blob_sha1"]
    ):
        return
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--quiet",
                "--ignore-cr-at-eol",
                "79d6a718e0d44122a763bb72f9c8ed929f39fd23",
                "--",
                manifest_entry["path"],
            ],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ForecastContractError(
            "Git is required to verify current HMC source bytes"
        ) from error
    if result.returncode != 0:
        raise ForecastContractError(
            f"current HMC source {manifest_entry['path']} drifts from reviewed bytes"
        )


@dataclass(frozen=True, slots=True)
class AlarmSlot:
    alarm_id: str
    family: str
    target: str
    severity: str


@dataclass(frozen=True, slots=True)
class CatalogueAction:
    action_id: str
    source_mode: str
    source_segment_index: int
    command_sha256: str
    command: CanonicalExternalCommand


@dataclass(frozen=True, slots=True)
class ForecastContracts:
    root: Path
    release_tier: str
    binding_sha256: str
    alarm_manifest_sha256: str
    action_catalogue_sha256: str
    development_profile_sha256: str
    development_record_contract_sha256: str
    hmc_contract: HMCContract
    reference_scenario: Scenario
    development_scenario: Scenario
    topology: ObservableTopology
    alarm_slots: tuple[AlarmSlot, ...]
    alarm_lifecycle_order: tuple[str, ...]
    actions: tuple[CatalogueAction, ...]
    record_contract: Mapping[str, Any]


def _load_hmc(root: Path, binding: Mapping[str, Any]) -> HMCContract:
    path = root / "contracts" / "habitat_v2_hmc_v1.json"
    raw = _strict_json(path)
    try:
        contract = HMCContract.from_mapping(raw)
    except Exception as error:  # HMC parser is the final authority for its schema.
        raise ForecastContractError("final HMC contract does not parse") from error
    named = (
        "hmc_contract_sha256",
        "snapshot_schema_sha256",
        "snapshot_verification_contract_sha256",
        "external_command_contract_sha256",
        "preflight_contract_sha256",
        "health_policy_sha256",
        "safety_policy_sha256",
        "proposal_receipt_schema_sha256",
        "arbitration_receipt_schema_sha256",
        "step_receipt_schema_sha256",
        "terminal_receipt_schema_sha256",
        "snapshot_verification_receipt_schema_sha256",
        "control_trace_schema_sha256",
    )
    for name in named:
        if getattr(contract, name) != binding[name]:
            raise ForecastContractError(f"final HMC {name} drifts from frozen binding")
    return contract


def _final_git_source(root: Path, path: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"79d6a718e0d44122a763bb72f9c8ed929f39fd23:{path}",
            ],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ForecastContractError(
            "Git is required to verify the reviewed HMC bytes"
        ) from error
    if result.returncode != 0:
        raise ForecastContractError(f"cannot read reviewed HMC source {path} from Git")
    return result.stdout


def _validate_binding(root: Path, value: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "release_tier",
        "final_hmc_commit_sha",
        "final_hmc_tree_sha",
        "hmc_contract_sha256",
        "snapshot_schema_sha256",
        "snapshot_verification_contract_sha256",
        "observable_topology_sha256",
        "external_command_contract_sha256",
        "preflight_contract_sha256",
        "health_policy_sha256",
        "safety_policy_sha256",
        "proposal_receipt_schema_sha256",
        "arbitration_receipt_schema_sha256",
        "step_receipt_schema_sha256",
        "terminal_receipt_schema_sha256",
        "snapshot_verification_receipt_schema_sha256",
        "control_trace_schema_sha256",
        "reference_scenario_path",
        "reference_scenario_sha256",
        "observable_topology",
        "hmc_source_files",
        "hmc_source_file_manifest_sha256",
        "binding_sha256",
    }
    _exact(value, fields, "HMC binding")
    if (
        value["schema_version"] != "aeolus_habitat_v2_forecast_hmc_binding_v1"
        or value["release_tier"] != RELEASE_TIER
    ):
        raise ForecastContractError("HMC binding schema/release tier is unsupported")
    _self_hash(value, "binding_sha256", _BINDING_SHA256)
    if (
        value["final_hmc_commit_sha"] != "79d6a718e0d44122a763bb72f9c8ed929f39fd23"
        or value["final_hmc_tree_sha"] != "91cea3b4c2334a4ece140bd1bf7144353f52ec0d"
    ):
        raise ForecastContractError(
            "HMC Git identity drifts from final reviewed source"
        )
    entries = value["hmc_source_files"]
    if type(entries) is not list or len(entries) != 27:
        raise ForecastContractError("HMC source manifest must contain exactly 27 files")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            raise ForecastContractError("HMC source manifest entry must be an object")
        _exact(entry, {"path", "sha256", "git_blob_sha1"}, f"HMC source entry {index}")
        path = entry["path"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ForecastContractError("HMC source manifest path is unsafe")
        paths.append(path)
        source = _final_git_source(root, path)
        if (
            hashlib.sha256(source).hexdigest()
            != _sha(entry["sha256"], label="source sha256")
            or _git_blob_sha1(source) != entry["git_blob_sha1"]
        ):
            raise ForecastContractError(f"bound HMC source {path} has drifted")
        _validate_current_source_bytes(root, entry)
    if len(paths) != len(set(paths)):
        raise ForecastContractError("HMC source manifest contains duplicate paths")
    if (
        hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
        != value["hmc_source_file_manifest_sha256"]
    ):
        raise ForecastContractError("HMC source manifest hash is invalid")


def _load_scenario(path: Path, *, expected_sha: str, label: str) -> Scenario:
    raw = _strict_json(path)
    try:
        scenario = Scenario.from_mapping(raw)
    except Exception as error:
        raise ForecastContractError(
            f"{label} scenario does not parse as closed V5"
        ) from error
    if scenario.scenario_sha256 != expected_sha:
        raise ForecastContractError(f"{label} scenario canonical identity drifts")
    return scenario


def _validate_alarm(
    value: Mapping[str, Any], binding: Mapping[str, Any]
) -> tuple[AlarmSlot, ...]:
    _exact(
        value,
        {
            "schema_version",
            "release_tier",
            "observable_topology_sha256",
            "health_policy_sha256",
            "lifecycle_order",
            "alarm_slots",
            "alarm_slots_sha256",
            "alarm_manifest_sha256",
        },
        "alarm manifest",
    )
    if (
        value["schema_version"] != "aeolus_habitat_v2_forecast_alarm_manifest_v1"
        or value["release_tier"] != RELEASE_TIER
    ):
        raise ForecastContractError("alarm manifest schema/release tier is unsupported")
    _self_hash(value, "alarm_manifest_sha256", _ALARM_SHA256)
    if (
        value["observable_topology_sha256"] != binding["observable_topology_sha256"]
        or value["health_policy_sha256"] != binding["health_policy_sha256"]
    ):
        raise ForecastContractError("alarm manifest identity drifts from HMC binding")
    if tuple(value["lifecycle_order"]) != ("ABSENT", "RAISED", "ACTIVE", "CLEARED"):
        raise ForecastContractError("alarm lifecycle order is not frozen")
    slots = value["alarm_slots"]
    if (
        type(slots) is not list
        or len(slots) != 287
        or hashlib.sha256(canonical_json_bytes(slots)).hexdigest()
        != value["alarm_slots_sha256"]
    ):
        raise ForecastContractError(
            "alarm slots are incomplete or have an invalid hash"
        )
    result: list[AlarmSlot] = []
    ids: set[str] = set()
    for index, slot in enumerate(slots):
        if type(slot) is not dict:
            raise ForecastContractError("alarm slot must be an object")
        _exact(
            slot, {"alarm_id", "family", "target", "severity"}, f"alarm slot {index}"
        )
        if any(type(slot[name]) is not str or not slot[name] for name in slot):
            raise ForecastContractError("alarm slot contains an invalid descriptor")
        if slot["alarm_id"] in ids:
            raise ForecastContractError("alarm manifest contains duplicate alarm IDs")
        ids.add(slot["alarm_id"])
        result.append(AlarmSlot(**slot))
    return tuple(result)


def _validate_catalogue(
    value: Mapping[str, Any], binding: Mapping[str, Any], scenario: Scenario
) -> tuple[CatalogueAction, ...]:
    _exact(
        value,
        {
            "schema_version",
            "release_tier",
            "observable_topology_sha256",
            "source_scenario_sha256",
            "actions",
            "catalogue_sha256",
        },
        "action catalogue",
    )
    if (
        value["schema_version"] != "aeolus_habitat_v2_forecast_action_catalogue_v1"
        or value["release_tier"] != RELEASE_TIER
    ):
        raise ForecastContractError(
            "action catalogue schema/release tier is unsupported"
        )
    _self_hash(value, "catalogue_sha256", _CATALOGUE_SHA256)
    if (
        value["observable_topology_sha256"] != binding["observable_topology_sha256"]
        or value["source_scenario_sha256"] != _REFERENCE_SCENARIO_SHA256
    ):
        raise ForecastContractError("action catalogue topology/source identity drifts")
    actions = value["actions"]
    if type(actions) is not list or len(actions) != 4:
        raise ForecastContractError(
            "action catalogue must contain exactly four actions"
        )
    result: list[CatalogueAction] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    modes: set[str] = set()
    for index, action in enumerate(actions):
        if type(action) is not dict:
            raise ForecastContractError("catalogue action must be an object")
        _exact(
            action,
            {
                "action_id",
                "source_mode",
                "source_segment_index",
                "command",
                "command_sha256",
            },
            f"action {index}",
        )
        if (
            type(action["action_id"]) is not str
            or not action["action_id"]
            or type(action["source_mode"]) is not str
            or action["source_mode"]
            not in {"dormant", "occupied", "eva_transition", "contingency"}
            or type(action["source_segment_index"]) is not int
            or isinstance(action["source_segment_index"], bool)
        ):
            raise ForecastContractError("catalogue action provenance is invalid")
        try:
            canonical = validate_external_command(scenario, action["command"])
        except Exception as error:
            raise ForecastContractError(
                "catalogue command fails production validation"
            ) from error
        if canonical.sha256 != action["command_sha256"]:
            raise ForecastContractError("catalogue command self hash is invalid")
        if (
            action["action_id"] in ids
            or canonical.sha256 in hashes
            or action["source_mode"] in modes
        ):
            raise ForecastContractError(
                "catalogue actions must have distinct IDs, commands and modes"
            )
        ids.add(action["action_id"])
        hashes.add(canonical.sha256)
        modes.add(action["source_mode"])
        result.append(
            CatalogueAction(
                action["action_id"],
                action["source_mode"],
                action["source_segment_index"],
                canonical.sha256,
                canonical,
            )
        )
    return tuple(result)


def _validate_profile(
    value: Mapping[str, Any],
    reference_raw: Mapping[str, Any],
    fixture_raw: Mapping[str, Any],
) -> None:
    _exact(
        value,
        {
            "schema_version",
            "release_tier",
            "source_scenario_path",
            "source_scenario_sha256",
            "fixture_scenario_path",
            "fixture_scenario_sha256",
            "allowed_top_level_differences",
            "unchanged_top_level_fields",
            "unchanged_top_level_values_sha256",
            "profile_manifest_sha256",
        },
        "development profile",
    )
    if (
        value["schema_version"] != "aeolus_habitat_v2_forecast_development_profile_v1"
        or value["release_tier"] != RELEASE_TIER
    ):
        raise ForecastContractError(
            "development profile schema/release tier is unsupported"
        )
    _self_hash(value, "profile_manifest_sha256", _PROFILE_SHA256)
    if (
        value["source_scenario_sha256"] != _REFERENCE_SCENARIO_SHA256
        or value["fixture_scenario_sha256"] != _FIXTURE_SCENARIO_SHA256
    ):
        raise ForecastContractError("development profile scenario identities drift")
    allowed = value["allowed_top_level_differences"]
    names = (
        tuple(item.get("field") for item in allowed)
        if type(allowed) is list and all(type(item) is dict for item in allowed)
        else ()
    )
    if names != ("name", "steps", "fault_profiles", "timeline"):
        raise ForecastContractError(
            "development profile allows unsupported scenario differences"
        )
    unchanged = tuple(value["unchanged_top_level_fields"])
    if (
        set(unchanged) | set(names) != set(reference_raw)
        or set(reference_raw) != set(fixture_raw)
        or set(unchanged) & set(names)
    ):
        raise ForecastContractError(
            "development profile top-level coverage is inconsistent"
        )
    unchanged_values = {field: reference_raw[field] for field in unchanged}
    if (
        any(reference_raw[field] != fixture_raw[field] for field in unchanged)
        or hashlib.sha256(canonical_json_bytes(unchanged_values)).hexdigest()
        != value["unchanged_top_level_values_sha256"]
    ):
        raise ForecastContractError(
            "development scenario changed a forbidden top-level field"
        )


def _validate_records(value: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "release_tier",
        "canonical_serialization",
        "strictness",
        "identity_framing",
        "record_self_hash_rule",
        "manifest_self_hash_rule",
        "development_publication_boundary",
        "nested_contracts",
        "records",
        "record_contract_sha256",
    }
    _exact(value, fields, "development record contract")
    if (
        value["schema_version"] != "aeolus_habitat_v2_forecast_development_records_v1"
        or value["release_tier"] != RELEASE_TIER
    ):
        raise ForecastContractError(
            "development record contract schema/release tier is unsupported"
        )
    _self_hash(value, "record_contract_sha256", _RECORDS_SHA256)
    records = value["records"]
    expected = {
        "family_clusters",
        "families",
        "scenario_members",
        "split_assignments",
        "control_runs",
        "control_traces",
        "replay_witnesses",
        "samples",
        "manifest",
    }
    if type(records) is not dict or set(records) != expected:
        raise ForecastContractError(
            "development record tables are not the closed contract"
        )
    for name, record in records.items():
        if (
            type(record) is not dict
            or type(record.get("path")) is not str
            or not record["path"].startswith("development-fixture-only/")
            or type(record.get("required_fields")) is not list
            or len(record["required_fields"]) != len(set(record["required_fields"]))
        ):
            raise ForecastContractError(f"development record {name} is malformed")
        if name != "manifest" and (
            type(record.get("id_field")) is not str
            or type(record.get("identity_domain")) is not str
            or type(record.get("identity_fields")) is not list
        ):
            raise ForecastContractError(
                f"development record {name} identity contract is malformed"
            )
    return _freeze(value)


def load_forecast_contracts(root: str | Path) -> ForecastContracts:
    """Load the only supported D1 fixture bundle, validating every binding."""
    root_path = Path(root).resolve()
    contract_dir = root_path / "contracts"
    binding = _strict_json(contract_dir / "habitat_v2_forecast_hmc_binding_v1.json")
    alarm = _strict_json(contract_dir / "habitat_v2_forecast_alarm_manifest_v1.json")
    catalogue = _strict_json(
        contract_dir / "habitat_v2_forecast_action_catalogue_v1.json"
    )
    profile = _strict_json(
        contract_dir / "habitat_v2_forecast_development_profile_v1.json"
    )
    records = _strict_json(
        contract_dir / "habitat_v2_forecast_development_records_v1.json"
    )
    _validate_binding(root_path, binding)
    hmc = _load_hmc(root_path, binding)
    reference_raw = _strict_json(root_path / binding["reference_scenario_path"])
    fixture_raw = _strict_json(
        root_path / "scenarios" / "habitat_v2_forecast_development.json"
    )
    reference = _load_scenario(
        root_path / binding["reference_scenario_path"],
        expected_sha=_REFERENCE_SCENARIO_SHA256,
        label="reference",
    )
    fixture = _load_scenario(
        root_path / "scenarios" / "habitat_v2_forecast_development.json",
        expected_sha=_FIXTURE_SCENARIO_SHA256,
        label="development",
    )
    topology = derive_observable_topology(fixture)
    if (
        topology.sha256 != binding["observable_topology_sha256"]
        or topology.to_mapping() != binding["observable_topology"]
        or derive_observable_topology(reference).sha256 != topology.sha256
    ):
        raise ForecastContractError("observable topology drifts from frozen binding")
    _validate_profile(profile, reference_raw, fixture_raw)
    alarms = _validate_alarm(alarm, binding)
    actions = _validate_catalogue(catalogue, binding, reference)
    for action in actions:
        try:
            if (
                validate_external_command(fixture, action.command.to_mapping()).sha256
                != action.command_sha256
            ):
                raise ForecastContractError(
                    "development scenario changes a catalogue command"
                )
        except ForecastContractError:
            raise
        except Exception as error:
            raise ForecastContractError(
                "development scenario rejects a catalogue command"
            ) from error
    return ForecastContracts(
        root_path,
        RELEASE_TIER,
        binding["binding_sha256"],
        alarm["alarm_manifest_sha256"],
        catalogue["catalogue_sha256"],
        profile["profile_manifest_sha256"],
        records["record_contract_sha256"],
        hmc,
        reference,
        fixture,
        topology,
        alarms,
        tuple(alarm["lifecycle_order"]),
        actions,
        _validate_records(records),
    )
