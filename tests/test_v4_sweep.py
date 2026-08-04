"""Fresh v4 development-only family contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.families import load_family_manifest, validate_manifest_disjointness
from aeolus.model_cycle_v4 import CANONICAL_V4_DEVELOPMENT_SPEC_SHA256
from aeolus.sweep import generate_sweep, parse_sweep_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v4_accepts_only_development_train_and_validation():
    path = REPO_ROOT / "scenarios" / "sweep-v4-development.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    parsed = parse_sweep_spec(document, source_path=path)

    assert parsed.schema_version == "aeolus_sweep_v4"
    assert parsed.suite_role == "development"
    assert tuple(parsed.splits) == ("train", "validation")
    assert parsed.splits["train"].seeds == tuple(range(700, 706))
    assert parsed.splits["validation"].seeds == tuple(range(900, 906))
    assert parsed.sha256 == CANONICAL_V4_DEVELOPMENT_SPEC_SHA256

    forbidden = json.loads(json.dumps(document))
    forbidden["suite_role"] = "final"
    forbidden["splits"] = {"final": forbidden["splits"]["validation"]}
    with pytest.raises(ValueError, match="v4.*development"):
        parse_sweep_spec(forbidden, source_path=path)


def test_checked_in_v4_generates_fresh_disjoint_declared_families(tmp_path: Path):
    v3_development_path = REPO_ROOT / "scenarios" / "sweep-v3-development.json"
    v3_final_path = REPO_ROOT / "scenarios" / "sweep-v3-final.json"
    v4_path = REPO_ROOT / "scenarios" / "sweep-v4-development.json"

    v3_development_dir = tmp_path / "v3-development"
    v3_final_dir = tmp_path / "v3-final"
    v4_dir = tmp_path / "v4-development"
    generate_sweep(v3_development_path, v3_development_dir)
    generate_sweep(v3_final_path, v3_final_dir)
    receipt = generate_sweep(v4_path, v4_dir)

    assert receipt["families_by_split"] == {"train": 360, "validation": 360}
    assert receipt["total_families"] == 720
    v3_development = load_family_manifest(v3_development_dir / "families.json")
    v3_final = load_family_manifest(v3_final_dir / "families.json")
    v4 = load_family_manifest(v4_dir / "families.json")
    validate_manifest_disjointness(v3_development, v4)
    validate_manifest_disjointness(v3_final, v4)
