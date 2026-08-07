"""V6 corpus identity and observable-label contracts."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from aeolus.corpus_v6 import generate_v6_corpus, validate_v6_corpus
from aeolus.families_v6 import load_v6_family_manifest
from aeolus.sweep_v6 import generate_v6_sweep, load_v6_sweep_spec

ROOT = Path(__file__).resolve().parents[1]
V6_SPEC = ROOT / "scenarios" / "sweep-v6-development.json"


def _small_v6_spec(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "scenarios" / "v6", tmp_path / "v6")
    document = json.loads(V6_SPEC.read_text())
    document["targets"] = ["cabin_a"]
    selected = []
    for role in ("fit", "calibration", "validation"):
        family = copy.deepcopy(next(item for item in document["room_families"] if item["role"] == role))
        family["seeds"] = family["seeds"][:1]
        family["fault_start_ticks"] = family["fault_start_ticks"][:1]
        family["operating_profiles"] = family["operating_profiles"][:1]
        family["gradual_profiles"] = family["gradual_profiles"][:1]
        family["blocked_effectiveness"] = family["blocked_effectiveness"][:1]
        selected.append(family)
    document["room_families"] = selected
    path = tmp_path / "sweep-v6-small.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_v6_corpus_binds_every_row_to_generated_room_family_inputs(tmp_path):
    spec_path = _small_v6_spec(tmp_path)
    generated = tmp_path / "generated"
    generate_v6_sweep(spec_path, generated)
    spec = load_v6_sweep_spec(spec_path)
    families = load_v6_family_manifest(
        generated / "families-v6.json", expected_sweep=spec
    )

    corpus = generate_v6_corpus(
        families, tmp_path / "corpus", window_ticks=10, stride_ticks=5
    )

    assert corpus["schema_version"] == "aeolus_corpus_v6"
    assert corpus["sweep_spec_sha256"] == spec.sha256
    assert corpus["family_manifest_sha256"] == families.manifest_sha256
    assert corpus["family_count"] == 9
    rows = validate_v6_corpus(tmp_path / "corpus", expected_families=families)
    assert {row["room_family_id"] for row in rows} == {
        "room-balanced",
        "room-capacity-constrained",
        "room-transition-heavy",
    }
    assert all(row["observable_context_version"] == "observable_context_v1" for row in rows)
