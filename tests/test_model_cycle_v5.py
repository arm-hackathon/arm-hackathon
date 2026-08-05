"""V5 model-cycle protocol binding."""

from __future__ import annotations

from pathlib import Path

import aeolus.model_cycle_v5 as v5


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v5_runner_binds_only_the_predeclared_protocol(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_run(spec_path: str | Path, output_dir: str | Path, **kwargs: object) -> dict[str, object]:
        captured["spec_path"] = spec_path
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"retained_method": "rule_baseline"}

    monkeypatch.setattr(v5, "_run_model_cycle", fake_run)

    result = v5.run_v5_development(
        REPO_ROOT / "scenarios" / "sweep-v5-development.json",
        tmp_path / "out",
        mlp_epochs=7,
        cnn_epochs=11,
    )

    assert result == {"retained_method": "rule_baseline"}
    assert captured["protocol_name"] == "v5"
    assert captured["expected_schema_version"] == "aeolus_sweep_v5"
    assert captured["expected_spec_sha256"] == v5.CANONICAL_V5_DEVELOPMENT_SPEC_SHA256
    assert captured["fit_seeds"] == (1100, 1101, 1102, 1103)
    assert captured["calibration_seeds"] == (1104, 1105)
    assert captured["validation_seeds"] == (1300, 1301, 1302, 1303, 1304, 1305)
    assert captured["report_schema_version"] == "aeolus_v5_development_evidence_v1"
    assert captured["report_filename"] == "v5-development-report.json"
    assert captured["mlp_epochs"] == 7
    assert captured["cnn_epochs"] == 11
    assert "scenarios/sweep-v5-development.json" in captured["source_paths"]
    assert "src/aeolus/model_cycle_v5.py" in captured["source_paths"]
    assert "src/aeolus/corpus.py" in captured["source_paths"]
    assert "src/aeolus/detector.py" in captured["source_paths"]
    assert "src/aeolus/model_input.py" in captured["source_paths"]
    assert "src/aeolus/plant.py" in captured["source_paths"]
    assert "src/aeolus/baseline.py" in captured["source_paths"]
    assert "src/aeolus/control.py" in captured["source_paths"]
    assert "src/aeolus/measurement.py" in captured["source_paths"]
    assert "src/aeolus/trace.py" in captured["source_paths"]
