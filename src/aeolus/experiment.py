"""One-command deterministic AEOLUS fault-detection experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from aeolus.corpus import generate_corpus_v2
from aeolus.detector import train_and_export
from aeolus.sweep import generate_sweep

USAGE = (
    "Usage: PYTHONPATH=src python -m aeolus.experiment "
    "<sweep.json> <experiment-output-dir> <artifacts-dir>"
)


def run_experiment(
    sweep_spec_path: str | Path,
    experiment_output_dir: str | Path,
    artifacts_dir: str | Path,
) -> dict[str, object]:
    """Generate scenarios/corpus, select candidates, and export final evidence."""
    output = Path(experiment_output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"experiment output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    sweep_dir = output / "sweep"
    corpus_dir = output / "corpus"
    sweep_receipt = generate_sweep(sweep_spec_path, sweep_dir)
    corpus_manifest = generate_corpus_v2(sweep_dir / "families.json", corpus_dir)
    metrics = train_and_export(
        corpus_dir / "corpus.jsonl",
        sweep_dir / "families.json",
        str(sweep_receipt["family_manifest_sha256"]),
        artifacts / "aeolus_fault_detector.json",
        artifacts / "aeolus_fault_detector.onnx",
        artifacts / "aeolus_fault_metrics.json",
    )
    artifact_hashes = {
        name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
        for name in (
            "aeolus_fault_detector.json",
            "aeolus_fault_detector.onnx",
            "aeolus_fault_metrics.json",
        )
    }
    receipt = {
        "schema_version": "aeolus_experiment_v1",
        "sweep": sweep_receipt,
        "corpus_manifest_sha256": corpus_manifest["manifest_sha256"],
        "selected_candidate": metrics["candidate_selection"]["selected_candidate"],
        "ai_advantage_demonstrated": metrics["evidence_conclusion"][
            "ai_advantage_demonstrated"
        ],
        "artifact_sha256": artifact_hashes,
    }
    (output / "experiment-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        receipt = run_experiment(*argv)
    except (ValueError, OSError, ImportError, json.JSONDecodeError) as exc:
        print(f"cannot run experiment: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
