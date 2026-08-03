"""One-command deterministic AEOLUS fault-detection experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
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
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_FILENAMES = (
    "aeolus_fault_detector.json",
    "aeolus_fault_detector.onnx",
    "aeolus_fault_metrics.json",
)


def _git_output(*args: str) -> str:
    """Return a repository-local Git command result for an evidence receipt."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _evidence_environment(onnx_path: Path) -> dict[str, object]:
    """Describe the source and exporter that produced an ONNX evidence artifact."""
    import onnx

    lock_path = REPOSITORY_ROOT / "uv.lock"
    if not lock_path.is_file():
        raise OSError(f"canonical evidence requires lock file: {lock_path}")
    onnx_model = onnx.load(onnx_path)
    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": importlib.metadata.version("numpy"),
        "onnx_version": importlib.metadata.version("onnx"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "onnx_ir_version": int(onnx_model.ir_version),
        "onnx_opsets": [
            {"domain": entry.domain, "version": int(entry.version)}
            for entry in onnx_model.opset_import
        ],
        "uv_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(_git_output("status", "--porcelain")),
    }


def run_experiment(
    sweep_spec_path: str | Path,
    experiment_output_dir: str | Path,
    artifacts_dir: str | Path,
) -> dict[str, object]:
    """Generate scenarios/corpus, select candidates, and export final evidence."""
    output = Path(experiment_output_dir)
    artifacts = Path(artifacts_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"experiment output directory is not empty: {output}")
    if artifacts.exists() and any(artifacts.iterdir()):
        raise ValueError(f"artifact output directory is not empty: {artifacts}")
    output.mkdir(parents=True, exist_ok=True)
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
        for name in ARTIFACT_FILENAMES
    }
    receipt = {
        "schema_version": "aeolus_experiment_v2",
        "environment": _evidence_environment(
            artifacts / "aeolus_fault_detector.onnx"
        ),
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
