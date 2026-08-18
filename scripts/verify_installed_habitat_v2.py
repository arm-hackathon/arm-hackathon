"""Verify an installed aeolus wheel with the Habitat V2 compound-fault scenario."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    return parser.parse_args(argv)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def wheel_sha256(wheel_path: Path) -> str:
    digest = hashlib.sha256()
    with wheel_path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_url_path(wheel_url: str) -> Path:
    parsed = urlparse(wheel_url)
    if parsed.scheme != "file":
        raise RuntimeError(f"installed distribution URL is not local: {wheel_url}")
    return Path(url2pathname(unquote(parsed.path))).resolve()


def assert_installed_wheel(*, source_root: Path, wheel_path: Path) -> None:
    import aeolus

    module_path = Path(aeolus.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    source_package_root = (source_root / "src").resolve()
    if is_within(module_path, source_package_root):
        raise RuntimeError(f"aeolus imported from the source tree: {module_path}")
    if not is_within(module_path, prefix):
        raise RuntimeError(
            f"aeolus imported outside the clean environment {prefix}: {module_path}"
        )

    distribution = importlib.metadata.distribution("aeolus")
    metadata_version = distribution.version
    if metadata_version != aeolus.__version__:
        raise RuntimeError(
            "package metadata/runtime version mismatch: "
            f"{metadata_version} != {aeolus.__version__}"
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("installed wheel has no direct_url.json provenance record")
    direct_url = json.loads(direct_url_text)
    installed_wheel = wheel_url_path(str(direct_url["url"]))
    if installed_wheel != wheel_path:
        raise RuntimeError(
            f"installed wheel does not match requested artifact: {installed_wheel} != {wheel_path}"
        )
    installed_hash = direct_url.get("archive_info", {}).get("hash")
    expected_hash = f"sha256={wheel_sha256(wheel_path)}"
    if installed_hash is not None and installed_hash != expected_hash:
        raise RuntimeError(
            f"installed wheel hash does not match requested artifact: {installed_hash} != {expected_hash}"
        )

    print(f"installed_module={module_path}")
    print(f"installed_version={metadata_version}")
    print(f"requested_wheel_sha256={expected_hash}")


def run_compound_fault_smoke(*, scenario_path: Path, trace_path: Path) -> None:
    from aeolus.habitat_v2.config import load_scenario_file
    from aeolus.habitat_v2.trace import validate_trace_bytes

    if trace_path.exists():
        raise RuntimeError(f"refusing to overwrite existing smoke trace: {trace_path}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "aeolus.habitat_v2",
            str(scenario_path),
            str(trace_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "compound-fault CLI smoke failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    scenario = load_scenario_file(scenario_path)
    rows = validate_trace_bytes(trace_path.read_bytes(), scenario=scenario)
    expected_fault_ids = [
        "airlock-supply-damper-jam",
        "fan-drive-degradation",
        "galley-primary-co2-drift",
        "laboratory-supply-blockage",
        "power-bay-secondary-temperature-stuck",
    ]
    actual_fault_ids = [
        fault["fault_id"] for fault in rows[1]["fault_receipt"]["active_faults"]
    ]
    if actual_fault_ids != expected_fault_ids:
        raise RuntimeError(
            f"compound-fault receipt changed: {actual_fault_ids} != {expected_fault_ids}"
        )
    if rows[-1]["fault_receipt"]["active_faults"] != []:
        raise RuntimeError("compound-fault receipt did not clear after the final step")

    print(completed.stdout, end="")
    print(f"compound_fault_trace={trace_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_root.resolve()
    wheel_path = args.wheel.resolve()
    scenario_path = args.scenario.resolve()
    trace_path = args.trace.resolve()

    for required_path in (wheel_path, scenario_path):
        if not required_path.is_file():
            raise RuntimeError(f"required file does not exist: {required_path}")
    if not source_root.is_dir():
        raise RuntimeError(f"source root does not exist: {source_root}")

    assert_installed_wheel(source_root=source_root, wheel_path=wheel_path)
    run_compound_fault_smoke(scenario_path=scenario_path, trace_path=trace_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
