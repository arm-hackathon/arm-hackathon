"""Run one V7 canonical development cycle from the repository root.

Usage (from the repository root, with the project venv active):

    env PYTHONPATH=src .venv/bin/python scripts/run_v7_canonical.py \
        scenarios/sweep-v7-development.json out/v7-<strategy>-canonical-YYYY-MM-DD-a

Wraps ``run_v7_development`` behind a ``__main__`` guard so the calibration
process pool works on platforms that spawn workers (Windows) as well as those
that fork (Linux/macOS). The runner requires a clean Git worktree and writes
its full receipt to ``<output-dir>/v7-development-report.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from aeolus.model_cycle_v7 import V7DevelopmentRequest, run_v7_development

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python scripts/run_v7_canonical.py <sweep-spec.json> <output-dir>"
        )
    request = V7DevelopmentRequest(
        sweep_spec_path=Path(sys.argv[1]).resolve(),
        output_dir=Path(sys.argv[2]).resolve(),
    )
    report = run_v7_development(request)
    print(
        f"development_gate_passed={report.get('development_gate_passed')} "
        f"selected_candidate={report.get('selected_candidate')}"
    )
