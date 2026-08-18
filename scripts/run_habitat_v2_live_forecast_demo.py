"""Deprecated alias kept for compatibility.

The command was renamed to ``scripts/run_habitat_v2_live_forecast.py``.
This wrapper forwards to it unchanged; new work should use the new name.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

target = Path(__file__).with_name("run_habitat_v2_live_forecast.py")
sys.stderr.write(
    "note: renamed to scripts/run_habitat_v2_live_forecast.py "
    "(this alias keeps working)\n"
)
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
