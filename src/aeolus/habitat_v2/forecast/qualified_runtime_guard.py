"""Live, fail-closed resource guard for sealed V2 qualification execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class QualifiedRuntimeGuardError(RuntimeError):
    """A live resource or exclusivity control cannot be established."""


@dataclass(frozen=True)
class QualifiedRuntimeLimits:
    wall_clock_seconds: float
    min_free_ram_bytes: int
    min_free_vram_bytes: int
    max_gpu_temperature_c: int
    min_free_disk_bytes: int


class QualifiedRuntimeGuard:
    """Atomic run lock plus live checks; it never kills processes or deletes data."""

    def __init__(self, run_root: Path, limits: QualifiedRuntimeLimits) -> None:
        self.run_root = run_root
        self.limits = limits
        self.started = time.monotonic()
        self.lock_path = run_root / ".aeolus-v2-qualified.lock"
        self._held = False

    def __enter__(self) -> "QualifiedRuntimeGuard":
        self.run_root.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise QualifiedRuntimeGuardError("exclusive qualified launcher lock already exists") from error
        self._held = True
        self.check("startup")
        return self

    def __exit__(self, *_: object) -> None:
        if self._held:
            try:
                self.lock_path.unlink()
            except OSError:
                pass
            self._held = False

    def _gpu(self) -> tuple[int, int]:
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=5)
            rows = [line.strip().split(",") for line in output.splitlines() if line.strip()]
            if len(rows) != 1 or len(rows[0]) != 2:
                raise ValueError("ambiguous GPU inventory")
            return int(rows[0][0].strip()) * 1024 * 1024, int(rows[0][1].strip())
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise QualifiedRuntimeGuardError("cannot obtain an unambiguous live GPU VRAM/temperature reading") from error

    def _other_training_process(self) -> bool:
        try:
            import psutil
        except ImportError as error:
            raise QualifiedRuntimeGuardError("psutil is required for live concurrent-process enforcement") from error
        for process in psutil.process_iter(("pid", "cmdline")):
            if process.info["pid"] == os.getpid():
                continue
            command = " ".join(process.info.get("cmdline") or ()).lower()
            if "aeolus" in command and any(token in command for token in ("train", "fitcal", "pilot_campaign", "run_v2_fitcal")):
                return True
        return False

    def check(self, phase: str) -> None:
        if not self._held:
            raise QualifiedRuntimeGuardError("live guard is not holding its lock")
        if time.monotonic() - self.started > self.limits.wall_clock_seconds:
            raise QualifiedRuntimeGuardError(f"wall-clock timeout exceeded at {phase}")
        if self._other_training_process():
            raise QualifiedRuntimeGuardError("another AEOLUS/training process is live; refusing concurrent execution")
        try:
            import psutil
            free_ram = psutil.virtual_memory().available
        except (ImportError, OSError) as error:
            raise QualifiedRuntimeGuardError("cannot obtain live free RAM") from error
        if free_ram < self.limits.min_free_ram_bytes:
            raise QualifiedRuntimeGuardError(f"free RAM below abort reserve at {phase}")
        free_vram, temperature = self._gpu()
        if free_vram < self.limits.min_free_vram_bytes:
            raise QualifiedRuntimeGuardError(f"free VRAM below abort reserve at {phase}")
        if temperature >= self.limits.max_gpu_temperature_c:
            raise QualifiedRuntimeGuardError(f"GPU temperature reaches abort threshold at {phase}")
        if shutil.disk_usage(self.run_root).free < self.limits.min_free_disk_bytes:
            raise QualifiedRuntimeGuardError(f"disk reserve below abort threshold at {phase}")
