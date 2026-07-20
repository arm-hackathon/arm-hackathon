"""JSONL replay trace for the ICARUS proof loop.

Each tick of a scenario run is persisted as one JSON object on its own line,
so a trace file can be diffed, replayed, or eyeballed line by line.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TickRecord:
    """The persisted state of one tick, plus the airflow that produced it."""

    tick: int
    room_a_co2: float
    room_b_co2: float
    main_fan_effectiveness: float
    airflow: float


class TraceWriter:
    """Append :class:`TickRecord` rows to a JSONL file, one record per line."""

    def __init__(self, path):
        self._path = Path(path)
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
        return self

    def write(self, record: TickRecord) -> None:
        if self._handle is None:
            raise RuntimeError("TraceWriter.write() called outside a 'with' block")
        # sort_keys keeps the byte layout stable so identical runs diff clean.
        self._handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def __exit__(self, *exc_info):
        if self._handle is not None:
            self._handle.close()
        return False
