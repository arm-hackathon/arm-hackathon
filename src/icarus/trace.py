"""JSONL replay trace for the ICARUS scenario graph.

Each tick of a scenario run is persisted as one JSON object on its own line,
so a trace file can be diffed, replayed, or eyeballed line by line.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TickRecord:
    """The persisted state of one tick.

    ``zones`` maps every zone id to its CO2 readings; the air_processing
    bay's entry also carries its cumulative ``captured_co2`` counter.
    ``connections`` maps every connection id to its actual ``airflow`` for
    the tick. Scenario fault labels and hidden effectiveness/health values are
    intentionally excluded from the persisted telemetry.
    """

    tick: int
    zones: dict[str, dict[str, float]] = field(default_factory=dict)
    connections: dict[str, dict[str, float]] = field(default_factory=dict)


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
        _validate_observable_telemetry(record)
        # sort_keys keeps the byte layout stable so identical runs diff clean.
        self._handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def __exit__(self, *exc_info):
        if self._handle is not None:
            self._handle.close()
        return False


def _validate_observable_telemetry(record: TickRecord) -> None:
    """Reject hidden scenario truth before it can enter a trace."""
    for zone_id, telemetry in record.zones.items():
        fields = set(telemetry)
        if "co2" not in fields or not fields <= {"co2", "captured_co2"}:
            raise ValueError(
                f"zone {zone_id!r} trace telemetry must contain only observable "
                f"co2/captured_co2 fields"
            )
    for connection_id, telemetry in record.connections.items():
        if set(telemetry) != {"airflow"}:
            raise ValueError(
                f"connection {connection_id!r} trace telemetry must contain only "
                f"observable airflow"
            )
