"""Local JSONL observation buffer. SQLite comes in a later phase if needed.

Offline-first: append-only file that survives API/database unavailability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from cropai.dataset.schema import ObservationRecord
from cropai.utils.paths import data_dir


class ObservationStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "processed" / "observations.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ObservationRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[ObservationRecord]:
        if not self.path.exists():
            return []
        out: list[ObservationRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                out.append(ObservationRecord(**data))
        return out

    def follow_ups(self, observation_id: str) -> list[ObservationRecord]:
        return [r for r in self.read_all() if r.follow_up_of == observation_id]

    def extend(self, records: Iterable[ObservationRecord]) -> int:
        n = 0
        for rec in records:
            self.append(rec)
            n += 1
        return n
