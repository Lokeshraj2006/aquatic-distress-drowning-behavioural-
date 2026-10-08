"""Multi-camera event index for the swimming-pool CCTV intelligence pipeline.

The existing detector and behavior engine remain unchanged. This module converts
those results into a camera-aware, queryable record and stores clarification
references separately so a later natural-language query can reuse them.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping


class EventIndex:
    """A SQLite index containing events from one or more cameras."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    person_id INTEGER,
                    behavior TEXT NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    confidence REAL,
                    evidence TEXT,
                    snapshot TEXT,
                    metadata_json TEXT,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (camera_id, event_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_camera_person ON events(camera_id, person_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_behavior_time ON events(behavior, start_s)"
            )
            connection.commit()
        finally:
            connection.close()

    def index_events(
        self,
        camera_id: str,
        video_path: str | Path,
        events: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert or replace events for one camera and return the inserted count."""
        rows = []
        for event in events:
            rows.append(
                (
                    int(event.get("event_id", len(rows) + 1)),
                    str(camera_id),
                    str(video_path),
                    event.get("entity_id") if event.get("entity_id") is not None else event.get("person_id"),
                    str(event.get("behavior", "unknown")),
                    float(event.get("start_s", 0.0)),
                    float(event.get("end_s", 0.0)),
                    float(event.get("confidence", 0.0)) if event.get("confidence") is not None else None,
                    str(event.get("evidence", "")),
                    event.get("snapshot") or event.get("snapshot_path"),
                    json.dumps(dict(metadata or {}), sort_keys=True),
                    event.get("indexed_at", ""),
                )
            )
        if not rows:
            return 0
        connection = self._connect()
        try:
            connection.executemany(
                """
                INSERT INTO events (
                    event_id, camera_id, video_path, person_id, behavior,
                    start_s, end_s, confidence, evidence, snapshot,
                    metadata_json, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id, event_id) DO UPDATE SET
                    video_path=excluded.video_path,
                    person_id=excluded.person_id,
                    behavior=excluded.behavior,
                    start_s=excluded.start_s,
                    end_s=excluded.end_s,
                    confidence=excluded.confidence,
                    evidence=excluded.evidence,
                    snapshot=excluded.snapshot,
                    metadata_json=excluded.metadata_json,
                    indexed_at=excluded.indexed_at
                """,
                rows,
            )
            connection.commit()
            return len(rows)
        finally:
            connection.close()

    def query(
        self,
        camera_id: str | None = None,
        person_id: int | None = None,
        behavior: str | None = None,
        since_s: float | None = None,
        until_s: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching event records in chronological order."""
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if person_id is not None:
            clauses.append("person_id = ?")
            params.append(int(person_id))
        if behavior:
            clauses.append("behavior = ?")
            params.append(behavior.lower())
        if since_s is not None:
            clauses.append("start_s >= ?")
            params.append(float(since_s))
        if until_s is not None:
            clauses.append("end_s <= ?")
            params.append(float(until_s))

        sql = "SELECT * FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY start_s, camera_id, event_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        connection = self._connect()
        try:
            rows = connection.execute(sql, params).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            connection.close()

    def index_file(
        self,
        camera_id: str,
        video_path: str | Path,
        events_path: str | Path,
    ) -> int:
        """Index an existing events.json document without rerunning detection."""
        document = json.loads(Path(events_path).read_text(encoding="utf-8"))
        events = document.get("events", [])
        metadata = document.get("meta", {})
        metadata.setdefault("source", str(events_path))
        return self.index_events(camera_id, video_path, events, metadata)

    def query_text(self, text: str, limit: int | None = 20) -> list[dict[str, Any]]:
        """Simple natural-language search over camera, person, behavior and evidence.

        This is intentionally conservative: it understands fixed terms such as
        "camera 2", "person 7", "drowning", and "last 10 minutes". Semantic
        retrieval can be layered on top without changing the indexed schema.
        """
        normalized = text.lower()
        camera_id = None
        person_id = None
        behavior = None
        since_s = None
        if "camera" in normalized:
            match = __import__("re").search(r"camera\s+([0-9]+)", normalized)
            if match:
                camera_id = f"camera_{match.group(1)}"
        match = __import__("re").search(r"person\s+([0-9]+)", normalized)
        if match:
            person_id = int(match.group(1))
        for value in ("drowning", "distress", "fall", "loitering", "running", "crowding", "zone_intrusion"):
            if value in normalized:
                behavior = value
                break
        match = __import__("re").search(r"last\s+(\d+)\s*(?:minutes?|mins?)", normalized)
        if match:
            since_s = max(0.0, float(match.group(1)) * 60.0)
        return self.query(camera_id, person_id, behavior, since_s, limit=limit)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["person_id"] = result.get("person_id")
        result["snapshot"] = result.get("snapshot")
        result["metadata"] = json.loads(result.get("metadata_json") or "{}")
        del result["metadata_json"]
        return result


class ClarifyMemory:
    """Persistent, camera-scoped references supplied through a clarification."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}
        else:
            self._data = {}

    def store(
        self,
        camera_id: str,
        reference: str,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        camera = self._data.setdefault(str(camera_id), {})
        camera["reference"] = str(reference)
        if aliases:
            camera.update({str(key): str(value) for key, value in aliases.items()})
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def resolve(self, camera_id: str, alias: str) -> str | None:
        camera = self._data.get(str(camera_id), {})
        normalized_alias = re.sub(r"[^a-z0-9]+", "_", str(alias).strip().lower()).strip("_")
        return camera.get(normalized_alias) or camera.get(str(alias))

    def get(self, camera_id: str) -> dict[str, str]:
        return dict(self._data.get(str(camera_id), {}))


def main() -> None:
    """Command-line helper: index an existing events.json file."""
    import argparse

    parser = argparse.ArgumentParser(description="Index an existing event report for a camera")
    parser.add_argument("--database", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    index = EventIndex(args.database)
    count = index.index_file(args.camera, args.video, args.events)
    print(f"Indexed {count} events for {args.camera}")


if __name__ == "__main__":
    main()
