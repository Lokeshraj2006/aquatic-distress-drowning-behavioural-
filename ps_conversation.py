"""Grounded conversational intelligence over the existing aquatic CV results.

This module intentionally does not invent cameras, identities, events, or evidence.
It reads the existing result artifacts produced by ``run.py`` and exposes a small
query contract for the Streamlit application.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


class PersistentReferenceMemory:
    """SQLite-backed, user-defined camera/reference mappings."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS \"references\" ("
                "reference_name TEXT PRIMARY KEY, camera_id TEXT NOT NULL, "
                "provenance TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def remember(self, reference_name: str, camera_id: str, provenance: str) -> None:
        if not reference_name.strip() or not camera_id.strip():
            raise ValueError("Both reference name and camera ID are required")
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO \"references\"(reference_name, camera_id, provenance, created_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(reference_name) DO UPDATE SET camera_id=excluded.camera_id, "
                "provenance=excluded.provenance, created_at=CURRENT_TIMESTAMP",
                (reference_name.strip().lower(), camera_id.strip(), provenance),
            )
            connection.commit()
        finally:
            connection.close()

    def lookup(self, reference_name: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT camera_id, provenance FROM \"references\" WHERE reference_name = ?",
                (reference_name.strip().lower(),),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {"camera_id": row[0], "provenance": row[1]}


class ConversationIndex:
    """Query only the real artifacts emitted by the aquatic CV engine."""

    def __init__(self, results_folder: Path, reference_memory: PersistentReferenceMemory | None = None):
        self.results_folder = Path(results_folder)
        self.reference_memory = reference_memory
        self.events_path = self.results_folder / "events.json"
        self.dashboard_path = self.results_folder / "pool_dashboard.json"
        self.tracking_path = self.results_folder / "stage1_tracking.jsonl"

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _read_tracking(self) -> list[dict[str, Any]]:
        if not self.tracking_path.is_file():
            return []
        records = []
        for line in self.tracking_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _format_time(self, seconds: float) -> str:
        minutes, remainder = divmod(max(0.0, float(seconds)), 60.0)
        return f"{minutes:02.0f}:{remainder:05.2f}"

    def _find_person(self, dashboard: dict[str, Any], person_id: int) -> dict[str, Any] | None:
        for person in dashboard.get("people", []):
            if int(person.get("person_id", -1)) == person_id:
                return person
        return None

    def _risk_at(self, dashboard: dict[str, Any], person_id: int, at_time: float) -> tuple[float, str] | None:
        risk_by_person = dashboard.get("risk", {})
        points = risk_by_person.get(str(person_id), [])
        if not points:
            return None
        selected = max((point for point in points if float(point[0]) <= at_time), key=lambda point: float(point[0]), default=None)
        if selected is None:
            return None
        return float(selected[1]), self._format_time(float(selected[0]))

    def _event_at(self, events: list[dict[str, Any]], at_time: float) -> list[dict[str, Any]]:
        return [
            event for event in events
            if float(event.get("start", -1)) <= at_time <= float(event.get("end", at_time))
        ]

    def _evidence_path(self, folder: Path, evidence: dict[str, Any]) -> str | None:
        for candidate in (
            evidence.get("snapshot"),
            evidence.get("speed_plot"),
            evidence.get("annotated_video"),
            evidence.get("evidence_frame"),
        ):
            if not candidate:
                continue
            path = (folder / str(candidate)).resolve()
            try:
                path.relative_to(folder.resolve())
            except ValueError:
                continue
            if path.is_file():
                return path.relative_to(Path.cwd()).as_posix() if path.is_relative_to(Path.cwd()) else str(path)
        for name in ("annotated.mp4", "highlights.mp4", "timeline.png", "heatmap.jpg"):
            path = folder / name
            if path.is_file():
                return str(path)
        return None

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def _parse_time(self, query: str) -> float | None:
        match = re.search(r"(?:at|around|between|from|to)\s+([0-1]?\d|2[0-3]):([0-5]\d)", self._normalize_query(query))
        if match:
            hours, minutes = map(int, match.groups())
            return hours * 60.0 + minutes
        return None

    def _parse_person_id(self, query: str) -> int | None:
        match = re.search(r"swimmer\s*(?:#|id)?\s*(\d+)|person\s*(?:#|id)?\s*(\d+)", self._normalize_query(query))
        if match:
            return int(next(group for group in match.groups() if group is not None))
        return None

    def _range(self, query: str) -> tuple[float, float] | None:
        match = re.search(r"between\s+(\d{2}:\d{2})\s+and\s+(\d{2}:\d{2})", self._normalize_query(query))
        if not match:
            return None
        start_h, start_m = map(int, match.group(1).split(":"))
        end_h, end_m = map(int, match.group(2).split(":"))
        return start_h * 60 + start_m, end_h * 60 + end_m

    def _clamp_time(self, value: float, duration: float) -> float:
        return max(0.0, min(float(duration), float(value)))

    def _record(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "source": "verified_pool_video",
            "camera_id": None,
            "timestamp": None,
            "track_id": None,
            "risk": None,
            "confidence": None,
            "event_state": None,
            "evidence_status": "unavailable",
            "evidence_path": None,
            "evidence": None,
            **kwargs,
        }

    def answer(self, query: str) -> dict[str, Any]:
        normalized = self._normalize_query(query)
        dashboard = self._read_json(self.dashboard_path)
        events_data = self._read_json(self.events_path)
        events = events_data.get("events", []) if isinstance(events_data.get("events"), list) else []
        duration = float(dashboard.get("duration_s", 0.0) or events_data.get("meta", {}).get("duration_s", 0.0))
        person_id = self._parse_person_id(query)
        target_time = self._parse_time(query)
        time_range = self._range(query)

        if re.search(r"highest risk|highest.*risk|risk.*highest", normalized):
            candidates = []
            for person in dashboard.get("people", []):
                pid = int(person.get("person_id", -1))
                points = dashboard.get("risk", {}).get(str(pid), [])
                if not points:
                    continue
                if target_time is None:
                    risk = max(float(point[1]) for point in points)
                    selected_time = max(float(point[0]) for point in points)
                else:
                    selected = max(
                        (point for point in points if float(point[0]) <= target_time),
                        key=lambda point: float(point[0]),
                        default=None,
                    )
                    if selected is None:
                        continue
                    risk = float(selected[1])
                    selected_time = float(selected[0])
                candidates.append((risk, pid, person.get("name", f"Swimmer #{pid}"), selected_time))
            if candidates:
                risk, pid, name, selected_time = max(candidates, key=lambda item: item[0])
                evidence_path = self._evidence_path(self.results_folder, {})
                return self._result(
                    f"{name} had the highest verified risk ({risk:.0%}) at {self._format_time(selected_time)} in this analysis.",
                    self._record(
                        track_id=pid,
                        risk=risk,
                        source=str(dashboard.get("video") or self._source_label()),
                        timestamp=self._format_time(selected_time),
                        evidence_status="available" if evidence_path else "unavailable",
                        evidence_path=evidence_path,
                    ),
                    "ok",
                )
            return self._result(
                "No verified risk curve was found in the current analysis.",
                self._record(evidence_status="unavailable"),
                "no_matches",
            )

        if "what happened" in normalized:
            if time_range is not None:
                start, end = time_range
                matching = [
                    event for event in events
                    if start <= float(event.get("start", 0)) <= end
                    or start <= float(event.get("end", 0)) <= end
                ]
                if matching:
                    event = matching[0]
                    evidence_path = self._evidence_path(self.results_folder, event)
                    return self._result(
                        f"Verified event state in {self._format_time(start)} to {self._format_time(end)}: "
                        f"event_id {event.get('event_id')} for Swimmer #{event.get('entity_id')}: "
                        f"{event.get('behavior_name', 'observed behavior')}.",
                        self._record(
                            timestamp=self._format_time(float(event.get("start", 0))),
                            track_id=int(event.get("entity_id", -1)),
                            event_state=event.get("behavior_name"),
                            evidence_status="available" if evidence_path else "unavailable",
                            evidence_path=evidence_path,
                            evidence=event.get("evidence"),
                        ),
                        "ok",
                    )
                return self._result(
                    f"No verified event was found in the {self._format_time(start)} to {self._format_time(end)} interval.",
                    self._record(timestamp=f"{self._format_time(start)} to {self._format_time(end)}", evidence_status="unavailable"),
                    "no_matches",
                )
            if target_time is not None:
                matching = self._event_at(events, target_time)
                if matching:
                    event = matching[0]
                    evidence_path = self._evidence_path(self.results_folder, event)
                    return self._result(
                        f"Verified event state at {self._format_time(target_time)}: event_id {event.get('event_id')} "
                        f"for Swimmer #{event.get('entity_id')}: {event.get('behavior_name', 'observed behavior')}.",
                        self._record(
                            timestamp=self._format_time(target_time),
                            track_id=int(event.get("entity_id", -1)),
                            event_state=event.get("behavior_name"),
                            evidence_status="available" if evidence_path else "unavailable",
                            evidence_path=evidence_path,
                            evidence=event.get("evidence"),
                        ),
                        "ok",
                    )
                return self._result(
                    f"No verified event was found at {self._format_time(target_time)}.",
                    self._record(timestamp=self._format_time(target_time), evidence_status="unavailable"),
                    "no_matches",
                )

        if person_id is not None:
            person = self._find_person(dashboard, person_id)
            if person is None:
                return self._result(
                    f"No verified swimmer record for #{person_id} in the current analysis.",
                    self._record(track_id=person_id, source=self._source_label(), timestamp=None,
                                 evidence_status="unavailable"),
                    "no_match",
                )
            risk = self._risk_at(dashboard, person_id, target_time or float(person["last_seen"]))
            events_for_person = [event for event in events if int(event.get("entity_id", -1)) == person_id]
            timeline = self._timeline_for_person(dashboard, person_id)
            source = str(dashboard.get("video") or self._source_label())
            answer = f"Swimmer #{person_id} was observed from {self._format_time(float(person['first_seen']))} to " \
                f"{self._format_time(float(person['last_seen']))}. Current state: {person.get('final_state', 'unknown')}."
            if risk is not None:
                answer += f" At {self._format_time(target_time or float(person['last_seen']))}, risk was {risk[0]:.0%}."
            if events_for_person:
                answer += f" Verified event state: {events_for_person[0].get('behavior_name', 'observed event')} at " \
                    f"{self._format_time(float(events_for_person[0].get('start', 0)))}."
            if timeline:
                answer += f" Behaviour timeline: {timeline}."
            evidence_path = self._evidence_path(self.results_folder, events_for_person[0] if events_for_person else {})
            return self._result(
                answer,
                self._record(
                    source=source,
                    timestamp=self._format_time(target_time or float(person["last_seen"])),
                    track_id=person_id,
                    risk=risk[0] if risk else None,
                    confidence=None,
                    event_state=events_for_person[0].get("behavior_name") if events_for_person else person.get("final_state"),
                    evidence_status="available" if evidence_path else "unavailable",
                    evidence_path=evidence_path,
                    evidence=events_for_person[0].get("evidence") if events_for_person else None,
                ),
                "ok",
            )

        if "deep end" in normalized or "where is" in normalized or "which camera" in normalized:
            reference = self.reference_memory.lookup("deep_end") if self.reference_memory else None
            if reference is None:
                return self._result(
                    "I do not have a verified camera mapping for 'deep end'. Which camera corresponds to it? "
                    "This is a user-defined reference and is not verified aquatic camera evidence.",
                    self._record(source="user_defined_reference", event_state="clarification_required",
                                 evidence_status="unavailable"),
                    "clarification_required",
                )
            return self._result(
                f"Saved user-defined reference: deep_end → {reference['camera_id']}. "
                f"Verified aquatic camera evidence is unavailable.",
                self._record(source="user_defined_reference", camera_id=reference["camera_id"],
                             event_state="mapped_reference", evidence_status="unavailable"),
                "ok",
            )

        if "visible" in normalized or "swimmers are currently" in normalized:
            people = dashboard.get("people", [])
            if people:
                names = ", ".join(f"{person.get('name', f'Swimmer #{person.get("person_id")}')} ({self._format_time(float(person['first_seen']))}–{self._format_time(float(person['last_seen']))})" for person in people)
                return self._result(
                    f"Verified visible swimmers in the current analysis: {names}.",
                    self._record(source=self._source_label(), evidence_status="available" if (self.results_folder / "annotated.mp4").is_file() else "unavailable"),
                    "ok",
                )

        return self._result(
            "I can answer questions about verified swimmers, tracker IDs, risk, time windows, events, and evidence from the current aquatic analysis. "
            "Questions about unverified cameras, global identity, or arbitrary visual attributes are not supported.",
            self._record(evidence_status="unavailable"),
            "unsupported",
        )

    def _timeline_for_person(self, dashboard: dict[str, Any], person_id: int) -> str:
        timeline = dashboard.get("timeline", {}).get(str(person_id), [])
        if not timeline:
            return ""
        return "; ".join(f"{self._format_time(float(item['t']))}: {item.get('note', '')}" for item in timeline)

    def _source_label(self) -> str:
        return "verified_pool_video"

    def _result(self, answer: str, record: dict[str, Any], status: str) -> dict[str, Any]:
        return {
            "status": status,
            "answer": answer,
            "record": record,
            "multi_camera": {
                "status": "unavailable",
                "source": "current_dataset",
                "cameras": [],
                "evidence": [],
                "message": "Multi-camera aquatic validation is unavailable with the current dataset.",
            },
            "memory": None,
        }
