"""Tests for the multi-camera event index and clarify-once memory."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_index import ClarifyMemory, EventIndex


class EventIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = EventIndex(self.root / "events.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_indexes_multiple_cameras_and_queries_by_person_and_behavior(self):
        events = [
            {
                "event_id": 1,
                "entity_id": 7,
                "behavior": "drowning",
                "start_s": 10.0,
                "end_s": 12.0,
                "confidence": 0.91,
                "evidence": "head remained submerged",
                "snapshot": "camera_1/1.png",
            },
            {
                "event_id": 2,
                "entity_id": 8,
                "behavior": "loitering",
                "start_s": 42.0,
                "end_s": 44.0,
                "confidence": 0.74,
                "evidence": "person remained stationary",
                "snapshot": None,
            },
        ]
        self.index.index_events("camera_1", "pool_1.mp4", events, {"scenario": "pool"})
        self.index.index_events("camera_2", "pool_2.mp4", [events[0]], {"scenario": "pool"})

        results = self.index.query(
            camera_id="camera_1",
            person_id=7,
            behavior="drowning",
            since_s=9.0,
            until_s=13.0,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["camera_id"], "camera_1")
        self.assertEqual(results[0]["person_id"], 7)
        self.assertEqual(results[0]["evidence"], "head remained submerged")
        self.assertEqual(results[0]["snapshot"], "camera_1/1.png")

        distinct = self.index.query(camera_id="camera_2")
        self.assertEqual(len(distinct), 1)
        self.assertEqual(distinct[0]["camera_id"], "camera_2")

    def test_query_returns_events_from_all_cameras_when_camera_id_is_omitted(self):
        self.index.index_events("camera_1", "pool_1.mp4", [{"event_id": 1, "entity_id": 3, "behavior": "fall", "start_s": 1.0, "end_s": 2.0, "confidence": 0.8, "evidence": "person fell"}], {})
        self.index.index_events("camera_2", "pool_2.mp4", [{"event_id": 2, "entity_id": 4, "behavior": "fall", "start_s": 2.0, "end_s": 3.0, "confidence": 0.8, "evidence": "person fell"}], {})

        results = self.index.query()
        self.assertEqual({row["camera_id"] for row in results}, {"camera_1", "camera_2"})

    def test_clear_and_memory_persist_aliases(self):
        memory = ClarifyMemory(self.root / "clarifications.json")
        memory.store(camera_id="camera_2", reference="deep end", aliases={"deep_end": "Camera 2 represents the deep end"})
        self.assertEqual(memory.resolve("camera_2", "deep end"), "Camera 2 represents the deep end")
        self.assertEqual(memory.resolve("camera_1", "deep end"), None)

        persisted = json.loads((self.root / "clarifications.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["camera_2"]["deep_end"], "Camera 2 represents the deep end")


if __name__ == "__main__":
    unittest.main()
