import json
import tempfile
import unittest
from pathlib import Path

from ps_conversation import ConversationIndex, PersistentReferenceMemory


class ConversationIndexTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.folder = Path(self.root.name)
        (self.folder / "events.json").write_text(
            json.dumps(
                {
                    "video": "samples/external_pool.mp4",
                    "events": [
                        {
                            "event_id": 1,
                            "entity_id": 13,
                            "behavior_name": "aquatic_distress",
                            "start": 3.2,
                            "end": 4.1,
                            "confidence": 0.83,
                            "snapshot": "snapshots/e1.jpg",
                            "evidence": "Swimmer remained in a risky behaviour state.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.folder / "pool_dashboard.json").write_text(
            json.dumps(
                {
                    "video": "samples/external_pool.mp4",
                    "people": [
                        {"person_id": 13, "name": "Swimmer #13", "first_seen": 0.0, "last_seen": 5.0,
                         "final_state": "WATCH"},
                        {"person_id": 27, "name": "Swimmer #27", "first_seen": 1.0, "last_seen": 2.0,
                         "final_state": "NORMAL"},
                    ],
                    "risk": {"13": [[3.2, 0.45], [4.1, 0.45]], "27": [[1.0, 0.25]]},
                    "timeline": {"13": [{"t": 3.2, "note": "watching"}]},
                }
            ),
            encoding="utf-8",
        )
        (self.folder / "annotated.mp4").touch()
        (self.folder / "snapshots").mkdir()
        (self.folder / "snapshots" / "e1.jpg").touch()

    def tearDown(self):
        self.root.cleanup()

    def test_grounded_swimmer_question_uses_real_track_and_risk(self):
        index = ConversationIndex(self.folder)
        answer = index.answer("Which swimmer has the highest risk around 00:04?")
        self.assertIn("Swimmer #13", answer["answer"])
        self.assertEqual(answer["record"]["track_id"], 13)
        self.assertEqual(answer["record"]["risk"], 0.45)
        self.assertEqual(answer["record"]["source"], "samples/external_pool.mp4")
        self.assertEqual(answer["record"]["evidence_status"], "available")

    def test_time_window_returns_only_verified_observations(self):
        index = ConversationIndex(self.folder)
        answer = index.answer("What happened between 00:03 and 00:04?")
        self.assertIn("event_id 1", answer["answer"])
        self.assertEqual(answer["record"]["event_state"], "aquatic_distress")

    def test_unavailable_event_is_explicit_not_invented(self):
        index = ConversationIndex(self.folder)
        answer = index.answer("What happened around 00:10?")
        self.assertIn("No verified event", answer["answer"])
        self.assertEqual(answer["record"]["evidence_status"], "unavailable")

    def test_reference_memory_persists_user_defined_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentReferenceMemory(Path(directory) / "references.sqlite")
            memory.remember("deep_end", "CAM04", "user clarification")
            self.assertEqual(memory.lookup("deep_end")["camera_id"], "CAM04")
            restarted = PersistentReferenceMemory(Path(directory) / "references.sqlite")
            self.assertEqual(restarted.lookup("deep_end")["camera_id"], "CAM04")


if __name__ == "__main__":
    unittest.main()
