"""Aquatic distress pipeline: stage 2 (behaviour), stage 3 (temporal reasoning), stage files, announcements."""
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

import announce
import distress
import features
import pool
import pool_behaviour
import utils

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import synthetic_pool  # noqa: E402


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """The synthetic pool story: detections + keypoints + zones, run through stages 2 and 3."""
    out = tmp_path_factory.mktemp("pool")
    synthetic_pool.make_files("synthetic_pool.mp4", out)
    cfg = utils.load_config(scenario="pool")
    res = utils.read_json(out / "detections.json")
    tracks = features.build_tracks(res, cfg)
    zones = utils.load_zones(HERE / "synthetic_pool_zones.json", res["proc_size"])
    kps = utils.read_json(out / "keypoints.json")["keypoints"]
    samples = pool_behaviour.behaviour_samples(tracks, cfg, zones, kps)
    last_seen = {tid: float(tr.t[-1]) for tid, tr in tracks.items()}
    stage3 = distress.reason(samples, cfg, last_seen=last_seen, video_end=res["duration_s"])
    return {"cfg": cfg, "res": res, "tracks": tracks, "zones": zones, "kps": kps, "samples": samples,
            "stage3": stage3, "out": out}


def _max_risk(stage3, pid):
    return max((r for _, r in stage3["risk"].get(pid, [])), default=0.0)


def test_only_the_child_gets_a_distress_alert_and_it_is_early(scene):
    events = scene["stage3"]["events"]
    distress_ev = [e for e in events if e["event"] == "high_risk_aquatic_distress"]
    assert [e["person_id"] for e in distress_ev] == [3]
    e = distress_ev[0]
    assert 15.5 <= e["start_time"] <= 19.0              # true distress starts at 16 s
    assert 20.0 <= e["alert_time"] <= 24.0              # alert hold_s (5 s) later, well before the head goes under at 28 s
    assert e["risk_score"] >= 0.85 and e["location"] == "Deep End"
    for key in ("vertical_posture", "low_displacement", "repeated_arm_motion", "transition_from_swimming"):
        assert key in e["evidence"]


def test_child_then_goes_under(scene):
    sub = [e for e in scene["stage3"]["events"] if e["event"] == "possible_submersion"]
    assert len(sub) == 1 and sub[0]["person_id"] == 3 and 27.5 <= sub[0]["start_time"] <= 31.0


def test_look_alikes_stay_calm(scene):
    st3 = scene["stage3"]
    assert _max_risk(st3, 1) < 0.55        # lap swimmer: big arm strokes but horizontal and moving
    assert _max_risk(st3, 2) < 0.3         # resting at the wall
    assert _max_risk(st3, 4) < 0.55        # diver: under water 2.5 s from normal swimming
    assert _max_risk(st3, 5) == 0.0        # lifeguard on the deck: outside the water zone
    assert all(e["person_id"] == 3 for e in st3["events"])


def test_timeline_tells_the_story_in_order(scene):
    notes = [x["note"] for x in scene["stage3"]["timeline"][3]]
    order = ["Normal swimming", "Movement slowing", "Vertical posture", "Repeated arm movement",
             "HIGH-RISK AQUATIC DISTRESS"]
    positions = [notes.index(n) for n in order]
    assert positions == sorted(positions)
    assert scene["stage3"]["timeline"][2] == []          # the resting teenager has nothing to report


def test_stage2_contract(scene):
    s = next(x for x in scene["samples"] if x["person_id"] == 3 and abs(x["timestamp"] - 20.0) < 0.3)
    for key in ("person_id", "timestamp", "posture", "movement", "arm_motion", "displacement"):
        assert key in s
    assert s["posture"] == "vertical" and s["movement"] == "low" and s["arm_motion"] == "repeated"
    assert s["displacement"] < 0.15 and s["head"] == "unstable" and s["arm_source"] == "pose"
    wall = next(x for x in scene["samples"] if x["person_id"] == 2 and abs(x["timestamp"] - 20.0) < 0.3)
    assert wall["near_wall"] and wall["arm_motion"] == "calm"


def test_without_pose_it_warns_but_does_not_cry_wolf(scene):
    samples = pool_behaviour.behaviour_samples(scene["tracks"], scene["cfg"], scene["zones"], None)
    st3 = distress.reason(samples, scene["cfg"])
    assert _max_risk(st3, 3) >= 0.55                     # the child still reaches WARNING from box signals
    assert all(e["person_id"] == 3 for e in st3["events"])


def test_risk_rules():
    cfg = utils.load_config(scenario="pool")
    p = cfg["pool"]
    base = {"posture": "vertical", "movement": "low", "arm_motion": "repeated", "displacement": 0.02,
            "head": "unstable", "near_wall": False, "in_water": True}
    full = distress.sample_risk(distress.sample_signals(base, p), True, p)
    assert full == pytest.approx(1.0)
    treading = dict(base, arm_motion="calm", head="stable")
    assert distress.sample_risk(distress.sample_signals(treading, p), True, p) <= p["treading_cap"]
    resting = dict(base, arm_motion="calm", near_wall=True)
    assert distress.sample_risk(distress.sample_signals(resting, p), False, p) < p["watch_risk"]
    swimming = dict(base, posture="horizontal", movement="high", displacement=0.8, head="stable")
    assert distress.sample_risk(distress.sample_signals(swimming, p), True, p) == 0.0   # strokes do not count
    deck = dict(base, in_water=False)
    assert distress.sample_risk(distress.sample_signals(deck, p), True, p) == 0.0


def test_stage_files_and_cli_modules(scene, tmp_path):
    pool.write_stage_files(tmp_path, scene["res"], scene["samples"], scene["stage3"], scene["cfg"])
    first = [json.loads(line) for line in open(tmp_path / "stage1_tracking.jsonl", encoding="utf-8")]
    assert "meta" in first[0] and {"person_id", "timestamp", "bbox"} <= set(first[1])
    out2 = tmp_path / "s2.jsonl"
    r = subprocess.run([sys.executable, "-B", "pool_behaviour.py", "--tracks", str(tmp_path / "stage1_tracking.jsonl"),
                        "--zones", str(HERE / "synthetic_pool_zones.json"), "--out", str(out2)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out3 = tmp_path / "s3.json"
    r = subprocess.run([sys.executable, "-B", "distress.py", "--behaviour", str(out2), "--out", str(out3)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "events" in utils.read_json(out3)


def test_ps07_events_and_announcement(scene, tmp_path):
    events = [pool.to_ps07_event(e, scene["tracks"], scene["cfg"]) for e in scene["stage3"]["events"]]
    for n, ev in enumerate(events, 1):
        ev["event_id"] = n
    assert {e["behavior"] for e in events} == {"aquatic_distress", "submersion"}
    final = {"events": events}
    text = announce.message_for(events[0], scene["cfg"])
    assert "Swimmer number 3" in text and "Deep End" in text and "percent" in text
    recs = announce.prepare(final, scene["cfg"], tmp_path, speech=False)        # tone only: fast, no voice
    assert len(recs) == 2 and (tmp_path / recs[0]["audio"]).exists()
    with wave.open(str(tmp_path / recs[0]["audio"])) as w:
        assert w.getframerate() == announce.RATE and w.getnframes() > announce.RATE   # at least a second of alarm
    assert events[0]["announcement"]["text"] == text


def test_pool_preset_turns_off_land_rules():
    cfg = utils.load_config(scenario="pool")
    assert cfg["pool"]["enabled"] and cfg["behaviors"]["aquatic_distress"]["enabled"]
    for key in ("loitering", "zone_intrusion", "running", "fall", "crowding", "near_miss"):
        assert not cfg["behaviors"][key]["enabled"]
