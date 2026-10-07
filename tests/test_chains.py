"""Tests for chains.py (event severity + incident chains) and for the report built from them.

Pure Python, no video, YOLO or OpenCV. Events are written by hand in the DESIGN 3.4 / 4.10 shape,
so each test reads like the story it checks ("loitering, then 1 s later inside the zone").
Thresholds are pinned in CFG so tuning config.yaml cannot break the tests.
"""
from __future__ import annotations

import json

import chains
import report
from chains import build_chains
from utils import DEFAULT_CONFIG, deep_merge, fmt_time

CFG = deep_merge(DEFAULT_CONFIG, {"chains": {"enabled": True, "max_gap_s": 15.0},
                                  "behaviors": {"baseline": {"z_threshold": 3.0},
                                                "crowding": {"min_count": 5}}})


def make_event(event_id, entity_id, behavior, start_s, end_s, confidence=0.8, metrics=None, zone=None,
               baseline=None, **extra):
    """One event dict with the keys chains.py and report.py read."""
    ev = {"event_id": event_id, "entity_id": entity_id, "behavior": behavior,
          "start_s": float(start_s), "end_s": float(end_s), "start": fmt_time(start_s), "end": fmt_time(end_s),
          "duration_s": round(end_s - start_s, 2), "confidence": confidence,
          "confidence_parts": {"margin": 0.7, "duration": 0.8, "detection": 0.8},
          "evidence": f"{behavior} evidence sentence.", "metrics": metrics or {}, "zone": zone,
          "baseline": baseline, "snapshot_time_s": float(start_s), "snapshot": None, "speed_plot": None,
          "verified": None, "pose": None, "severity": None,
          "entities": [entity_id] if entity_id is not None else [], "other_entity": None,
          "entity_name": f"Person #{entity_id}" if entity_id is not None else None,
          "behavior_name": None}
    ev.update(extra)
    return ev


def loiter_then_zone(gap=1.0):
    """Person #3 loiters 12-30 s, then is inside the zone `gap` s later for 5 s."""
    return [make_event(1, 3, "loitering", 12, 30),
            make_event(2, 3, "zone_intrusion", 30 + gap, 35 + gap, zone="restricted",
                       metrics={"zone": "restricted"})]


# --------------------------------------------------------------------------- chains

def test_loitering_then_zone_is_one_high_chain():
    events = loiter_then_zone(gap=1.0)
    found = build_chains(events, CFG)
    assert len(found) == 1
    c = found[0]
    assert c["chain_id"] == 1 and c["entity_id"] == 3 and c["event_ids"] == [1, 2]
    assert c["pattern"] == "loitering -> zone_intrusion"
    assert c["title"].startswith("Waited nearby")
    assert c["severity"] == "high"
    assert (c["start_s"], c["end_s"], c["start"], c["end"]) == (12.0, 36.0, "00:12", "00:36")
    # story names the person, both steps, the gap and the stay (DESIGN 4.1 example)
    assert c["story"] == ("Person #3 loitered 00:12-00:30, then entered zone 'restricted' at 00:31 "
                          "(1 s later) and stayed 5 s.")
    # every event of the chain is raised to the chain severity
    assert [e["severity"] for e in events] == ["high", "high"]


def test_running_then_near_miss_title():
    events = [make_event(1, 4, "running", 10, 14),
              make_event(2, 4, "near_miss", 15, 16, entities=[4, 9], other_entity=9,
                         metrics={"other_class": 2, "peak_time_s": 15.5, "contact": False})]
    (c,) = build_chains(events, CFG)
    assert c["title"] == "Ran and nearly collided"
    assert c["pattern"] == "running -> near_miss"
    assert c["severity"] == "high"
    assert "nearly collided with" in c["story"] and "00:15" in c["story"]


def test_other_known_patterns_have_their_titles():
    cases = {("running", "zone_intrusion"): "Rushed into the restricted zone",
             ("zone_intrusion", "running"): "Ran away after entering the restricted zone",
             ("loitering", "running"): "Waited, then suddenly ran",
             ("near_miss", "running"): "Near miss, then fled",
             ("zone_intrusion", "near_miss"): "Entered the danger zone and nearly got hit"}
    for (a, b), title in cases.items():
        events = [make_event(1, 1, a, 0, 4), make_event(2, 1, b, 5, 8)]
        (c,) = build_chains(events, CFG)
        assert c["title"] == title, (a, b)


def test_unknown_pattern_gets_the_default_title_and_medium_severity():
    events = [make_event(1, 1, "running", 0, 4), make_event(2, 1, "running", 8, 12)]
    (c,) = build_chains(events, CFG)
    assert c["title"] == "Repeated running"  # same behaviour twice is named, not "suspicious"
    assert c["severity"] == "medium"        # no zone intrusion in it

    mixed = [make_event(1, 2, "fall", 0, 4), make_event(2, 2, "loitering", 6, 18)]
    (c,) = build_chains(mixed, CFG)
    assert c["title"] == "Repeated suspicious activity"


def test_overlapping_loitering_and_zone_is_lingering_inside_the_zone():
    # Real footage: a person stands still inside the zone, so both events start together.
    events = [make_event(1, 1, "zone_intrusion", 6.0, 18.4), make_event(2, 1, "loitering", 6.3, 19.0)]
    (c,) = build_chains(events, CFG)
    assert c["title"] == "Lingered inside the restricted zone"
    assert c["severity"] == "high"


def test_most_alarming_pattern_names_a_longer_chain():
    events = [make_event(1, 1, "loitering", 0, 12), make_event(2, 1, "running", 14, 18),
              make_event(3, 1, "zone_intrusion", 19, 24)]
    (c,) = build_chains(events, CFG)
    assert c["pattern"] == "loitering -> running -> zone_intrusion"
    assert c["title"] == "Rushed into the restricted zone"
    assert c["event_ids"] == [1, 2, 3]
    assert c["severity"] == "high"


def test_events_more_than_15_seconds_apart_are_not_chained():
    far = [make_event(1, 3, "loitering", 0, 10), make_event(2, 3, "zone_intrusion", 25.5, 28, zone="z",
                                                          metrics={"zone": "z"})]
    assert build_chains(far, CFG) == []
    edge = [make_event(1, 3, "loitering", 0, 10), make_event(2, 3, "zone_intrusion", 25.0, 28, zone="z",
                                                           metrics={"zone": "z"})]
    assert len(build_chains(edge, CFG)) == 1            # exactly 15 s still counts


def test_a_long_pause_splits_one_entity_into_two_chains():
    events = [make_event(1, 5, "loitering", 0, 10), make_event(2, 5, "running", 12, 15),
              make_event(3, 5, "loitering", 100, 112), make_event(4, 5, "zone_intrusion", 113, 118, zone="z",
                                                                  metrics={"zone": "z"})]
    found = build_chains(events, CFG)
    assert [c["event_ids"] for c in found] == [[1, 2], [3, 4]]
    assert [c["chain_id"] for c in found] == [1, 2]


def test_different_entities_are_never_chained_together():
    events = [make_event(1, 1, "loitering", 0, 12), make_event(2, 2, "zone_intrusion", 13, 18, zone="z",
                                                             metrics={"zone": "z"})]
    assert build_chains(events, CFG) == []


def test_crowding_with_no_entity_is_never_chained():
    crowd = {"max_count": 6, "min_count": 5, "mean_count": 5.5}
    events = [make_event(1, None, "crowding", 0, 10, metrics=crowd, zone="hall", entities=[1, 2, 3, 4, 5, 6]),
              make_event(2, None, "crowding", 12, 20, metrics=crowd, zone="hall", entities=[1, 2, 3, 4, 5, 6]),
              make_event(3, 2, "running", 5, 8)]
    assert build_chains(events, CFG) == []
    assert all(e["severity"] for e in events)           # but they are still rated


def test_overlapping_events_of_one_person_chain_and_say_so():
    events = [make_event(1, 2, "loitering", 0, 60), make_event(2, 2, "zone_intrusion", 10, 15, zone="z",
                                                             metrics={"zone": "z"})]
    (c,) = build_chains(events, CFG)
    assert "(while doing so)" in c["story"]
    assert c["end_s"] == 60.0


def test_chain_gap_counts_from_the_latest_end_so_far():
    # loitering keeps going until 60, so a run at 70 is only 10 s after it, not 55 s after the zone event
    events = [make_event(1, 2, "loitering", 0, 60), make_event(2, 2, "zone_intrusion", 10, 15, zone="z",
                                                             metrics={"zone": "z"}),
              make_event(3, 2, "running", 70, 74)]
    (c,) = build_chains(events, CFG)
    assert c["event_ids"] == [1, 2, 3]


def test_chains_can_be_switched_off_but_severity_is_still_set():
    cfg = deep_merge(CFG, {"chains": {"enabled": False}})
    events = loiter_then_zone()
    assert build_chains(events, cfg) == []
    assert [e["severity"] for e in events] == ["low", "medium"]


def test_story_uses_the_scenario_words():
    cfg = deep_merge(CFG, {"scenario": {"entity_word": "Animal",
                                        "behavior_names": {"zone_intrusion": "Left the pen",
                                                           "loitering": "Not moving"}}})
    events = [make_event(1, 8, "loitering", 0, 12, entity_name="Animal #8"),
              make_event(2, 8, "zone_intrusion", 14, 20, zone="pen", metrics={"zone": "pen"},
                         entity_name="Animal #8")]
    (c,) = build_chains(events, cfg)
    assert c["story"].startswith("Animal #8 not moving 00:00-00:12")
    assert "left the pen 00:14-00:20" in c["story"]
    assert "Person" not in c["story"]


# --------------------------------------------------------------------------- severity

def sev(behavior, **kw):
    """Severity of one lone event of the given behaviour."""
    ev = make_event(1, 1, behavior, 0, 10, **kw)
    build_chains([ev], CFG)
    return ev["severity"]


def test_severity_defaults_per_behaviour():
    assert sev("loitering") == "low"
    assert sev("zone_intrusion") == "medium"
    assert sev("running", confidence=0.8) == "medium"
    assert sev("running", confidence=0.6) == "medium"
    assert sev("running", confidence=0.59) == "low"
    assert sev("fall", metrics={"fell": True}) == "high"
    assert sev("fall", confidence=0.1) == "high"
    assert sev("near_miss", confidence=0.1) == "high"


def test_crowding_severity_is_high_at_twice_the_limit():
    assert sev("crowding", metrics={"max_count": 6, "min_count": 5}) == "medium"
    assert sev("crowding", metrics={"max_count": 9, "min_count": 5}) == "medium"
    assert sev("crowding", metrics={"max_count": 10, "min_count": 5}) == "high"
    assert sev("crowding", metrics={"max_count": 10}) == "high"      # limit falls back to the config


def test_an_unusual_baseline_raises_severity_one_level():
    unusual = {"anomaly_score": 4.2, "note": "moved 2.6x faster than typical"}
    normal = {"anomaly_score": 0.5, "note": "typical"}
    assert sev("loitering", baseline=unusual) == "medium"
    assert sev("running", confidence=0.9, baseline=unusual) == "high"
    assert sev("zone_intrusion", baseline=unusual) == "high"
    assert sev("fall", baseline=unusual) == "high"                   # already at the top
    assert sev("loitering", baseline=normal) == "low"
    assert sev("loitering", baseline={"anomaly_score": 0.0, "unusual": True}) == "medium"


def test_chain_severity_is_never_below_its_most_serious_event():
    events = [make_event(1, 1, "running", 0, 4, baseline={"anomaly_score": 5.0, "note": "x"}),
              make_event(2, 1, "running", 6, 9)]
    (c,) = build_chains(events, CFG)
    assert events[0]["severity"] == "high"
    assert c["severity"] == "high" and events[1]["severity"] == "high"


# --------------------------------------------------------------------------- report

CROWD = {"duration_s": 12.0, "zone": "hall", "max_count": 6, "mean_count": 5.4, "min_count": 5,
         "min_duration_s": 5.0, "peak_time_s": 50.0, "mean_det_conf": 0.8}
FALL = {"duration_s": 8.0, "max_aspect": 1.9, "upright_aspect_before": 0.4, "fell": True, "transition_s": 1.2,
        "min_down_s": 2.0, "peak_time_s": 61.0, "mean_det_conf": 0.8}
NEAR = {"min_distance_bh": 0.12, "peak_time_s": 15.5, "max_closing_speed_bh_s": 2.4, "min_ttc_s": None,
        "rel_speed_at_closest_bh_s": 2.1, "other_id": 9, "other_class": 2, "approach_s": 1.4, "contact": True,
        "duration_s": 1.0, "mean_det_conf": 0.8, "min_duration_s": 0.0}


def full_final(contact=True):
    """A final dict with all six behaviours, a chain, a highlight reel entry and near-miss notes."""
    near = dict(NEAR, contact=contact)
    events = [
        make_event(1, 3, "loitering", 12, 30, metrics={"duration_s": 18.0, "max_radius_bh": 0.3,
                                                      "radius_threshold_bh": 0.5, "min_duration_s": 10.0,
                                                      "mean_speed_bh_s": 0.1, "peak_time_s": 21.0}),
        make_event(2, 3, "zone_intrusion", 31, 36, zone="restricted",
                   metrics={"duration_s": 5.0, "zone": "restricted", "max_depth_bh": 0.6, "min_duration_s": 1.0,
                            "peak_time_s": 33.0}),
        make_event(3, 4, "running", 40, 46, metrics={"duration_s": 6.0, "peak_speed_bh_s": 3.4,
                                                   "mean_speed_bh_s": 2.9, "start_threshold_bh_s": 1.8,
                                                   "end_threshold_bh_s": 1.4, "min_duration_s": 1.0,
                                                   "peak_time_s": 43.0}),
        make_event(4, None, "crowding", 45, 57, metrics=CROWD, zone="hall", entities=[1, 2, 3, 4, 5, 6],
                   entity_name="Group of 6"),
        make_event(5, 5, "fall", 58, 66, metrics=FALL),
        make_event(6, 7, "near_miss", 70, 71, metrics=near, entities=[7, 9], other_entity=9),
    ]
    incidents = build_chains(events, CFG)
    return {"events": events, "incidents": incidents,
            "near_misses": [{"track_id": 6, "behavior": "loitering", "value": 8.2, "threshold": 10.0, "unit": "s",
                             "note": "Person #6 stood still for 8.2 s (rule: 10 s). Not flagged."}],
            "unusual_tracks": [{"entity_id": 4, "anomaly_score": 4.2, "note": "Moved 2.6x faster", "has_event": True}],
            "baseline": {"enabled": True, "active": True, "n_tracks": 9, "median_speed_bh_s": 0.8,
                         "median_dwell_s": 1.5},
            "highlight_reel": "highlights.mp4"}


META = {"video": "samples/yard.mp4", "duration_s": 75.0, "fps": 25.0, "stride": 2, "device": "cpu",
        "model": "yolo11n.pt", "n_people": 9, "zones": ["restricted"], "privacy": True, "processing_fps": 12.34,
        "scenario": {"name": "campus", "title": "Campus corridor (restricted zone)"}}
TRACKS = [{"id": 3, "first_seen": 10.0, "last_seen": 40.0, "duration_s": 30.0, "median_speed_bh_s": 0.4,
           "p90_speed_bh_s": 1.0, "max_dwell_s": 18.0}]


def test_write_outputs_with_all_behaviours_incidents_reel_and_privacy(tmp_path):
    final = full_final()
    assert {e["behavior"] for e in final["events"]} == {"loitering", "zone_intrusion", "running", "fall",
                                                        "crowding", "near_miss"}
    result = report.write_outputs(final, dict(META, config=CFG), TRACKS, tmp_path)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    incident = final["incidents"][0]

    assert incident["title"] in page                                   # incident title
    assert "Incident 1" in page and 'href="#event-1"' in page and 'href="#event-2"' in page
    assert 'id="event-2"' in page and 'id="incident-1"' in page
    assert "POSSIBLE CONTACT" in page                                  # near miss with contact
    assert "Group of 6" in page                                        # crowding
    assert "HIGH SEVERITY" in page and "MEDIUM SEVERITY" in page
    assert "Person #7 and Person #9" in page or "Person #7 and Car #9" in page
    assert "highlights.mp4" in page and "Watch the highlight reel" in page   # reel link
    assert "Privacy mode: faces blurred in all outputs" in page
    assert "Processed at 12.3 frames/s on cpu" in page
    assert "Campus corridor (restricted zone)" in page
    assert "Went from upright to lying" in page                        # fall metrics
    assert "src=\"http" not in page and "href=\"http" not in page      # self-contained: no external requests
    for colour in ("#ffa500", "#e60000", "#c800e6", "#ff4500", "#00a0a0", "#dc143c"):   # 6 behaviour colours
        assert colour in page

    doc = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    for key in ("video", "meta", "events", "near_misses", "unusual_tracks", "baseline", "tracks"):
        assert key in doc                                              # old keys survive
    assert doc["incidents"][0]["title"] == incident["title"]
    assert doc["highlight_reel"] == "highlights.mp4"
    assert doc["events"][0]["severity"] in ("low", "medium", "high")
    assert result["report_html"].endswith("report.html")


def test_no_possible_contact_badge_without_contact(tmp_path):
    report.write_outputs(full_final(contact=False), dict(META, config=CFG), TRACKS, tmp_path)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "POSSIBLE CONTACT" not in page


def test_report_without_reel_or_privacy_has_neither_note(tmp_path):
    final = full_final()
    final.pop("highlight_reel")
    meta = dict(META, config=CFG, privacy=False, processing_fps=None)
    report.write_outputs(final, meta, TRACKS, tmp_path)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Watch the highlight reel" not in page
    assert "Privacy mode" not in page
    assert "Processed at" not in page


def test_report_builds_the_incidents_itself_when_final_has_none(tmp_path):
    final = {"events": loiter_then_zone(), "near_misses": [], "unusual_tracks": [], "baseline": {}}
    report.write_outputs(final, dict(META, config=CFG), TRACKS, tmp_path)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Waited nearby, then entered the restricted zone" in page
    assert "incidents" not in final                                    # the caller's dict is left alone


def test_report_with_no_events_still_writes(tmp_path):
    final = {"events": [], "incidents": [], "near_misses": [], "unusual_tracks": [], "baseline": {}}
    result = report.write_outputs(final, dict(META, config=CFG, privacy=False), [], tmp_path)
    assert "No incidents" in result["summary"]
    assert (tmp_path / "report.html").exists()


def test_summary_lists_incidents_first_then_loose_events_then_notes():
    final = full_final()
    text = report.make_summary(final, dict(META, config=CFG), CFG)
    lines = text.splitlines()
    assert lines[0].startswith("Video yard.mp4 (01:15), 9 people seen. 5 incidents:")
    bullets = [ln for ln in lines if ln.startswith("- ")]
    assert "Waited nearby, then entered the restricted zone" in bullets[0]     # the chain comes first
    assert bullets[0].endswith("[HIGH]")
    assert "Person #5" in bullets[1] and "fell and stayed down" in bullets[1]      # then loose events, worst first
    assert "POSSIBLE CONTACT" in bullets[2] and bullets[2].endswith("[HIGH]")
    assert "...and 2 more" in bullets[3]
    assert any("almost-flagged" in ln for ln in lines)
    assert text.rstrip().endswith("Privacy mode: faces blurred in all outputs.")
    assert len(lines) <= 7


def test_summary_without_events_says_all_normal():
    final = {"events": [], "incidents": [], "near_misses": [], "unusual_tracks": [], "baseline": {}}
    text = report.make_summary(final, dict(META, privacy=False), CFG)
    assert "No incidents. 9 people seen" in text


def test_summary_and_report_use_the_scenario_wording(tmp_path):
    cfg = deep_merge(CFG, {"scenario": {"name": "livestock", "title": "Livestock monitoring", "entity_word": "Animal",
                                        "unit_word": "body-lengths",
                                        "behavior_names": {"zone_intrusion": "Left the pen",
                                                           "loitering": "Not moving"}}})
    events = [make_event(1, 8, "loitering", 0, 12, entity_name="Animal #8"),
              make_event(2, 8, "zone_intrusion", 14, 20, zone="pen", metrics={"zone": "pen", "max_depth_bh": 0.4,
                                                                            "duration_s": 6.0},
                         entity_name="Animal #8")]
    final = {"events": events, "incidents": build_chains(events, cfg), "near_misses": [], "unusual_tracks": [],
             "baseline": {}}
    meta = {"video": "pen.mp4", "duration_s": 30.0, "n_people": 12, "config": cfg,
            "scenario": {"name": "livestock", "title": "Livestock monitoring"}}
    result = report.write_outputs(final, meta, [], tmp_path)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "12 animals seen" in result["summary"]
    assert "[Livestock monitoring]" in result["summary"]
    assert "Person" not in result["summary"] and "Person" not in page
    assert "Left the pen" in page and "LEFT THE PEN" in page
    assert "Animals seen" in page and "body-lengths" in page


def test_chains_module_does_not_import_the_vision_stack():
    """chains.py is pure Python (DESIGN 4.1); report.py reads images but never needs OpenCV."""
    for mod in (chains, report):
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import cv2", "import torch", "import ultralytics", "from ultralytics", "from cv2"):
            assert banned not in src, f"{mod.__name__} must not contain '{banned}'"
