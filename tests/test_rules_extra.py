"""Tests for the rules added in DESIGN section 4: fall, crowding and near miss (plus names and units).

Like test_logic.py these use SYNTHETIC tracker output (no video, YOLO or OpenCV). Every object is a
function t -> box, so a test reads like the story it checks: "a person stands 5 s, then the box turns
sideways within 0.5 s and stays that way for 4 s". Thresholds are pinned here (not read from
config.yaml) so tuning config.yaml cannot break the tests.

Frame: 1280 x 1080 processing pixels, 30 fps, every 2nd frame processed, upright person box 60 x 150 px
(so one body-height is 150 px).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from behaviors import compute_metrics, detect_behaviors, down_mask
from events import build_events
from features import build_tracks
from utils import DEFAULT_CONFIG, behavior_name, deep_merge, entity_name, unit_name

FPS, STRIDE, BH = 30.0, 2, 150.0
PROC_SIZE = [1280, 1080]
ZONE = {"name": "restricted", "points": [(100.0, 300.0), (1100.0, 300.0), (1100.0, 900.0), (100.0, 900.0)]}

TEST_CFG = deep_merge(DEFAULT_CONFIG, {
    "features": {"smoothing_s": 0.5, "speed_window_s": 0.5, "max_gap_s": 1.0,
                 "min_track_s": 1.0, "min_box_h_px": 20, "scale": "max_side"},
    "behaviors": {
        "loitering": {"enabled": True, "radius_bh": 0.5, "min_duration_s": 10.0},
        "zone_intrusion": {"enabled": True, "min_duration_s": 1.0},
        "running": {"enabled": True, "start_speed_bh_s": 1.8, "end_speed_bh_s": 1.4, "min_duration_s": 1.0},
        "baseline": {"enabled": True, "min_tracks": 4, "z_threshold": 3.0},
        "fall": {"enabled": True, "down_min_aspect": 1.2, "upright_max_aspect": 0.8,
                 "transition_s": 2.0, "min_down_s": 2.0},
        "crowding": {"enabled": True, "min_count": 5, "min_duration_s": 5.0, "whole_frame": False},
        "near_miss": {"enabled": True, "vulnerable_classes": [0], "other_classes": [0, 1, 2, 3, 5, 7],
                      "near_distance_bh": 0.6, "min_rel_speed_bh_s": 0.8, "min_closing_speed_bh_s": 0.3,
                      "ttc_s": 1.0, "contact_bh": 0.15, "max_scale_diff": 0.35, "min_duration_s": 0.0},
    },
    "events": {"merge_gap_s": 1.5, "near_miss_ratio": 0.7,
               "confidence_weights": {"margin": 0.4, "duration": 0.3, "detection": 0.3}},
})
CAR_CFG = deep_merge(TEST_CFG, {"model": {"classes": [0, 2]}})      # a scene with people AND cars


# --------------------------------------------------------------------------- synthetic data helpers

def box_at(x, y, w, h):
    """Box (x1, y1, x2, y2) whose bottom-centre (the foot point) is (x, y)."""
    return (x - w / 2.0, y - h, x + w / 2.0, y)


def rows_for(tid, t_end, box_fn, t0=0.0, cls=None, conf=0.85):
    """Detection rows [frame, t, id, x1, y1, x2, y2, conf(, cls)] for t0..t_end; box_fn(t) -> box."""
    rows = []
    for frame in range(int(round(t0 * FPS)), int(round(t_end * FPS)) + 1):
        if frame % STRIDE:
            continue
        t = frame / FPS
        row = [frame, t, tid, *box_fn(t), conf]
        if cls is not None:
            row.append(cls)                                  # 9-element row = with a class id
        rows.append(row)
    return rows


def standing(x, y, w=60.0, h=BH):
    """box_fn: stands still at foot point (x, y)."""
    return lambda t: box_at(x, y, w, h)


def walking(x0, y0, vx_px_s, vy_px_s=0.0, w=60.0, h=BH, t_start=0.0):
    """box_fn: moves in a straight line at constant velocity (px/s) from (x0, y0) at time t_start."""
    return lambda t: box_at(x0 + vx_px_s * (t - t_start), y0 + vy_px_s * (t - t_start), w, h)


def falling(x, y, t_fall=5.0, ramp=0.5, t_up=None, slide_px_s=0.0):
    """box_fn: upright 50 x 150, then within `ramp` s lying 150 x 50 (bottom-centre fixed).

    t_up: if given, the person is upright again from that time on (a brief bend, not a fall).
    slide_px_s: the lying box slides sideways at this speed for 1.5 s after the fall.
    """
    def fn(t):
        k = min(1.0, max(0.0, (t - t_fall) / ramp))
        if t_up is not None and t >= t_up:
            k = 0.0
        slid = slide_px_s * min(max(t - t_fall, 0.0), 1.5)
        return box_at(x + slid, y, 50.0 + 100.0 * k, 150.0 - 100.0 * k)
    return fn


def make_result(*row_groups, duration_s=None):
    """Wrap detection rows into a TrackerResult dict (DESIGN 3.1)."""
    rows = [r for group in row_groups for r in group]
    if duration_s is None:
        duration_s = max((r[1] for r in rows), default=0.0)
    return {"video": "synthetic.mp4", "fps": FPS, "stride": STRIDE, "orig_size": PROC_SIZE,
            "proc_size": PROC_SIZE, "n_frames_read": int(duration_s * FPS), "duration_s": duration_s,
            "frames_processed": len(rows), "device": "cpu", "model": "yolo11n.pt", "detections": rows}


def run_logic(result, zones=(), cfg=None):
    """features -> behaviors -> events, exactly like run.py wires them."""
    cfg = cfg or TEST_CFG
    tracks = build_tracks(result, cfg)
    raw = detect_behaviors(tracks, list(zones), cfg)
    final = build_events(raw, tracks, list(zones), cfg)
    return tracks, raw, final


def of_kind(final, behavior):
    return [e for e in final["events"] if e["behavior"] == behavior]


# --------------------------------------------------------------------------- features: scale, class, aspect, velocity

def test_track_class_comes_from_the_ninth_column_and_defaults_to_person():
    car = rows_for(7, 5, walking(100, 500, 60, w=160, h=100), cls=2)
    person_old_format = rows_for(1, 5, walking(100, 700, 60))                  # 8-element rows
    tracks = build_tracks(make_result(car, person_old_format), TEST_CFG)       # mixed row lengths
    assert tracks[7].cls == 2 and tracks[1].cls == 0
    assert isinstance(tracks[7].cls, int)


def test_body_scale_is_the_longest_side_so_a_lying_person_keeps_their_scale():
    lying = build_tracks(make_result(rows_for(1, 6, standing(400, 500, w=150, h=50))), TEST_CFG)[1]
    assert lying.height.mean() == pytest.approx(150.0, abs=1.0)                # max(150, 50), not 50
    assert lying.aspect.mean() == pytest.approx(3.0, abs=0.05)
    box_h_cfg = deep_merge(TEST_CFG, {"features": {"scale": "box_h"}})
    legacy = build_tracks(make_result(rows_for(1, 6, standing(400, 500, w=150, h=50))), box_h_cfg)[1]
    assert legacy.height.mean() == pytest.approx(50.0, abs=1.0)                # old behaviour: box height
    car = build_tracks(make_result(rows_for(2, 6, walking(100, 500, 240, w=160, h=100), cls=2)), TEST_CFG)[2]
    assert car.height.mean() == pytest.approx(160.0, abs=1.0)
    assert np.nanmedian(car.speed) == pytest.approx(240.0 / 160.0, abs=0.1)    # 1.5 body-lengths/s


def test_track_velocity_in_pixels_per_second():
    tr = build_tracks(make_result(rows_for(1, 8, walking(100, 500, 120.0, -30.0))), TEST_CFG)[1]
    assert tr.vel.shape == (len(tr.t), 2)
    assert np.isnan(tr.vel[0]).all()                                           # no history at the very start
    mid = tr.vel[len(tr.t) // 2]
    assert mid[0] == pytest.approx(120.0, abs=3.0) and mid[1] == pytest.approx(-30.0, abs=3.0)


# --------------------------------------------------------------------------- (a) fall

def test_person_falls_and_stays_down_is_one_fall_event():
    rows = rows_for(1, 9.5, falling(400, 500, t_fall=5.0, ramp=0.5))           # lying from 5.5 to 9.5 s
    _, _, final = run_logic(make_result(rows))
    falls = of_kind(final, "fall")
    assert len(falls) == 1
    ev = falls[0]
    m = ev["metrics"]
    assert m["fell"] is True and m["upright_aspect_before"] <= 0.8 and m["max_aspect"] >= 2.5
    assert 5.0 <= ev["start_s"] <= 6.2 and ev["end_s"] == pytest.approx(9.5, abs=0.2)
    assert m["duration_s"] >= 3.0
    assert ev["snapshot_time_s"] == pytest.approx(ev["start_s"] + 0.5, abs=0.01)
    assert ev["entity_id"] == 1 and ev["entities"] == [1] and ev["other_entity"] is None
    assert ev["evidence"].startswith("Box went from upright (w/h 0.") and "stayed down" in ev["evidence"]
    assert "within 2.0 s" in ev["evidence"]
    assert ev["zone"] is None and ev["behavior_name"] == "Fall / person down"
    # a person who dropped to the floor is a stronger case than one who was simply seen lying
    assert ev["confidence_parts"]["margin"] >= 0.8


def test_person_first_seen_lying_is_a_fall_event_but_not_fell():
    _, _, final = run_logic(make_result(rows_for(1, 6, standing(400, 500, w=150, h=50))))
    ev = of_kind(final, "fall")[0]
    assert ev["metrics"]["fell"] is False and ev["metrics"]["upright_aspect_before"] is None
    assert ev["evidence"].startswith("Lying (w/h 3.0) for")


def test_brief_bend_is_not_a_fall():
    # box turns sideways for 0.8 s (< min_down_s 2 s), then the person is upright again
    rows = rows_for(1, 12, falling(400, 500, t_fall=5.0, ramp=0.01, t_up=5.8))
    _, _, final = run_logic(make_result(rows))
    assert of_kind(final, "fall") == []
    assert [nm for nm in final["near_misses"] if nm["behavior"] == "fall"] == []     # 0.8 s is too short to be an "almost"


def test_lying_for_nearly_the_minimum_is_a_too_short_near_miss():
    rows = rows_for(1, 12, falling(400, 500, t_fall=5.0, ramp=0.01, t_up=6.7))       # down about 1.7 s of the 2 s rule
    _, _, final = run_logic(make_result(rows))
    assert of_kind(final, "fall") == []
    near = [nm for nm in final["near_misses"] if nm["behavior"] == "fall"]
    assert len(near) == 1 and 1.4 <= near[0]["value"] < 2.0
    assert "not flagged" in near[0]["note"].lower() and near[0]["note"].startswith("Person #1")


@pytest.mark.parametrize("lying_box", [(0.0, 450.0, 150.0, 500.0),             # touches the left edge
                                       (350.0, 1030.0, 500.0, 1080.0)])        # touches the bottom edge
def test_box_cut_by_the_frame_edge_is_not_a_fall(lying_box):
    rows = rows_for(1, 8, lambda t: lying_box)
    tracks, _, final = run_logic(make_result(rows))
    assert tracks[1].at_edge.all() and not down_mask(tracks[1], TEST_CFG).any()
    assert of_kind(final, "fall") == []


def test_the_same_lying_box_away_from_the_edge_is_a_fall():
    # control for the test above: only the position changed
    _, _, final = run_logic(make_result(rows_for(1, 8, lambda t: (300.0, 450.0, 450.0, 500.0))))
    assert len(of_kind(final, "fall")) == 1


def test_a_fall_is_not_a_run_but_the_same_slide_is_running_when_the_fall_rule_is_off():
    # walks 3 s, drops, and the lying box slides at 2.5 body-lengths/s for 1.5 s, then rests 3 s
    rows = rows_for(1, 8, falling(300, 500, t_fall=3.0, ramp=0.3, slide_px_s=2.5 * BH))
    _, _, final = run_logic(make_result(rows))
    assert of_kind(final, "running") == []
    assert len(of_kind(final, "fall")) == 1
    off = deep_merge(TEST_CFG, {"behaviors": {"fall": {"enabled": False}}})
    _, _, final_off = run_logic(make_result(rows), cfg=off)
    assert of_kind(final_off, "fall") == [] and len(of_kind(final_off, "running")) == 1


def test_vehicles_are_never_judged_as_fallen():
    car = rows_for(7, 8, standing(500, 600, w=300, h=120), cls=2)           # a parked car: always wider than tall
    tracks, _, final = run_logic(make_result(car), cfg=CAR_CFG)
    assert not down_mask(tracks[7], CAR_CFG).any()
    assert of_kind(final, "fall") == []


# --------------------------------------------------------------------------- (b) crowding

def _people(n, t_end=8.0, first_id=1):
    """n people standing still in the zone, 180 px (1.2 body-heights) apart."""
    return [rows_for(first_id + i, t_end, standing(250 + 180 * (i % 3), 500 + 150 * (i // 3)))
            for i in range(n)]


def test_six_people_in_a_zone_for_8s_is_one_crowding_event():
    _, _, final = run_logic(make_result(*_people(6)), zones=[ZONE])
    crowd = of_kind(final, "crowding")
    assert len(crowd) == 1
    ev = crowd[0]
    assert ev["entity_id"] is None and ev["entities"] == [1, 2, 3, 4, 5, 6] and ev["other_entity"] is None
    assert ev["metrics"]["max_count"] == 6 and ev["metrics"]["min_count"] == 6
    assert ev["metrics"]["mean_count"] == pytest.approx(6.0) and ev["metrics"]["zone"] == "restricted"
    assert ev["zone"] == "restricted" and ev["entity_name"] == "Group of 6"
    assert ev["evidence"] == ("6 people (avg 6.0) inside 'restricted' for 8 s. Rule: >= 5 for >= 5 s.")
    assert ev["duration_s"] == pytest.approx(8.0, abs=0.2)
    assert ev["confidence_parts"]["margin"] == pytest.approx(0.4)          # (6/5 - 1) / 0.5
    assert ev["baseline"] is None and ev["severity"] is None
    # the same people also each trigger their own zone-intrusion event; events stay sorted and numbered
    assert len(of_kind(final, "zone_intrusion")) == 6
    assert [e["event_id"] for e in final["events"]] == list(range(1, 8))
    json.dumps(final)


def test_four_people_in_a_zone_is_not_crowding():
    _, raw, final = run_logic(make_result(*_people(4)), zones=[ZONE])
    assert of_kind(final, "crowding") == []
    assert [iv for iv in raw["intervals"] if iv["behavior"] == "crowding"] == []


def test_a_crowd_that_lasts_under_the_minimum_is_dropped():
    _, _, final = run_logic(make_result(*_people(6, t_end=4.0)), zones=[ZONE])
    assert of_kind(final, "crowding") == []


def test_crowding_counts_only_people_inside_the_zone():
    outside = [rows_for(10 + i, 8.0, standing(300 + 150 * i, 1000)) for i in range(3)]   # below the zone (y > 900)
    _, _, final = run_logic(make_result(*_people(4), *outside), zones=[ZONE])
    assert of_kind(final, "crowding") == []


def test_whole_frame_crowding_needs_the_setting_and_no_zone():
    result = make_result(*_people(6))
    _, _, off = run_logic(result)                                           # no zone, whole_frame false
    assert of_kind(off, "crowding") == []
    on_cfg = deep_merge(TEST_CFG, {"behaviors": {"crowding": {"whole_frame": True}}})
    _, _, on = run_logic(result, cfg=on_cfg)
    assert len(of_kind(on, "crowding")) == 1 and of_kind(on, "crowding")[0]["zone"] == "whole frame"
    _, _, zoned = run_logic(result, zones=[ZONE], cfg=on_cfg)                # a drawn zone wins over the whole frame
    assert [e["zone"] for e in of_kind(zoned, "crowding")] == ["restricted"]


def test_crowding_uses_the_scenario_entity_word():
    cfg = deep_merge(TEST_CFG, {"scenario": {"entity_word": "Animal"}})
    _, _, final = run_logic(make_result(*_people(6)), zones=[ZONE], cfg=cfg)
    assert of_kind(final, "crowding")[0]["evidence"].startswith("6 animals (avg 6.0) inside")
    assert of_kind(final, "zone_intrusion")[0]["entity_name"] == "Animal #1"


# --------------------------------------------------------------------------- (c) near miss

def _car_crossing(offset_px, car_id=7):
    """Person (id 1) walks right at 0.8 bh/s; a car (class 2) crosses its path at 3 bh/s (450 px/s).

    The car passes the person's path `offset_px` to the right of where the person will be at t = 3 s.
    """
    person = rows_for(1, 6, walking(200, 400, 0.8 * BH))
    car = rows_for(car_id, 3.8, walking(560 + offset_px, 400, 0.0, 3.0 * BH, w=160, h=60, t_start=3.0),
                   t0=2.4, cls=2)
    return make_result(person, car)


def test_person_and_fast_car_passing_close_is_one_near_miss():
    _, raw, final = run_logic(_car_crossing(45), cfg=CAR_CFG)             # closest approach about 0.3 body-heights
    near = of_kind(final, "near_miss")
    assert len(near) == 1
    ev = near[0]
    m = ev["metrics"]
    assert ev["entity_id"] == 1 and ev["other_entity"] == 7 and ev["entities"] == [1, 7]
    assert m["other_id"] == 7 and m["other_class"] == 2
    assert 0.15 < m["min_distance_bh"] < 0.45 and m["contact"] is False
    assert m["min_ttc_s"] is not None and 0.0 < m["min_ttc_s"] < 1.5
    assert m["max_closing_speed_bh_s"] > 2.0 and m["rel_speed_at_closest_bh_s"] > 2.0
    assert m["approach_s"] > 0.3 and m["min_duration_s"] == 0.0 and m["mean_det_conf"] == pytest.approx(0.85, abs=0.01)
    assert ev["snapshot_time_s"] == m["peak_time_s"] and 2.8 <= m["peak_time_s"] <= 3.3
    assert ev["entity_name"] == "Person #1"
    assert ev["evidence"].startswith("Person #1 and Car #7 came within 0.")
    assert "(time-to-collision" in ev["evidence"] and "possible contact" not in ev["evidence"]
    assert "Rule: closer than 0.6 body-heights while moving >= 0.8 body-heights/s relative." in ev["evidence"]
    assert 0.0 < ev["confidence"] <= 1.0
    # raw interval carries the pair
    iv = [i for i in raw["intervals"] if i["behavior"] == "near_miss"]
    assert iv and iv[0]["track_id"] == 1 and iv[0]["other_id"] == 7 and iv[0]["entities"] == [1, 7]


def test_a_car_driving_through_the_person_is_possible_contact():
    _, _, final = run_logic(_car_crossing(0), cfg=CAR_CFG)
    ev = of_kind(final, "near_miss")[0]
    assert ev["metrics"]["contact"] is True and ev["metrics"]["min_distance_bh"] <= 0.15
    assert "possible contact" in ev["evidence"] and "time-to-collision" not in ev["evidence"]
    assert ev["confidence_parts"]["margin"] >= 0.9


def test_car_passing_far_from_the_person_is_not_a_near_miss():
    _, _, final = run_logic(_car_crossing(400), cfg=CAR_CFG)                 # about 2.5 body-heights away
    assert of_kind(final, "near_miss") == []


def test_two_people_walking_together_are_not_a_near_miss():
    a = rows_for(1, 10, walking(200, 400, 0.8 * BH))
    b = rows_for(2, 10, walking(200, 460, 0.8 * BH))                          # 0.4 body-heights beside the first
    _, _, final = run_logic(make_result(a, b))
    assert of_kind(final, "near_miss") == []


def test_two_people_walking_into_each_other_is_a_near_miss():
    a = rows_for(1, 6, walking(200, 400, 0.8 * BH))
    b = rows_for(2, 6, walking(1000, 430, -0.8 * BH))                         # head-on in a corridor, 0.2 bh apart
    _, _, final = run_logic(make_result(a, b))
    near = of_kind(final, "near_miss")
    assert len(near) == 1 and near[0]["entity_id"] == 1 and near[0]["other_entity"] == 2
    assert near[0]["metrics"]["other_class"] == 0
    assert near[0]["evidence"].startswith("Person #1 and Person #2 came within")


def test_a_much_smaller_person_overlapping_in_the_image_is_not_a_near_miss():
    # B is half the size (a person far behind A): the feet cross in the image, but not in the world
    a = rows_for(1, 6, walking(200, 400, 0.8 * BH))
    b = rows_for(2, 6, walking(1000, 415, -150.0, w=30.0, h=75.0))
    result = make_result(a, b)
    _, _, final = run_logic(result)
    assert of_kind(final, "near_miss") == []
    # control: without the depth guard the same data IS a near miss, so it was the guard that suppressed it
    no_guard = deep_merge(TEST_CFG, {"behaviors": {"near_miss": {"max_scale_diff": 0.9}}})
    _, _, final_no_guard = run_logic(result, cfg=no_guard)
    assert len(of_kind(final_no_guard, "near_miss")) == 1


def test_parked_car_next_to_a_standing_person_is_not_a_near_miss():
    person = rows_for(1, 8, standing(400, 400))
    car = rows_for(7, 8, standing(460, 400, w=160, h=60), cls=2)              # 0.4 body-heights away, both still
    _, _, final = run_logic(make_result(person, car), cfg=CAR_CFG)
    assert of_kind(final, "near_miss") == []


def test_cars_alone_never_form_a_near_miss_pair():
    # near_miss pairs need a vulnerable object (a person) as A
    c1 = rows_for(7, 6, walking(200, 400, 450.0, w=160, h=60), cls=2)
    c2 = rows_for(8, 6, walking(1000, 410, -450.0, w=160, h=60), cls=2)
    _, _, final = run_logic(make_result(c1, c2), cfg=CAR_CFG)
    assert of_kind(final, "near_miss") == []


def test_disabled_rules_produce_nothing():
    cfg = deep_merge(CAR_CFG, {"behaviors": {"near_miss": {"enabled": False}, "crowding": {"enabled": False},
                                             "fall": {"enabled": False}}})
    _, raw, _ = run_logic(_car_crossing(45), cfg=cfg)
    assert not [iv for iv in raw["intervals"] if iv["behavior"] in ("near_miss", "crowding", "fall")]


# --------------------------------------------------------------------------- (d) entity names and units

def test_entity_name_uses_the_class_when_the_scene_has_several_classes():
    assert entity_name(CAR_CFG, 7, 2) == "Car #7"
    assert entity_name(CAR_CFG, 3, 0) == "Person #3"
    assert entity_name(TEST_CFG, 7, 2) == "Person #7"                         # one-class scene: the entity word
    assert entity_name(TEST_CFG, None) == "Group"


def test_events_and_near_miss_notes_use_entity_names():
    # a car creeping at 1.5 car-lengths/s (70% of the 1.8 running rule) and a person standing 8 s (80% of 10 s)
    car = rows_for(7, 5, walking(40, 600, 240.0, w=160, h=100), cls=2)
    person = rows_for(1, 8, standing(900, 800))
    _, _, final = run_logic(make_result(car, person), cfg=CAR_CFG)
    notes = {nm["behavior"]: nm for nm in final["near_misses"]}
    assert notes["running"]["track_id"] == 7 and notes["running"]["note"].startswith("Car #7 reached 1.")
    assert notes["running"]["unit"] == "bh/s"
    assert notes["loitering"]["note"].startswith("Person #1 stood still")
    # a real car event carries the car's name too
    fast = rows_for(7, 5, walking(40, 600, 450.0, w=160, h=100), cls=2)
    _, _, final2 = run_logic(make_result(fast), cfg=CAR_CFG)
    run_ev = of_kind(final2, "running")[0]
    assert run_ev["entity_name"] == "Car #7" and run_ev["entities"] == [7]


def test_units_follow_the_scenario():
    cfg = deep_merge(CAR_CFG, {"scenario": {"unit_word": "body-lengths"}})
    assert unit_name(cfg) == "body-lengths"
    fast = rows_for(7, 5, walking(40, 600, 450.0, w=160, h=100), cls=2)
    _, _, final = run_logic(make_result(fast), cfg=cfg)
    ev = of_kind(final, "running")[0]
    assert "body-lengths/s" in ev["evidence"] and "Rule: >= 1.8 bl/s" in ev["evidence"]
    _, _, final_near = run_logic(_car_crossing(45), cfg=deep_merge(cfg, {}))
    assert "body-lengths of each other" in of_kind(final_near, "near_miss")[0]["evidence"]


# --------------------------------------------------------------------------- (e) scenario display names

def test_behavior_name_comes_from_the_scenario_preset():
    preset = {"scenario": {"name": "traffic", "behavior_names": {"running": "Speeding", "fall": "Lying down"}}}
    cfg = deep_merge(TEST_CFG, preset)
    assert behavior_name(cfg, "running") == "Speeding" and behavior_name(cfg, "fall") == "Lying down"
    assert behavior_name(cfg, "crowding") == "Crowding" and behavior_name(TEST_CFG, "running") == "Running"
    rows = rows_for(1, 10, walking(40, 500, 2.5 * BH))
    _, _, final = run_logic(make_result(rows), cfg=cfg)
    assert [e["behavior_name"] for e in of_kind(final, "running")] == ["Speeding"]
    _, _, default = run_logic(make_result(rows))
    assert [e["behavior_name"] for e in of_kind(default, "running")] == ["Running"]


# --------------------------------------------------------------------------- event format and API

def test_every_new_event_has_the_full_key_set_and_serialises():
    people = _people(6, first_id=30)
    faller = rows_for(20, 9.5, falling(900, 700, t_fall=5.0))
    _, _, final = run_logic(_merge_results(make_result(*people, faller), _car_crossing(45)), zones=[ZONE], cfg=CAR_CFG)
    kinds = {e["behavior"] for e in final["events"]}
    assert {"fall", "crowding", "near_miss", "zone_intrusion"} <= kinds
    expected = {"event_id", "entity_id", "behavior", "start_s", "end_s", "start", "end", "duration_s", "confidence",
                "confidence_parts", "evidence", "metrics", "zone", "baseline", "snapshot_time_s", "snapshot",
                "speed_plot", "verified", "pose", "severity", "entities", "other_entity", "entity_name",
                "behavior_name"}
    for e in final["events"]:
        assert set(e) == expected
        assert e["entities"] and isinstance(e["entities"], list)
        assert e["entity_name"] and e["behavior_name"] and e["evidence"].endswith(".")
        assert 0.0 <= e["confidence"] <= 1.0
        assert e["metrics"]["peak_time_s"] == e["snapshot_time_s"]
    assert [e["event_id"] for e in final["events"]] == list(range(1, len(final["events"]) + 1))
    starts = [e["start_s"] for e in final["events"]]
    assert starts == sorted(starts)
    json.dumps(final)


def _merge_results(a, b):
    """Combine two synthetic TrackerResults (distinct track ids) into one."""
    out = dict(a)
    out["detections"] = a["detections"] + b["detections"]
    out["duration_s"] = max(a["duration_s"], b["duration_s"])
    return out


def test_metric_keys_per_new_behavior():
    fall_rows = make_result(rows_for(1, 9.5, falling(400, 500)))
    tracks = build_tracks(fall_rows, TEST_CFG)
    fall = compute_metrics(tracks[1], "fall", 5.5, 9.5, [], TEST_CFG)
    assert set(fall) == {"duration_s", "max_aspect", "upright_aspect_before", "fell", "transition_s", "min_down_s",
                         "peak_time_s", "mean_det_conf"}
    assert fall["peak_time_s"] == pytest.approx(6.0)                       # start of the run + 0.5 s
    crowd_tracks = build_tracks(make_result(*_people(6)), TEST_CFG)
    crowd = compute_metrics(None, "crowding", 0.0, 8.0, [ZONE], TEST_CFG, tracks=crowd_tracks)
    assert set(crowd) == {"duration_s", "zone", "max_count", "mean_count", "min_count", "min_duration_s",
                          "peak_time_s", "mean_det_conf"}
    pair = build_tracks(_car_crossing(45), CAR_CFG)
    near = compute_metrics(pair[1], "near_miss", 2.8, 3.3, [], CAR_CFG, other=pair[7])
    assert set(near) == {"min_distance_bh", "peak_time_s", "max_closing_speed_bh_s", "min_ttc_s",
                         "rel_speed_at_closest_bh_s", "other_id", "other_class", "approach_s", "contact",
                         "duration_s", "mean_det_conf", "min_duration_s"}
    with pytest.raises(ValueError):
        compute_metrics(None, "crowding", 0.0, 8.0, [ZONE], TEST_CFG)       # needs all tracks
    with pytest.raises(ValueError):
        compute_metrics(pair[1], "near_miss", 2.8, 3.3, [], CAR_CFG)        # needs the other track


def test_baseline_compares_like_with_like():
    # five ordinary walkers plus ONE fast car: the car must not make the walkers look slow or "unusual"
    walkers = [rows_for(i + 1, 20, walking(40, 100 + 100 * i, v * BH)) for i, v in enumerate([0.70, 0.80, 0.85, 0.90, 0.95])]
    car = rows_for(9, 20, walking(40, 800, 3.0 * BH, w=160, h=100), cls=2)
    tracks, raw, _ = run_logic(make_result(*walkers, car), cfg=CAR_CFG)
    bl = raw["baseline"]
    assert bl["active"] and bl["tracks"] and 9 not in bl["tracks"]            # one car is too few to define "normal"
    assert not any(info["unusual"] for info in bl["tracks"].values())


def test_runtime_is_fast():
    t0 = time.perf_counter()
    people = _people(6)
    run_logic(_merge_results(make_result(*people), _car_crossing(45)), zones=[ZONE], cfg=CAR_CFG)
    assert time.perf_counter() - t0 < 5.0
