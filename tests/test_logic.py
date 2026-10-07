"""Unit tests for the reasoning core (features, behaviors, events) on SYNTHETIC tracker output.

No video, YOLO or OpenCV is needed: we fabricate a TrackerResult dict (DESIGN 3.1) that looks like what
tracker.py would produce for a 30 fps clip processed every 2nd frame at 640x360, with a person box
about 150 px tall. A person's path is a list of "legs" (duration_s, speed_in_body_heights_per_s), so a
test reads like the story it checks: "walk 6 s, run 5 s, walk 6 s".

Thresholds are pinned in TEST_CFG (not read from config.yaml) so tuning config.yaml during the
hackathon cannot break the tests.
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np
import pytest

from behaviors import compute_metrics, detect_behaviors, running_intervals
from events import build_events
from features import Track, build_tracks, stationary_runs, track_summary
from utils import DEFAULT_CONFIG, deep_merge, fmt_time, load_config

FPS, STRIDE, BH = 30.0, 2, 150.0
PROC_SIZE = [640, 360]

TEST_CFG = deep_merge(DEFAULT_CONFIG, {
    "features": {"smoothing_s": 0.5, "speed_window_s": 0.5, "max_gap_s": 1.0,
                 "min_track_s": 1.0, "min_box_h_px": 20},
    "behaviors": {
        "loitering": {"enabled": True, "radius_bh": 0.5, "min_duration_s": 10.0},
        "zone_intrusion": {"enabled": True, "min_duration_s": 1.0},
        "running": {"enabled": True, "start_speed_bh_s": 1.8, "end_speed_bh_s": 1.4, "min_duration_s": 1.0},
        "baseline": {"enabled": True, "min_tracks": 4, "z_threshold": 3.0},
    },
    "events": {"merge_gap_s": 1.5, "near_miss_ratio": 0.7,
               "confidence_weights": {"margin": 0.4, "duration": 0.3, "detection": 0.3}},
})

ZONE = {"name": "restricted", "points": [(280.0, 250.0), (640.0, 250.0), (640.0, 350.0), (280.0, 350.0)]}


# --------------------------------------------------------------------------- synthetic data helpers

def person_rows(tid, legs, t0=0.0, start=(40.0, 300.0), direction=(1.0, 0.0), jitter_px=0.0,
                conf=0.85, height=BH, seed=None):
    """Detection rows for one person walking along a straight line.

    legs = [(duration_s, speed_bh_s), ...]; speed 0 means standing still. `jitter_px` adds uniform
    +-jitter noise to the foot position (detector box jitter). Rows follow DESIGN 3.1:
    [frame_idx, t, track_id, x1, y1, x2, y2, conf], only on frames with frame % STRIDE == 0.
    """
    rng = np.random.default_rng(tid if seed is None else seed)
    durations = np.array([d for d, _ in legs], dtype=float)
    leg_start = np.cumsum(durations) - durations
    rows = []
    for frame in range(int(round(t0 * FPS)), int(round((t0 + durations.sum()) * FPS)) + 1):
        if frame % STRIDE:
            continue
        t = frame / FPS
        local = t - t0
        # distance walked so far = sum over legs of speed * height * time spent in that leg
        dist = sum(v * height * max(0.0, min(local - s, d)) for (d, v), s in zip(legs, leg_start))
        x = start[0] + direction[0] * dist
        y = start[1] + direction[1] * dist
        if jitter_px:
            x += rng.uniform(-jitter_px, jitter_px)
            y += rng.uniform(-jitter_px, jitter_px)
        w = 0.2 * height
        rows.append([frame, t, tid, x - w, y - height, x + w, y, conf])
    return rows


def make_result(*row_groups, duration_s=None):
    """Wrap detection rows into a TrackerResult dict (DESIGN 3.1)."""
    rows = [r for group in row_groups for r in group]
    if duration_s is None:
        duration_s = max((r[1] for r in rows), default=0.0)
    return {"video": "synthetic.mp4", "fps": FPS, "stride": STRIDE, "orig_size": [1280, 720],
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


def hand_track(speed, track_id=1, dt=1 / 15, conf=0.9):
    """A Track built directly from a speed signal (to unit-test the running rule in isolation)."""
    n = len(speed)
    t = np.arange(n) * dt
    return Track(track_id=track_id, t=t, frame=np.rint(t * FPS).astype(int), box=np.zeros((n, 4)),
                 conf=np.full(n, conf), foot=np.column_stack((np.cumsum(np.ones(n)), np.full(n, 300.0))),
                 height=np.full(n, BH), speed=np.asarray(speed, dtype=float), segment=np.zeros(n, dtype=int))


# --------------------------------------------------------------------------- features

def test_build_tracks_basic_shape_and_speed():
    result = make_result(person_rows(7, [(10, 0.8)], jitter_px=1.0))
    tracks = build_tracks(result, TEST_CFG)
    assert list(tracks) == [7]
    tr = tracks[7]
    n = len(tr.t)
    assert tr.t.shape == (n,) and tr.box.shape == (n, 4) and tr.foot.shape == (n, 2)
    assert tr.frame.dtype.kind == "i"
    assert np.all(np.diff(tr.t) > 0)
    assert tr.start_s == pytest.approx(0.0) and tr.duration_s == pytest.approx(10.0, abs=0.1)
    # foot is the bottom-centre of the box: y is the box bottom
    assert tr.foot[n // 2, 1] == pytest.approx(300.0, abs=2.0)
    # first samples have no speed yet (less than half a window of history)
    assert math.isnan(tr.speed[0])
    mid = tr.speed[np.isfinite(tr.speed)]
    assert np.median(mid) == pytest.approx(0.8, abs=0.1)


def test_build_tracks_filters_and_dedupes():
    good = person_rows(1, [(5, 0.8)])
    short = person_rows(2, [(0.5, 0.8)])                                   # 0.5 s < min_track_s
    two_samples = [r for r in person_rows(3, [(5, 0.8)])][:2]              # fewer than 3 samples
    tiny = person_rows(4, [(5, 0.8)], height=15.0)                         # box height 15 px < 20
    # duplicate time stamps: the higher-confidence box must win
    low = [list(r) for r in good[:10]]
    for r in low:
        r[7], r[3], r[5] = 0.30, r[3] + 40, r[5] + 40
    result = make_result(good, short, two_samples, tiny, low)
    tracks = build_tracks(result, TEST_CFG)
    assert list(tracks) == [1]
    # the low-confidence duplicates (id 1, same times, shifted 40 px) were discarded
    assert tracks[1].conf.min() == pytest.approx(0.85)
    assert len(tracks[1].t) == len(good)


def test_build_tracks_sorts_unsorted_input():
    rows = person_rows(1, [(5, 0.8)])
    shuffled = list(reversed(rows))
    tr = build_tracks(make_result(shuffled), TEST_CFG)[1]
    assert np.all(np.diff(tr.t) > 0)


def test_gap_splits_segments_and_blocks_speed():
    first = person_rows(1, [(4, 0.8)])
    second = person_rows(1, [(4, 0.8)], t0=6.0, start=(40.0 + 0.8 * BH * 6, 300.0))   # 2 s gap > 1 s
    tr = build_tracks(make_result(first, second), TEST_CFG)[1]
    assert set(np.unique(tr.segment)) == {0, 1}
    # the sample right after the gap must not get a speed that spans the gap
    first_of_seg1 = int(np.flatnonzero(tr.segment == 1)[0])
    assert math.isnan(tr.speed[first_of_seg1])
    assert np.all(np.diff(tr.segment) >= 0)


def test_stationary_runs_values():
    still = build_tracks(make_result(person_rows(1, [(15, 0.0)], jitter_px=2.0)), TEST_CFG)[1]
    assert stationary_runs(still, 0.5) == pytest.approx(15.0, abs=0.2)
    walker = build_tracks(make_result(person_rows(2, [(15, 0.8)], jitter_px=2.0)), TEST_CFG)[2]
    # a walker leaves a 0.5-bh circle (diameter 1 bh) in about 1 / 0.8 = 1.25 s
    assert 0.8 < stationary_runs(walker, 0.5) < 1.8


def test_stationary_runs_does_not_cross_a_gap():
    a = person_rows(1, [(3, 0.0)])
    b = person_rows(1, [(3, 0.0)], t0=5.0)         # same spot, but 2 s later
    tr = build_tracks(make_result(a, b), TEST_CFG)[1]
    assert stationary_runs(tr, 0.5) == pytest.approx(3.0, abs=0.2)


def test_track_summary_keys():
    tr = build_tracks(make_result(person_rows(5, [(6, 0.8), (6, 0.0)])), TEST_CFG)[5]
    s = track_summary(tr, TEST_CFG)
    assert set(s) == {"id", "first_seen", "last_seen", "duration_s", "median_speed_bh_s",
                      "p90_speed_bh_s", "max_dwell_s"}
    assert s["id"] == 5 and s["duration_s"] == pytest.approx(12.0, abs=0.1)
    assert s["max_dwell_s"] > 5.0
    assert s["p90_speed_bh_s"] >= s["median_speed_bh_s"]
    json.dumps(s)                                  # plain Python types only


# --------------------------------------------------------------------------- behaviours end to end

def test_steady_walker_has_no_events():
    _, raw, final = run_logic(make_result(person_rows(1, [(25, 0.8)], jitter_px=1.5)))
    assert final["events"] == []
    assert final["near_misses"] == []
    assert raw["intervals"] == []


def test_standing_still_15s_is_one_loitering_event():
    # walk in 5 s, stand still 15 s (+-2 px jitter), walk away 5 s: truth is 5.0 .. 20.0
    rows = person_rows(1, [(5, 0.8), (15, 0.0), (5, 0.8)], jitter_px=2.0)
    _, _, final = run_logic(make_result(rows))
    assert len(final["events"]) == 1
    ev = final["events"][0]
    assert ev["behavior"] == "loitering" and ev["entity_id"] == 1
    assert abs(ev["start_s"] - 5.0) <= 1.0
    assert abs(ev["end_s"] - 20.0) <= 1.0
    assert ev["metrics"]["max_radius_bh"] <= 0.5
    # the walk-in / walk-out steps push max_radius_bh up to the limit, but the person really stood
    # still, so the confidence margin (based on the 90th-percentile radius) must stay high
    assert ev["metrics"]["p90_radius_bh"] < 0.1
    assert ev["confidence_parts"]["margin"] >= 0.8 and ev["confidence"] >= 0.75
    assert "Stayed within" in ev["evidence"] and "Rule: within 0.5 bh for >= 10 s." in ev["evidence"]
    assert ev["snapshot_time_s"] == pytest.approx(0.5 * (ev["start_s"] + ev["end_s"]), abs=0.1)
    assert ev["zone"] is None


def test_runner_between_walking_is_one_running_event():
    rows = person_rows(1, [(6, 0.8), (5, 2.5), (6, 0.8)], jitter_px=1.5)    # runs from 6.0 to 11.0
    _, _, final = run_logic(make_result(rows))
    assert len(final["events"]) == 1
    ev = final["events"][0]
    assert ev["behavior"] == "running"
    assert abs(ev["start_s"] - 6.0) <= 0.7
    assert abs(ev["end_s"] - 11.0) <= 0.7
    m = ev["metrics"]
    assert m["peak_speed_bh_s"] >= m["mean_speed_bh_s"] >= 1.8
    assert m["start_threshold_bh_s"] == 1.8 and m["end_threshold_bh_s"] == 1.4
    assert 6.0 <= ev["snapshot_time_s"] <= 11.7
    assert "body-heights/s" in ev["evidence"] and "Rule: >= 1.8 bh/s for >= 1 s" in ev["evidence"]


def test_foot_in_zone_for_3s_is_one_zone_intrusion():
    # walks left to right at y=320; x in [280, 640] (the zone) from t=1.5 s to t=4.5 s
    rows = person_rows(1, [(8, 0.8)], start=(100.0, 320.0), jitter_px=1.0)
    _, _, final = run_logic(make_result(rows), zones=[ZONE])
    assert [e["behavior"] for e in final["events"]] == ["zone_intrusion"]
    ev = final["events"][0]
    assert ev["zone"] == "restricted" and ev["metrics"]["zone"] == "restricted"
    assert ev["duration_s"] == pytest.approx(3.0, abs=0.6)
    assert abs(ev["start_s"] - 1.5) <= 0.6
    assert 0.0 < ev["metrics"]["max_depth_bh"] <= 0.25            # foot is 30 px (0.2 bh) above the bottom edge
    assert "Feet inside zone 'restricted'" in ev["evidence"] and "Rule: >= 1 s." in ev["evidence"]
    assert ev["snapshot_time_s"] == ev["metrics"]["peak_time_s"]


def test_no_zones_means_no_zone_events():
    rows = person_rows(1, [(8, 0.8)], start=(100.0, 320.0))
    _, _, final = run_logic(make_result(rows), zones=[])
    assert final["events"] == []


def test_short_speed_blips_do_not_create_running_events():
    # 0.4 s bursts at 3 bh/s are far shorter than the 1 s minimum
    rows = person_rows(1, [(4, 0.8), (0.4, 3.0), (4, 0.8), (0.4, 3.0), (4, 0.8)], jitter_px=1.0)
    _, raw, final = run_logic(make_result(rows))
    assert of_kind(final, "running") == []
    assert final["events"] == []
    for iv in raw["intervals"]:                      # whatever raw pieces exist are all under 1 s
        assert iv["end_s"] - iv["start_s"] < 1.0


def test_hysteresis_keeps_running_between_the_two_thresholds():
    # speed climbs over 1.8, dips to 1.5 (above the 1.4 stop threshold), then falls below 1.4
    speed = [0.8] * 5 + [2.0] * 5 + [1.5] * 5 + [2.0] * 5 + [1.0] * 5
    ivs = running_intervals(hand_track(speed), TEST_CFG)
    assert len(ivs) == 1                              # the 1.5 dip did not end the run
    dt = 1 / 15
    assert ivs[0][0] == pytest.approx(5 * dt) and ivs[0][1] == pytest.approx(19 * dt)
    # a dip below the stop threshold does split it
    speed2 = [0.8] * 3 + [2.0] * 5 + [1.0] * 3 + [2.0] * 5 + [0.8] * 3
    assert len(running_intervals(hand_track(speed2), TEST_CFG)) == 2
    # NaN speeds change nothing
    speed3 = [0.8] * 3 + [2.0] * 3 + [np.nan] * 3 + [2.0] * 3 + [0.8] * 3
    assert len(running_intervals(hand_track(speed3), TEST_CFG)) == 1


def test_two_running_pieces_one_second_apart_merge():
    # hand-made raw intervals 1.0 s apart (< merge_gap_s 1.5) on a real runner track
    tracks = build_tracks(make_result(person_rows(1, [(20, 2.5)])), TEST_CFG)
    zones = []
    raw = {"intervals": [{"track_id": 1, "behavior": "running", "start_s": 4.0, "end_s": 8.0},
                         {"track_id": 1, "behavior": "running", "start_s": 9.0, "end_s": 12.0}],
           "near_misses": [], "baseline": {"enabled": True, "active": False, "n_tracks": 1, "tracks": {}}}
    final = build_events(raw, tracks, zones, TEST_CFG)
    assert len(final["events"]) == 1
    ev = final["events"][0]
    assert (ev["start_s"], ev["end_s"]) == (4.0, 12.0)
    assert ev["duration_s"] == pytest.approx(8.0)
    # ...but pieces 2 s apart stay separate
    raw["intervals"][1]["start_s"] = 10.0
    assert len(build_events(raw, tracks, zones, TEST_CFG)["events"]) == 2


def test_running_pieces_with_short_dip_become_one_event_end_to_end():
    # run 3 s, slow to a walk for 1.2 s, run 3 s again
    rows = person_rows(1, [(5, 0.8), (3, 2.5), (1.2, 0.8), (3, 2.5), (5, 0.8)], jitter_px=1.0)
    _, raw, final = run_logic(make_result(rows))
    assert len(of_kind(final, "running")) == 1
    ev = of_kind(final, "running")[0]
    assert ev["duration_s"] > 6.5                     # the two bursts plus the dip between them


def test_piece_shorter_than_minimum_after_merge_is_dropped_but_exact_minimum_passes():
    tracks = build_tracks(make_result(person_rows(1, [(10, 2.5)])), TEST_CFG)
    base = {"near_misses": [], "baseline": {"enabled": True, "active": False, "n_tracks": 1, "tracks": {}}}
    ok = build_events({**base, "intervals": [{"track_id": 1, "behavior": "running", "start_s": 3.0, "end_s": 4.0}]},
                      tracks, [], TEST_CFG)
    assert len(ok["events"]) == 1                      # exactly 1.0 s passes the 1.0 s rule
    short = build_events({**base, "intervals": [{"track_id": 1, "behavior": "running", "start_s": 3.0, "end_s": 3.9}]},
                         tracks, [], TEST_CFG)
    assert short["events"] == []


def test_standing_still_8s_is_a_loitering_near_miss():
    rows = person_rows(1, [(4, 0.8), (8, 0.0), (4, 0.8)], jitter_px=2.0)
    _, _, final = run_logic(make_result(rows))
    assert final["events"] == []
    near = [nm for nm in final["near_misses"] if nm["behavior"] == "loitering"]
    assert len(near) == 1
    nm = near[0]
    assert nm["track_id"] == 1 and nm["threshold"] == 10.0 and nm["unit"] == "s"
    assert 7.0 <= nm["value"] < 10.0
    assert "Not flagged" in nm["note"] and "#1" in nm["note"]


def test_walker_speed_near_miss_for_running():
    # 1.5 bh/s: above 70% of 1.8 (1.26) but below the running threshold
    _, _, final = run_logic(make_result(person_rows(1, [(10, 1.5)], jitter_px=1.0)))
    assert final["events"] == []
    near = [nm for nm in final["near_misses"] if nm["behavior"] == "running"]
    assert len(near) == 1 and near[0]["unit"] == "bh/s" and 1.26 <= near[0]["value"] < 1.8


def test_short_zone_touch_is_a_too_short_near_miss():
    # crosses a narrow part of the zone: foot inside for roughly 0.9 s (< the 1 s rule)
    narrow = {"name": "restricted", "points": [(300.0, 250.0), (400.0, 250.0), (400.0, 350.0), (300.0, 350.0)]}
    rows = person_rows(1, [(6, 0.8)], start=(100.0, 320.0))
    _, _, final = run_logic(make_result(rows), zones=[narrow])
    assert final["events"] == []
    near = [nm for nm in final["near_misses"] if nm["behavior"] == "zone_intrusion"]
    assert len(near) == 1 and 0.7 <= near[0]["value"] < 1.0
    assert "zone" in near[0]["note"] and "not flagged" in near[0]["note"].lower()


def test_near_miss_dropped_when_the_pair_has_an_event():
    tracks = build_tracks(make_result(person_rows(1, [(10, 2.5)])), TEST_CFG)
    raw = {"intervals": [{"track_id": 1, "behavior": "running", "start_s": 3.0, "end_s": 6.0}],
           "near_misses": [{"track_id": 1, "behavior": "running", "value": 1.5, "threshold": 1.8, "unit": "bh/s",
                            "note": "x"},
                           {"track_id": 1, "behavior": "loitering", "value": 8.0, "threshold": 10.0, "unit": "s",
                            "note": "y"}],
           "baseline": {"enabled": True, "active": False, "n_tracks": 1, "tracks": {}}}
    final = build_events(raw, tracks, [], TEST_CFG)
    assert [(n["track_id"], n["behavior"]) for n in final["near_misses"]] == [(1, "loitering")]


def test_disabled_behavior_is_not_reported():
    cfg = deep_merge(TEST_CFG, {"behaviors": {"loitering": {"enabled": False}}})
    rows = person_rows(1, [(5, 0.8), (15, 0.0), (5, 0.8)], jitter_px=2.0)
    _, raw, final = run_logic(make_result(rows), cfg=cfg)
    assert final["events"] == [] and raw["intervals"] == []
    assert not [nm for nm in final["near_misses"] if nm["behavior"] == "loitering"]


# --------------------------------------------------------------------------- baseline / unusual

def _crowd_with_runner():
    """Five ordinary walkers (0.70-0.95 bh/s) and one runner (2.5 bh/s) over 20 s."""
    walker_speeds = [0.70, 0.80, 0.85, 0.90, 0.95]
    groups = [person_rows(i + 1, [(20, v)], start=(40.0, 60.0 + 50 * i), jitter_px=1.0, seed=i)
              for i, v in enumerate(walker_speeds)]
    groups.append(person_rows(99, [(20, 2.5)], start=(40.0, 330.0), jitter_px=1.0, seed=99))
    return make_result(*groups)


def test_baseline_flags_the_runner_as_unusual():
    _, raw, final = run_logic(_crowd_with_runner())
    bl = raw["baseline"]
    assert bl["enabled"] and bl["active"] and bl["n_tracks"] == 6
    assert bl["tracks"][99]["unusual"] is True
    assert bl["tracks"][99]["anomaly_score"] >= 3.0
    assert "faster" in bl["tracks"][99]["note"]
    assert [i for i in range(1, 6) if bl["tracks"][i]["unusual"]] == []
    # final events: only the runner, with the baseline note attached
    assert [(e["entity_id"], e["behavior"]) for e in final["events"]] == [(99, "running")]
    assert final["events"][0]["baseline"]["anomaly_score"] >= 3.0
    assert [u["entity_id"] for u in final["unusual_tracks"]] == [99]
    assert final["unusual_tracks"][0]["has_event"] is True
    assert "tracks" not in final["baseline"] and final["baseline"]["active"] is True
    assert final["baseline"]["median_speed_bh_s"] == pytest.approx(0.88, abs=0.1)


def test_baseline_inactive_with_too_few_tracks():
    rows = [person_rows(i, [(10, 0.8)], start=(40.0, 60.0 + 50 * i)) for i in range(1, 4)]
    _, raw, final = run_logic(make_result(*rows))
    assert raw["baseline"]["active"] is False and raw["baseline"]["n_tracks"] == 3
    assert raw["baseline"]["tracks"] == {}
    assert final["unusual_tracks"] == []


def test_unusual_track_without_event_has_event_false():
    # one person stands still ~6 s among walkers: not a loitering event (needs 10 s) but unusual vs the scene
    groups = [person_rows(i + 1, [(10, v)], start=(40.0, 60.0 + 50 * i), jitter_px=1.0, seed=i)
              for i, v in enumerate([0.8, 0.85, 0.9, 0.95, 0.75])]
    groups.append(person_rows(50, [(2, 0.8), (6, 0.0), (2, 0.8)], start=(40.0, 330.0), jitter_px=1.0, seed=50))
    _, _, final = run_logic(make_result(*groups))
    assert [e for e in final["events"] if e["entity_id"] == 50 and e["behavior"] == "loitering"] == []
    unusual = {u["entity_id"]: u for u in final["unusual_tracks"]}
    assert 50 in unusual and unusual[50]["has_event"] is False
    assert "Stayed near one spot" in unusual[50]["note"] or "faster" in unusual[50]["note"]


# --------------------------------------------------------------------------- robustness / format

def test_empty_detections_do_not_crash():
    for zones in ([], [ZONE]):
        tracks, raw, final = run_logic(make_result([], duration_s=10.0), zones=zones)
        assert tracks == {}
        assert raw["intervals"] == [] and raw["near_misses"] == []
        assert final["events"] == [] and final["near_misses"] == [] and final["unusual_tracks"] == []
        assert final["baseline"]["active"] is False
    # a result with no "detections" key at all
    assert build_tracks({"fps": 30.0}, TEST_CFG) == {}


def test_single_sample_and_tiny_tracks_do_not_crash():
    rows = [[0, 0.0, 1, 100.0, 100.0, 140.0, 250.0, 0.9]]
    _, _, final = run_logic(make_result(rows))
    assert final["events"] == []


def test_time_strings_match_seconds():
    # a long clip so the strings cross the one-minute mark: runs from 60 s to 65 s
    rows = person_rows(1, [(60, 0.8), (5, 2.5), (5, 0.8)], jitter_px=1.0)
    _, _, final = run_logic(make_result(rows))
    assert len(final["events"]) == 1
    ev = final["events"][0]
    assert ev["start"] == fmt_time(ev["start_s"]) and ev["end"] == fmt_time(ev["end_s"])
    assert ev["start"] == "01:00" and ev["end"] == "01:05"
    assert re.fullmatch(r"\d{2}:\d{2}", ev["start"]) and re.fullmatch(r"\d{2}:\d{2}", ev["end"])
    assert fmt_time(65.0) == "01:05" and fmt_time(5.9) == "00:05"


def test_events_are_sorted_numbered_and_well_formed():
    # two people, three events; person 2's loitering starts earlier than person 1's run
    a = person_rows(1, [(30, 0.8), (4, 2.5), (6, 0.8)], jitter_px=1.0)
    b = person_rows(2, [(2, 0.8), (14, 0.0), (6, 0.8)], jitter_px=1.5, start=(40.0, 200.0))
    c = person_rows(3, [(20, 0.8)], start=(100.0, 320.0), jitter_px=1.0)
    _, _, final = run_logic(make_result(a, b, c), zones=[ZONE])
    events = final["events"]
    assert [e["event_id"] for e in events] == list(range(1, len(events) + 1))
    assert [(e["start_s"], e["entity_id"]) for e in events] == sorted((e["start_s"], e["entity_id"]) for e in events)
    assert {e["behavior"] for e in events} == {"loitering", "zone_intrusion", "running"}
    expected_keys = {"event_id", "entity_id", "behavior", "start_s", "end_s", "start", "end", "duration_s",
                     "confidence", "confidence_parts", "evidence", "metrics", "zone", "baseline",
                     "snapshot_time_s", "snapshot", "speed_plot", "verified", "pose"}
    expected_keys |= {"severity", "entities", "other_entity", "entity_name", "behavior_name"}   # DESIGN 4.10
    for e in events:
        assert set(e) == expected_keys
        assert e["snapshot"] is None and e["speed_plot"] is None and e["verified"] is None and e["pose"] is None
        assert e["duration_s"] == pytest.approx(e["end_s"] - e["start_s"], abs=0.011)
        assert e["metrics"]["peak_time_s"] == e["snapshot_time_s"]
        assert 0.0 < e["metrics"]["mean_det_conf"] <= 1.0
        assert e["evidence"] and e["evidence"].endswith(".")
    assert set(final) == {"events", "near_misses", "unusual_tracks", "baseline", "ignored_outside_hours"}
    json.dumps(final)                                        # plain Python types, serialisable as-is


def test_metric_keys_per_behavior():
    tracks = build_tracks(make_result(person_rows(1, [(30, 0.0)], jitter_px=1.0, start=(460.0, 300.0))), TEST_CFG)
    tr = tracks[1]
    common = {"duration_s", "mean_det_conf", "peak_time_s"}
    loiter = compute_metrics(tr, "loitering", 0.0, 29.0, [ZONE], TEST_CFG)
    # DESIGN keys plus one extra (p90_radius_bh) used for the loitering confidence margin
    assert set(loiter) >= common | {"max_radius_bh", "radius_threshold_bh", "min_duration_s", "mean_speed_bh_s"}
    assert set(loiter) - (common | {"max_radius_bh", "radius_threshold_bh", "min_duration_s", "mean_speed_bh_s"})         == {"p90_radius_bh"}
    assert loiter["p90_radius_bh"] <= loiter["max_radius_bh"] <= 0.1
    zone = compute_metrics(tr, "zone_intrusion", 0.0, 29.0, [ZONE], TEST_CFG)
    assert set(zone) == common | {"zone", "max_depth_bh", "min_duration_s"}
    assert zone["zone"] == "restricted" and zone["max_depth_bh"] > 0.2
    run = compute_metrics(tr, "running", 0.0, 29.0, [ZONE], TEST_CFG)
    assert set(run) == common | {"peak_speed_bh_s", "mean_speed_bh_s", "start_threshold_bh_s",
                                 "end_threshold_bh_s", "min_duration_s"}
    with pytest.raises(ValueError):
        compute_metrics(tr, "dancing", 0.0, 1.0, [], TEST_CFG)


def test_confidence_in_range_and_follows_the_formula():
    rows = [person_rows(1, [(6, 0.8), (5, 2.5), (6, 0.8)], jitter_px=1.5, start=(40.0, 120.0)),
            person_rows(2, [(5, 0.8), (15, 0.0), (5, 0.8)], jitter_px=2.0, start=(40.0, 200.0), conf=0.6),
            person_rows(3, [(8, 0.8)], start=(100.0, 320.0), conf=0.95)]
    _, _, final = run_logic(make_result(*rows), zones=[ZONE])
    assert len(final["events"]) == 3
    w = TEST_CFG["events"]["confidence_weights"]
    for e in final["events"]:
        parts = e["confidence_parts"]
        assert set(parts) == {"margin", "duration", "detection"}
        assert all(0.0 <= v <= 1.0 for v in parts.values()) and 0.0 <= e["confidence"] <= 1.0
        expected = w["margin"] * parts["margin"] + w["duration"] * parts["duration"] + w["detection"] * parts["detection"]
        assert e["confidence"] == pytest.approx(expected, abs=0.02)      # parts are rounded to 2 decimals
        assert e["confidence"] == round(e["confidence"], 2)
    by_kind = {e["behavior"]: e for e in final["events"]}
    assert by_kind["loitering"]["confidence_parts"]["detection"] == pytest.approx(0.6)
    assert by_kind["zone_intrusion"]["confidence_parts"]["detection"] == pytest.approx(0.95)


def test_real_config_yaml_smoke():
    """The shipped config.yaml must work with the logic (catches missing or renamed keys)."""
    cfg = load_config()
    rows = person_rows(1, [(6, 0.8), (5, 2.5), (6, 0.8)], jitter_px=1.5)
    _, _, final = run_logic(make_result(rows), cfg=cfg)
    assert [e["behavior"] for e in final["events"]] == ["running"]


def test_logic_modules_do_not_import_the_vision_stack():
    """DESIGN section 1: the core stays NumPy-only so it runs anywhere."""
    import behaviors
    import events
    import features
    for mod in (features, behaviors, events):
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import cv2", "import torch", "import ultralytics", "from ultralytics", "from cv2"):
            assert banned not in src, f"{mod.__name__} must not contain '{banned}'"


def test_runtime_is_fast():
    t0 = time.perf_counter()
    run_logic(_crowd_with_runner())
    assert time.perf_counter() - t0 < 5.0
