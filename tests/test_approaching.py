"""Tests for the approaching (looming) alert used by the wearable-camera 'navigation' preset."""
import numpy as np

import approaching
import features
import utils

FPS, W, H = 30.0, 640, 360


def _cfg():
    cfg = utils.load_config(scenario="navigation")
    cfg["video"]["frame_stride"] = 1
    return cfg


def _result(rows):
    return {"video": "synthetic", "fps": FPS, "stride": 1, "orig_size": [W, H], "proc_size": [W, H],
            "n_frames_read": int(max(r[0] for r in rows)) + 1, "duration_s": max(r[1] for r in rows),
            "frames_processed": len(rows), "device": "cpu", "model": "yolo11n.pt", "detections": rows}


def _walker(tid, dist_fn, cx_fn, duration_s=4.0, cls=0, conf=0.85):
    """Rows for an object whose image height is 100 px at 6 m (h = 600 / distance)."""
    rows = []
    for f in range(int(duration_s * FPS)):
        t = f / FPS
        h = 600.0 / dist_fn(t)
        w = 0.4 * h
        cx = cx_fn(t)
        y2 = min(H - 2.0, 200 + 0.5 * h)
        rows.append([f, t, tid, cx - w / 2, y2 - h, cx + w / 2, y2, conf, cls])
    return rows


def _track(rows, cfg):
    tracks = features.build_tracks(_result(rows), cfg)
    assert len(tracks) == 1
    return next(iter(tracks.values()))


def test_person_walking_straight_at_the_camera_triggers_an_alert():
    cfg = _cfg()
    # Closing at 2 m/s from 7 m: true time to contact at distance Z is Z / 2.
    tr = _track(_walker(1, lambda t: 7.0 - 2.0 * t, lambda t: W / 2 + 10 * np.sin(t), duration_s=2.5), cfg)
    runs = approaching.approaching_intervals(tr, cfg, (W, H))
    assert runs, "expected an approaching alert"
    m = approaching.approaching_metrics(tr, runs[0][0], runs[-1][1], cfg, (W, H))
    assert m["min_ttc_s"] is not None and m["min_ttc_s"] < 2.5
    assert m["direction"] == "12 o'clock"
    assert m["alert_text"].startswith("Person approaching from 12 o'clock")
    # The alert must come BEFORE contact: at the first alert, there is still >= ~1 s to go.
    first_t = runs[0][0]
    assert (7.0 - 2.0 * first_t) / 2.0 >= 1.0
    assert 0.0 <= approaching.approaching_margin(m, cfg) <= 1.0


def test_moving_away_never_alerts():
    cfg = _cfg()
    tr = _track(_walker(2, lambda t: 2.0 + 1.5 * t, lambda t: W / 2), cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_object_passing_far_to_the_side_does_not_alert():
    cfg = _cfg()
    # Getting closer, but stays at the right edge (about 2 o'clock) and is not drifting to the centre.
    tr = _track(_walker(3, lambda t: 7.0 - 2.0 * t, lambda t: W * 0.92, duration_s=2.5), cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_far_away_object_does_not_alert_even_if_approaching():
    cfg = _cfg()
    # 30 m -> 26 m: growing, but tiny in the frame (size << 25% of the frame height).
    tr = _track(_walker(4, lambda t: 30.0 - 1.0 * t, lambda t: W / 2), cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_slow_drift_is_not_an_alert():
    cfg = _cfg()
    # Closing at 0.2 m/s from 3 m: time to contact ~15 s, far above the 2.5 s rule.
    tr = _track(_walker(5, lambda t: 3.0 - 0.2 * t, lambda t: W / 2), cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_rule_is_off_outside_the_navigation_preset():
    cfg = utils.load_config()                       # campus defaults: fixed camera
    tr = _track(_walker(6, lambda t: 7.0 - 2.0 * t, lambda t: W / 2, duration_s=2.5), cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_clock_direction():
    assert approaching.clock_direction(0.0) == "12 o'clock"
    assert approaching.clock_direction(-1.0) == "10 o'clock"
    assert approaching.clock_direction(-0.5) == "11 o'clock"
    assert approaching.clock_direction(0.5) == "1 o'clock"
    assert approaching.clock_direction(1.0) == "2 o'clock"



def test_person_in_the_next_lane_who_will_pass_beside_you_is_not_flagged():
    cfg = _cfg()
    # Walking towards each other in parallel lanes 0.8 m apart: image x = centre + 0.8 * 600 / distance,
    # so they look central when far and drift right as they get close, then pass beside you.
    tr = _track(_walker(7, lambda t: 7.0 - 2.0 * t, lambda t: W / 2 + 0.8 * 600.0 / (7.0 - 2.0 * t),
                        duration_s=2.5), cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_raising_arms_grows_the_box_but_is_not_an_approach():
    cfg = _cfg()
    rows = []
    for f in range(int(3.0 * FPS)):
        t = f / FPS
        h = 250.0 * (1.0 + 0.6 * min(1.0, max(0.0, (t - 1.0) / 0.5)))  # height +60% in 0.5 s
        w = 100.0                                                     # width unchanged
        rows.append([f, t, 8, W / 2 - w / 2, 340 - h, W / 2 + w / 2, 340.0, 0.85, 0])
    tr = _track(rows, cfg)
    assert approaching.approaching_intervals(tr, cfg, (W, H)) == []


def test_one_alert_per_person_even_if_the_signal_flickers():
    cfg = _cfg()
    tr = _track(_walker(9, lambda t: 7.0 - 2.0 * t, lambda t: W / 2 + 25 * np.sin(9 * t), duration_s=2.5), cfg)
    runs = approaching.approaching_intervals(tr, cfg, (W, H))
    assert len(runs) == 1
