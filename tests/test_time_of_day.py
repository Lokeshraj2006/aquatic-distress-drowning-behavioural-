"""Time-of-day rules: --start-time + per-rule active_hours (e.g. the elderly preset's night-time exit)."""
import subprocess
import sys
from pathlib import Path

import pytest

import behaviors
import events
import features
import utils

ROOT = Path(__file__).resolve().parents[1]
FPS, STRIDE, W, H = 30.0, 2, 640, 360
ZONE = [{"name": "exit", "points": [(250, 250), (450, 250), (450, 350), (250, 350)]}]


def _final(start_time="", hours="21:00-06:00", scenario=None):
    """A person stands in the exit zone from t = 5 s to t = 9 s of the video."""
    rows = []
    for f in range(0, int(12 * FPS), STRIDE):
        t = f / FPS
        x = 350 if 5.0 <= t <= 9.0 else 100 + 10 * t
        rows.append([f, t, 1, x - 25, 150, x + 25, 300, 0.9, 0])
    res = {"video": "synthetic", "fps": FPS, "stride": STRIDE, "orig_size": [W, H], "proc_size": [W, H],
           "n_frames_read": int(12 * FPS), "duration_s": 12.0, "frames_processed": len(rows),
           "device": "cpu", "model": "yolo11n.pt", "detections": rows}
    cfg = utils.load_config(scenario=scenario)
    cfg["behaviors"]["zone_intrusion"]["active_hours"] = hours
    cfg["behaviors"]["loitering"]["enabled"] = False
    cfg["time_of_day"]["start_time"] = start_time
    tracks = features.build_tracks(res, cfg)
    return events.build_events(behaviors.detect_behaviors(tracks, ZONE, cfg), tracks, ZONE, cfg)


def test_clock_parsing_and_windows_that_wrap_midnight():
    assert utils.parse_clock("22:05") == 22 * 3600 + 5 * 60
    assert utils.parse_clock("06:00:30") == 6 * 3600 + 30
    assert utils.parse_clock("") is None
    for bad in ("25:00", "7pm", "12:60", "1:2:3:4"):
        with pytest.raises(ValueError):
            utils.parse_clock(bad)
    night = utils.parse_hours("21:00-06:00")
    assert utils.in_hours(utils.parse_clock("23:30"), night)
    assert utils.in_hours(utils.parse_clock("02:00"), night)
    assert not utils.in_hours(utils.parse_clock("12:00"), night)
    two = utils.parse_hours("08:00-12:00, 14:00-18:00")
    assert utils.in_hours(utils.parse_clock("15:00"), two) and not utils.in_hours(utils.parse_clock("13:00"), two)
    assert utils.fmt_clock(utils.parse_clock("23:59:50") + 20) == "00:00:10"


def test_event_outside_active_hours_is_ignored_and_listed():
    final = _final(start_time="20:58:00")             # zone entry at 20:58:05, before the 21:00 night window
    assert not [e for e in final["events"] if e["behavior"] == "zone_intrusion"]
    assert final["ignored_outside_hours"] and final["ignored_outside_hours"][0]["clock_start"] == "20:58:05"


def test_event_inside_active_hours_is_kept_with_clock_times():
    final = _final(start_time="23:10:00")
    (ev,) = [e for e in final["events"] if e["behavior"] == "zone_intrusion"]
    assert ev["clock_start"] == "23:10:05" and "active hours (21:00-06:00)" in ev["evidence"]
    assert ev["event_id"] == 1


def test_without_start_time_rules_run_all_day():
    final = _final(start_time="")
    assert [e for e in final["events"] if e["behavior"] == "zone_intrusion"]
    assert final["ignored_outside_hours"] == []


def test_bad_start_time_is_a_clean_error():
    proc = subprocess.run([sys.executable, "-B", "run.py", "--video", "samples/synthetic.mp4", "--start-time", "7pm"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "--start-time" in (proc.stdout + proc.stderr)


@pytest.mark.parametrize("name", ["elderly", "agriculture"])
def test_presets_with_night_rules_parse(name):
    cfg = utils.load_config(scenario=name)
    assert utils.parse_hours(cfg["behaviors"]["zone_intrusion"]["active_hours"])
