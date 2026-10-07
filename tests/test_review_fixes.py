"""Regression tests for the issues found in the adversarial review (one test per fix)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import behaviors
import events
import features
import utils

ROOT = Path(__file__).resolve().parents[1]
FPS, STRIDE, W, H = 30.0, 2, 640, 360


def _result(rows):
    return {"video": "synthetic", "fps": FPS, "stride": STRIDE, "orig_size": [W, H], "proc_size": [W, H],
            "n_frames_read": int(max(r[0] for r in rows)) + 1, "duration_s": max(r[1] for r in rows),
            "frames_processed": len(rows), "device": "cpu", "model": "yolo11n.pt", "detections": rows}


def _run(rows, cfg=None, zones=()):
    cfg = cfg or utils.load_config()
    tracks = features.build_tracks(_result(rows), cfg)
    final = events.build_events(behaviors.detect_behaviors(tracks, list(zones), cfg), tracks, list(zones), cfg)
    return final, tracks


def test_short_dropout_does_not_hide_loitering():
    # Standing still 0-14 s, but YOLO misses the person for 1.2 s at t=5 (e.g. hidden by a pillar).
    rows = []
    for f in range(0, int(14 * FPS), STRIDE):
        t = f / FPS
        if 5.0 <= t < 6.2:
            continue
        jitter = 1.5 * np.sin(f)
        rows.append([f, t, 1, 300 + jitter, 150, 350 + jitter, 300, 0.85, 0])
    final, _ = _run(rows)
    loiter = [e for e in final["events"] if e["behavior"] == "loitering"]
    assert len(loiter) == 1 and loiter[0]["duration_s"] >= 10.0


def test_long_absence_or_moving_away_still_splits_loitering():
    # Gone for 5 s and back somewhere else: two short stays, no loitering event.
    rows = []
    for f in range(0, int(16 * FPS), STRIDE):
        t = f / FPS
        if 6.0 <= t < 11.0:
            continue
        x = 300 if t < 6 else 500
        rows.append([f, t, 1, x, 150, x + 50, 300, 0.85, 0])
    final, _ = _run(rows)
    assert not [e for e in final["events"] if e["behavior"] == "loitering"]


def test_near_miss_approach_time_ignores_standing_still():
    # A person stands 2 body-heights from a parked car for 20 s, then dashes past it in about 0.7 s.
    rows = []
    for f in range(0, int(22 * FPS), STRIDE):
        t = f / FPS
        rows.append([f, t, 2, 400, 200, 520, 300, 0.9, 2])                     # parked car, foot (460, 300)
        x = 160.0 if t < 20.0 else min(560.0, 160.0 + 3.0 * 150.0 * (t - 20.0))  # 3 bh/s dash
        rows.append([f, t, 1, x - 25, 150, x + 25, 300, 0.85, 0])              # person, 150 px tall
    final, _ = _run(rows)
    nm = [e for e in final["events"] if e["behavior"] == "near_miss"]
    assert nm, "expected a near miss"
    assert nm[0]["metrics"]["approach_s"] < 2.0
    assert "approached for 2" not in nm[0]["evidence"]


def test_fmt_time_does_not_drop_a_second_on_frame_rounding():
    assert utils.fmt_time(898 / 59.94) == "00:15"
    assert utils.fmt_time(41.9) == "00:41"


@pytest.mark.parametrize("bad", [
    "not json at all",
    json.dumps({"image_size": [0, 0], "zones": [{"name": "z", "points": [[1, 1], [5, 1], [5, 5]]}]}),
    json.dumps({"zones": [{"name": "z", "points": [[1, 1, 7], [5, 1, 7], [5, 5, 7]]}]}),
])
def test_invalid_zones_file_is_a_clean_error(tmp_path, bad):
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(ROOT / "outputs" / "synthetic" / "detections.json", out / "detections.json")
    zones = tmp_path / "zones.json"
    zones.write_text(bad, encoding="utf-8")
    proc = subprocess.run([sys.executable, "-B", "run.py", "--video", "samples/synthetic.mp4", "--zones", str(zones),
                           "--out", str(out), "--reuse", "--no-video"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "invalid" in (proc.stdout + proc.stderr) and "Traceback" not in proc.stderr


def test_missing_config_file_is_a_clean_error():
    proc = subprocess.run([sys.executable, "-B", "run.py", "--video", "samples/synthetic.mp4",
                           "--config", "no_such_config.yaml"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "config file not found" in (proc.stdout + proc.stderr)


def test_reuse_refuses_a_cache_that_covers_less_than_asked():
    import run
    det = utils.read_json(ROOT / "outputs" / "synthetic" / "detections.json")
    cfg = utils.load_config()
    cfg["video"]["max_seconds"] = 20.0
    short = dict(det, duration_s=10.0, n_frames_read=250,
                 detections=[r for r in det["detections"] if r[1] < 10.0])
    path = ROOT / "outputs" / "synthetic" / "_short_detections_test.json"
    utils.write_json(path, short)
    try:
        res, reason = run.load_reusable_detections(path, Path("samples/synthetic.mp4"), cfg, n_frames_container=750)
    finally:
        path.unlink(missing_ok=True)
    assert res is None and "less than" in reason
