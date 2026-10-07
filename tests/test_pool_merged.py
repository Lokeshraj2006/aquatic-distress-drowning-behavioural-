"""Features merged from the team's behaviour_analysis.py prototype into pool_behaviour.py / distress.py."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import distress
import features
import pool_behaviour as pb
import utils

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synthetic_pool  # noqa: E402

FPS = 25.0


@pytest.fixture(scope="module")
def story(tmp_path_factory):
    out = tmp_path_factory.mktemp("merged")
    synthetic_pool.make_files("synthetic_pool.mp4", out)
    cfg = utils.load_config(scenario="pool")
    res = utils.read_json(out / "detections.json")
    tracks = features.build_tracks(res, cfg)
    zones = utils.load_zones(HERE / "synthetic_pool_zones.json", res["proc_size"])
    kps = utils.read_json(out / "keypoints.json")["keypoints"]
    return cfg, tracks, zones, kps, pb.behaviour_samples(tracks, cfg, zones, kps)


def _at(samples, pid, t):
    return min((s for s in samples if s["person_id"] == pid), key=lambda s: abs(s["timestamp"] - t))


def test_peak_finder_counts_a_sine():
    t = np.arange(0, 4, 0.1)
    x = np.sin(2 * np.pi * 1.25 * t)                       # 5 peaks in 4 s
    assert len(pb.find_peaks(x, 0.5)) == 5
    assert len(pb.find_peaks(0.02 * x, 0.5)) == 0          # tiny wiggle: no peaks


def test_activity_hint_and_stroking(story):
    _, _, _, _, s = story
    child, wall, lap = _at(s, 3, 21.0), _at(s, 2, 20.0), _at(s, 1, 20.0)
    assert child["activity"] == "floundering"
    assert wall["activity"] == "treading"                  # upright, calm, head steady (and at the wall)
    assert lap["arm_motion"] == "stroking" and lap["activity"] == "swimming"
    assert lap["progress_ratio"] > 0.8 and child["progress_ratio"] < 0.5
    assert child["arm_hz"] >= 0.6 and 0 < child["confidence"] <= 1


def _person(kind, seconds=8.0, hips=True):
    """One synthetic swimmer with full keypoints (hips visible) for the torso-angle path."""
    rows, kp = [], {}
    for i, f in enumerate(range(0, int(seconds * FPS), 2)):
        t = f / FPS
        k = [[0.0, 0.0, 0.0] for _ in range(17)]
        if kind == "upright" or (kind == "blip" and not (3.0 <= t < 4.0)):
            x, y = 300, 200
            k[5], k[6] = [x - 15, y - 60, .9], [x + 15, y - 60, .9]
            k[11], k[12] = [x - 12, y, .9 if hips else 0], [x + 12, y, .9 if hips else 0]
            box = [x - 30, y - 100, x + 30, y + 20]
        else:                                               # lying flat (or the blip)
            x, y = 300, 200
            k[5], k[6] = [x, y - 8, .9], [x, y + 8, .9]
            k[11], k[12] = [x - 60, y - 8, .9 if hips else 0], [x - 60, y + 8, .9 if hips else 0]
            box = [x - 80, y - 30, x + 30, y + 30]
        k[0] = [x, y - 85, .9]
        k[9], k[10] = [x - 30, y - 40, .9], [x + 30, y - 40, .9]
        kp[str(f)] = k
        rows.append([f, t, 1, *box, 0.9, 0])
    return rows, {"1": kp}


def _samples(rows, kps):
    cfg = utils.load_config(scenario="pool")
    res = {"fps": FPS, "stride": 2, "proc_size": [640, 360], "detections": rows}
    return pb.behaviour_samples(features.build_tracks(res, cfg), cfg, [], kps)


def test_torso_angle_gives_posture_when_hips_are_visible():
    s = _samples(*_person("upright"))
    assert {x["posture"] for x in s} == {"vertical"} and s[-1]["state_source"] == "torso_angle"
    assert s[-1]["torso_angle"] > 60
    flat = _samples(*_person("flat"))
    assert flat[-1]["posture"] == "horizontal" and flat[-1]["torso_angle"] < 30


def test_posture_hold_ignores_a_short_blip():
    s = _samples(*_person("blip"))                         # 1 s of "horizontal" inside an upright stretch
    assert all(x["posture"] == "vertical" for x in s)
    assert any(x["posture_raw"] == "horizontal" for x in s)   # it was seen, just not reported


def test_id_switch_guard_marks_and_stage3_ignores(story):
    cfg, tracks, zones, kps, _ = story
    tr = tracks[1]
    jumped = features.Track(track_id=9, t=tr.t.copy(), frame=tr.frame.copy(), box=tr.box.copy(), conf=tr.conf.copy(),
                            foot=tr.foot.copy(), height=tr.height.copy(), speed=tr.speed.copy(),
                            segment=tr.segment.copy(), frame_size=tr.frame_size)
    k = len(jumped.t) // 2
    jumped.box[k:] += np.array([300.0, 0, 300.0, 0])         # teleports 300 px: another person got this ID
    s = pb.behaviour_samples({9: jumped}, cfg, zones, None)
    bad = [x for x in s if not x["track_valid"]]
    assert bad and all(abs(x["timestamp"] - jumped.t[k]) <= cfg["pool"]["window_s"] + 0.6 for x in bad)
    res = distress.reason([dict(x, posture="vertical", displacement=0.0, arm_motion="repeated", head="unstable")
                           for x in bad], cfg)
    assert res["events"] == [] and res["risk"][9] == []       # invalid windows never count


def test_diagonal_counts_as_upright_only_when_going_nowhere():
    p = utils.load_config(scenario="pool")["pool"]
    base = {"posture": "diagonal", "movement": "low", "arm_motion": "calm", "head": "stable",
            "near_wall": False, "in_water": True}
    assert distress.sample_signals(dict(base, displacement=0.02), p)["vertical_posture"]
    assert not distress.sample_signals(dict(base, displacement=0.6), p)["vertical_posture"]


def test_signal_plots(story, tmp_path):
    _, _, _, _, s = story
    files = pb.plot_signals(s, tmp_path)
    assert files and all(Path(f).stat().st_size > 5000 for f in files)
