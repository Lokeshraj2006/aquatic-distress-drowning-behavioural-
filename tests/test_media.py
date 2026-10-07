"""Tests for the picture / video side: privacy blur, render_all, draw_frame and the highlight reel.

The end-to-end tests use the synthetic clip (samples/synthetic.mp4 + outputs/synthetic/detections.json,
made by tests/synthetic.py) with HAND-BUILT events, so every behaviour (loitering, zone intrusion,
running, fall, crowding, near miss) and an incident chain are drawn without running YOLO.
Generated files go to outputs/_test_media/ (wiped at the start of every test that uses it).
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import features
import highlights
import privacy
import render
import utils
from utils import BEHAVIOR_COLORS_BGR, deep_merge, load_config

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "samples" / "synthetic.mp4"
DETECTIONS = ROOT / "outputs" / "synthetic" / "detections.json"
ZONES = ROOT / "tests" / "synthetic_zones.json"
OUT_ROOT = ROOT / "outputs" / "_test_media"

needs_clip = pytest.mark.skipif(not (VIDEO.exists() and DETECTIONS.exists()),
                                reason="synthetic clip not generated (python tests/synthetic.py)")


def fresh_out(name: str) -> Path:
    """An empty output folder under outputs/_test_media/."""
    out = OUT_ROOT / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def video_seconds(path: Path) -> float:
    """Duration of a video file read back through OpenCV."""
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1.0)
    finally:
        cap.release()


# =========================================================================== privacy.anonymize

def test_privacy_is_enabled_follows_the_config():
    assert not privacy.is_enabled(load_config())
    assert privacy.is_enabled(deep_merge(load_config(), {"privacy": {"enabled": True}}))
    assert not privacy.is_enabled({})            # a config without a privacy section


@pytest.mark.parametrize("style", ["pixelate", "blur"])
def test_anonymize_changes_only_the_head_region(style):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    before = frame.copy()
    cfg = deep_merge(load_config(), {"privacy": {"enabled": True, "mode": "head", "head_fraction": 0.28,
                                                  "style": style}})
    out = privacy.anonymize(frame, [(100, 50, 160, 250)], cfg)

    assert out is frame                                      # works in place and returns the frame
    head = (slice(50, 106), slice(100, 160))                 # top 28 % of a 200 px tall box = 56 px
    assert np.abs(frame[head].astype(int) - before[head].astype(int)).mean() > 20     # head is destroyed
    untouched = np.ones(frame.shape[:2], dtype=bool)
    untouched[head] = False
    assert np.array_equal(frame[untouched], before[untouched])                       # nothing else changed


def test_anonymize_body_mode_and_lying_person_blur_the_whole_box():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    body_cfg = {"privacy": {"mode": "body"}}
    frame = base.copy()
    privacy.anonymize(frame, [(50, 20, 110, 180)], body_cfg)
    assert not np.array_equal(frame[170:180, 50:110], base[170:180, 50:110])         # feet area changed too
    assert np.array_equal(frame[:20], base[:20])

    frame = base.copy()                                      # head mode, box wider than tall = person lying down
    privacy.anonymize(frame, [(50, 100, 200, 160)], {"privacy": {"mode": "head"}})
    assert not np.array_equal(frame[150:160, 60:190], base[150:160, 60:190])         # the whole box is hidden


def test_anonymize_clamps_and_survives_odd_input():
    frame = np.full((120, 160, 3), 127, dtype=np.uint8)
    frame[:, ::2] = 0                                        # something to destroy
    cfg = load_config()
    for boxes in ([(-50, -20, 40, 60)],                      # sticks out on the left / top
                  [(140, 100, 400, 500)],                    # sticks out on the right / bottom
                  [(500, 500, 600, 700)],                    # entirely outside
                  [(10, 10, 10, 10)],                        # empty
                  [(float("nan"), 0, 5, 5)],                 # broken
                  [], None):
        out = privacy.anonymize(frame.copy(), boxes, cfg)
        assert out.shape == frame.shape
    tiny = np.zeros((6, 9, 3), dtype=np.uint8)               # any frame size
    assert privacy.anonymize(tiny, [(0, 0, 9, 6)], cfg).shape == (6, 9, 3)
    gray = np.random.default_rng(2).integers(0, 255, (80, 80), dtype=np.uint8)
    assert privacy.anonymize(gray, [(10, 10, 40, 70)], cfg).shape == (80, 80)


# =========================================================================== draw_frame on hand-made tracks

def make_pair_tracks(cfg, class_b=0):
    """Two people walking towards each other (A left to right, B right to left): near misses at ~1.3 s."""
    fps, stride, h = 30.0, 2, 150.0
    rows = []
    for frame in range(0, 91, stride):
        t = frame / fps
        for tid, x, y in ((1, 100 + 150 * t, 300.0), (2, 540 - 150 * t, 270.0)):
            rows.append([frame, t, tid, x - 30, y - h, x + 30, y, 0.9] + ([class_b] if tid == 2 else [0]))
    res = {"video": "x.mp4", "fps": fps, "stride": stride, "orig_size": [640, 360], "proc_size": [640, 360],
           "n_frames_read": 91, "duration_s": 91 / fps, "frames_processed": len(range(0, 91, stride)),
           "detections": rows}
    return res, features.build_tracks(res, cfg)


def pair_event(behavior, **extra):
    ev = {"event_id": 1, "entity_id": 1, "behavior": behavior, "start_s": 1.0, "end_s": 1.6, "start": "00:01",
          "end": "00:01", "duration_s": 0.6, "confidence": 0.8, "metrics": {}, "snapshot_time_s": 1.3,
          "entities": [1], "other_entity": None, "zone": None, "entity_name": None, "behavior_name": None,
          "severity": None, "snapshot": None, "speed_plot": None}
    ev.update(extra)
    return ev


def test_near_miss_line_is_red_when_near_and_amber_when_far():
    assert render._link_color(0.2, 0.6) == render.LINK_NEAR_BGR
    assert render._link_color(0.6, 0.6) == render.LINK_NEAR_BGR
    assert render._link_color(10.0, 0.6) == render.LINK_FAR_BGR
    mid = render._link_color(0.6 * (1 + render.LINK_FADE) / 2, 0.6)
    assert mid[0] == 0 and 0 < mid[1] < render.LINK_FAR_BGR[1]

    cfg = load_config()
    res, tracks = make_pair_tracks(cfg)
    ev = pair_event("near_miss", entities=[1, 2], other_entity=2, entity_name="Person #1")
    ctx = render.make_context(res, tracks, {"events": [ev]}, [], cfg)
    frame_idx = 40                                           # t = 1.33 s: the two feet are ~0.4 body-heights apart
    img = render.draw_frame(np.zeros((360, 640, 3), np.uint8), frame_idx, frame_idx / 30.0, ctx)
    ia, ib = dict(ctx.lookup[frame_idx])[1], dict(ctx.lookup[frame_idx])[2]
    pa, pb = np.array(tracks[1].foot[ia]), np.array(tracks[2].foot[ib])
    x, y = (pa + 0.3 * (pb - pa)).round().astype(int)         # on the line, away from the distance label
    patch = img[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).astype(int)
    b, g, r = patch[patch[:, 2].argmax()]
    assert r > 200 and g < 80 and b < 80, f"line pixel {b, g, r} should be red"


def test_draw_frame_new_behaviours_use_their_colours():
    cfg = load_config()
    res, tracks = make_pair_tracks(cfg)
    frame_idx = 20
    t = frame_idx / 30.0
    base = render.draw_frame(np.zeros((360, 640, 3), np.uint8), frame_idx, t, render.make_context(res, tracks, {"events": []}, [], cfg))

    # fall: the person's box is drawn in the fall colour and the label says what scenario wording says
    ev = pair_event("fall", start_s=0.2, end_s=1.2)
    ctx = render.make_context(res, tracks, {"events": [ev]}, [], cfg)
    img = render.draw_frame(np.zeros((360, 640, 3), np.uint8), frame_idx, t, ctx)
    i = dict(ctx.lookup[frame_idx])[1]
    x1, y1, x2, y2 = (int(round(v)) for v in tracks[1].box[i])
    edge = img[(y1 + y2) // 2 - 1:(y1 + y2) // 2 + 2, x1 - 1:x1 + 2].reshape(-1, 3).astype(int)
    assert np.abs(edge - np.array(BEHAVIOR_COLORS_BGR["fall"])).sum(axis=1).min() < 40
    assert not np.array_equal(img, base)

    # crowding: the zone gets a strong teal fill; without the event it is only lightly tinted
    zone = {"name": "hall", "points": [(0.0, 0.0), (640.0, 0.0), (640.0, 360.0), (0.0, 360.0)]}
    quiet = render.draw_frame(np.zeros((360, 640, 3), np.uint8), frame_idx, t,
                              render.make_context(res, tracks, {"events": []}, [zone], cfg))
    crowd = pair_event("crowding", entity_id=None, entities=[1, 2], zone="hall", entity_name="Group of 2",
                       start_s=0.2, end_s=1.2)
    loud = render.draw_frame(np.zeros((360, 640, 3), np.uint8), frame_idx, t,
                             render.make_context(res, tracks, {"events": [crowd]}, [zone], cfg))
    patch = (slice(100, 140), slice(300, 340))               # empty floor inside the zone, away from people / labels
    assert loud[patch][..., 1].mean() > quiet[patch][..., 1].mean() + 25        # teal has a strong green part


def test_scenario_wording_reaches_the_overlays():
    """behavior_name / entity_name come from the scenario, not from hardcoded strings."""
    cfg = deep_merge(load_config(), {"scenario": {"entity_word": "Animal", "behavior_names": {"fall": "Lying down"}}})
    res, tracks = make_pair_tracks(cfg)
    ev = pair_event("fall", start_s=0.2, end_s=1.2)
    assert render._beh_label(cfg, "fall") == "LYING DOWN"
    assert render._who(ev, cfg, tracks) == "Animal #1"
    assert render._who(pair_event("crowding", entity_id=None, entities=[1, 2, 3]), cfg, tracks) == "Group of 3"


def test_pack_rows_stacks_overlapping_spans():
    rows, n = render._pack_rows([(0, 10), (5, 12), (11, 14), (20, 21)])
    assert n == 2 and rows[0] != rows[1]                    # the overlapping pair is stacked
    assert rows[2] == rows[0] and rows[3] == rows[0]        # later spans fall back into the first sub-row
    assert render._pack_rows([]) == ([], 1)


# =========================================================================== end to end on the synthetic clip

def build_scene(privacy_on: bool):
    """tracks + zones + a hand-built `final` (all six behaviours and one incident chain) for the synthetic clip."""
    cfg = deep_merge(load_config(), {"privacy": {"enabled": privacy_on},
                                     "highlights": {"pad_s": 1.5, "max_segment_s": 8.0, "title_s": 2.0}})
    res = utils.read_json(DETECTIONS)
    tracks = features.build_tracks(res, cfg)
    zones = utils.load_zones(ZONES, res["proc_size"])

    def ev(i, beh, a, b, entity, **kw):
        base = {"event_id": i, "entity_id": entity, "behavior": beh, "start_s": a, "end_s": b,
                "start": utils.fmt_time(a), "end": utils.fmt_time(b), "duration_s": round(b - a, 2),
                "confidence": 0.8, "confidence_parts": {}, "evidence": "test", "metrics": {},
                "zone": None, "baseline": None, "snapshot_time_s": 0.5 * (a + b), "snapshot": None,
                "speed_plot": None, "verified": None, "pose": None, "severity": "medium",
                "entities": [entity] if entity is not None else [], "other_entity": None,
                "entity_name": None, "behavior_name": None}
        base.update(kw)
        return base

    events = [
        ev(1, "crowding", 8.0, 8.9, None, entities=[1, 2, 3], zone="whole frame", entity_name="Group of 3",
           severity="medium", snapshot_time_s=8.4,
           metrics={"zone": "whole frame", "max_count": 3, "mean_count": 3.0, "min_count": 3, "peak_time_s": 8.4}),
        ev(2, "near_miss", 9.0, 9.6, 2, entities=[2, 3], other_entity=3, severity="high", snapshot_time_s=9.2,
           metrics={"min_distance_bh": 1.0, "peak_time_s": 9.2, "contact": False, "other_id": 3}),
        ev(3, "loitering", 14.5, 16.0, 4, severity="low"),
        ev(4, "zone_intrusion", 15.0, 17.0, 4, zone="restricted", severity="medium"),
        ev(5, "running", 18.0, 19.0, 4, severity="medium"),
        ev(6, "fall", 22.5, 28.0, 5, severity="high",
           metrics={"max_aspect": 1.8, "fell": True, "peak_time_s": 23.0}),
    ]
    chain = {"chain_id": 1, "entity_id": 4, "event_ids": [3, 4, 5], "pattern": "loitering -> zone_intrusion",
             "title": "Waited nearby, then entered the restricted zone", "start_s": 14.5, "end_s": 19.0,
             "start": "00:14", "end": "00:19", "severity": "high", "story": "test story"}
    final = {"events": events, "near_misses": [], "unusual_tracks": [], "baseline": {}, "incidents": [chain]}
    return cfg, res, tracks, zones, final


# What the hand-built scene should produce, worked out by hand (pad 1.5 s, max segment 8 s, title 2 s):
#   segment 1: crowding 8.0-8.9 and near miss 9.0-9.6 overlap once padded      -> [6.5, 11.1]   4.6 s
#   segment 2: the chain 14.5-19.0                                              -> [13.0, 20.5]  7.5 s
#   segment 3: the fall 22.5-28.0 -> [21.0, 29.5] = 8.5 s > 8 -> fast-forward x2 -> 4.25 s
EXPECTED_REEL_S = 2.0 + 4.6 + 7.5 + 4.25


@needs_clip
def test_plan_segments_pads_merges_and_fast_forwards():
    cfg, res, tracks, zones, final = build_scene(False)
    items, segments = highlights.plan_segments(final, res, cfg, tracks)
    assert len(items) == 4                                    # the chain + crowding + near miss + fall
    assert [(round(s["start_s"], 2), round(s["end_s"], 2), s["ff"]) for s in segments] == \
        [(6.5, 11.1, 1), (13.0, 20.5, 1), (21.0, 29.5, 2)]
    chain_item = next(it for it in items if it["title"])
    assert chain_item["event_ids"] == [3, 4, 5] and chain_item["severity"] == "high"
    assert chain_item["behaviors"] == ["LOITERING", "ZONE INTRUSION", "RUNNING"]
    assert chain_item["who"] == "Person #4"


@needs_clip
@pytest.mark.parametrize("privacy_on", [False, True])
def test_render_all_and_highlight_reel_on_the_synthetic_clip(privacy_on):
    cfg, res, tracks, zones, final = build_scene(privacy_on)
    out = fresh_out("privacy_on" if privacy_on else "privacy_off")

    render.render_all(str(VIDEO), res, tracks, final, zones, cfg, out)
    name = highlights.make_highlight_reel(str(VIDEO), res, tracks, final, zones, cfg, out)

    for rel in ("annotated.mp4", "heatmap.jpg", "timeline.png", "highlights.mp4"):
        assert (out / rel).stat().st_size > 1000, rel
    assert name == "highlights.mp4" and final["highlight_reel"] == "highlights.mp4"
    for e in final["events"]:                                 # one snapshot and one plot per event, all six kinds
        assert e["snapshot"] and (out / e["snapshot"]).stat().st_size > 1000, e["behavior"]
        assert e["speed_plot"] and (out / e["speed_plot"]).stat().st_size > 1000, e["behavior"]
    assert not (out / "annotated_tmp.mp4").exists() and not (out / "highlights_tmp.mp4").exists()

    # the reel is the title card plus the (fast-forwarded) segments, with H.264-friendly even sizes
    assert abs(video_seconds(out / "highlights.mp4") - EXPECTED_REEL_S) < 1.0
    cap = cv2.VideoCapture(str(out / "highlights.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) % 2 == 0 and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) % 2 == 0
    cap.release()
    # the annotated video keeps one frame per processed frame
    assert abs(video_seconds(out / "annotated.mp4") - res["duration_s"]) < 1.0


@needs_clip
def test_privacy_changes_only_head_pixels_in_the_snapshot():
    """Same event rendered with privacy off and on: the pictures differ only around the heads."""
    shots = {}
    for on in (False, True):
        cfg, res, tracks, zones, final = build_scene(on)
        out = fresh_out(f"snap_privacy_{on}")
        render.render_all(str(VIDEO), res, tracks, final, zones, deep_merge(cfg, {"output": {"annotated_video": False}}), out)
        crowd = next(e for e in final["events"] if e["behavior"] == "crowding")
        shots[on] = cv2.imread(str(out / crowd["snapshot"])).astype(int)
    diff = np.abs(shots[True] - shots[False]).max(axis=2)
    assert diff.max() > 30                                    # privacy did blur something

    k = shots[True].shape[1] / res["proc_size"][0]            # snapshot is k x the processing size
    frame_idx = int(round(8.4 * res["fps"] / res["stride"])) * res["stride"]
    allowed = np.zeros(diff.shape, dtype=bool)
    margin = 20                                               # JPEG blocks smear a change a little
    for tr in tracks.values():
        for f, box in zip(tr.frame, tr.box):
            if f == frame_idx:
                x1, y1, x2, y2 = box * k
                y_head = y1 + 0.28 * (y2 - y1)
                allowed[max(0, int(y1) - margin):int(y_head) + margin, max(0, int(x1) - margin):int(x2) + margin] = True
    assert diff[~allowed].max() <= 25, "privacy mode changed pixels outside the head regions"


@needs_clip
def test_render_and_reel_with_nothing_to_show():
    cfg = load_config()
    res = utils.read_json(DETECTIONS)
    res = dict(res, detections=[])                            # no people at all
    out = fresh_out("empty")
    final = {"events": [], "near_misses": [], "unusual_tracks": [], "baseline": {}, "incidents": []}
    render.render_all(str(VIDEO), res, {}, final, [], cfg, out)
    name = highlights.make_highlight_reel(str(VIDEO), res, {}, final, [], cfg, out)
    for rel in ("annotated.mp4", "heatmap.jpg", "timeline.png"):
        assert (out / rel).stat().st_size > 500, rel
    assert not (out / "snapshots").exists() or not list((out / "snapshots").iterdir())
    assert name == "highlights.mp4" and final["highlight_reel"] == "highlights.mp4"
    assert abs(video_seconds(out / "highlights.mp4") - cfg["highlights"]["title_s"]) < 0.5   # just the "No incidents" card


@needs_clip
def test_reel_can_be_switched_off_and_never_raises():
    cfg = deep_merge(load_config(), {"highlights": {"enabled": False}})
    out = fresh_out("reel_off")
    final = {"events": []}
    assert highlights.make_highlight_reel(str(VIDEO), {}, {}, final, [], cfg, out) is None
    assert final["highlight_reel"] is None
    # an unreadable video with incidents: a warning, not an exception
    cfg2, res, tracks, zones, final2 = build_scene(False)
    assert highlights.make_highlight_reel(str(out / "missing.mp4"), res, tracks, final2, zones, cfg2, out) is None
    assert final2["highlight_reel"] is None


@needs_clip
def test_events_with_missing_tracks_do_not_break_rendering():
    cfg, res, tracks, zones, final = build_scene(False)
    final = copy.deepcopy(final)
    final["events"][1]["other_entity"] = 99                   # a near miss with a track that does not exist
    final["events"][1]["entities"] = [2, 99]
    final["events"][5]["entity_id"] = 98                      # an event for a vanished person
    out = fresh_out("missing_tracks")
    render.render_all(str(VIDEO), res, tracks, final, zones, deep_merge(cfg, {"output": {"annotated_video": False}}), out)
    assert (out / "timeline.png").exists()
    assert final["events"][1]["speed_plot"] is None            # nothing to plot, but no crash
