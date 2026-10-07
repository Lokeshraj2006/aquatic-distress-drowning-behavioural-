"""Synthetic POOL clip with ground truth: the team's story, acted out by simple drawn swimmers.

    python tests/synthetic_pool.py
    python run.py --video samples/synthetic_pool.mp4 --zones tests/synthetic_pool_zones.json \\
                  --scenario pool --out outputs/synthetic_pool --reuse

It writes the video, the detections (what YOLO + ByteTrack would give), the pose keypoints (what the
pose model would give) and the ground-truth labels, so stages 2-4 can be tested end to end without
real distress footage (which must never be recorded for real; stage it with a lifeguard present).

Scene: 1280x720, 25 fps, 40 s, camera high on the wall looking over the pool.
  #1 lap swimmer, horizontal, arms moving a lot (freestyle)        -> must stay NORMAL
  #2 teenager resting at the right wall, upright, calm             -> must stay NORMAL (resting)
  #3 child: swims (0-10 s), slows (10-14), turns upright (14), arms pressing down and head bobbing
     (16-28), head goes under (28-30), lost from view (30)         -> DISTRESS, then SUBMERSION
  #4 swimmer who dives under for 2.5 s at 18 s and comes back up   -> must stay NORMAL (dive)
  #5 lifeguard standing on the deck (outside the water)            -> ignored
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

W, H, FPS, DURATION_S, STRIDE, POSE_EVERY = 1280, 720, 25.0, 40.0, 2, 4
K = 0.5                                   # processing size 640x360
WATER_TOP = 250
ZONES = [
    {"name": "water", "points": [(0, WATER_TOP), (W, WATER_TOP), (W, H), (0, H)]},
    {"name": "Deep End", "points": [(640, WATER_TOP), (W, WATER_TOP), (W, H), (640, H)]},
    {"name": "Shallow End", "points": [(0, WATER_TOP), (640, WATER_TOP), (640, H), (0, H)]},
    {"name": "Pool edge (right wall)", "points": [(1160, WATER_TOP), (W, WATER_TOP), (W, H), (1160, H)]},
]


def _lerp(t, t0, t1, a, b):
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return a + (b - a) * u


def scene(t: float) -> list[dict]:
    """Every visible object at time t: id, box (original px), kind, and what its arms / head do."""
    objs = []
    # 1: lap swimmer, 200x70, laps at 150 px/s between x=150 and x=1050
    period = 2 * 900 / 150.0
    ph = (t % period) / period
    x = 150 + 900 * (2 * ph if ph < 0.5 else 2 - 2 * ph)
    objs.append({"id": 1, "box": (x - 100, 385, x + 100, 455), "kind": "swim", "arms": (0.30, 1.0), "head": 0.01})
    # 2: resting at the right wall (inside the edge zone), head and shoulders above the water
    objs.append({"id": 2, "box": (1180, 355, 1250, 445), "kind": "upright", "arms": (0.005, 0.3), "head": 0.01})
    # 3: the child
    if t < 10:
        x = _lerp(t, 0, 10, 250, 850)
        objs.append({"id": 3, "box": (x - 80, 470, x + 80, 530), "kind": "swim", "arms": (0.25, 1.0), "head": 0.01})
    elif t < 14:
        x = _lerp(t, 10, 14, 850, 890)
        objs.append({"id": 3, "box": (x - 80, 470, x + 80, 530), "kind": "swim", "arms": (0.06, 0.5), "head": 0.02})
    elif t < 28:
        jit = 2 * math.sin(3 * t)
        distress = t >= 16
        objs.append({"id": 3, "box": (860 + jit, 450, 930 + jit, 550), "kind": "upright",
                     "arms": (0.20, 1.3) if distress else (0.03, 0.5), "head": 0.12 if distress else 0.02})
    elif t < 30:
        s = _lerp(t, 28, 30, 1.0, 0.6)
        objs.append({"id": 3, "box": (895 - 30 * s, 500 - 30 * s, 895 + 30 * s, 500 + 30 * s), "kind": "head",
                     "arms": (0.20, 1.3), "head": None})
    # 4: diver: swims right at 200 px/s, under water 18.0-20.5 s, then swims on
    if t < 18 or 20.5 <= t < 30:
        x = 100 + 200 * (t if t < 18 else t - 2.5)
        if x < W - 100:
            objs.append({"id": 4, "box": (x - 90, 600, x + 90, 660), "kind": "swim", "arms": (0.30, 1.1), "head": 0.01})
    # 5: lifeguard on the deck
    objs.append({"id": 5, "box": (300, 60, 360, 230), "kind": "deck", "arms": (0.0, 0.0), "head": 0.0})
    return objs


def keypoints_for(o: dict, t: float) -> list:
    """17x3 COCO keypoints (processing px) for one object: nose, eyes, shoulders, wrists only."""
    x1, y1, x2, y2 = (v * K for v in o["box"])
    w, h = x2 - x1, y2 - y1
    scale = max(w, h)
    cx = (x1 + x2) / 2
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    if o["head"] is None:                                     # head under the water: no face keypoints
        amp, hz = o["arms"]
        wy = y1 + 0.3 * h - amp * scale * math.sin(2 * math.pi * hz * t)
        kp[9], kp[10] = [cx - 0.3 * w, wy, 0.6], [cx + 0.3 * w, wy, 0.6]
        return kp
    head_y = (y1 + 0.18 * h if o["kind"] != "swim" else y1 + 0.4 * h) + o["head"] * scale * math.sin(2 * math.pi * 1.3 * t)
    head_x = cx if o["kind"] != "swim" else x2 - 0.12 * w
    kp[0] = [head_x, head_y, 0.9]
    kp[1], kp[2] = [head_x - 3, head_y - 2, 0.9], [head_x + 3, head_y - 2, 0.9]
    sh_y = y1 + 0.5 * h
    kp[5], kp[6] = [cx - 0.3 * w, sh_y, 0.85], [cx + 0.3 * w, sh_y, 0.85]
    amp, hz = o["arms"]
    wy = sh_y - (0.1 + amp * math.sin(2 * math.pi * hz * t)) * scale
    kp[9], kp[10] = [cx - 0.35 * w, wy, 0.8], [cx + 0.35 * w, wy, 0.8]
    return [[round(a, 1), round(b, 1), c] for a, b, c in kp]


def make_files(video_name: str, out_dir: Path, seed: int = 5) -> None:
    """detections.json + keypoints.json (in the format run.py --reuse expects) + zones + labels."""
    rng = np.random.default_rng(seed)
    rows, kps = [], {}
    n_frames = int(FPS * DURATION_S)
    for f in range(0, n_frames, STRIDE):
        t = f / FPS
        for o in scene(t):
            j = rng.normal(0, 0.4, 4)
            box = [o["box"][i] * K + j[i] for i in range(4)]
            rows.append([f, round(t, 4), o["id"], *[round(v, 2) for v in box], round(0.8 + 0.1 * rng.random(), 3), 0])
            if f % POSE_EVERY == 0:
                kps.setdefault(str(o["id"]), {})[str(f)] = keypoints_for(o, t)
    res = {"video": f"samples/{video_name}", "fps": FPS, "stride": STRIDE, "orig_size": [W, H],
           "proc_size": [int(W * K), int(H * K)], "n_frames_read": n_frames, "duration_s": n_frames / FPS,
           "frames_processed": len(range(0, n_frames, STRIDE)), "device": "synthetic", "model": "yolo11n.pt",
           "classes": [0], "detections": rows}
    utils.write_json(out_dir / "detections.json", res)
    key = {"video": video_name, "stride": STRIDE, "pose_hz": 6.0, "proc_size": [int(W * K), int(H * K)]}
    utils.write_json(out_dir / "keypoints.json", {"key": key, "keypoints": kps})
    utils.save_zones(HERE / "synthetic_pool_zones.json", ZONES, (W, H))
    (HERE / "synthetic_pool_labels.csv").write_text(
        "clip,entity_description,behavior,start_s,end_s\n"
        f"{video_name},child #3 (distress: upright + arms pressing + no progress),aquatic_distress,16.0,28.0\n"
        f"{video_name},child #3 head goes under,submersion,28.0,30.0\n", encoding="utf-8")


def write_video(path: Path) -> None:
    """Draw the pool and the swimmers."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    base = np.zeros((H, W, 3), np.uint8)
    base[:WATER_TOP] = (190, 195, 200)
    base[WATER_TOP:] = (200, 140, 40)
    for y in range(WATER_TOP + 90, H, 110):
        cv2.line(base, (0, y), (W, y), (230, 230, 230), 2)               # lane ropes
    cv2.putText(base, "DEEP END", (980, WATER_TOP + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(base, "SHALLOW END", (40, WATER_TOP + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    skin, cap = (150, 180, 230), {1: (0, 0, 200), 2: (0, 160, 0), 3: (0, 200, 255), 4: (200, 0, 200), 5: (40, 40, 40)}
    for f in range(int(FPS * DURATION_S)):
        img = base.copy()
        t = f / FPS
        for o in scene(t):
            x1, y1, x2, y2 = (int(v) for v in o["box"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if o["kind"] == "swim":
                cv2.ellipse(img, (cx, cy), ((x2 - x1) // 2, (y2 - y1) // 3), 0, 0, 360, (120, 90, 60), -1)
                cv2.circle(img, (x2 - 18, cy), 16, skin, -1)
                cv2.circle(img, (x2 - 18, cy - 6), 16, cap[o["id"]], -1)
                arm = int(25 * math.sin(2 * math.pi * o["arms"][1] * t))
                cv2.line(img, (cx, cy), (cx + 20, cy - 25 + arm), skin, 6)
            elif o["kind"] == "upright":
                cv2.rectangle(img, (x1 + 8, cy), (x2 - 8, y2), (120, 90, 60), -1)
                hy = y1 + 20 + int(o["head"] * 100 * math.sin(2 * math.pi * 1.3 * t))
                cv2.circle(img, (cx, hy), 20, skin, -1)
                cv2.circle(img, (cx, hy - 8), 20, cap[o["id"]], -1)
                a = int(o["arms"][0] * 120 * math.sin(2 * math.pi * o["arms"][1] * t))
                cv2.line(img, (x1 + 8, cy), (x1 - 15, cy - 20 + a), skin, 6)
                cv2.line(img, (x2 - 8, cy), (x2 + 15, cy - 20 + a), skin, 6)
            elif o["kind"] == "head":
                cv2.circle(img, (cx, cy), max(6, (x2 - x1) // 2), cap[o["id"]], -1)
                cv2.ellipse(img, (cx, cy + 10), ((x2 - x1), 10), 0, 0, 360, (240, 220, 200), 2)  # ripple
            else:
                cv2.rectangle(img, (x1 + 15, y1 + 40), (x2 - 15, y2), (40, 40, 40), -1)
                cv2.circle(img, (cx, y1 + 22), 20, skin, -1)
            cv2.putText(img, "SYNTHETIC TEST SCENE (staged story)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (30, 30, 30), 2)
        writer.write(img)
    writer.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic pool clip + detections + keypoints + labels")
    ap.add_argument("--out-video", default=str(ROOT / "samples" / "synthetic_pool.mp4"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "synthetic_pool"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)
    video = Path(args.out_video)
    video.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        write_video(video)
    make_files(video.name, Path(args.out_dir))
    print(f"wrote {video}, {args.out_dir}/detections.json + keypoints.json, tests/synthetic_pool_zones.json, "
          f"tests/synthetic_pool_labels.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
