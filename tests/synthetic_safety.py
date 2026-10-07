"""Synthetic WORKPLACE-SAFETY clip: a near miss (cyclist vs pedestrian) and a fall.

Real footage of near misses and falls is hard to get, so this script draws a simple scene and writes
the matching detections (what YOLO + ByteTrack would output), like tests/synthetic.py does for the
corridor clip. It lets anyone run the headline features end to end without a GPU or a real accident:

    python tests/synthetic_safety.py
    python run.py --video samples/synthetic_safety.mp4 --scenario workplace --out outputs/synthetic_safety --reuse

Scene (1280x720, 25 fps, 30 s):
  #1 pedestrian walks left -> right across the floor (normal walking, ~0.8 body-heights/s)
  #2 cyclist rides right -> left fast and passes within ~0.1 body-heights of #1 at about 6.3 s -> NEAR MISS
  #3 worker walks in, falls at ~18 s and stays down ~5 s                                       -> FALL
  #4 a second pedestrian walks calmly in the background (control: must not be flagged)
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

W, H, FPS, DURATION_S, STRIDE = 1280, 720, 25.0, 30.0, 2
PROC_W, PROC_H = 640, 360
K = PROC_W / W
CLASSES = [0, 1, 2, 3, 5, 7]               # the workplace preset's classes (so --reuse accepts the file)


def _lerp(t, t0, t1, a, b):
    """Linear interpolation of a -> b over [t0, t1] (clamped)."""
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return a + (b - a) * u


def boxes_at(t: float) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """(track id, class, box in ORIGINAL pixels) for every visible object at time t."""
    out = []
    # 1: pedestrian, 180 px tall, foot y 600, walks 100 -> 1000 px over 0..6.25 s then on (0.8 bh/s = 144 px/s)
    if t <= 12.0:
        x = 100 + 144.0 * t
        out.append((1, 0, (x - 30, 420, x + 30, 600)))
    # 2: cyclist, 150 px tall x 190 wide, foot y 600, rides from x=1500 at 3.4 bh/s (612 px/s) from t=5.5 s
    if 5.5 <= t <= 10.0:
        x = 1500 - 612.0 * (t - 5.5)
        out.append((2, 1, (x - 95, 450, x + 95, 600)))
    # 3: worker walks in 13..18 s, falls over 0.6 s, lies 18.6..24 s, then gets up and leaves
    if 13.0 <= t <= 27.0:
        if t < 18.0:
            x = _lerp(t, 13.0, 18.0, 200, 560)
            out.append((3, 0, (x - 30, 400, x + 30, 580)))
        elif t < 18.6:                       # falling: box gets wider and lower
            u = (t - 18.0) / 0.6
            w2, h = 30 + 60 * u, 180 - 120 * u
            out.append((3, 0, (560 - w2, 580 - h, 560 + w2, 580)))
        elif t < 24.0:                       # lying on the floor
            out.append((3, 0, (470, 520, 650, 580)))
        else:                                # gets up and walks off
            x = _lerp(t, 24.0, 27.0, 560, 300)
            out.append((3, 0, (x - 30, 400, x + 30, 580)))
    # 4: background walker, smaller (farther), calm
    if 2.0 <= t <= 28.0:
        x = _lerp(t, 2.0, 28.0, 1150, 250)
        out.append((4, 0, (x - 18, 330, x + 18, 440)))
    return out


def make_tracker_result(video_name: str, seed: int = 3) -> dict:
    """TrackerResult dict (DESIGN 3.1) with small box jitter, boxes in processing pixels."""
    rng = np.random.default_rng(seed)
    rows = []
    n_frames = int(FPS * DURATION_S)
    for f in range(0, n_frames, STRIDE):
        t = f / FPS
        for tid, cls, (x1, y1, x2, y2) in boxes_at(t):
            j = rng.normal(0, 1.0, 4)
            box = [max(0.0, x1 * K + j[0]), max(0.0, y1 * K + j[1]), min(PROC_W - 1.0, x2 * K + j[2]),
                   min(PROC_H - 1.0, y2 * K + j[3])]
            if box[2] - box[0] < 4:              # off screen
                continue
            rows.append([f, round(t, 4), tid, *[round(v, 2) for v in box], round(0.82 + 0.1 * rng.random(), 3), cls])
    return {"video": f"samples/{video_name}", "fps": FPS, "stride": STRIDE, "orig_size": [W, H],
            "proc_size": [PROC_W, PROC_H], "n_frames_read": n_frames, "duration_s": n_frames / FPS,
            "frames_processed": len(range(0, n_frames, STRIDE)), "device": "synthetic", "model": "yolo11n.pt",
            "classes": CLASSES, "detections": rows}


def write_video(path: Path) -> None:
    """Draw the scene with OpenCV: floor, walkers (body + head), a cyclist (two wheels), a fallen worker."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    base = np.full((H, W, 3), (70, 70, 72), np.uint8)
    base[:380] = (150, 160, 170)
    for x in range(0, W, 160):
        cv2.line(base, (x, 380), (x - 300, H), (90, 90, 92), 2)
    cv2.rectangle(base, (380, 360), (760, 600), (0, 200, 255), 3)          # yellow safety marking
    for f in range(int(FPS * DURATION_S)):
        img = base.copy()
        t = f / FPS
        for tid, cls, (x1, y1, x2, y2) in boxes_at(t):
            x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))
            col = {1: (200, 120, 40), 2: (40, 40, 200), 3: (30, 140, 255), 4: (120, 160, 90)}[tid]
            if cls == 1:                                                     # cyclist
                r = (y2 - y1) // 4
                cv2.circle(img, (x1 + r, y2 - r), r, (20, 20, 20), 4)
                cv2.circle(img, (x2 - r, y2 - r), r, (20, 20, 20), 4)
                cv2.rectangle(img, ((x1 + x2) // 2 - 15, y1 + 20), ((x1 + x2) // 2 + 15, y2 - r), col, -1)
                cv2.circle(img, ((x1 + x2) // 2, y1 + 15), 15, (180, 200, 230), -1)
            elif (x2 - x1) > (y2 - y1):                                     # lying person
                cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, col, -1)
                cv2.circle(img, (x1 + 15, (y1 + y2) // 2), 15, (180, 200, 230), -1)
            else:                                                            # standing person
                head = max(8, (x2 - x1) // 2)
                cv2.rectangle(img, (x1, y1 + 2 * head), (x2, y2), col, -1)
                cv2.circle(img, ((x1 + x2) // 2, y1 + head), head, (180, 200, 230), -1)
        cv2.putText(img, "SYNTHETIC TEST SCENE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)
    writer.release()


def write_labels(path: Path, clip: str) -> None:
    """Ground truth for evaluate.py."""
    path.write_text("clip,entity_description,behavior,start_s,end_s\n"
                    f"{clip},pedestrian #1 and cyclist #2,near_miss,5.9,6.7\n"
                    f"{clip},cyclist #2 riding fast on the shop floor,running,6.0,8.2\n"
                    f"{clip},worker #3 (orange),fall,18.2,24.0\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-video", default=str(ROOT / "samples" / "synthetic_safety.mp4"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "synthetic_safety"))
    ap.add_argument("--no-video", action="store_true", help="only write detections.json and the labels")
    args = ap.parse_args(argv)
    video = Path(args.out_video)
    video.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        write_video(video)
    utils.write_json(Path(args.out_dir) / "detections.json", make_tracker_result(video.name))
    write_labels(HERE / "synthetic_safety_labels.csv", video.name)
    print(f"wrote {video}, {Path(args.out_dir) / 'detections.json'} and {HERE / 'synthetic_safety_labels.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
