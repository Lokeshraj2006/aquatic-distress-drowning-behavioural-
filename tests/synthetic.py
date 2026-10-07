"""Synthetic corridor clip + matching tracker output, so the whole pipeline can run without YOLO.

Run from the project root:
    python tests/synthetic.py --out-video samples/synthetic.mp4 --out-dir outputs/synthetic

It writes
  * the video (25 fps, 30 s, 1280x720, grey floor, simple coloured people),
  * <out-dir>/detections.json   a TrackerResult (DESIGN 3.1) built from the same motion paths,
  * tests/synthetic_zones.json  the restricted zone (original-pixel coordinates),
  * tests/synthetic_labels.csv  ground truth for evaluate.py.
Then the pipeline can be tried with:
    python run.py --video samples/synthetic.mp4 --zones tests/synthetic_zones.json --out outputs/synthetic --reuse

Who is in the clip (foot-point paths are straight lines between waypoints, so the ground truth
is exact; speeds are expressed in body-heights per second like the real thresholds):
  #1 walker    crosses left -> right at 0.8 bh/s                      (normal)
  #2 loiterer  walks in, stands still ~15 s, walks out                (loitering)
  #3 runner    sprints left -> right along the back wall at 2.5 bh/s   (running)
  #4 intruder  walks into the restricted rectangle for ~4 s, leaves   (zone intrusion)
  #5 walker    crosses right -> left at 0.75 bh/s                     (normal)

Only NumPy is needed for the data; OpenCV is imported lazily, just for drawing the video.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

# ----------------------------------------------------------------------------- video geometry
FPS = 25.0
DURATION_S = 30.0
W, H = 1280, 720
N_FRAMES = int(FPS * DURATION_S)
STRIDE = 2                       # same as config.yaml video.frame_stride
PROC_W, PROC_H = 640, 360        # same as config.yaml video.resize_width
SCALE = PROC_W / W

HORIZON_Y = 430                  # wall above, floor below
ZONE_NAME = "restricted"
ZONE_POINTS = [(800, 470), (1200, 470), (1200, 620), (800, 620)]   # original pixels (foot points)


def person_height(foot_y: float) -> float:
    """Box height in pixels for a person whose feet are at foot_y (simple perspective)."""
    return 100.0 + 0.35 * (foot_y - 450.0)


def person_width(h: float) -> float:
    return 0.36 * h


# ----------------------------------------------------------------------------- the people
@dataclass
class Person:
    track_id: int
    role: str
    shirt_bgr: tuple
    colour_word: str
    waypoints: list                       # [(t, x, y)] foot point in original pixels
    stand: tuple | None = None            # (t0, t1): standing still, adds a tiny natural sway
    _cum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        pts = np.array([(x, y) for _, x, y in self.waypoints], dtype=float)
        steps = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
        self._cum = np.r_[0.0, np.cumsum(steps)]       # path length at each waypoint (drives the walking cycle)

    @property
    def t_start(self) -> float:
        return self.waypoints[0][0]

    @property
    def t_end(self) -> float:
        return self.waypoints[-1][0]

    def visible(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def pos(self, t: float):
        """Foot point (x, y) at time t, or None when the person is not in the scene."""
        if not self.visible(t):
            return None
        ts = [w[0] for w in self.waypoints]
        x = float(np.interp(t, ts, [w[1] for w in self.waypoints]))
        y = float(np.interp(t, ts, [w[2] for w in self.waypoints]))
        if self.stand and self.stand[0] <= t <= self.stand[1]:   # small sway while standing still
            x += 3.0 * math.sin(2 * math.pi * t / 4.1) + 2.0 * math.sin(2 * math.pi * t / 2.3 + 1.0)
            y += 1.5 * math.sin(2 * math.pi * t / 3.3 + 0.5)
        return x, y

    def path_length(self, t: float) -> float:
        return float(np.interp(t, [w[0] for w in self.waypoints], self._cum))


def _walk_time(t0: float, x0: float, x1: float, y: float, speed_bh_s: float) -> float:
    """Time at which a person walking from x0 to x1 at a given bh/s speed arrives."""
    return t0 + abs(x1 - x0) / (speed_bh_s * person_height(y))


def build_people() -> list[Person]:
    """The five people of the clip (see the module docstring)."""
    # 1) walker, front lane, left -> right at 0.8 bh/s
    y1 = 690.0
    walker = Person(1, "walker", (110, 160, 40), "green",
                    [(1.0, 60, y1), (_walk_time(1.0, 60, 1220, y1, 0.8), 1220, y1)])

    # 2) loiterer: walks in at 0.9 bh/s, stands still until t=20 s, walks out again
    y2, stand_end = 600.0, 20.0
    arrive = _walk_time(3.0, 40, 320, y2, 0.9)
    leave = _walk_time(stand_end, 320, 40, y2, 0.9)
    loiterer = Person(2, "loiterer", (30, 120, 235), "orange",
                      [(3.0, 40, y2), (arrive, 320, y2), (stand_end, 320, y2), (leave, 40, y2)],
                      stand=(arrive, stand_end))

    # 3) runner along the back wall at 2.5 bh/s for about 5 s
    y3 = 445.0
    runner = Person(3, "runner", (200, 90, 20), "blue",
                    [(8.0, 30, y3), (_walk_time(8.0, 30, 1250, y3, 2.5), 1250, y3)])

    # 4) intruder: front lane from the right, steps up into the zone, leaves again through the front
    intruder = Person(4, "intruder", (50, 50, 205), "red",
                      [(13.0, 1245, 690), (14.5, 1050, 690), (16.5, 980, 560), (18.5, 880, 540), (20.5, 850, 690)])

    # 5) second walker, front lane, right -> left at 0.75 bh/s
    y5 = 700.0
    walker2 = Person(5, "walker2", (150, 120, 120), "grey-purple",
                     [(21.0, 1230, y5), (_walk_time(21.0, 1230, 50, y5, 0.75), 50, y5)])
    return [walker, loiterer, runner, intruder, walker2]


def _inside_zone(x: float, y: float) -> bool:
    return utils.point_in_polygon(x, y, ZONE_POINTS)


def ground_truth(people: list[Person] | None = None) -> list[dict]:
    """Exact behaviour intervals (seconds) derived from the motion paths."""
    people = people or build_people()
    by_id = {p.track_id: p for p in people}
    rows = []

    lo = by_id[2]
    t0, t1 = lo.stand
    rows.append({"behavior": "loitering", "start_s": t0, "end_s": t1,
                 "entity_description": "person in an orange shirt standing still near the left wall"})

    ru = by_id[3]
    rows.append({"behavior": "running", "start_s": ru.t_start, "end_s": ru.t_end,
                 "entity_description": "person in a blue shirt sprinting along the back wall"})

    zi = by_id[4]
    ts = np.arange(zi.t_start, zi.t_end, 0.01)
    inside = [t for t in ts if _inside_zone(*zi.pos(float(t)))]
    rows.append({"behavior": "zone_intrusion", "start_s": float(inside[0]), "end_s": float(inside[-1]),
                 "entity_description": "person in a red shirt stepping into the marked restricted area"})
    for r in rows:
        r["start_s"], r["end_s"] = round(r["start_s"], 1), round(r["end_s"], 1)
    return rows


# ----------------------------------------------------------------------------- tracker output
def make_tracker_result(video_path: str, seed: int = 7, dropout: float = 0.02,
                        people: list[Person] | None = None) -> dict:
    """Build a TrackerResult (DESIGN 3.1) from the motion paths.

    Boxes are in processing pixels (640x360) with ~0.8 px Gaussian jitter on every corner,
    confidence ~0.85 +/- 0.03 and a small fraction of missed detections, like a real detector.
    """
    people = people or build_people()
    rng = np.random.default_rng(seed)
    dets = []
    for frame_idx in range(0, N_FRAMES, STRIDE):
        t = frame_idx / FPS
        for p in people:
            pos = p.pos(t)
            if pos is None or rng.random() < dropout:
                continue
            x, y = pos
            h = person_height(y)
            w = person_width(h)
            box = np.array([x - w / 2, y - h, x + w / 2, y]) * SCALE + rng.normal(0.0, 0.8, 4)
            conf = float(np.clip(0.85 + rng.normal(0.0, 0.03), 0.6, 0.99))
            dets.append([frame_idx, round(t, 4), p.track_id] + [round(float(v), 2) for v in box] + [round(conf, 3)])
    return {
        "video": Path(video_path).as_posix(),
        "fps": FPS, "stride": STRIDE,
        "orig_size": [W, H], "proc_size": [PROC_W, PROC_H],
        "n_frames_read": N_FRAMES, "duration_s": N_FRAMES / FPS,
        "frames_processed": len(range(0, N_FRAMES, STRIDE)),
        # `model` is the configured detector name so run.py --reuse accepts the file (it refuses a different
        # detector); `source` records that these boxes come from the motion paths, not from YOLO.
        "device": "cpu", "model": utils.load_config()["model"]["weights"],
        "source": "synthetic ground-truth boxes from tests/synthetic.py (not YOLO output)",
        "detections": dets,
    }


# ----------------------------------------------------------------------------- drawing
def _background(cv2) -> np.ndarray:
    """Static corridor: beige wall with doors and lights, grey tiled floor, marked restricted zone."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:HORIZON_Y] = (196, 192, 184)                       # wall
    img[:70] = (226, 226, 226)                              # ceiling strip
    for x in (180, 640, 1100):                              # ceiling lights
        cv2.rectangle(img, (x - 90, 22), (x + 90, 40), (250, 250, 245), -1)
    for x0 in (110, 470, 830, 1080):                        # doors
        cv2.rectangle(img, (x0, 170), (x0 + 120, HORIZON_Y), (60, 85, 120), -1)
        cv2.rectangle(img, (x0, 170), (x0 + 120, HORIZON_Y), (40, 55, 80), 3)
        cv2.circle(img, (x0 + 100, 310), 6, (70, 190, 220), -1, cv2.LINE_AA)
    cv2.rectangle(img, (620, 200), (760, 300), (230, 235, 240), -1)     # notice board
    cv2.rectangle(img, (620, 200), (760, 300), (120, 130, 140), 3)
    cv2.line(img, (0, HORIZON_Y - 8), (W, HORIZON_Y - 8), (150, 150, 145), 8)   # skirting board

    rows = np.arange(HORIZON_Y, H)                          # floor: darker far away, lighter near
    shade = (112 + 30 * (rows - HORIZON_Y) / (H - HORIZON_Y)).astype(np.uint8)
    img[HORIZON_Y:] = shade[:, None, None]
    vp = (W / 2, 120.0)                                     # vanishing point of the tile lines
    for xb in range(-1400, 2700, 230):
        a = (xb, H)
        k = (H - HORIZON_Y) / (H - vp[1])
        b = (xb + (vp[0] - xb) * k, HORIZON_Y)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (96, 96, 96), 1, cv2.LINE_AA)
    k_i = 1
    while HORIZON_Y + 6 * k_i ** 1.6 < H:
        y = int(HORIZON_Y + 6 * k_i ** 1.6)
        cv2.line(img, (0, y), (W, y), (96, 96, 96), 1, cv2.LINE_AA)
        k_i += 1

    # restricted zone: translucent yellow fill, dashed outline, label
    pts = np.array(ZONE_POINTS, np.int32)
    tint = img.copy()
    cv2.fillPoly(tint, [pts], (0, 215, 255))
    img = cv2.addWeighted(tint, 0.14, img, 0.86, 0)
    for i in range(len(ZONE_POINTS)):
        (x0, y0), (x1, y1) = ZONE_POINTS[i], ZONE_POINTS[(i + 1) % len(ZONE_POINTS)]
        length = math.hypot(x1 - x0, y1 - y0)
        n_dash = int(length // 28)
        for d in range(n_dash):
            a, b = d / n_dash, (d + 0.55) / n_dash
            cv2.line(img, (int(x0 + (x1 - x0) * a), int(y0 + (y1 - y0) * a)),
                     (int(x0 + (x1 - x0) * b), int(y0 + (y1 - y0) * b)), (0, 200, 240), 4, cv2.LINE_AA)
    cv2.putText(img, "RESTRICTED AREA", (ZONE_POINTS[0][0] + 14, ZONE_POINTS[0][1] + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 175, 215), 2, cv2.LINE_AA)
    cv2.putText(img, "synthetic test clip", (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def _draw_person(cv2, img, p: Person, t: float) -> None:
    """Simple person: shadow, swinging legs and arms, shirt, head. Fits the analytic box."""
    pos = p.pos(t)
    if pos is None:
        return
    x, y = pos
    h = person_height(y)
    # walking speed from a short finite difference decides how much the limbs swing
    a, b = p.pos(max(p.t_start, t - 0.05)), p.pos(min(p.t_end, t + 0.05))
    speed_bh = math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1 / h if a and b else 0.0
    moving = speed_bh > 0.15
    amp = h * min(0.15, 0.04 + 0.045 * speed_bh) if moving else 0.0
    phase = 2 * math.pi * p.path_length(t) / (0.9 * h)
    sw = amp * math.sin(phase)
    trousers, skin, hair = (70, 55, 45), (150, 185, 230), (35, 35, 45)
    shirt = p.shirt_bgr
    dark = tuple(int(c * 0.7) for c in shirt)
    ix = lambda v: int(round(v))  # noqa: E731

    cv2.ellipse(img, (ix(x), ix(y)), (ix(0.22 * h), max(2, ix(0.035 * h))), 0, 0, 360, (70, 70, 70), -1, cv2.LINE_AA)
    for sgn in (-1, 1):                                                     # legs
        hip = (ix(x + sgn * 0.045 * h), ix(y - 0.47 * h))
        foot = (ix(x + sgn * 0.045 * h + sgn * sw), ix(y))
        cv2.line(img, hip, foot, trousers, max(3, ix(0.11 * h)), cv2.LINE_AA)
        cv2.ellipse(img, (foot[0] + ix(0.02 * h * sgn), foot[1]), (ix(0.06 * h), max(2, ix(0.025 * h))), 0, 0, 360,
                    (25, 25, 25), -1, cv2.LINE_AA)
    for sgn in (-1, 1):                                                     # arms swing opposite to the legs
        sh_ = (ix(x + sgn * 0.14 * h), ix(y - 0.80 * h))
        hand = (ix(x + sgn * 0.15 * h - sgn * 0.6 * sw), ix(y - 0.50 * h))
        cv2.line(img, sh_, hand, dark, max(3, ix(0.07 * h)), cv2.LINE_AA)
        cv2.circle(img, hand, max(2, ix(0.03 * h)), skin, -1, cv2.LINE_AA)
    torso = np.array([(x - 0.15 * h, y - 0.82 * h), (x + 0.15 * h, y - 0.82 * h),
                      (x + 0.12 * h, y - 0.45 * h), (x - 0.12 * h, y - 0.45 * h)], np.float32)
    cv2.fillConvexPoly(img, torso.astype(np.int32), shirt, cv2.LINE_AA)
    cv2.circle(img, (ix(x), ix(y - 0.915 * h)), ix(0.075 * h), skin, -1, cv2.LINE_AA)       # head
    cv2.ellipse(img, (ix(x), ix(y - 0.925 * h)), (ix(0.078 * h), ix(0.078 * h)), 0, 180, 360, hair, -1, cv2.LINE_AA)


def write_video(path: Path, people: list[Person]) -> None:
    """Render the clip with OpenCV (mp4v, which OpenCV reads everywhere)."""
    import cv2  # lazy: the data side of this file works without OpenCV

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    bg = _background(cv2)
    try:
        for f in range(N_FRAMES):
            t = f / FPS
            frame = bg.copy()
            for p in sorted(people, key=lambda q: (q.pos(t) or (0, -1))[1]):   # far people first
                _draw_person(cv2, frame, p, t)
            writer.write(frame)
    finally:
        writer.release()


# ----------------------------------------------------------------------------- outputs
def write_labels(path: Path, clip_name: str, rows: list[dict]) -> None:
    """Ground-truth CSV in the evaluate.py format: clip,entity_description,behavior,start_s,end_s."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "entity_description", "behavior", "start_s", "end_s"])
        for r in rows:
            w.writerow([clip_name, r["entity_description"], r["behavior"], r["start_s"], r["end_s"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic corridor clip and its tracker output.")
    ap.add_argument("--out-video", default=str(ROOT / "samples" / "synthetic.mp4"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "synthetic"),
                    help="where detections.json is written (the run.py --out directory)")
    ap.add_argument("--zones-out", default=str(HERE / "synthetic_zones.json"))
    ap.add_argument("--labels-out", default=str(HERE / "synthetic_labels.csv"))
    ap.add_argument("--seed", type=int, default=7, help="random seed for box jitter / missed detections")
    ap.add_argument("--no-video", action="store_true", help="only write the JSON / CSV files (no OpenCV needed)")
    args = ap.parse_args(argv)

    people = build_people()
    out_video = Path(args.out_video)
    if not args.no_video:
        write_video(out_video, people)
        print(f"video      -> {out_video}  ({W}x{H}, {FPS:g} fps, {DURATION_S:g} s)")

    result = make_tracker_result(args.out_video, seed=args.seed, people=people)
    utils.write_json(Path(args.out_dir) / "detections.json", result)
    print(f"detections -> {Path(args.out_dir) / 'detections.json'}  ({len(result['detections'])} boxes)")

    utils.save_zones(args.zones_out, [{"name": ZONE_NAME, "points": ZONE_POINTS}], (W, H))
    print(f"zones      -> {args.zones_out}")

    rows = ground_truth(people)
    write_labels(Path(args.labels_out), out_video.name, rows)
    print(f"labels     -> {args.labels_out}")
    for r in rows:
        print(f"  truth: {r['behavior']:<15} {r['start_s']:5.1f} - {r['end_s']:5.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
