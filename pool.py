"""Pool pipeline orchestrator: Aquatic Distress Behaviour Intelligence (HackNex HNX26PSI07).

    stage 1  tracking   (tracker.py: YOLO + ByteTrack)          -> stage1_tracking.jsonl
    stage 2  behaviour  (pool_behaviour.py, pose keypoints here) -> stage2_behaviour.jsonl
    stage 3  reasoning  (distress.py: risk + state machine)      -> stage3_events.json
    stage 4  dashboard  (pool_dashboard.json + report/UI/announcement)

run_pool() is called by run.py when the scenario has pool.enabled. It returns the alerts as normal
PS07 events (behaviours "aquatic_distress" and "submersion"), so the evidence cards, snapshots,
highlight reel, report and web UI all work unchanged. The stage files follow the team's data contract,
so each teammate's module can also be run (and replaced) on its own.
"""
from __future__ import annotations

from pathlib import Path

import distress
import pool_behaviour
import utils


# --------------------------------------------------------------------------- pose keypoints (optional)

def extract_keypoints(video_path, tracker_result: dict, tracks: dict, cfg: dict) -> dict:
    """Pose keypoints for every swimmer at pool.pose_hz: {track_id: {frame_idx: 17x3 list}} in processing px.

    Runs yolo11n-pose on a padded crop around each tracked box. Frames are read sequentially (one pass).
    Imports cv2 / ultralytics lazily. Raises on a missing model or video; run_pool turns that into a warning.
    """
    import cv2
    from ultralytics import YOLO
    import tracker as trk
    from pose_verify import padded_crop_box

    pcfg = cfg["pool"]
    fps = float(tracker_result["fps"]) or 30.0
    stride = int(tracker_result.get("stride") or 1)
    every = max(stride, int(round(fps / float(pcfg["pose_hz"]) / stride)) * stride)
    wanted: dict[int, list] = {}
    for tid, tr in tracks.items():
        for f, box in zip(tr.frame, tr.box):
            if int(f) % every == 0:
                wanted.setdefault(int(f), []).append((tid, box))
    if not wanted:
        return {}
    model = YOLO(trk.resolve_weights(cfg["model"]["pose_weights"]))
    device = trk.resolve_device(tracker_result.get("device") or "cpu")
    cap, _, _, _ = trk.open_video(video_path)
    proc = tracker_result["proc_size"]
    out: dict[int, dict] = {}
    last = max(wanted)
    f = 0
    try:
        while f <= last:
            if f not in wanted:
                if not cap.grab():
                    break
                f += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            img = trk.resize_frame(frame, proc)
            for tid, box in wanted[f]:
                x1, y1, x2, y2 = padded_crop_box(box, img.shape[1], img.shape[0], 0.3)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                res = model.predict(crop, verbose=False, device=device, conf=0.2)[0]
                kp = _best_keypoints(res, (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1))
                if kp is not None:
                    out.setdefault(int(tid), {})[f] = [[round(x + x1, 1), round(y + y1, 1), round(c, 3)]
                                                       for x, y, c in kp]
            f += 1
    finally:
        cap.release()
    return out


def _best_keypoints(result, target):
    """17x3 keypoints of the detection whose centre is inside the tracked box (else the most confident)."""
    import numpy as np
    from tracker import to_numpy
    boxes, kps = getattr(result, "boxes", None), getattr(result, "keypoints", None)
    if boxes is None or kps is None or len(boxes) == 0 or kps.xy is None or kps.conf is None:
        return None
    conf = to_numpy(boxes.conf).reshape(-1)
    xyxy = to_numpy(boxes.xyxy).reshape(-1, 4)
    xy, kc = to_numpy(kps.xy), to_numpy(kps.conf)
    cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    inside = (cx >= target[0]) & (cx <= target[2]) & (cy >= target[1]) & (cy <= target[3])
    pool = np.where(inside)[0] if inside.any() else np.arange(len(conf))
    best = int(pool[np.argmax(conf[pool])])
    return [(float(x), float(y), float(c)) for (x, y), c in zip(xy[best], kc[best])]


def load_or_extract_keypoints(video_path, tracker_result, tracks, cfg, out_dir, reuse: bool):
    """keypoints.json in out_dir when it matches this video/stride/pose rate (and --reuse), else run the pose model."""
    path = Path(out_dir) / "keypoints.json"
    key = {"video": Path(str(tracker_result.get("video", video_path))).name, "stride": tracker_result.get("stride"),
           "pose_hz": cfg["pool"]["pose_hz"], "proc_size": tracker_result.get("proc_size")}
    if reuse and path.exists():
        try:
            data = utils.read_json(path)
            if data.get("key") == key:
                print(f"      reusing pose keypoints from {path}")
                return data["keypoints"]
        except Exception:
            pass
    kps = extract_keypoints(video_path, tracker_result, tracks, cfg)
    utils.write_json(path, {"key": key, "keypoints": {str(k): {str(f): v for f, v in d.items()} for k, d in kps.items()}})
    return kps


# --------------------------------------------------------------------------- the whole pool stage

def run_pool(video_path, tracker_result: dict, tracks: dict, zones: list, cfg: dict, out_dir, reuse: bool = False) -> dict:
    """Stages 2-4 for one video. Returns {"events": [PS07 events], "dashboard": {...}, "stage3": {...}}."""
    out_dir = Path(out_dir)
    pcfg = cfg["pool"]
    keypoints, pose_note = None, "pose not used"
    if pcfg.get("pose", True) and tracks:
        try:
            keypoints = load_or_extract_keypoints(video_path, tracker_result, tracks, cfg, out_dir, reuse)
            n = sum(len(v) for v in keypoints.values())
            pose_note = f"pose keypoints for {len(keypoints)} swimmer(s), {n} frames"
        except Exception as exc:                       # no model / no video: box-based signals still work
            pose_note = f"pose unavailable ({type(exc).__name__}: {exc}); using box-based arm/head signals"
    print(f"      {pose_note}")

    det_classes = {}
    for row in tracker_result.get("detections") or []:
        det_classes[(int(row[2]), int(row[0]))] = int(row[8]) if len(row) >= 9 else 0
    samples = pool_behaviour.behaviour_samples(tracks, cfg, zones, keypoints, det_classes)
    last_seen = {int(tid): float(tr.t[-1]) for tid, tr in tracks.items() if len(tr.t)}
    stage3 = distress.reason(samples, cfg, last_seen=last_seen, video_end=float(tracker_result.get("duration_s") or 0))

    write_stage_files(out_dir, tracker_result, samples, stage3, cfg)
    events = [to_ps07_event(e, tracks, cfg) for e in stage3["events"]]
    for ev in events:                                  # risk curve around the event, for the evidence plot
        curve = stage3["risk"].get(ev["entity_id"], [])
        lo, hi = ev["start_s"] - 15.0, ev["end_s"] + 5.0
        ev["metrics"]["risk_series"] = [p for p in curve if lo <= p[0] <= hi]
    dashboard = build_dashboard(tracker_result, tracks, stage3, cfg, pose_note)
    utils.write_json(out_dir / "pool_dashboard.json", dashboard)
    return {"events": events, "dashboard": dashboard, "stage3": stage3, "samples": samples}


def write_stage_files(out_dir: Path, tracker_result: dict, samples: list, stage3: dict, cfg: dict) -> None:
    """The team's data contract, one file per stage."""
    state_map = {int(k): str(v) for k, v in (cfg["pool"].get("state_classes") or {}).items()}
    rows = [{"meta": {"fps": tracker_result.get("fps"), "stride": tracker_result.get("stride"),
                      "proc_size": tracker_result.get("proc_size"), "video": tracker_result.get("video")}}]
    for r in tracker_result.get("detections") or []:
        cls = int(r[8]) if len(r) >= 9 else 0
        rec = {"person_id": int(r[2]), "timestamp": round(float(r[1]), 3), "frame": int(r[0]),
               "bbox": [round(float(v), 1) for v in r[3:7]], "conf": round(float(r[7]), 3), "class": cls}
        if state_map:
            rec["state"] = state_map.get(cls, "unknown")
        rows.append(rec)
    pool_behaviour.write_jsonl(out_dir / "stage1_tracking.jsonl", rows)
    pool_behaviour.write_jsonl(out_dir / "stage2_behaviour.jsonl", samples)
    utils.write_json(out_dir / "stage3_events.json", {"events": stage3["events"], "timeline": stage3["timeline"]})


def to_ps07_event(e: dict, tracks: dict, cfg: dict) -> dict:
    """A stage-3 event as a normal PS07 event dict (DESIGN 3.4 keys), so render/report/UI handle it."""
    import numpy as np
    behavior = "submersion" if e["event"] == "possible_submersion" else "aquatic_distress"
    pid = int(e["person_id"])
    tr = tracks.get(pid)
    start, end = float(e["start_time"]), max(float(e["end_time"]), float(e["start_time"]))
    if end - start < 0.5:
        end = start + 0.5
    conf = 0.85
    if tr is not None and len(tr.t):
        sel = (tr.t >= start - 0.5) & (tr.t <= end + 0.5)
        conf = float(np.mean(tr.conf[sel])) if sel.any() else float(np.mean(tr.conf))
    who = utils.entity_name(cfg, pid, getattr(tr, "cls", 0) if tr is not None else 0)
    checklist = "; ".join(distress.EVIDENCE_TEXT.get(k, k) for k in e["evidence"])
    if behavior == "aquatic_distress":
        evidence = (f"{who} showed a distress pattern from {utils.fmt_time_precise(start)}, held for "
                    f"{float(e['alert_time']) - start:.1f} s before the alert at {utils.fmt_time_precise(e['alert_time'])} "
                    f"in the {e['location']}. Evidence: {checklist}. Risk {e['risk_score']:.0%}.")
    else:
        evidence = (f"{who} was in trouble and then {('went under' if 'head_submerged' in e['evidence'] else 'was lost from view')} "
                    f"at {utils.fmt_time_precise(start)} in the {e['location']}. Check immediately.")
    snap_t = float(e["alert_time"]) if behavior == "aquatic_distress" else start
    if tr is not None and len(tr.t):
        snap_t = float(min(max(snap_t, tr.t[0]), tr.t[-1]))
    metrics = {"duration_s": round(end - start, 2), "risk_score": e["risk_score"], "alert_time_s": e["alert_time"],
               "location": e["location"], "evidence_keys": e["evidence"], "peak_time_s": round(snap_t, 2),
               "mean_det_conf": round(conf, 3), "event_type": e["event"]}
    return {"event_id": 0, "entity_id": pid, "behavior": behavior, "start_s": round(start, 2), "end_s": round(end, 2),
            "start": utils.fmt_time(start), "end": utils.fmt_time(end), "duration_s": round(end - start, 2),
            "confidence": round(float(e["risk_score"]), 2),
            "confidence_parts": {"margin": round(float(e["risk_score"]), 2), "duration": 1.0, "detection": round(conf, 2)},
            "evidence": evidence, "metrics": metrics, "zone": e["location"], "baseline": None,
            "snapshot_time_s": round(snap_t, 2), "snapshot": None, "speed_plot": None, "verified": None, "pose": None,
            "severity": "high", "entities": [pid], "other_entity": None, "entity_name": who,
            "behavior_name": utils.behavior_name(cfg, behavior)}


def build_dashboard(tracker_result: dict, tracks: dict, stage3: dict, cfg: dict, pose_note: str) -> dict:
    """Everything the lifeguard dashboard needs: people, their risk curves, timelines and the alerts."""
    people = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        people.append({"person_id": int(tid), "name": utils.entity_name(cfg, tid, getattr(tr, "cls", 0)),
                       "first_seen": round(float(tr.t[0]), 2), "last_seen": round(float(tr.t[-1]), 2),
                       "final_state": stage3["status"].get(int(tid), "NORMAL")})
    p = cfg["pool"]
    return {"video": tracker_result.get("video"), "duration_s": tracker_result.get("duration_s"),
            "people": people, "risk": {str(k): v for k, v in stage3["risk"].items()},
            "timeline": {str(k): v for k, v in stage3["timeline"].items()}, "alerts": stage3["events"],
            "thresholds": {"watch": p["watch_risk"], "warning": p["warning_risk"], "alert": p["alert_risk"]},
            "pose": pose_note}
