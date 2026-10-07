# PS07 Design Spec (the contract every module follows)

**Problem:** HackNEX 2026 PS07, Autonomous Vision and Behaviour Understanding.
**Scenario:** a restricted zone in a college or workplace corridor.
**What the system does:** a video goes in. Out comes a report of **which person (stable ID) did what (loitering / zone intrusion / running), when (start and end time), and why it is unusual (measured numbers, confidence, snapshot)**. It never just lists detections.

## 1. Pipeline

```
video ─► tracker.py ─► features.py ─► behaviors.py ─► events.py ─► (pose_verify.py) ─► render.py ─► report.py
          YOLO11n +      per-ID tracks    raw intervals     final events     optional --pose     video, snapshots,   events.json,
          ByteTrack      smoothed foot,   + near misses     merge, filter,   true/false/         plots, heatmap,     report.html,
          detections     height, speed    + baseline        confidence       uncertain           timeline            summary.txt
```

`run.py` wires everything together. `evaluate.py` compares `events.json` with a hand-written `labels.csv`.

**Core logic is NumPy only.** `utils.py`, `features.py`, `behaviors.py`, `events.py` and `evaluate.py` must not import OpenCV, torch or ultralytics, so they can be unit-tested anywhere. Only `tracker.py`, `pose_verify.py`, `render.py`, `zone_picker.py` and `run.py` may import `cv2`. Only `tracker.py` and `pose_verify.py` may import `ultralytics` or `torch`, and they import it **inside functions** so `run.py --reuse` works without them.

All config comes from `utils.load_config()`. See `config.yaml`; every key used is listed there. Never hardcode a threshold.

## 2. Coordinates and time
- **Processing frame:** the original frame resized to `video.resize_width` wide, with height scaled to keep the aspect ratio (`proc_size = [w, h]`). All boxes, foot points and zone points used in logic are in processing pixels.
- **Frame index:** the original 0-based frame number in the source video. **Time:** `t = frame_idx / fps`. `fps` comes from the video metadata. If it is 0 or NaN, use 30.0 and print a warning.
- Only frames with `frame_idx % frame_stride == 0` are processed.
- **Foot point:** the bottom-centre of the box, `((x1 + x2) / 2, y2)`. **Body-height (bh):** box height `y2 - y1` in pixels. **Speed** is in bh per second.

## 3. Data contracts

### 3.1 TrackerResult (`tracker.py` output; also saved as `<out>/detections.json`)
```python
{
  "video": "samples/clip.mp4",      # path as given
  "fps": 25.0,                       # source fps (after fallback)
  "stride": 2,
  "orig_size": [1280, 720],          # source W, H
  "proc_size": [640, 360],           # processing W, H
  "n_frames_read": 1500,             # source frames read (honours max_seconds)
  "duration_s": 60.0,                # n_frames_read / fps
  "frames_processed": 750,
  "device": "cpu",
  "model": "yolo11n.pt",
  "detections": [[frame_idx, t, track_id, x1, y1, x2, y2, conf], ...]   # processing pixels; only boxes WITH a track id
}
```

### 3.2 Track (`features.py`)
```python
@dataclass
class Track:
    track_id: int
    t: np.ndarray        # (n,) seconds, strictly increasing
    frame: np.ndarray    # (n,) int original frame indices
    box: np.ndarray      # (n, 4) raw x1, y1, x2, y2 (processing px)
    conf: np.ndarray     # (n,)
    foot: np.ndarray     # (n, 2) SMOOTHED foot point (moving_average over features.smoothing_s, per segment)
    height: np.ndarray   # (n,) SMOOTHED box height px
    speed: np.ndarray    # (n,) bh/s, NaN where unknown (start of a segment or across a gap)
    segment: np.ndarray  # (n,) int, +1 at every time gap > features.max_gap_s
    # properties: start_s, end_s, duration_s
```

`build_tracks(tracker_result: dict, cfg: dict) -> dict[int, Track]`
- Group detections by track id and sort them by time. If there are duplicate times, keep the highest conf.
- Drop samples with box height < `features.min_box_h_px`. Drop tracks whose total duration is < `features.min_track_s`, or that have fewer than 3 samples.
- Smooth the foot point and height **within each segment** using `utils.moving_average`.
- **Speed at i:** let `j` be the earliest index in the same segment with `t[j] >= t[i] - speed_window_s`. If `t[i] - t[j] < 0.5 * speed_window_s`, speed is NaN. Otherwise `speed = ||foot[i] - foot[j]|| / (t[i] - t[j]) / height[i]`.

`stationary_runs(track, radius_bh) -> longest_s` is a helper used for near misses and the baseline. It returns the longest time span over which all smoothed foot points stay within `radius_bh * median(height in span)` of the span's centroid. It works within segments. O(n²) using NumPy is fine for n ≤ 3000.

`track_summary(track) -> dict` gives `{"id", "first_seen", "last_seen", "duration_s", "median_speed_bh_s", "p90_speed_bh_s", "max_dwell_s"}`. Here `max_dwell_s = stationary_runs(track, loitering.radius_bh)`.

### 3.3 Behaviours (`behaviors.py`)
`detect_behaviors(tracks: dict[int, Track], zones: list[dict], cfg: dict) -> dict` returns:
```python
{
  "intervals": [{"track_id": 3, "behavior": "running", "start_s": 41.2, "end_s": 48.0}],   # RAW, unmerged
  "near_misses": [{"track_id": 5, "behavior": "loitering", "value": 8.2, "threshold": 10.0,
                   "unit": "s", "note": "Person #5 stood still for 8.2 s (rule: 10 s). Not flagged."}],
  "baseline": {"enabled": True, "active": True, "n_tracks": 7, "median_speed_bh_s": 0.82,
               "median_dwell_s": 1.5,
               "tracks": {3: {"median_speed_bh_s": 2.1, "max_dwell_s": 0.0, "speed_z": 4.2,
                              "dwell_z": -0.3, "anomaly_score": 4.2, "unusual": True,
                              "note": "Moved 2.6x faster than the scene's typical person"}}}
}
```

Rules (each only if `enabled`):
- **loitering:** for each sample i, take the window `[t[i] - min_duration_s, t[i]]` within one segment. The window must actually cover ≥ 0.9 × min_duration_s. If every smoothed foot point in it is within `radius_bh × median(height in window)` of the window centroid, mark samples j..i. The runs of marked samples (`utils.mask_to_intervals` with `max_gap_s`) are the intervals.
- **zone_intrusion:** a sample is inside if `utils.point_in_polygon(foot)` is true for any zone. Each run of inside samples is one interval. Record which zone. **Do not** filter by duration here; events.py does that after merging.
- **running:** hysteresis over `speed` (NaN means no change of state). The state turns on when `speed >= start_speed_bh_s` and off when `speed < end_speed_bh_s`. Each on-run is one interval. No duration filter here.
- **near misses:** only for (track, behaviour) pairs that produced **no** raw interval:
  - loitering: `stationary_runs` ≥ ratio × min_duration_s
  - running: max speed ≥ ratio × start_speed_bh_s
  - zone: was inside some zone, but the longest inside run is < min_duration_s

  Here ratio = `events.near_miss_ratio`. events.py later also drops near misses for pairs whose intervals were all filtered out. It keeps them, and then the note says it was too short.
- **baseline (scene-relative "normal"):** use tracks with duration ≥ 2 s. If there are fewer than `baseline.min_tracks`, set `active: False` and give no per-track z-scores. Otherwise:
  - `speed_z` = `utils.robust_z` of per-track median speed
  - `dwell_z` = `robust_z` of max_dwell_s
  - `anomaly_score = max(speed_z, dwell_z, 0)`
  - `unusual = anomaly_score >= z_threshold`

  The note uses ratios to the scene median.

`compute_metrics(track, behavior, start_s, end_s, zones, cfg) -> dict` is called by events.py on the **merged** span:
- loitering: `{"duration_s", "max_radius_bh", "radius_threshold_bh", "min_duration_s", "mean_speed_bh_s"}`
- zone_intrusion: `{"duration_s", "zone", "max_depth_bh", "min_duration_s"}`. Depth is the distance of the foot point inside the polygon divided by the height.
- running: `{"duration_s", "peak_speed_bh_s", "mean_speed_bh_s", "start_threshold_bh_s", "end_threshold_bh_s", "min_duration_s"}`
- all of them also: `"mean_det_conf"`, and `"peak_time_s"` (running: time of peak speed; zone: time of max depth; loitering: the middle).

### 3.4 Events (`events.py`)
`build_events(behavior_result, tracks, zones, cfg) -> dict` returns
`{"events": [...], "near_misses": [...], "unusual_tracks": [...], "baseline": {... without per-track dict ...}}`.
1. For each (track, behaviour), merge raw intervals with `utils.merge_intervals(gap=events.merge_gap_s)`. Zone intervals only merge if they are in the same zone.
2. Drop merged intervals shorter than that behaviour's `min_duration_s`. Use a small epsilon of 1e-6 so that 1.0 s passes the 1.0 s rule.
3. Compute metrics, then confidence, where all three scores are in [0, 1]:
   - margin:
     - running: `clip01((mean_speed / start_threshold - 1) / 0.5)`
     - loitering: `clip01(1 - max_radius_bh / radius_bh)`
     - zone: `clip01(max_depth_bh / 0.5)`
   - duration: `clip01(duration_s / (2 × min_duration_s))`
   - detection: `mean_det_conf`
   - `confidence = round(w_margin·margin + w_duration·duration + w_detection·detection, 2)`
4. Sort by start_s, then entity id. Number the events from 1 as `event_id`.

Event dict (exact keys; render and pose fill the null fields):
```python
{"event_id": 1, "entity_id": 3, "behavior": "running",
 "start_s": 41.2, "end_s": 48.0, "start": "00:41", "end": "00:48", "duration_s": 6.8,
 "confidence": 0.86, "confidence_parts": {"margin": 0.9, "duration": 1.0, "detection": 0.71},
 "evidence": "Moved at 2.9 body-heights/s (peak 3.4) for 6.8 s. Rule: >= 1.8 bh/s for >= 1 s; walking is about 0.6-1.0.",
 "metrics": {...compute_metrics...},
 "zone": None,                    # zone name for zone_intrusion
 "baseline": {"anomaly_score": 4.2, "note": "..."} or None,
 "snapshot_time_s": 44.0,         # = metrics.peak_time_s
 "snapshot": None, "speed_plot": None,      # filled by render.py (paths relative to the out dir)
 "verified": None, "pose": None}            # filled by pose_verify.py
```

Evidence templates (plain English, with numbers):
- loitering: `Stayed within {max_radius:.2f} body-heights of one spot for {dur:.0f} s. Rule: within {R} bh for >= {min:.0f} s.`
- zone: `Feet inside zone '{zone}' for {dur:.1f} s, up to {depth:.2f} body-heights deep. Rule: >= {min:.0f} s.`
- running: as in the example above.

`unusual_tracks` are baseline tracks with `unusual=True`: `[{"entity_id", "anomaly_score", "note", "has_event": bool}]`.

### 3.5 Pose check (`pose_verify.py`, only with `--pose`)
`verify_events(video_path, tracker_result, tracks, final, cfg) -> final` works on running and loitering events only:
- Sample up to `pose.max_frames_per_event` processed frames evenly within the event.
- Read each frame by seeking `cv2.CAP_PROP_POS_FRAMES`, then resize to proc_size.
- Crop the person box with 20% padding and run `yolo11n-pose`. Take the most confident person.
- Keypoints (COCO): 5/6 shoulders, 11/12 hips, 15/16 ankles. A frame is usable if all of these have conf ≥ `min_keypoint_conf`.
- Compute:
  - `lean_deg`: angle of the hip-mid→shoulder-mid vector from vertical
  - `leg_spread_bh`: |ankle_l − ankle_r| / box height
- **running:** agrees if `max(leg_spread) >= run_leg_spread_bh` or `median(lean) >= run_lean_deg`.
- **loitering:** agrees if `median(lean) <= upright_max_lean_deg`.
- If there are fewer than `min_pose_frames` usable frames, the result is `"uncertain"`.
- Set `event["verified"] = True | False | "uncertain"`, and `event["pose"] = {"frames_checked", "usable_frames", "median_lean_deg", "max_leg_spread_bh", "agrees", "note"}`.
- Never crash the run. On any error, set uncertain and record the error in the note.

### 3.6 Render (`render.py`, uses cv2 + matplotlib with the Agg backend; no GUI)
`render_all(video_path, tracker_result, tracks, final, zones, cfg, out_dir) -> final` writes:
- **`annotated.mp4`** (if `output.annotated_video`):
  - one frame per processed frame, fps = fps / stride
  - zone polygons, semi-transparent
  - boxes in `utils.id_color`, labelled `#ID`
  - a trail of the last `trail_s` seconds of smoothed foot points
  - active events as a coloured label over the person (`BEHAVIOR_LABELS` / `BEHAVIOR_COLORS_BGR`)
  - a top-left timestamp `mm:ss.s` and an active-event counter

  Write with `cv2.VideoWriter` using fourcc `mp4v` to a temp file. If `ffmpeg` is on PATH, transcode to H.264 (`-vcodec libx264 -pix_fmt yuv420p -movflags +faststart`) so it plays in browsers and Colab. Otherwise keep the mp4v file.
- **`snapshots/e{event_id}.jpg`:** the frame nearest `snapshot_time_s`, with:
  - the event person's box thick, in the behaviour colour
  - the person's full trail during the event (smoothed foot points)
  - the zone, if any
  - a caption bar: `Event {id} | #{entity} {LABEL} | {start}-{end} | conf {c}`

  Set `event["snapshot"]`.
- **`plots/e{event_id}.png`:** matplotlib speed (bh/s) vs time for that person across the event ±3 s. Shade the event span and draw threshold lines:
  - running: start and end thresholds
  - loitering: plot distance from the event centroid in bh, with the radius line
  - zone: plot depth inside the zone

  Set `event["speed_plot"]`.
- **`heatmap.jpg`:** foot-point density (Gaussian-blurred 2D histogram, colormap) blended over the first processed frame.
- **`timeline.png`:** one horizontal bar per track (thin grey = visible), with coloured blocks for events. The x-axis is mm:ss.
- Reading frames: same resize/stride as the tracker. Use sequential `cap.read()` / `cap.grab()`, not per-frame seeking, for the video pass.
- Works when there are 0 events and 0 tracks: still writes the video, heatmap and timeline. Skip snapshots.

### 3.7 Report (`report.py`, no cv2 needed)
`write_outputs(final, meta, tracks_summary, out_dir) -> dict` writes:
- **`events.json`:** `{"video", "meta", "events", "near_misses", "unusual_tracks", "baseline", "tracks"}`. `tracks` is a list of `track_summary` dicts.
- **`summary.txt`** comes from `make_summary(final, meta) -> str`. It is a 3–6 line guard summary, e.g.:
  ```
  Video corridor.mp4 (1:00), 4 people seen. 3 incidents:
  - 00:12-00:34 Person #1 loitering 22 s near one spot.
  - ...
  No other unusual activity.
  ```
  With 0 incidents: "No incidents. N people seen, all behaviour within normal range."
- **`report.html`:** a single self-contained file (images embedded as base64 data URIs, inline CSS, no external requests). It shows:
  - a header with the video name, duration, people count and incident count
  - the guard summary
  - an **evidence card per event**: snapshot, speed plot, behaviour badge in its colour, times, evidence sentence, confidence bar with its parts, baseline note, pose verdict
  - the timeline image, the heatmap, the near-misses table, the unusual-tracks table, and a settings table (key thresholds)

  It should look clean in light mode and print well. Missing images are skipped gracefully.

### 3.8 run.py CLI
```
python run.py --video PATH [--zones zones.json] [--config config.yaml] [--out outputs/<stem>]
              [--device auto|cpu|0] [--stride N] [--max-seconds S] [--pose] [--no-video] [--reuse]
python run.py --webcam SECONDS [...]     # records the laptop camera to samples/webcam_<n>.mp4, then runs on it
```
- `--reuse`: if `<out>/detections.json` exists and is for the same video, stride and resize, skip YOLO. This allows fast threshold tuning.
- Prints the steps with timings and a final table: `# | ID | behaviour | start-end | conf | evidence`. Also prints where the outputs are.
- Exit code 0 even with 0 people or 0 events. Exits with a clear message if the video can't be opened.

### 3.9 zone_picker.py
- `python zone_picker.py --video X [--out zones.json] [--name restricted]`: opens an OpenCV window on the first frame. Left-click adds a point, right-click undoes, Enter or S saves, Esc quits. Points are saved in **original** video pixels via `utils.save_zones`.
- `--grid`: no GUI (for Colab). Writes `zone_grid.jpg`, the first frame with a labelled coordinate grid every 50 px of the original frame.
- `--points "x1,y1 x2,y2 x3,y3 x4,y4"`: no GUI. Writes zones.json directly and `zone_preview.jpg` showing the polygon.

### 3.10 evaluate.py
`labels.csv` columns: `clip,entity_description,behavior,start_s,end_s`. A row with behavior `none` means the clip should have no events.

```
python evaluate.py --labels labels.csv --outputs outputs/      # each outputs/<clip_stem>/events.json
python evaluate.py --labels labels.csv --events outputs/x/events.json --clip x.mp4
```

**Matching:** per clip and behaviour, a prediction matches a label if the intervals overlap after widening the label by `--tolerance` seconds (default 2) on both sides. Greedy one-to-one, largest overlap first.

**Reports:**
- per behaviour and overall: TP, FP, FN, precision, recall, F1
- mean |start error| and mean |end error| for matches
- false alarms on clips labelled `none`

Prints a table and writes `eval_report.txt` and `eval_report.json` next to the labels (or `--report-dir`). Entity IDs aren't compared, because labels describe people by appearance.

## 4. Differentiators (added after the core build)

### 4.1 Incident chains: `chains.py` (pure Python, no cv2)
`build_chains(events, cfg) -> list[dict]`. It also sets `event["severity"]` on every event.
- **Per-event severity:**
  - zone_intrusion → `medium`
  - running → `medium` if confidence ≥ 0.6, else `low`
  - loitering → `low`
  - Raise one level (low→medium→high) if the event's `baseline` says the track is unusual.
- **Chain:** the same `entity_id` with ≥ 2 events in start order, where each next event starts ≤ `chains.max_gap_s` (default 15) after the previous one ends. A sequence that keeps meeting this condition becomes one chain.
- **Known patterns** (matched on consecutive behaviours) give a human title:
  - `loitering → zone_intrusion`: "Waited nearby, then entered the restricted zone"
  - `running → zone_intrusion`: "Rushed into the restricted zone"
  - `zone_intrusion → running`: "Ran away after entering the restricted zone"
  - `loitering → running`: "Waited, then suddenly ran"
  - anything else: "Repeated suspicious activity"
- **Chain severity:** `high` if it contains a zone_intrusion plus any other behaviour, else `medium`. All events in a chain are raised to at least the chain severity.
- **Chain dict:**
  ```python
  {"chain_id": 1, "entity_id": 3, "event_ids": [1, 2], "pattern": "loitering -> zone_intrusion",
   "title": "...", "start_s", "end_s", "start", "end", "severity": "high",
   "story": "Person #3 loitered 00:12-00:30, then entered zone 'restricted' at 00:31 (1 s later) and stayed 5 s."}
  ```
  The story is built from the events in order, with gaps in seconds.
- **Config:** `chains: {enabled: true, max_gap_s: 15}`.
- **Outputs:** `final["incidents"] = chains`. `events.json` gets `"incidents"`. `report.html` shows an **Incidents** section above the event cards: title, severity badge, story, and links to the event cards (`#event-N`). Severity badges also appear on each event card. `summary.txt` lists incidents (chains) first, then any events not in a chain. The table printed by `run.py` gets a severity column.

### 4.2 Highlight reel: `highlights.py` (cv2)
`make_highlight_reel(video_path, tracker_result, tracks, final, zones, cfg, out_dir) -> str | None` writes `highlights.mp4`, H.264 via ffmpeg when available (same helper as render).
- **Segments:**
  - For each incident chain, use [chain start − `pad_s`, chain end + `pad_s`].
  - For each event not in a chain, use [event start − `pad_s`, event end + `pad_s`].
  - Clamp to the video length, merge overlaps, and order by time.
- **Long segments:** if a segment is longer than `max_segment_s`, play it fast-forward. Take every k-th processed frame so it fits, and show a "FAST-FORWARD xk" tag.
- **Title card:** an opening card (2 s, dark background): "HIGHLIGHT REEL", the video name, "N incidents in mm:ss of video". With 0 events, write a 2 s card saying "No incidents".
- **Frames:** annotated exactly like `annotated.mp4` (reuse render's per-frame drawing function), plus a bottom caption bar: "Incident k/N | #ID BEHAVIOUR(S) | mm:ss-mm:ss | severity". Read segments by seeking to the segment's start frame, then reading sequentially.
- **Config:** `highlights: {enabled: true, pad_s: 1.5, max_segment_s: 8.0, title_s: 2.0}`.
- **Outputs:** `final["highlight_reel"] = "highlights.mp4"`, shown at the top of `report.html` as a link plus a note. In `colab_run.ipynb`, the results cell plays highlights.mp4 first.

### 4.3 Privacy mode: `privacy.py` (cv2 + numpy)
`anonymize(frame, boxes, cfg) -> frame` (works in place and returns the frame). It blurs each person box's top `head_fraction` (default 0.28), or the whole box when `mode: body`. It uses a strong Gaussian blur or pixelation (`style: pixelate|blur`), and the box is clamped to the frame.
- **Where it applies:** when enabled (`--privacy` flag or `privacy.enabled: true`), on every rendered image *before* overlays are drawn:
  - annotated video
  - highlight reel
  - snapshots
  - heatmap background (using that frame's boxes)

  Detection, tracking and pose run on the original frames. Privacy affects outputs only.
- **Config:** `privacy: {enabled: false, mode: head, head_fraction: 0.28, style: pixelate}`.
- **Report:** `report.html` and `summary.txt` say "Privacy mode: faces blurred in all outputs" when it is on.

### 4.4 Speed line
`run.py` adds `meta["processing_fps"]` = frames_processed / tracking seconds (null with `--reuse`). The report header shows "Processed at X frames/s on <device>".

### 4.5 Scenario presets: `scenarios/<name>.yaml` + `--scenario <name>`
A preset is a **partial config** deep-merged over `config.yaml` by `utils.load_config(path, scenario=name)`. It may change any key. It also has a `scenario:` block used only for wording:
```yaml
scenario:
  name: livestock
  title: "Livestock monitoring"
  entity_word: "Animal"          # "Person #3" becomes "Animal #3"
  behavior_names:                # display names; internal behaviour keys never change
    loitering: "Not moving"
    zone_intrusion: "Left the pen"
    running: "Stampede / distress"
    fall: "Lying down"
    crowding: "Bunching"
model:
  classes: [17, 18, 19]          # COCO ids tracked (default [0] = person)
```
Use `utils.entity_name(cfg, track_id)` ("Person #3") and `utils.behavior_name(cfg, key)` ("Running") **everywhere** text is shown: evidence, notes, summary, report, overlays, run table. Never hardcode "Person" or a behaviour title. The `scenario.*` keys are wording only. The thresholds sit in the normal sections.

The five presets to ship are campus (= defaults), workplace, elderly, livestock and traffic.

**COCO ids:**
- people and vehicles: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
- animals: 14 bird, 15 cat, 16 dog, 17 horse, 18 sheep, 19 cow

**Detections carry a class:**
- TrackerResult detection rows may have a 9th element, the class id. Old 8-element rows mean class 0.
- `Track.cls` is the most common class id of the track.
- The tracker passes `classes=cfg["model"]["classes"]`.

**Body scale (changed for all presets):** `features.scale: max_side` (default) measures speed and distance in units of `max(box_h, box_w)` instead of box height.
- For an upright person this is the same as box height.
- For a lying person or a car it is the body length, so speeds stay sane.

`Track.height` keeps its name but stores this scale. Text still says "body-heights" for people. For other classes it says "body-lengths", via `utils.unit_name(cfg)`.

### 4.6 Fall / person-down rule: behaviour key `fall`
Every sample has an aspect ratio `w / h` from the raw box, smoothed like the foot point.
- **Down:** a sample is down when aspect ≥ `fall.down_min_aspect` (1.2) **and** the box does not touch the frame edge (≥ 3 px margin, so cut-off boxes don't count).
- **Event:** a run of down samples lasting ≥ `fall.min_down_s`. Raw runs go out from `detect_behaviors`; events.py merges them and filters by duration like the others.
- **Metrics:** `{"duration_s", "max_aspect", "upright_aspect_before", "fell": bool, "transition_s", "min_down_s", "peak_time_s", "mean_det_conf"}`. `fell` is true if, within `fall.transition_s` before the run, some sample had aspect ≤ `fall.upright_max_aspect`. `peak_time_s` is the start of the run plus 0.5 s.
- **Evidence:**
  - when `fell`: `Box went from upright (w/h {before:.1f}) to lying (w/h {max:.1f}) within {transition:.1f} s and stayed down {dur:.0f} s.`
  - otherwise: `Lying (w/h {max:.1f}) for {dur:.0f} s.`
- **Confidence margin:** `clip01((max_aspect / down_min_aspect - 1) / 0.5)`, plus 0.2 if `fell`, capped at 1.
- **Interactions:**
  - Running ignores samples where the person is down. A fall's speed spike is not a run.
  - Loitering still works on a person who is down.
  - Severity is always `high`.
- **Config:** `fall: {enabled: true, down_min_aspect: 1.2, upright_max_aspect: 0.8, transition_s: 2.0, min_down_s: 2.0}`.

### 4.7 Crowding rule: behaviour key `crowding` (a scene-level event)
- **Count:** for each processed time, count the tracks whose foot point is inside a zone. If there are no zones and `crowding.whole_frame` is true, count the whole frame instead and use zone name `"whole frame"`.
- **Event:** a run where count ≥ `crowding.min_count` lasting ≥ `crowding.min_duration_s`. Raw runs go out, and events.py merges and filters.
- **Event fields:**
  - `entity_id: null`
  - `entities: [ids present during the event]`
  - `zone`
  - metrics `{"duration_s", "zone", "max_count", "mean_count", "min_count", "min_duration_s", "peak_time_s", "mean_det_conf"}`, where peak is the time of max count
- **Evidence:** `{max} people (avg {mean:.1f}) inside '{zone}' for {dur:.0f} s. Rule: >= {min_count} for >= {min_dur:.0f} s.` (use the entity word)
- **Confidence margin:** `clip01((max_count / min_count - 1) / 0.5)`.
- **Handling the null entity:**
  - render: snapshot shows all entity boxes and the zone highlighted
  - report: card says "Group of N"
  - chains: skips null-entity events
  - evaluate: treats it like any other behaviour
- **Severity:** `medium`, or `high` if max_count ≥ 2 × min_count.
- **Config:** `crowding: {enabled: true, min_count: 5, min_duration_s: 5.0, whole_frame: false}`.

`BEHAVIORS` becomes `("loitering", "zone_intrusion", "running", "fall", "crowding")`. Colours:
- fall: red-orange `#ff4500` / BGR (0, 69, 255)
- crowding: teal `#00a0a0` / BGR (160, 160, 0)

### 4.8 Near-miss intelligence: behaviour key `near_miss` (pairwise, the headline feature)
A near miss is two tracked objects that came dangerously close **while moving relative to each other**, even though nothing visibly happened. Typical cases are a person and a car, bike or truck, or two people colliding in a corridor.

- **Pairs:** a pair (A, B) qualifies if A's class is in `near_miss.vulnerable_classes` (default [0], person) and B's class is in `near_miss.other_classes`. B may also be a person. Each unordered pair is evaluated once. When both are vulnerable, A is the one with the smaller id.
- **Alignment:** compare only times present in both tracks, by matching frame indices.
- **Signals at each common time t:**
  - Scale `s(t)`: A's scale (Track.height). If both are people, use the mean of the two.
  - Distance `d(t) = ||footA - footB|| / s(t)`, using smoothed foot points.
  - Relative speed `rel(t) = ||velA - velB|| / s(t)`. Each velocity is the smoothed foot displacement over `features.speed_window_s`, the same way as speed.
  - Closing speed `vc(t) = -(d(t) - d(t-w)) / w` over the same window (positive while approaching).
  - Time-to-collision `ttc(t) = d(t) / vc(t)` when `vc > near_miss.min_closing_speed_bh_s`, else inf.
- **Depth guard (people pairs only):** skip samples where `|hA - hB| / max(hA, hB) > near_miss.max_scale_diff`. People at very different sizes are at different depths and only overlap in the image.
- **Condition:** the sample is "near" when `d <= near_distance_bh` **and** `rel >= min_rel_speed_bh_s`. It is also near when `ttc <= ttc_s` **and** `d <= 1.5 * near_distance_bh`. Pairs walking together have low relative speed, so they are never flagged.
- **Runs:** runs of near samples (merged by events.py like the others) are raw intervals with `track_id: A`, `entities: [A, B]`, `other_id: B`. A run of a single sample is padded +-0.25 s so it has a span.
- **Metrics:** `{"min_distance_bh", "peak_time_s" (time of min distance), "max_closing_speed_bh_s", "min_ttc_s" (null if inf), "rel_speed_at_closest_bh_s", "other_id", "other_class", "approach_s" (how long d was falling before the closest moment), "contact": min_distance_bh <= contact_bh, "duration_s", "mean_det_conf", "min_duration_s"}`.
- **Event fields:** `entity_id = A`, `entities = [A, B]`, `other_entity = B`.
- **Evidence:** `{A} and {B} came within {dmin:.2f} {unit} of each other at {mm:ss.s}, closing at {vc:.1f} {unit}/s (time-to-collision {ttc:.1f} s). They approached for {approach:.1f} s, then separated. Rule: closer than {near} {unit} while moving >= {rel} {unit}/s relative.` Replace the TTC clause with "possible contact" when `contact` is true. Use `utils.entity_name(cfg, id, cls)`.
- **Confidence margin:** `clip01(1 - dmin / near_distance_bh)`, plus `0.3 * clip01((1.5 - min_ttc) / 1.5)` when the TTC is finite. Cap at 1.
- **Severity:** always `high`. When `contact` is true, the report shows "POSSIBLE CONTACT".
- **Chains:** the event belongs to entity A. Extra pattern titles:
  - `running -> near_miss`: "Ran and nearly collided"
  - `near_miss -> running`: "Near miss, then fled"
  - `zone_intrusion -> near_miss`: "Entered the danger zone and nearly got hit"
- **Rendering:** draw a line between the two foot points coloured by distance (red when near), plus both boxes. The snapshot is taken at the closest moment. The plot shows `d(t)` and the closing speed over the event +-3 s, with the near-distance line.
- **Config (under `behaviors:`):** `near_miss: {enabled, vulnerable_classes: [0], other_classes: [0,1,2,3,5,7], near_distance_bh: 0.6, min_rel_speed_bh_s: 0.8, min_closing_speed_bh_s: 0.3, ttc_s: 1.0, contact_bh: 0.15, max_scale_diff: 0.35, min_duration_s: 0.0}`.
- **Colour:** `near_miss` is crimson `#dc143c` / BGR (60, 20, 220). Display name: "Near miss".
- **Entity names for mixed classes:** `utils.entity_name(cfg, id, cls)` returns `CLASS_NAMES[cls] + " #id"` when the scenario tracks more than one class. Otherwise it uses `entity_word`.

`BEHAVIORS` is now `(loitering, zone_intrusion, running, fall, crowding, near_miss)`, and `utils` has all the colours and names.

### 4.9 Timeline fix
When a track has overlapping events (e.g. loitering and zone_intrusion at the same time), draw them as stacked sub-rows inside that track's row so neither hides the other.

### 4.10 Event keys added by section 4 (all optional, present on every event, null when unused)
- `severity` (set by chains.py)
- `entities` (list of ids; `[entity_id]` for single-entity events)
- `other_entity` (near_miss only)
- `zone` (zone_intrusion and crowding)
- `entity_name` (display string, e.g. "Car #7" or "Group of 6")
- `behavior_name` (scenario display name)


### 4.11 Approaching alert: behaviour key `approaching` (wearable / moving camera, `approaching.py`)
On a moving camera, positions and speeds are meaningless, but **looming** still works:
- The image size `s` of an object is proportional to 1 / distance.
- So `g = d(ln s)/dt` gives the time to contact, `TTC = 1/g`.

The **signals**, all computed per track:
- **growth `g`:** the rate of `ln(Track.height)`, measured over `window_s` within one segment.
- **uniform growth:** `g_w` and `g_h` (the growth of box width and height) must both be > 0, and their ratio must be within `uniform_growth_ratio`.
- **centre offset:** in `[-1, 1]` across the image.
- **predicted offset at contact:** `offset + vx * TTC / (W/2)`.
- **edge guard:** samples marked `at_edge` are ignored.

An **alert** fires when all of these are true at once:
- `TTC <= ttc_s`
- `|predicted offset| <= center_band`
- growth is uniform
- the object is not at the edge
- `size / frame_h >= min_size_frac`

Runs shorter than `min_duration_s` are filtered by events.py. Runs within `cooldown_s` merge into one alert per object.

**Metrics:** `{"duration_s", "min_ttc_s", "max_growth_per_s", "size_frac_at_alert", "center_offset", "direction" ("11 o'clock"), "alert_text", "ttc_threshold_s", "min_duration_s", "peak_time_s", "mean_det_conf"}`.

**Evidence** (assembled in events.py and approaching.py): `"{who} grew 45%/s in view at 12 o'clock (time to contact about 1.6 s) at mm:ss.s. Rule: ..."`.
**Confidence margin:** `clip01((ttc_s - min_ttc) / ttc_s + 0.3)`.
**Severity:** `high` if min_ttc <= 1 s, else `medium`.
**Track.frame_size** is set by build_tracks.

The rule is **off by default**. Only `scenarios/navigation.yaml` turns it on, and that preset turns off all the fixed-camera rules plus chains.

`BEHAVIORS` is now `(loitering, zone_intrusion, running, fall, crowding, near_miss, approaching)`.

### 4.12 Review-round changes
- `behaviors.loitering.bridge_gap_s`: a dropout of up to 3 s at the same spot does not reset the loitering clock (`features.loiter_segments`).
- Near-miss `approach_s` only counts time while the closing speed is above `min_closing_speed_bh_s`.
- Privacy: before a track is confirmed and after it ends, the whole region it occupied is pixelated for 5 processed frames, so there is no face leak on the first sighting.
- run.py:
  - a missing `--config` or an invalid `zones.json` is a clean error (exit 2)
  - `--reuse` refuses caches shorter than `--max-seconds`
  - a stride sparser than `speed_window_s` gives a warning
  - `--no-video` deletes stale videos


### 4.13 Time-of-day rules and three more presets
- **Settings:** `time_of_day.start_time` (`HH:MM[:SS]`, or `--start-time`) is the clock time of the first frame. Every behaviour can have `active_hours`: `""` means always, or windows like `"21:00-06:00"` or `"08:00-12:00, 14:00-18:00"`, which may wrap past midnight.
- **Filter:** `events._apply_active_hours` runs after merging and scoring. It drops each event whose span does not touch its rule's windows (checked every second), and records it in `final["ignored_outside_hours"]` as `{behavior, entity_id, clock_start, clock_end, active_hours}`.
- **Kept events** get `clock_start` / `clock_end` and are renumbered. If they belong to a timed rule, the evidence gets one extra sentence.
- **No start time:** nothing is filtered, and run.py warns.
- **Helpers:** `utils.parse_clock`, `parse_hours`, `in_hours`, `span_in_hours`, `fmt_clock`.
- **New presets:**
  - `public_safety`: whole-frame crowding of 10+, fleeing, collapse, and near misses with bikes
  - `disaster`: person down, not moving 30 s, danger zone, crowd building up, fleeing
  - `agriculture`: people and animals entering a field 19:00-06:00
- **Changed preset:** `elderly` now limits "Left through the exit" to 21:00-06:00.


### 4.14 Aquatic distress (pool) and speaker announcements
The full design is in POOL.md.

Modules:
- `pool_behaviour.py`: stage 2. NumPy only.
- `distress.py`: stage 3. Pure Python.
- `pool.py`: orchestration, pose keypoints cached in `keypoints.json`, and the stage files.
- `pool_ui.py`: stage 4 inside `app.py`.
- `announce.py`: spoken alerts.

`run.py` calls `pool.run_pool` after `build_events` when `pool.enabled`. Pool alerts become normal events with behaviours `aquatic_distress` and `submersion`; `confidence` = risk score, severity is always high, and `metrics` contains `risk_score`, `alert_time_s`, `location`, `evidence_keys` and `risk_series`. Before `write_outputs`, `announce.prepare` writes `alerts/alert_e<id>.wav` and `announcements.json`. With `--announce` they are played after the console report. All thresholds are in `config.yaml` under `pool:` and `announce:`.
