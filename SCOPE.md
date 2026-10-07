# Scope: what is done, what is stretch, what is not done

> **Other systems draw boxes. Ours writes incident reports, proves every claim, and catches accidents that almost happened.**

Problem: HackNEX 2026 PS07, Autonomous Vision and Behaviour Understanding.
Scenario: one fixed camera over a restricted zone in a college or workplace corridor, extended with scenario presets
(workplace, elderly care, livestock, traffic) that re-use the same engine.
Goal: for every unusual event say **which person or object** (stable ID), **what** it did, **when**, and **why** it counts, and
catch the interactions that **almost became accidents**, even when nothing visibly happens in the video.

## MVP (done)

| Part | What it does |
|---|---|
| Detection and tracking | YOLO11n finds objects, ByteTrack gives each a stable ID, and every box keeps its COCO class (`tracker.py`) |
| Per-object features | Smoothed foot point, body size (longest side of the box), aspect ratio, speed and velocity in body sizes per second (`features.py`) |
| Loitering | Stays within 0.5 body-heights of one spot for 10 s or more |
| Zone intrusion | Feet inside a drawn polygon for 1 s or more (`zone_picker.py`) |
| Running | Speed at or above 1.8 body-heights/s for 1 s or more, with hysteresis (stops at 1.4) |
| Near-miss intelligence | Two objects that came within 0.6 body-sizes of each other while moving relative to each other: distance, relative speed, closing speed, time-to-collision, depth guard for people pairs, "possible contact" flag. Works when nothing was hit |
| Fall / person down | Box turns from upright to lying (aspect ratio) and stays down 2 s; running ignores the fall's speed spike |
| Crowding | Count of tracks inside a zone at or above a limit for 5 s or more; one event for the group |
| Event building | Merge pieces, drop too-short ones, confidence in three parts, plain-English evidence (`events.py`) |
| Incident chains | Severity per event; one entity's events linked into an incident with a title and a story (`chains.py`) |
| Highlight reel | `highlights.mp4` with only the incidents, title card, caption bar, fast-forward for long ones (`highlights.py`) |
| Privacy mode | `--privacy` blurs faces in the annotated video, the highlights, the snapshots and the heatmap (`privacy.py`) |
| Scenario presets | `--scenario campus / workplace / elderly / livestock / traffic`: classes, thresholds and wording; your own YAML works too |
| Almost-flagged cases | Cases that reached 70% of a threshold are listed, so a reader sees what the rules ignored |
| Outputs | `events.json`, `summary.txt`, `report.html`, `highlights.mp4`, annotated video, snapshots, plots, heatmap, timeline |
| Tooling | `run.py` CLI (`--reuse`, `--scenario`, `--list-scenarios`, `--privacy`), processing-speed line, `evaluate.py` with `labels.csv` for all six behaviours, unit tests, Colab notebook with a near-miss demo |

## Stretch (built, optional)

| Part | Notes |
|---|---|
| Pose check (`--pose`) | A pose model looks at a few frames of each running or loitering event and says `true`, `false` or `uncertain`. It is a second opinion, never a gate: events are kept either way |
| Webcam (`--webcam SECONDS`) | Records the laptop camera to `samples/webcam_<n>.mp4` with the measured frame rate, then analyses it. Good for a live demo |
| Scene baseline | Compares each entity with the others in the same video (median and MAD). Marks one as unusual when far from the scene's normal and raises its severity one level. Needs at least 4 tracks |
| Approaching alert (`--scenario navigation`) | Prototype for a WEARABLE camera: time to contact from how fast an object grows in view (1 / growth rate), only if it will be in the walking path at contact, grows uniformly, and is already close. One alert per object per 3 s. Unit-tested (10 tests), offline only; real false-alarm rate still to be measured with the 5-walk protocol in DEMO.md |
| Synthetic safety clip (`tests/synthetic_safety.py`) | A drawn workplace scene with ground truth (near miss, speeding cyclist, fall) so the headline features can be shown and scored without real accidents: precision 1.00, recall 1.00 |

## Domains the presets target, and their real status

| Domain | Preset | Status |
|---|---|---|
| Workplace / industrial safety | `workplace` | tested on a synthetic clip with ground truth (near miss, fall); not on real factory footage; forklifts need a custom detector (COCO has none) |
| Elderly care | `elderly` | preset only (night-time exit rule via `--start-time`) |
| Agriculture | `agriculture` | preset only: people and animals entering a field at night; crop health not covered |
| Transportation | `traffic` | preset only; run end to end on the Intel person-bicycle-car clip |
| Public safety | `public_safety` | loitering, zone intrusion and running tested; crowding (whole frame) unit-tested only |
| Education / campus | `campus` | tested (Intel clips plus a synthetic clip with ground truth) |
| Logistics / warehouse | `workplace` | preset only; forklifts and pallet trucks need a custom detector |
| Water safety (pools) | `pool` | **team scenario**: stages 2-4 built and tested on a synthetic staged story (precision/recall 1.00); swimmer detector training kit ready; needs a trained model and staged real clips (POOL.md) |
| Disaster response | `disaster` | preset only (fixed camera); drones and body cams need the moving-camera work |
| Animal / livestock | `livestock` | preset only |
| Accessibility | `navigation` | prototype: approaching alert for a wearable camera, unit-tested, not yet measured on real chest-camera walks |

"Tested" means unit tests plus a run on real sample footage of that kind. "Preset only" means the preset loads and the
rules are unit-tested, but it was not validated on real footage of the domain. "Design only" means something is missing.
The full table with the rules per domain is in [README.md](README.md#which-domains-this-covers-honestly).

## Not done (on purpose)

| Part | Why not |
|---|---|
| Forklift / machine detector | Needs a custom-trained model and labelled footage from the real site. The rules do not change: swap `model.weights` |
| Ground-plane calibration, depth estimation | Needs camera calibration or a depth model. Distances stay in the image plane, in body sizes, so nothing has to be measured on site |
| Real-world units (km/h, metres) | Follows from the line above; speeds are body sizes per second |
| Learning the thresholds | They are set by hand and explained in the report. There is no labelled near-miss dataset to learn from |
| Face recognition, identity tracking across days | Out of scope and against the privacy goal. IDs are per video |
| Streamlit timeline app | The static `report.html` already carries the evidence and needs no server |
| Wrong-way rule | Needs a defined flow direction per camera; out of scope for the first version |
| ONNX export | PyTorch weights are enough for this project; faster deployment formats can come later |
| Re-identification after long occlusion | An object that leaves the view for many seconds gets a new ID. ByteTrack keeps IDs through short gaps only (`track_buffer: 60` frames) |
| Multi-camera fusion | One fixed camera per run |

## Known limitations

- **Distances are 2D image distances.** No ground-plane calibration and no depth. Two objects that overlap in the picture can be far apart in the world (a car passing behind a person) and the reverse. The depth guard compares box sizes of two people only; for a person and a vehicle there is no depth cue, so a vehicle passing behind a pedestrian can raise a false near miss. The near-miss thresholds come from geometry and were not tuned on real near-miss footage.
- **No forklift class in COCO.** Forklifts, pallet trucks and machines are missed or called "truck" or "car". Industrial near-miss monitoring needs a custom detector.
- **Presets are starting points,** not validated products. Only the campus scenario was checked on real footage; thresholds for the other presets are reasoned from the geometry of the domain.
- **Fall is a box-shape rule.** Sitting, bending, crawling, or a person cut off by the frame edge can look like a fall or hide one. It is off for animals and vehicles, whose boxes are wide all the time.
- **Crowding needs a zone** (or `crowding.whole_frame`) and counts foot points; dense crowds cause missed detections.
- **Privacy mode is a box-based blur,** not a face detector and not an anonymisation guarantee. A person the detector misses stays visible. The original video is untouched.
- **ID switches when objects cross or hide behind each other.** The same object can get a new ID, which splits one event in two. Events of the same behaviour closer than 1.5 s are merged again, which repairs short breaks. Incident chains link events by ID, so an ID switch can split a chain.
- **Walking straight toward the camera looks slow.** Speed is measured on the image plane, so motion along the camera axis changes the box size more than the foot position. Running toward the camera can be missed.
- **Low light, motion blur and heavy compression** lower detection confidence and make boxes jump. The confidence score includes the detection confidence for this reason.
- **Crowded scenes** (many overlapping people) cause missed detections and ID switches. The system is designed for a few objects at a time.
- **The camera must be fixed.** Speed, distance and zones assume the background does not move. A panning or shaking camera breaks all three.
- **Body size is the longest side of the box,** so an object that is cut off by the frame edge or an obstacle changes the unit. Boxes smaller than 20 px are ignored.
- **Thresholds are set by hand,** not learned. Tune them for each camera in `config.yaml` (or in a preset) and measure the result with `evaluate.py`.
- **Speed.** About 6 to 7 processed frames per second on a laptop CPU with every 2nd frame at 640 px (measured); a GPU is much faster. The report header shows the real figure.
- **Classes are the stock COCO set.** Tracked classes are whatever the preset lists (person, bicycle, car, motorbike, bus, truck, dog, horse, sheep, cow, ...); anything else is not analysed.
