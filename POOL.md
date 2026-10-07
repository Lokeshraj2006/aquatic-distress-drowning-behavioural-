# Aquatic Distress Behaviour Intelligence (HackNex HNX26PSI07)

We are implementing HackNex HNX26PSI07, Autonomous Vision & Behaviour Understanding. Our scenario is Aquatic Distress Behaviour Intelligence for swimming pools.

What the system does:
- detects and persistently tracks swimmers
- extracts behavioural signals
- analyses behaviour over time
- identifies the transition from normal aquatic activity to high-risk distress
- produces an evidence-backed alert with the person ID, location, timestamp, risk score and behavioural evidence
- can **announce it on a speaker**

It is **not** a person detector and **not** a single-frame swimming/drowning classifier. It is an **early-warning assistant for lifeguards**. It does not replace them.

## Run it

```powershell
python run.py --video pool.mp4 --zones pool_zones.json --scenario pool --announce
python tests/synthetic_pool.py                    # the staged story, drawn, with ground truth
python run.py --video samples/synthetic_pool.mp4 --zones tests/synthetic_pool_zones.json --scenario pool --out outputs/synthetic_pool --reuse
streamlit run app.py                              # Lifeguard dashboard: scenario "Aquatic distress (swimming pool)"
```

Result on the synthetic story:
- Swimmer #3 is flagged **HIGH-RISK AQUATIC DISTRESS** with the alert at 00:22. The behaviour began at 00:17, about 6 s before the head goes under.
- Then **POSSIBLE SUBMERSION** at 00:29.8.
- Risk is 98%.
- The lap swimmer, the swimmer resting at the wall, the diver and the lifeguard on the deck all stay normal.
- Scored against the ground truth: precision 1.00, recall 1.00.

## The common pipeline

```
VIDEO -> [1 DETECTION + TRACKING] -> Swimmer #1 #2 #3 -> [2 BEHAVIOUR] -> posture, motion, arms, head
      -> [3 TEMPORAL REASONING] -> normal -> distress transition -> [RISK / EVENT] -> EARLY WARNING
      -> [4 DASHBOARD + SPEAKER]
```

| Stage | Owner | Code | Output file (data contract) | Run on its own |
|---|---|---|---|---|
| 1 Detection + tracking | Loki (model) | `tracker.py` (YOLO + ByteTrack), `training/` | `stage1_tracking.jsonl` | `python tracker.py --video X --out detections.json` |
| 2 Behaviour | Minje | `pool_behaviour.py` (+ pose keypoints in `pool.py`) | `stage2_behaviour.jsonl` | `python pool_behaviour.py --tracks stage1_tracking.jsonl --zones z.json` |
| 3 Temporal reasoning | Person 3 | `distress.py` | `stage3_events.json` | `python distress.py --behaviour stage2_behaviour.jsonl` |
| 4 Dashboard + alert | Person 4 | `pool_ui.py` (inside `app.py`), `announce.py` | `pool_dashboard.json`, `alerts/*.wav`, `announcements.json` | `streamlit run app.py` |

Each stage reads only the previous stage's file. A teammate can rewrite their module (with any AI) and the rest still works, as long as the file format below stays the same.

## Data contract (exact fields)

**Stage 1**: one line per box. The first line is `{"meta": {...}}`.
```json
{"person_id": 3, "timestamp": 84.2, "frame": 2105, "bbox": [120, 80, 250, 400], "conf": 0.87, "class": 1, "state": "swimmer_vertical"}
```
`state` is only present with a custom swimmer detector (`pool.state_classes`). bbox is in processing pixels (frame resized to 640 wide).

**Stage 2**: one line per swimmer every 0.5 s, looking at the last 2 s. The contract fields come first, then the extras.
```json
{"person_id": 3, "timestamp": 84.2, "posture": "vertical", "movement": "low", "arm_motion": "repeated",
 "displacement": 0.05, "head": "unstable", "near_wall": false, "in_water": true, "location": "Deep End",
 "speed": 0.08, "progress_ratio": 0.09, "net_displacement": 0.1, "path_length": 1.1,
 "arm_hz": 1.04, "arm_regularity": 0.1, "arm_amplitude": 0.14, "head_visibility": 1.0, "torso_angle": null,
 "activity": "floundering", "confidence": 0.8, "track_valid": true,
 "state_source": "box_shape", "arm_source": "pose", "posture_raw": "vertical", "bbox": [...]}
```

The fields:
- **posture:** horizontal / diagonal / vertical / head_only / underwater / out_of_water. The source, in order of preference:
  1. the trained detector
  2. the **torso angle**, when the hips are visible
  3. the box shape

  A new posture must last **1 s** before it is reported (`posture_raw` holds the unheld value).
- **movement:** low / medium / high.
- **displacement:** net progress in body-heights **per second**. **progress_ratio** = net ÷ path (1 = straight, about 0 = going nowhere).
- **arm_motion:** repeated / **stroking** (rhythmic arms while really travelling = swimming) / calm / irregular. It comes from **wrist peaks per second** (`arm_hz`) and their regularity.
- **head:** stable / unstable / submerged / unknown. It comes from head visibility, bobbing, and (when upright) whether the head is held clearly above the shoulders.
- **activity:** a *hint* only (swimming / floating / treading / floundering / submerged / unclear). Stage 3 never decides from it.
- **track_valid:** false for 2 s after the box jumps (an ID switch is suspected). Stage 3 skips those windows.

Several of these ideas come from Minje's `behaviour_analysis.py` prototype:
- torso angle and the posture hold
- progress ratio
- peak-counted arm rate and regularity
- head visibility
- ID-switch guard and keypoint smoothing
- confidence, activity hint and the signal plots

Smoothed keypoints feed the slow signals; raw keypoints feed the rhythm signals, because smoothing damps the oscillation we look for. `python pool_behaviour.py --tracks stage1_tracking.jsonl --plots` draws one signal plot per swimmer.

**Stage 3**: events, plus a timeline per swimmer.
```json
{"person_id": 3, "event": "high_risk_aquatic_distress", "start_time": 17.0, "alert_time": 22.0, "end_time": 29.5,
 "risk_score": 0.98, "location": "Deep End",
 "evidence": ["vertical_posture", "low_displacement", "repeated_arm_motion", "unstable_head", "transition_from_swimming"]}
```
`event` is `high_risk_aquatic_distress` or `possible_submersion`. The timeline reads, for example: `2.0 s Normal swimming -> 11.5 s Movement slowing -> 14.5 s Vertical posture -> 17.0 s Repeated arm movement -> 22.0 s HIGH-RISK AQUATIC DISTRESS`.

**Stage 4**: `pool_dashboard.json` holds the people, a risk curve per swimmer, the timelines, the alerts and the thresholds. The UI just displays it.

## How the decision is made (stage 3)

1. **Signals:** each sample gets a **risk** equal to the sum of the warning signs present. The weights are in `config.yaml` under `pool.weights`:

   | Signal | Weight | Note |
   |---|---|---|
   | Vertical posture | 0.25 | |
   | Little or no forward progress | 0.25 | |
   | Repeated arm motion | 0.25 | counted only while **upright**; swimming strokes are normal |
   | Unstable or submerged head | 0.15 | |
   | Transition | 0.10 | the swimmer was swimming normally within the last 30 s |

2. **Look-alike damping:**
   - **Resting at the wall** (wall zone, calm arms): risk × 0.3.
   - **Treading water** (upright, calm arms, steady head): risk capped at 0.45.
   - **On the deck** (outside the water zone): risk 0.
3. **State machine:** NORMAL → WATCH (0.3) → WARNING (0.55) → **DISTRESS** when risk ≥ 0.7 is **held for 5 s**. The alert is raised then; the behaviour start time is recorded too. The run only ends below 0.5 (hysteresis).
4. **Submersion:** a swimmer in WARNING or DISTRESS who is **lost from view for 1.5 s**, or whose **head is submerged for 1 s**, triggers POSSIBLE SUBMERSION. A dive from normal swimming doesn't count, because no warning came before it.

No single frame and no single signal can raise an alert. Every number is in `config.yaml` (section `pool:`).

## Speaker announcement

- **What it says:** "Attention lifeguard. Swimmer number 3 may be in distress at the Deep End. Risk 98 percent." For a submersion: "Urgent. Swimmer number 3 may have gone under at the Deep End. Check now."
- **The audio:** an alarm tone followed by the voice. Every alert gets `alerts/alert_e<id>.wav`.
- **Voices:** offline. Windows uses its built-in voices (System.Speech); macOS uses `say`; Linux uses `espeak`. If no voice is found, the alert is the alarm tone only.
- **Playing it:**
  - `--announce` plays it on this computer's speaker, twice (`announce.repeat`).
  - The dashboard plays it once in the browser, through the device showing it.
- **Settings:** `announce.message`, `voice`, `rate` and `min_risk` in `config.yaml`.

## Zones for a pool (optional, recommended)

Draw them with `zone_picker.py` and name them with these words:
- `Deep End` / `Shallow End`: the location shown in the alert.
- `Pool edge` or `wall`: resting at the wall is normal there.
- `water`: only people inside it are judged; the deck is ignored.

## Current limits (say them honestly)

- **Tested on synthetic data only.** Stages 2–4 are tested on a drawn, staged story with ground truth. There is no real distress footage, and there must never be. Real accuracy has to come from **staged** clips (below).
- **The stock detector isn't a swimmer detector.** COCO YOLO misses swimmers (splash, refraction, only a head visible). The fine-tuned model from `training/` is the fix.
- **The pose model struggles in water.** Without keypoints, the arm and head signals fall back to box movement, which is weaker. On the synthetic clip, #3 then only reaches WARNING, not the alert. That's the safe direction.
- **One fixed camera.** Distances are in the image (2D), not metres. Glare and occlusion by other swimmers reduce tracking quality.

## Filming protocol (staged clips, lifeguard present)

Never film a real emergency for this. **Stage** each scenario with good swimmers, a lifeguard in the water nearby, and shallow-enough water for the actor's safety.
- **Camera:** fixed, high (2–4 m) on the side wall, looking down at about 30–45°. Use 1080p at 25–30 fps, avoid direct sun glare, and keep each clip 40–90 s.

| # | Scenario (actors) | Expected |
|---|---|---|
| 1 | Normal laps, 2–3 swimmers | no alert |
| 2 | Treading water and chatting in the deep end | no alert (treading) |
| 3 | Resting at the wall, holding the edge | no alert (resting) |
| 4 | Kids splashing and playing | no alert |
| 5 | A dive under for 3–5 s, then surfacing | no alert (dive) |
| 6 | **Staged distress:** swim, slow down, turn upright, press the arms down repeatedly, little progress, head low, then the actor ducks under | alert about 5 s after the struggle starts, then possible submersion |

- **Labels:** write them like `tests/synthetic_pool_labels.csv`: `clip, who, aquatic_distress/submersion/none, start_s, end_s`.
- **Scoring:** `python evaluate.py --labels labels.csv --outputs outputs/`.
- **For the judges, report:** alerts caught, false alarms on clips 1–5, and seconds of warning before the actor ducked under.
