# PS07: Autonomous Vision and Behaviour Understanding

> **Team scenario: Aquatic Distress Behaviour Intelligence (swimming pools).** See **[POOL.md](POOL.md)** for the
> pool pipeline (detection + tracking -> behaviour -> temporal distress reasoning -> lifeguard dashboard + speaker
> announcement), the data contract and the filming protocol. Run it with `--scenario pool`. The training kit
> for the swimmer detector is in [training/](training/README.md). The rest of this README describes the shared engine.

**HackNEX 2026 | NEXUS Club, Karunya**

> **Other systems draw boxes. Ours writes incident reports, proves every claim, and catches accidents that almost happened.**

A fixed camera watches a corridor, a shop floor, a road or a barn. You give the system a video. It does not just list
detections. It returns a short report that says **which person or object** (a stable ID) **did what** (near miss, fall,
crowding, loitering, zone intrusion, running), **when** (start and end time), and **why it counts** (the measured numbers,
a confidence score, a severity and a snapshot). Every flagged event is an *evidence card* a guard or a safety officer can
read in ten seconds. Events of one entity that belong together are linked into an *incident chain* with a plain-English
story, and a `highlights.mp4` plays only the incidents. Events that almost reached a rule but did not are listed too, so
a reader can see what was ignored and why.

The headline feature is **near-miss intelligence**. Instead of detecting an accident after it happens, the system reads the
*trajectories*: where each worker and each vehicle was, how fast, in which direction, how close they got, and how fast they
were closing. It reports "this interaction almost became an accident" even when nothing visibly happens in the video, which
a conventional incident detector would miss entirely.

```
Worker  ------->
                  \
                   !  NEAR MISS   (distance, relative speed, closing speed, time-to-collision)
                  /
Vehicle ------->
```

Everything runs on a pretrained detector with plain, explainable rules: no training, no external API, and every threshold
lives in one file (`config.yaml`). Scenario presets (`scenarios/*.yaml`) re-target the same engine to another domain.

An evidence card (illustrative numbers, not a measured result):

```
Event 2 | Person #5 | RUNNING | 00:41 - 00:48 (6.8 s) | confidence 0.83 | severity medium
Why:       Moved at 2.5 body-heights/s (peak 3.1) for 6.8 s.
           Rule: >= 1.8 bh/s for >= 1 s; walking is about 0.6-1.0.
Confidence: margin 0.80, duration 1.00, detection 0.70
            = 0.4*0.80 + 0.3*1.00 + 0.3*0.70 = 0.83
Baseline:  Moved 2.9x faster than the scene's typical person
Pose:      verified = true   (only with --pose)
Evidence:  snapshots/e2.jpg   plots/e2.png
```

A near-miss card (illustrative numbers, not a measured result):

```
Event 4 | Person #3 and Car #7 | NEAR MISS | 00:14 - 00:15 | confidence 0.76 | severity high
Why:       Person #3 and Car #7 came within 0.42 body-lengths of each other at 00:14.2, closing at
           2.1 body-lengths/s (time-to-collision 0.2 s). They approached for 1.6 s, then separated.
           Rule: closer than 0.6 body-lengths while moving >= 0.8 body-lengths/s relative.
Evidence:  snapshots/e4.jpg (taken at the closest moment)   plots/e4.png (distance and closing speed over time)
```

## Pipeline

```
video -> tracker.py -> features.py -> behaviors.py -> events.py -> (pose_verify.py) -> chains.py -> render.py -> highlights.py -> report.py
         YOLO11n +     per-object     raw intervals   merge, filter,  optional --pose    severity +    video, snapshots,  highlights.mp4   events.json,
         ByteTrack     smoothed foot, + almost-       confidence,     true/false/        incident      plots, heatmap,    (incidents only) report.html,
         detections    scale, speed,  flagged cases   evidence        uncertain          chains        timeline                            summary.txt
                       class, aspect  + baseline
```

`run.py` wires the steps together. `evaluate.py` compares `events.json` with a hand-written `labels.csv`.
The binding contract between modules (function names, dict keys, file formats) is [DESIGN.md](DESIGN.md).

Interactive diagrams (open in a browser):

- [Architecture](docs/diagrams/architecture.html)
- [Data pipeline](docs/diagrams/data-pipeline.html)
- [Workflow](docs/diagrams/workflow.html)

## Quick start

### Local (Windows, macOS, Linux; Python 3.11)

```powershell
git clone https://github.com/Lokeshraj2006/aquatic-distress-drowning-behavioural-.git
cd aquatic-distress-drowning-behavioural-
py -3.11 -m venv .venv
.venv\Scripts\activate              # macOS / Linux: source .venv/bin/activate
# NVIDIA GPU? Install torch first from https://pytorch.org (pick your CUDA version).
pip install -r requirements.txt
python get_samples.py               # Intel sample clips (CC BY 4.0) + synthetic test clips with ground truth
python run.py --video samples/one-by-one-person-detection.mp4
```

Already-computed results are committed in [examples/](examples/) (open `report.html` there), so you can see the
output before installing anything.

Open `outputs/one-by-one-person-detection/report.html` in a browser (and play `highlights.mp4` next to it). The YOLO
weights (`yolo11n.pt`) download automatically the first time and are cached afterwards.

The near-miss demo. Real near misses are rare in public footage, so `get_samples.py` draws a synthetic workplace
clip with known ground truth (a cyclist nearly hits a pedestrian at about 6 s, a worker falls at about 18 s):

```powershell
python run.py --list-scenarios
python run.py --video samples/synthetic_safety.mp4 --scenario workplace --out outputs/synthetic_safety --reuse
python evaluate.py --labels tests/synthetic_safety_labels.csv --events outputs/synthetic_safety/events.json --clip synthetic_safety.mp4
```

Result: near miss, cyclist speeding and fall all found, precision 1.00, recall 1.00, timing error under 0.2 s.
On real footage, `python run.py --video samples/person-bicycle-car-detection.mp4 --scenario traffic` correctly reports
**no incidents**: in that car park nobody comes closer than about 4.7 body-heights to anyone (a true negative).
Record your own clip (a bike or trolley passing close to a walking person) for a real near miss.

With a restricted zone (draw it once with the mouse; see [Zones](#zones)):

```powershell
python zone_picker.py --video samples/worker-zone-detection.mp4        # click corners, Enter to save
python run.py --video samples/worker-zone-detection.mp4 --zones zones.json
```

Faster tuning loop: after one full run, change `config.yaml` and re-run only the rules (YOLO is skipped):

```powershell
python run.py --video samples/one-by-one-person-detection.mp4 --reuse
```

Run the unit tests (no GPU and no YOLO weights needed; they use the packages from `requirements.txt`, including
OpenCV for the rendering tests). Run `python get_samples.py --no-intel` first so the synthetic clips exist:

```powershell
pip install pytest
python -B -m pytest -p no:cacheprovider -q tests
```

### Google Colab (free T4 GPU)

1. Zip the `ps07` folder (leave out `.venv` and `outputs`), or push it to a git repository.
2. Open `colab_run.ipynb` in Colab (`File > Upload notebook`), then `Runtime > Change runtime type > T4 GPU`.
3. Run the cells in order. The notebook uploads or clones the code, installs the requirements, downloads the Intel
   sample videos, lets you pick a **scenario** and the **privacy** switch, lets you define a zone from a coordinate
   grid, runs the analysis, and has a **near-miss demo** (workplace preset on the synthetic safety clip, plus the real
   car-park clip as a true-negative check). The results
   cell plays `highlights.mp4` first, then `report.html`, then `annotated.mp4`. A last cell zips the results.

## Web UI (Streamlit)

The same pipeline with a browser front end. It is the main demo surface; the command line stays the fallback.

```powershell
pip install -r requirements.txt              # includes streamlit
streamlit run app.py                         # opens http://localhost:8501
```

The UI does not re-implement anything. **Run** starts `run.py` as a subprocess with exactly the flags you could type
yourself (the command is shown under "Command used"), streams its output live, and then reads the files that `run.py`
wrote. It never imports torch or ultralytics, so the page stays fast. Settings are in `.streamlit/config.toml` (light
theme, uploads up to 500 MB, and the first-start e-mail prompt is switched off so a fresh laptop does not stop at the
terminal).

A short tour:

1. **Sidebar, 1. Video.** Pick a clip from `samples/`, or upload one. Uploads are saved to `samples/uploads/` under a
   safe file name (spaces and odd characters are replaced), and can be picked again later. A file that is not a video
   is refused and deleted at once.
2. **2. Scenario.** One of the presets in `scenarios/`, shown by its title, with the classes it tracks and the rules that
   are on.
3. **3. Zone (optional).** None, a **rectangle** (four sliders in original video pixels, drawn live on the first frame;
   written to `outputs/_ui/<video>_zones.json`), or a **zones file** (any `*zones*.json` in `samples/`, `tests/` or
   `examples/`, or an uploaded one).
4. **4. Options.** Privacy mode, device (auto, cpu, 0), frame stride (its default comes from the scenario: the
   navigation preset uses 1), max seconds, *reuse cached detections* (on by default: when the same video was analysed
   before, YOLO is skipped and a run takes seconds instead of minutes), *skip videos* (faster), and the pose check
   (downloads a 6 MB model the first time).
5. **Main area.** The first frame with the zone drawn on it, next to the video facts (duration, fps, size). Press
   **Run analysis**. A progress bar follows the `[k/N]` steps and the log scrolls live. If `run.py` fails, the last 20 lines
   are shown as an error and the app keeps working.
6. **Results**, as metric tiles (incidents, events, people or objects, video length, processing speed, scenario) and tabs:
   *Summary* (the guard summary and the incident chains with severity badges), *Highlight reel* (`highlights.mp4`),
   *Incidents* (one card per event: severity, who, behaviour, time, confidence and its three parts, the evidence
   sentence, snapshot, plot, pose verdict; below them the almost-flagged cases and the unusual entities), *Timeline &
   heatmap*, *Full report* (`report.html` embedded), *Annotated video* and *Downloads* (`events.json`, `summary.txt`,
   `report.html`, `highlights.mp4` and a zip of the whole folder). A file that is missing from a folder is skipped
   with a one-line note.
7. **Open existing results** (bottom of the sidebar) lists every `outputs/*/` and `examples/*/` folder that holds an
   `events.json`. This is the demo fallback: it shows finished results without running anything.

Fast demo of the headline feature: pick `synthetic_safety.mp4`, scenario *Industrial / workplace safety*, press Run. The
cached detections in `outputs/synthetic_safety/` are reused (the app copies them into `outputs/synthetic_safety_workplace/`),
so the near miss, the speeding cyclist and the fall appear in about 10 seconds.

What the page does when something goes wrong (each of these has a test in `tests/test_app.py`):

- **A reload keeps the results.** The last run (command, log, results folder, video and scenario) is saved to
  `outputs/_ui/last_run.json`. A new browser tab, or a restarted server, shows it again and puts its video and scenario
  back in the sidebar. If the sidebar later points to another video or scenario, a note says that the results below belong to the
  earlier run.
- **Cancel** by changing any setting, pressing Stop (top right) or closing the tab. `run.py` and the programs it
  started (ffmpeg, and the real Python behind the venv launcher) are stopped, so nothing keeps running in the background.
  While `run.py` is silent the title shows the seconds since Run was pressed. The log keeps the last 5000 lines and cuts
  a line after 2000 characters.
- **Bad input is explained, not crashed.** A zones file with the wrong layout, with NaN, infinite or absurdly large
  numbers, or without any zone disables Run and says why; so does a rectangle with no area (swapped corners are simply
  sorted). A zone that lies completely outside the frame gets a warning. The same file name with a different video gets
  its own file (a short tag is added), because cached detections are matched by file name. An `events.json` with odd or
  missing parts shows what it can instead of a traceback.
- **Cached detections are predicted and protected.** The page applies the same checks as `run.py --reuse` (name, stride,
  classes, frame size, resize width, detector, `--max-seconds`; a test compares its answer with `run.py` itself). It
  says whether YOLO will run and roughly how long it takes, names the presets that do have cached detections for the
  chosen video, and keeps a copy of old detections in `outputs/_ui/cache/` before a run replaces them (campus and
  workplace runs of one video share `outputs/<video>/`, so a wrong preset must not destroy the fast path).

The UI code is `app.py` (the page) and `ui_helpers.py` (plain Python: finding files, building the `run.py` command,
reading results; it has its own tests). `tests/test_app.py` drives the page headless with Streamlit's `AppTest`,
including a real run with `--reuse`.

## Near-miss intelligence

A **near miss** is two tracked objects that came dangerously close **while moving relative to each other**, even though
nothing visibly happened. Typical cases: a worker and a vehicle, a pedestrian and a car or bicycle, two people colliding
in a corridor. Nothing has to go wrong in the video, so an accident detector would not fire.

**How it works** (pure geometry on the tracks; no extra model):

1. **Pairs.** A pair (A, B) qualifies when A's class is in `behaviors.near_miss.vulnerable_classes` (default: person) and B's
   class is in `other_classes` (default: person, bicycle, car, motorbike, bus, truck). Each unordered pair is checked
   once. Only the times when both were tracked are compared.
2. **Distance in body-heights.** `d(t)` is the distance between the two smoothed foot points divided by A's body size
   (the longest side of its box). Using the vulnerable object's own size means the numbers do not depend on how far away
   the pair is from the camera. If both are people the mean of the two sizes is used.
3. **Relative speed.** `rel(t)` is the length of the difference of the two velocity vectors (smoothed foot displacement over
   `features.speed_window_s`), in body-heights per second. Two people walking side by side have a relative speed near
   zero, so they are never flagged.
4. **Closing speed.** `vc(t)` is how fast the distance is shrinking, `-(d(t) - d(t-w)) / w`. It is positive while the two
   approach each other.
5. **Time-to-collision.** `ttc(t) = d(t) / vc(t)` when the closing speed is above `min_closing_speed_bh_s`, otherwise
   infinite (they are not really approaching).
6. **Depth guard (people pairs only).** If the two boxes differ by more than `max_scale_diff` (35 percent) in size, the
   people are at very different depths and only overlap in the picture, so that moment is skipped.
7. **Rule.** A moment is "near" when the distance is at most `near_distance_bh` (0.6) **and** the relative speed is at least
   `min_rel_speed_bh_s` (0.8). It is also near when the time-to-collision is at most `ttc_s` (1.0) and the distance is
   within 1.5 times `near_distance_bh`. Runs of near moments are merged like every other behaviour into one event.
8. **Evidence.** The card gives the closest distance and its time, the closing speed, the time-to-collision, and how long
   the pair approached before separating. Closer than `contact_bh` (0.15) is shown as **POSSIBLE CONTACT**. The snapshot is
   taken at the closest moment and draws a line between the two feet, red when near. Severity is always **high**.
9. **Confidence.** margin = `1 - closest distance / near_distance_bh`, plus `0.3 x clip01((1.5 - ttc) / 1.5)` when the
   time-to-collision is finite (capped at 1), combined with the duration and detection scores like every other event.

The near-miss rule is on by default in `campus` (person versus person only, since only people are tracked there) and is the
headline of the `workplace` and `traffic` presets, which also track vehicles.

## Other rules added in round 2

**Fall / person down (`fall`).** The aspect ratio `width / height` of the box is smoothed like the foot point. A person is
*down* when it is at least `down_min_aspect` (1.2) and the box does not touch the frame edge (a cut-off box does not
count). A fall event is a run of down samples lasting at least `min_down_s` (2 s). It says whether the person went from
upright (aspect at most 0.8) to lying within `transition_s` (2 s) before. The running rule ignores samples where the person
is down (a fall's speed spike is not a run); loitering still works on a person who is down. Severity is always **high**.
It is the headline of the `elderly` preset. It is switched off for animals and vehicles, whose boxes are wide all the time.

**Crowding (`crowding`).** At each time the tracks whose foot point is inside a zone are counted (or the whole frame when no
zone is drawn and `crowding.whole_frame` is true). A run with at least `min_count` (5) for at least `min_duration_s` (5 s)
is one event. It belongs to a group, not to one entity: the card says "Group of N" and the snapshot shows everyone present.
Severity is medium, or high from twice the minimum count.

**Incident chains (`chains.py`).** Every event gets a severity (zone intrusion medium; running medium at confidence 0.6 or
more, else low; loitering low; fall and near miss high; one level up when the scene baseline says the entity is unusual).
Events of one entity that follow each other within `chains.max_gap_s` (15 s) become one *incident* with a title, a severity
(high when it holds a zone intrusion plus another behaviour, or any fall or near miss; otherwise medium) and a story with the gaps in seconds:

| Sequence | Title |
|---|---|
| loitering -> zone intrusion | Waited nearby, then entered the restricted zone |
| running -> zone intrusion | Rushed into the restricted zone |
| zone intrusion -> running | Ran away after entering the restricted zone |
| loitering -> running | Waited, then suddenly ran |
| running -> near miss | Ran and nearly collided |
| near miss -> running | Near miss, then fled |
| zone intrusion -> near miss | Entered the danger zone and nearly got hit |
| anything else | Repeated suspicious activity |

Incidents are listed above the event cards in `report.html`, first in `summary.txt`, under `incidents` in `events.json` and
below the events table in the console.

**Highlight reel (`highlights.py`).** `highlights.mp4` contains only the incidents: each incident chain (or each event that
is not in a chain) with `pad_s` (1.5 s) of context on both sides, overlaps merged, in time order. A segment longer than
`max_segment_s` (8 s) is played fast-forward with a "FAST-FORWARD xk" tag. It opens with a 2 s title card ("HIGHLIGHT REEL",
the video name, "N incidents in mm:ss of video"; or "No incidents"), and every frame has the same overlays as
`annotated.mp4` plus a caption bar: incident number, who, behaviour, time span, severity. `report.html` links to it at the top.
`--no-video` skips it.

**Privacy mode (`privacy.py`, `--privacy`).** Blurs (pixelates) the top `head_fraction` (28 percent) of every person box in
the annotated video, the highlight reel, the snapshots and the heatmap background, *before* the overlays are drawn. With
`privacy.mode: body` the whole person box is blurred. Detection, tracking and the pose check still use the original frames,
so the results are identical; only the images that leave the system change. `report.html` and `summary.txt` say
"Privacy mode: faces blurred in all outputs". This is a box-based blur, not a face detector: see the limitations.

**Speed line.** The console and the report header show how fast the detector ran ("Processed at X frames/s on <device>"). It is
unknown with `--reuse`, because YOLO is skipped.

## Approaching alert (navigation prototype, wearable camera)

Every rule above assumes a **fixed** camera. The `navigation` preset is for a camera worn on the chest of a person
who is blind or has low vision. There, everything slides across the image as the wearer walks, so positions and
speeds mean nothing. What still works is **looming**: an object that is coming at you gets bigger in the picture.

* Growth rate `g = d(ln size) / dt`, and **time to contact = 1 / g** (image size is inversely proportional to distance).
  This uses size change only, so it works while the wearer walks.
* An alert fires when time to contact <= 2.5 s, the object will be **in the walking path at the moment of contact**
  (its predicted horizontal position, so someone in the next lane who will pass beside you is not flagged), its width and
  height **grow together** (raising arms or bending is not an approach), it is already close (>= 25% of the frame height),
  and this holds for 0.3 s. One alert per object per 3 s (no spam in a crowd).
* Output, ready to be spoken: *"Person approaching from 11 o'clock, about 1.6 seconds away."*

```powershell
python run.py --video my_chest_cam_walk.mp4 --scenario navigation
```

Status: **prototype**. It is unit-tested (10 tests: head-on approach alerts at least 1 s before contact; walking away,
next lane, far away, slow drift and raised arms do not). It runs offline on a recorded walk; a live audio version
needs real-time inference on a phone. Real camera shake and crowded pavements still need measuring; see
[DEMO.md](DEMO.md) for a 5-walk recording protocol that gives a real false-alarm rate. It is a heads-up aid, not a
replacement for a cane or a guide dog.

## Scenario presets

A preset is a **partial config** in `scenarios/<name>.yaml`, merged over `config.yaml` by `--scenario NAME`. It can change what
is tracked (`model.classes`), any threshold, and the wording (`scenario.*`: title, entity word, unit word, display names of
the behaviours). The internal behaviour keys never change. Output goes to `outputs/<video>_<preset>/` (campus, the
default, uses `outputs/<video>/`).

| Preset | Title | Tracks | Headline rules | Wording examples |
|---|---|---|---|---|
| `campus` | Campus corridor (restricted zone) | person | loitering, zone intrusion, running (plus fall, crowding and person-person near miss) | the defaults |
| `workplace` | Industrial / workplace safety | person, bicycle, car, motorbike, bus, truck | **near miss** (person versus vehicle), fall, zone intrusion, running, loitering 20 s | "Entered machine danger zone", "Running on the shop floor" |
| `elderly` | Elderly care | person | **fall**, no movement for 60 s, left through the exit **at night (21:00-06:00, needs `--start-time`)**; running, crowding and near miss off | "No movement", "Left through the exit" |
| `livestock` | Livestock monitoring | horse, sheep, cow, dog | not moving 60 s, left the pen, stampede, bunching of 8 or more; fall and near miss off | "Not moving", "Left the pen", "Stampede / distress", "Bunching" |
| `public_safety` | Public safety (crowds, fleeing, falls) | person, bicycle, motorbike | **crowd gathering** (10+ people, whole frame, no zone needed), fleeing/running, person collapsed, loitering 30 s, near miss with bikes | "Crowd gathering", "Fleeing / running", "Person collapsed" |
| `disaster` | Disaster response (people down or trapped) | person | **person down**, not moving 30 s (possibly trapped), inside the danger zone, crowd building up (6+), fleeing | "Person down", "Not moving (possibly trapped)", "Crowd building up" |
| `agriculture` | Agriculture (field intrusion by people or animals) | person, dog, horse, sheep, cow | **entered the field at night (19:00-06:00, needs `--start-time`)**, lingering, herd in the field; fall off | "Entered the field", "Herd or group in the field" |
| `pool` | **Aquatic distress (swimming pool)** | person (or a custom swimmer model) | **high-risk aquatic distress** (normal swimming -> slowing -> vertical -> repeated arm motion -> no progress, held 5 s) and **possible submersion**; land rules off. See POOL.md | "High-risk aquatic distress", "Possible submersion" |
| `traffic` | Traffic and pedestrians | person, bicycle, car, motorbike, bus, truck | **near miss** (pedestrian versus vehicle), speeding, stopped too long 30 s, restricted lane, congestion of 8 or more | "Speeding", "Stopped too long", "Entered restricted lane", "Congestion" |

When a scenario tracks several classes, names come from the detected class ("Car #7", "Person #3"); otherwise from the
scenario's entity word ("Animal #4"). Speeds and distances are in **body-lengths** (the longest side of the box) for
animals and vehicles and in body-heights for people; the report uses the preset's unit word.

**Time-of-day rules.** Any rule can be limited to certain hours with `active_hours: "22:00-06:00"` (several windows:
`"08:00-12:00, 14:00-18:00"`; windows may wrap past midnight). A video has no clock, so give the clock time of its first
frame with `--start-time HH:MM` (or the "Video start time" box in the UI). Events outside their rule's hours are dropped
and listed in `events.json` under `ignored_outside_hours`; every kept event gets `clock_start` / `clock_end`. Without a
start time the rules run all day and `run.py` prints a warning. The `elderly` (exit at night) and `agriculture` (field at
night) presets use this.

```powershell
python run.py --video hallway.mp4 --zones exit.json --scenario elderly --start-time 23:10
python run.py --list-scenarios
python run.py --video samples/person-bicycle-car-detection.mp4 --scenario traffic
python run.py --video barn.mp4 --zones pen.json --scenario livestock
python run.py --video clip.mp4 --scenario my_preset.yaml        # your own preset: copy one from scenarios/ and edit it
```

### Which domains this covers, honestly

The same engine (track objects, measure geometry over time, apply explainable rules, write evidence) maps onto many
domains. The status column says how much of that is real today:

* **tested** = the rules have unit tests and the pipeline was run on real sample footage of that kind.
* **preset only** = a preset exists and loads, and the rules are unit-tested, but it was not validated on real footage of that domain.
* **design only** = needs something that is not built.

| Domain | Preset | Rules that apply | Status |
|---|---|---|---|
| Workplace / industrial safety | `workplace` | near miss worker versus vehicle, machine danger zone, fall, running | tested on a **synthetic** clip with ground truth (near miss, speeding cyclist and fall all found, precision and recall 1.00); not on real factory footage. COCO has no forklift class (trucks and cars stand in); **forklifts need a custom detector** |
| Elderly care | `elderly` | fall, no movement (loitering rule), left through the exit **at night** (zone rule + time of day) | preset only. Fall rule works on box shape (synthetic clip tested); not validated on real care footage |
| Agriculture | `agriculture` | people **and** animals entering a field at night (zone rule + time of day), lingering, herd in the field | preset only; no farm footage tested. Crop or plant health is not covered |
| Transportation | `traffic` | near miss pedestrian versus vehicle, speeding, stopped too long, restricted lane, congestion | preset only. Run end to end on `samples/person-bicycle-car-detection.mp4`; speeds are image-plane, not km/h |
| Public safety | `public_safety` | crowd gathering (whole frame), fleeing, person collapsed, loitering, near miss with bikes | loitering, zone intrusion and running tested on the Intel clips; crowding unit-tested only (dense crowds make the detector miss people) |
| Education / campus | `campus` | loitering, zone intrusion, running, person-person near miss in corridors | **tested** (the original target scenario: Intel clips plus a synthetic clip with known ground truth) |
| Logistics / warehouse | `workplace` | near miss between people and trucks or carts, no-go aisle (zone rule) | preset only; **forklifts and pallet trucks need a custom detector** |
| Disaster response | `disaster` | person down, not moving (possibly trapped), inside the danger zone, crowd building up, fleeing | preset only (fixed camera); no disaster footage. Drones and body cams move, so they need the moving-camera work first |
| Animal / livestock | `livestock` | not moving, left the pen, stampede, bunching | preset only. COCO covers horse, sheep, cow and dog; no animal footage was tested |
| Accessibility | `navigation` | approaching alert for a wearable camera (person, bike, car or dog coming at the wearer, with clock direction and time to contact) | prototype: unit-tested, offline on recorded walks, not yet measured on real chest-camera footage. Wheelchairs, canes and guide dogs as objects are not COCO classes |
| Water safety (pools) | none | upright, inside the water zone, no forward progress for >= 10 s (zone + no-movement + pose rules) | design only. Detectors miss swimmers (splash, refraction, mostly under water) and there is no footage; a life-safety claim needs validation we cannot do here |

## CLI reference

### `run.py`

| Flag | Meaning |
|---|---|
| `--video PATH` | Input video file |
| `--webcam SECONDS` | Instead of `--video`: record the laptop camera to `samples/webcam_<n>.mp4` (real measured frame rate), then analyse it |
| `--zones zones.json` | Zone polygons (see [Zones](#zones)). Without it the zone rule has nothing to check, and crowding only counts the whole frame when `crowding.whole_frame` is on |
| `--scenario NAME` | Preset from `scenarios/` (or a path to your own YAML): classes, rules, thresholds, wording. Output goes to `outputs/<video>_<NAME>` |
| `--list-scenarios` | Print the presets (title, tracked classes, rules) and exit; no video needed |
| `--privacy` | Blur faces in the annotated video, highlights, snapshots and heatmap (analysis unchanged) |
| `--announce` | Speak alerts on this computer's speaker (alarm tone + voice; pool alerts by default) |
| `--start-time HH:MM` | Clock time of the first frame: enables time-of-day rules (`active_hours`) and adds clock times to events |
| `--config config.yaml` | Settings file (default: `config.yaml` next to `run.py`); a preset is merged on top of it |
| `--out DIR` | Output folder (default `outputs/<video name>`, plus `_<preset>` for every preset except campus) |
| `--device auto\|cpu\|0` | Where YOLO runs (default from `config.yaml`: `auto` picks the GPU when there is one) |
| `--stride N` | Process every Nth frame (default 2) |
| `--max-seconds S` | Only analyse the first S seconds |
| `--pose` | Second opinion with a pose model for running and loitering events |
| `--no-video` | Skip `annotated.mp4` and `highlights.mp4` (faster) |
| `--reuse` | Reuse `<out>/detections.json` if it matches the video, stride, resize width, detector and tracked classes; skips YOLO |

The console shows each step with its time, the detector speed in frames per second, and a final table
`# | who | behaviour | severity | start-end | conf | evidence`, followed by the incident chains. The exit code is 0 even when
nobody or nothing unusual is found; an unreadable video stops with a clear message.

### `zone_picker.py`

| Command | Meaning |
|---|---|
| `python zone_picker.py --video X [--out zones.json] [--name restricted]` | Window on the first frame: left-click adds a corner, right-click undoes, Enter or S saves, Esc quits |
| `... --grid` | No window (Colab): writes `zone_grid.jpg`, the first frame with a labelled grid every 50 px |
| `... --points "x1,y1 x2,y2 x3,y3 x4,y4"` | No window: writes `zones.json` directly and `zone_preview.jpg` to check it |

### `evaluate.py`

| Flag | Meaning |
|---|---|
| `--labels labels.csv` | Ground truth (see [Evaluation](#evaluation)) |
| `--outputs outputs/` | Folder with one `<clip name>/events.json` per labelled clip (a `<clip name>_<preset>` folder is found too) |
| `--events FILE --clip NAME` | Score a single `events.json` instead |
| `--tolerance S` | Seconds of slack on each side of a label (default 2) |
| `--report-dir DIR` | Where to write `eval_report.txt` and `eval_report.json` (default: next to the labels) |

## Outputs

Everything for a video goes to `outputs/<video name>/` (or `outputs/<video name>_<preset>/`).

| File | What it is |
|---|---|
| `report.html` | One self-contained page (images embedded): header, link to the highlight reel, guard summary, **incidents**, an evidence card per event with a severity badge, timeline, heatmap, almost-flagged cases, unusual entities, settings. Prints cleanly |
| `highlights.mp4` | Only the incidents, with a title card and a caption bar (H.264 when `ffmpeg` is on the PATH) |
| `summary.txt` | The 3-6 line summary a guard can read: incidents first, then single events |
| `events.json` | All data: `video`, `meta` (scenario, privacy, processing speed, settings), `events`, `incidents`, `near_misses`, `unusual_tracks`, `baseline`, `tracks` |
| `annotated.mp4` | Zones, boxes with IDs, trails, event labels, timestamp (H.264 when `ffmpeg` is on the PATH, so browsers can play it) |
| `snapshots/e<N>.jpg` | The frame at the peak of event N (the closest moment for a near miss): thick box, trail, zone, caption bar |
| `plots/e<N>.png` | Speed (running), distance from the spot (loitering), depth in the zone (intrusion), or distance and closing speed (near miss) over time, with the thresholds |
| `heatmap.jpg` | Where things were, blended over the first frame |
| `timeline.png` | One bar per track (grey = visible) with coloured blocks for events; overlapping events are stacked |
| `detections.json` | The cached YOLO + ByteTrack result (with the class of every box), used by `--reuse` |

Every event in `events.json` has the keys of DESIGN 3.4 plus `severity`, `entities`, `other_entity` (near miss), `zone`,
`entity_name` ("Car #7", "Group of 6") and `behavior_name` (the scenario's wording). The key `near_misses` in `events.json`
holds the *almost-flagged* cases (behaviour that reached 70 percent of a threshold); it is not the near-miss behaviour.

## How each rule works

**Coordinates and time.** Each frame is resized to 640 px wide (`video.resize_width`) and only every 2nd frame is
processed (`video.frame_stride`). Time is `frame number / fps`, using the video's own fps. An object's position is the
**foot point**, the bottom-centre of the detection box. The unit of distance is the **body size**: the longest side of the
box (`features.scale: max_side`). For an upright person that is the box height (a *body-height*, bh); for a person lying
down, a car or an animal it is the body *length*, so speeds stay sane.

**Why body-heights.** A person 180 px tall who moves 180 px in one second moves at 1.0 bh/s. A person far away, 60 px
tall, who moves 60 px in one second also moves at 1.0 bh/s. Speeds and distances in bh are comparable near and far from
the camera, on any resolution, with no camera calibration. Walking is about 0.6 to 1.0 bh/s.

**Smoothing and speed.** Detection boxes jitter, so the foot point, the body size and the aspect ratio are averaged over 0.5 s
(`features.smoothing_s`), separately for each stretch of the track (a gap longer than `features.max_gap_s` starts a new
stretch). Speed at a moment = distance the smoothed foot point moved over the last 0.5 s, divided by the elapsed time,
divided by the body size. It is unknown (not zero) at the start of a stretch.

**Loitering.** Look at the last 10 s (`loitering.min_duration_s`). If every smoothed foot point in that window stays
within 0.5 body-heights (`loitering.radius_bh`) of the window's centre, the person is standing around. The window must
really be covered by observations. Example: a person 200 px tall must stay inside a 100 px circle for 10 s.

**Zone intrusion.** A foot point is inside when it lies inside a zone polygon (ray-casting point-in-polygon test).
One continuous stay inside is one interval. The event records the zone name and how deep the feet went, in body-heights
from the zone edge.

**Running with hysteresis.** The state "running" turns on when speed reaches 1.8 bh/s (`start_speed_bh_s`) and turns off
only when speed falls below 1.4 bh/s (`end_speed_bh_s`). Two different thresholds stop the label flickering when the
speed hovers near one value. Unknown speed changes nothing. (In the `traffic` preset the same rule is shown as "Speeding".)

**Merging and filtering.** The rules produce raw pieces. For each entity and behaviour, pieces separated by less than
1.5 s (`events.merge_gap_s`) are merged into one event (zone pieces merge only inside the same zone). Only then are
events shorter than the behaviour's minimum duration dropped. This way an object that is briefly lost by the tracker is
still one event.

**Almost-flagged cases.** For every entity and behaviour that produced no event, the system checks how close it came: loitering
still for at least 70% of the minimum time, running speed of at least 70% of the start threshold, or a zone visit that was
too short. These are listed as "almost flagged", with the number and the rule, so a reader can judge the thresholds
(`events.near_miss_ratio`).

**Confidence.** Three scores, each between 0 and 1, are combined with the weights in `events.confidence_weights`:

| Score | Weight | Meaning |
|---|---|---|
| margin | 0.4 | how far past the threshold: running `(mean speed / start threshold - 1) / 0.5`; loitering `1 - max radius / radius limit`; zone `max depth / 0.5 bh`; fall `(max aspect / 1.2 - 1) / 0.5` (+0.2 if it was a fall from upright); crowding `(max count / min count - 1) / 0.5`; near miss as described above |
| duration | 0.3 | `duration / (2 x minimum duration)` |
| detection | 0.3 | mean YOLO confidence of the object during the event |

`confidence = 0.4 x margin + 0.3 x duration + 0.3 x detection`, rounded to 2 decimals. The parts are stored in the event
and shown on the card, so the number is never a black box.

**Baseline: what is normal here.** With at least 4 tracks in the video (`baseline.min_tracks`), each one is compared
with the others using a robust z-score (median and median absolute deviation, which one outlier cannot distort) on
median speed and on the longest time standing still. `anomaly_score = max(speed z, dwell z, 0)`. An entity at or above
3.0 (`baseline.z_threshold`) is marked unusual, with a note such as "Moved 2.9x faster than the scene's typical person".
This adds context to events (and raises their severity one level) and can also flag an entity that broke no fixed rule.

**Pose check (`--pose`).** For running and loitering events a pose model (`yolo11n-pose`) looks at up to 12 frames
spread over the event. From the shoulders, hips and ankles it measures the torso lean from vertical and the leg
spread in body-heights. Running agrees when the legs spread widely or the torso leans forward; loitering agrees when the
body is upright. Too few usable frames gives `"uncertain"`. The result is `verified = true | false | "uncertain"` on
the event. It never removes an event and never stops the run. (The pose weights `yolo11n-pose.pt` download only when
`--pose` is used.)

## Configuration

All thresholds are in `config.yaml` (also listed in `DEFAULT_CONFIG` in `utils.py`). The important ones:

| Key | Default | Effect |
|---|---|---|
| `video.resize_width` | 640 | frames are resized to this width before detection |
| `video.frame_stride` | 2 | process every Nth frame (1 = all frames, slower and finer) |
| `model.conf` | 0.30 | minimum detection confidence |
| `model.classes` | `[0]` | COCO class ids tracked (0 person, 1 bicycle, 2 car, 3 motorbike, 5 bus, 7 truck, 16 dog, 17 horse, 18 sheep, 19 cow) |
| `features.smoothing_s` | 0.5 | smoothing window for foot point, size and aspect |
| `features.min_track_s` | 1.0 | tracks shorter than this are ghost detections and are dropped |
| `features.scale` | `max_side` | body size = longest side of the box (box height for upright people) |
| `behaviors.loitering.radius_bh` / `min_duration_s` | 0.5 / 10 s | stay within this radius for this long |
| `behaviors.zone_intrusion.min_duration_s` | 1 s | feet inside a zone at least this long |
| `behaviors.running.start_speed_bh_s` / `end_speed_bh_s` | 1.8 / 1.4 | hysteresis thresholds |
| `behaviors.fall.down_min_aspect` / `min_down_s` | 1.2 / 2 s | width/height at which a box counts as lying, and for how long |
| `behaviors.crowding.min_count` / `min_duration_s` / `whole_frame` | 5 / 5 s / false | how many inside a zone, for how long, count the whole frame when no zone |
| `behaviors.near_miss.near_distance_bh` | 0.6 | closer than this (in body sizes) while moving relative to each other = near miss |
| `behaviors.near_miss.min_rel_speed_bh_s` / `ttc_s` / `contact_bh` | 0.8 / 1.0 s / 0.15 | relative speed needed, time-to-collision limit, possible-contact distance |
| `behaviors.near_miss.vulnerable_classes` / `other_classes` | `[0]` / `[0,1,2,3,5,7]` | who can get hurt, and what they can nearly hit |
| `behaviors.near_miss.max_scale_diff` | 0.35 | depth guard for pairs of people |
| `behaviors.baseline.min_tracks` / `z_threshold` | 4 / 3.0 | tracks needed, and how far from normal |
| `events.merge_gap_s` | 1.5 s | pieces closer than this become one event |
| `events.near_miss_ratio` | 0.7 | report almost-flagged cases reaching this fraction of a threshold |
| `events.confidence_weights` | 0.4 / 0.3 / 0.3 | margin / duration / detection |
| `chains.max_gap_s` | 15 s | next event of the same entity must start within this to join an incident |
| `highlights.pad_s` / `max_segment_s` | 1.5 / 8 s | context around an incident, longest segment before fast-forward |
| `privacy.enabled` / `mode` / `head_fraction` / `style` | false / head / 0.28 / pixelate | what `--privacy` blurs and how |
| `scenario.*` | campus | wording only: title, entity word, unit word, behaviour display names |
| `pose.*` | see file | pose check limits |
| `output.*` | see file | annotated video, trail length, heatmap, timeline |

`--reuse` stays valid while `video.frame_stride`, `video.resize_width`, `model.weights` and `model.classes` are unchanged.
Change anything else and re-run with `--reuse`: only the rules run, which takes about a second.

## Zones

A zone is a polygon with at least 3 corners. `zones.json` stores the corners in the **original video pixels** together
with the image size, so the pipeline can scale them to the processing frame itself (see `zones.example.json`):

```json
{"image_size": [1280, 720],
 "zones": [{"name": "restricted", "points": [[820, 200], [1180, 200], [1180, 620], [820, 620]]}]}
```

Make one with `zone_picker.py` (window, or `--grid` plus `--points` when there is no screen) or by hand. Several zones
are allowed; an event records which zone was entered. The same file serves as the machine danger zone, the exit door, the
pen, the restricted lane or the area to count a crowd in, depending on the preset.

## Evaluation

Watch a clip and write the true incidents into `labels.csv` (template: `labels_template.csv`; its rows are examples to
replace):

```csv
clip,entity_description,behavior,start_s,end_s
corridor.mp4,man in a red jacket,loitering,12.0,34.0
corridor.mp4,woman with a backpack,running,41.0,48.0
quiet_clip.mp4,no unusual activity,none,,
```

`behavior` is `loitering`, `zone_intrusion`, `running`, `fall`, `crowding`, `near_miss`, or `none` (the clip should have no
events). Times are seconds or `mm:ss`. Label every incident you want counted: an event with no matching label counts as a
false alarm.

```powershell
python evaluate.py --labels labels.csv --outputs outputs/
python evaluate.py --labels labels.csv --events outputs/corridor/events.json --clip corridor.mp4
```

A prediction matches a label when they have the same behaviour in the same clip and their time intervals overlap after
the label is widened by the tolerance (2 s). Matching is one-to-one, largest overlap first. Entity IDs are not compared,
because labels describe entities by appearance. The report gives, per behaviour and overall, true positives, false
positives, false negatives, precision, recall and F1, the mean absolute start and end error of matched events, and the
false alarms on `none` clips. It is printed and saved as `eval_report.txt` and `eval_report.json`.

## Project structure

```
ps07/
  run.py               CLI: wires all steps together, prints timings, the events table and the incidents
  tracker.py           YOLO11n detection + ByteTrack IDs (any COCO classes) -> detections.json
  features.py          per-object tracks: smoothed foot point, size, aspect, speed, velocity, class
  behaviors.py         rules: loitering, zone intrusion, running, fall, crowding, near miss, almost-flagged, baseline
  events.py            merge, filter, confidence, evidence sentences
  chains.py            severity per event + incident chains (pure Python)
  pose_verify.py       optional pose check (--pose)
  render.py            annotated video, snapshots, plots, heatmap, timeline
  highlights.py        highlights.mp4 (incidents only)
  privacy.py           face blur for every rendered image (--privacy)
  report.py            events.json, summary.txt, report.html
  zone_picker.py       draw the zone (window, grid, points)
  app.py               Streamlit web UI (streamlit run app.py): wraps run.py, shows the results
  ui_helpers.py        plain-Python helpers for the UI: files, run.py command, reading results (no Streamlit)
  evaluate.py          score events.json against labels.csv
  utils.py             config, presets, names, time, geometry, intervals, JSON helpers (NumPy only)
  config.yaml          every threshold
  .streamlit/          config.toml: theme and upload limit of the web UI
  scenarios/           presets: campus, workplace, elderly, livestock, traffic (partial configs)
  bytetrack_custom.yaml  ByteTrack settings (longer track buffer)
  DESIGN.md            the contract between modules
  SCOPE.md             done / stretch / not done / limitations
  colab_run.ipynb      Google Colab notebook (T4 GPU)
  requirements.txt     pip packages
  labels_template.csv  example labels      zones.example.json  example zone file
  samples/             input videos (not committed)
  tests/               unit tests (core logic needs no GPU), including test_scenarios.py for the presets;
                       test_app.py and test_ui_helpers.py cover the web UI
```

## Data Pipeline

1. **Collect.** One video from a fixed camera: a file, or a clip recorded with `--webcam`. The Intel sample videos are
   used for demos. There is no dataset to prepare and no training.
2. **Preprocess.** OpenCV reads the video. The fps comes from the file (30 if it is missing, with a warning). Only every
   2nd frame is processed, each resized to 640 px wide. Time is computed from the original frame number.
3. **Perceive.** YOLO11n (pretrained on COCO, the classes chosen by the scenario) finds objects. ByteTrack links them
   across frames and gives each a stable ID. The class of every box is stored. The result is saved as `detections.json`,
   so the later steps never need the video or the network again.
4. **Reason.** Per-object tracks are smoothed, speeds and velocities are computed in body sizes per second, and the rules
   (loitering, zone, running, fall, crowding, near miss) plus the scene baseline run. Raw pieces are merged, short ones
   dropped, and each event gets numbers, a confidence, and an evidence sentence. Severity and incident chains are added.
5. **Output.** `events.json` (data), `summary.txt` (3-6 lines), `report.html` (incidents and evidence cards),
   `highlights.mp4`, plus the annotated video, snapshots, plots, heatmap and timeline. With `--privacy` every image has
   the faces blurred.

## Sample input and output

Committed in [examples/](examples/) (open each `report.html` in a browser; `highlights.mp4` plays the incidents):

**1. Real footage, campus preset.** Input: Intel `one-by-one-person-detection.mp4` (2:19, 10 fps, 768x432) with the
zone in `examples/campus_one-by-one/zones.json` drawn around the table. Command:
`python run.py --video samples/one-by-one-person-detection.mp4 --zones samples/one-by-one_zones.json`.
Processing took about 97 s on a laptop CPU (5-6 frames/s). `summary.txt`:

```
Video one-by-one-person-detection.mp4 (02:19), 6 people seen. 6 incidents:
- 00:06-00:19 Person #1: Lingered inside the restricted zone [HIGH]
- 00:23-00:44 Person #5: Lingered inside the restricted zone [HIGH]
- 00:50-01:09 Person #7: Lingered inside the restricted zone [HIGH]
- ...and 3 more (see report.html).
```

Each person in this clip walks up to the table, stands there 10-26 s and leaves, so each one is one incident chain
(loitering while inside the zone). An evidence line from `events.json`:
`Stayed within 0.46 body-heights of one spot for 26 s. Rule: within 0.5 bh for >= 10 s.` (confidence 0.96).

**2. Synthetic workplace clip, workplace preset** (near miss + fall, with ground truth). Command:
`python run.py --video samples/synthetic_safety.mp4 --scenario workplace --out outputs/synthetic_safety --reuse`.

```
Video synthetic_safety.mp4 (00:30) [Industrial / workplace safety], 4 objects seen. 3 incidents:
- 00:06-00:06 Person #1 and Bicycle #2: Near miss - came within 0.07 body-heights, closing at 2.9 body-heights/s (POSSIBLE CONTACT) [HIGH]
- 00:18-00:24 Person #3: Fall / person down - fell and stayed down 6 s [HIGH]
- 00:06-00:08 Bicycle #2: Running on the shop floor - moved at 3.0 body-heights/s (peak 3.3) [MEDIUM]
```

Scored with `evaluate.py` against `tests/synthetic_safety_labels.csv`: 3 TP, 0 FP, 0 FN (precision 1.00, recall 1.00),
mean start error 0.15 s, mean end error 0.11 s. The calm background walker (#4) is correctly not flagged.

**3. Real car park, traffic preset:** `person-bicycle-car-detection.mp4` gives **no incidents**, which is correct (nobody
comes closer than about 4.7 body-heights to anyone).

## Live demonstration

Step-by-step demo script with timings, a fallback plan and likely jury questions: [DEMO.md](DEMO.md).

## Declared resources

| Resource | Use | License |
|---|---|---|
| Ultralytics YOLO11 (`yolo11n.pt`) and YOLO11 pose (`yolo11n-pose.pt`) weights | pretrained COCO detector and pose model, used as they are; weights download once from Ultralytics on first run | AGPL-3.0 |
| ByteTrack, through the built-in Ultralytics tracker (`bytetrack_custom.yaml`) | tracking by association with stable IDs (Zhang et al., ECCV 2022) | AGPL-3.0 (Ultralytics implementation) |
| `lap` | linear assignment for the tracker | BSD-2-Clause |
| PyTorch | runs the models (installed with Ultralytics, preinstalled on Colab) | BSD-3-Clause |
| OpenCV (`opencv-python`) | video reading and writing, drawing, face blur | Apache-2.0 |
| NumPy | all numeric work in the core logic | BSD-3-Clause |
| PyYAML | reads `config.yaml` and the scenario presets | MIT |
| Matplotlib | speed and distance plots and timeline | Matplotlib license (BSD-style) |
| Streamlit | the web UI (`app.py`); runs locally, no cloud | Apache-2.0 |
| Intel IoT DevKit sample videos (`one-by-one-person-detection.mp4`, `worker-zone-detection.mp4`, `person-bicycle-car-detection.mp4`) | demo input, credit Intel Corporation; fetched by `get_samples.py` | CC BY 4.0 |
| FFmpeg (optional, found on PATH; preinstalled on Colab) | converts the output videos to H.264 so they play in browsers | LGPL / GPL (used as an external program) |
| Synthetic test clips (`tests/synthetic.py`, `tests/synthetic_safety.py`) | drawn by our own code, with ground-truth labels | ours |

- No external APIs or cloud services are called while the system runs. The only network use is the one-time download
  of the model weights and the optional download of the sample videos.
- No model was trained or fine-tuned. The behaviour rules are written by hand and use no learned classifier.
- Licence of this project: the team still has to pick one. Because Ultralytics is AGPL-3.0, the simplest compatible
  choice for a public repository is AGPL-3.0 (add a `LICENSE` file before submitting).
- An AI coding assistant (Claude Code) helped write and document this code. The team reviewed it, ran the tests, and
  can explain every module.

## Known limitations

- **Distances are in the image plane (2D).** There is no ground-plane calibration and no depth estimate. Two objects that
  overlap in the picture can be far apart in the real world (a car passing behind a person), and two that are close in the
  world can look far apart. The depth guard only compares box sizes of two *people*; for a person and a vehicle there is no
  depth cue, so a vehicle passing behind a pedestrian can raise a false near miss. Speeds are body sizes per second, not
  km/h. The camera must be fixed. The near-miss thresholds come from geometry and were not tuned on real near-miss footage.
- **No forklift class.** The stock COCO detector knows person, bicycle, car, motorbike, bus and truck, not forklifts, pallet
  trucks or machines. In industrial footage those are usually missed or called "truck" or "car". Real forklift near-miss
  monitoring needs a custom-trained detector (replace `model.weights`; the rules stay the same).
- **Presets are starting points.** Only the campus scenario was validated on real footage; the other presets use the same
  engine with different thresholds and were not tuned on footage of their own domain.
- **Fall is a box-shape rule.** A box that turns wider than tall for 2 s. Sitting, bending, crawling, a person partly hidden or
  cut off by the frame edge can look similar or be missed. It is switched off for animals and vehicles.
- **Crowding counts foot points inside a zone.** It needs a zone (or `crowding.whole_frame`) and suffers when the detector
  misses people in a dense crowd.
- **Privacy mode is a box-based blur, not a guarantee.** It blurs the top part of every detected person box. A person the
  detector misses, or a box that is off, leaves a visible face. The original video is never changed, and `events.json` and
  the console still contain IDs and times.
- **Speed.** Measured on a laptop CPU the detector processes about 6 to 7 frames per second at 640 px with every 2nd frame
  (slower than real time); a GPU is much faster. The report header shows the real number for each run.
- IDs can switch when objects cross or hide each other; walking straight toward the camera looks slow; low light, blur and
  crowded scenes reduce accuracy; thresholds are set by hand and need tuning per camera. Details are in [SCOPE.md](SCOPE.md).

---

HackNEX 2026, PS07. Team NEXUS Club, Karunya.
