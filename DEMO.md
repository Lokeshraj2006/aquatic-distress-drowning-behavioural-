# Live demo script (about 4 minutes) + fallback + jury Q&A

## The demo surface: the web UI
Run the demo from the browser: `streamlit run app.py` (http://localhost:8501). Everything below can be done there:
pick the sample and the scenario in the sidebar, press **Run analysis**, then show the tabs (Highlight reel, Incidents,
Timeline & heatmap, Full report). With *Reuse cached detections* on (the default) a run takes seconds. The commands in
each step below are the **command-line fallback**: the UI starts the same `run.py` with the same flags (it only chooses the output folder itself, `outputs/<video>_<scenario>`; see "Command used" on the page).
If a run is slow or fails, open a finished folder with **Open existing results** at the bottom of the sidebar
(`examples/workplace_near-miss_fall` and `examples/campus_one-by-one` always work).

## Before you walk in
- [ ] `python get_samples.py` has run and `samples/` has the 3 Intel clips and the 2 synthetic clips.
- [ ] `streamlit run app.py` is running in a browser tab, and each demo below has been run once in it (so the cached detections exist). Reloading the tab (F5) keeps the last results: the page restores them from `outputs/_ui/last_run.json`.
- [ ] In the UI, pick the preset **before** pressing Run. If the facts box says "YOLO will run (roughly N min)", another preset has the cached detections: the lightbulb line names it.
- [ ] Every demo run below has been done **once already**. The jury run then uses `--reuse`, which takes seconds instead of minutes on CPU.
- [ ] These are open in browser tabs:
  - [ ] `examples/campus_one-by-one/report.html`
  - [ ] `examples/workplace_near-miss_fall/report.html`
  - [ ] `docs/diagrams/architecture.html`
- [ ] Each `highlights.mp4` is downloaded locally, in case there is no internet.
- [ ] Laptop is on the charger, extra apps are closed, and the terminal font is large.

## The pitch (15 s)
> "Detecting a person is easy. Understanding what they **do**, **when**, and whether it is **dangerous** is the hard part.
> Other systems draw boxes. Ours writes an incident report, proves every claim with numbers and a snapshot, and catches
> accidents that **almost** happened."

## 1. How it works (30 s): open `docs/diagrams/architecture.html`
- The flow is YOLO11n (people and vehicles) → ByteTrack (a stable ID per object) → measurements per second (speed, dwell, distance between objects) → explainable rules → evidence.
- No training and no cloud. It runs on a laptop CPU at 5-6 frames/s, and faster on a GPU.
- Every distance is in **body-heights**, so the same rule works near and far from the camera and on any camera.

## 2. The headline: near-miss intelligence (60 s)
In the UI: sample `synthetic_safety.mp4`, scenario *Industrial / workplace safety*, **Run analysis**. Command-line equivalent:
```
python run.py --video samples/synthetic_safety.mp4 --scenario workplace --out outputs/synthetic_safety --reuse
```
- Play `outputs/synthetic_safety/highlights.mp4`. A 30 s clip is reduced to only its incidents, each one captioned.
- Open the near-miss card in `report.html`. Read the evidence aloud:
  > "Person #1 and Bicycle #2 came within 0.07 body-heights ... closing at 2.9 body-heights/s (POSSIBLE CONTACT)."
- Point to the plot: distance falls, crosses the 0.6 line, then rises again. Nothing visibly happened, and the system still caught it.
- Then show the fall card: the box went from upright (w/h 0.3) to lying (w/h 3.1) and stayed down 6 s.

## 3. Real footage (60 s)
In the UI: sample `one-by-one-person-detection.mp4`, scenario *Campus corridor*, zone type *Zones file* (`samples/one-by-one_zones.json` is preselected), **Run analysis**. Command-line equivalent:
```
python run.py --video samples/one-by-one-person-detection.mp4 --zones samples/one-by-one_zones.json --reuse
```
- Read `summary.txt`: "Person #5 lingered inside the restricted zone 00:23-00:44 [HIGH]".
- Open one evidence card. Show the snapshot with the trail, the distance-from-spot plot under the 0.5 line, and the confidence parts.
- Show the **timeline** (who was visible when) and the **"almost flagged"** list. That list explains why someone was **not** flagged.

## 4. It generalises (30 s)
```
python run.py --list-scenarios
```
- One engine with presets for campus, workplace, elderly (fall), livestock, traffic, and a navigation prototype for a wearable camera.
- Run `--privacy` once and show the pixelated heads. The analysis is unchanged; only the outputs are blurred.

## 5. Honesty (15 s): the accuracy table
```
python evaluate.py --labels tests/synthetic_safety_labels.csv --events outputs/synthetic_safety/events.json --clip synthetic_safety.mp4
```
- Precision 1.00, recall 1.00, timing error under 0.2 s on the clip with ground truth.
- The real car-park clip gives **no incidents**, which is correct, because nobody comes close there.
- Name the limitations before the jury asks: distances are in 2D, there is no forklift class, and presets other than campus are not validated on their own footage.

## If something fails (fallback)
| Problem | Do this |
|---|---|
| The jury's video is slow on CPU | Add `--max-seconds 30 --stride 3`, or run it on Colab with a T4 while you show the `examples/` reports |
| No internet | Everything in `examples/` is pre-computed. The YOLO weights are already in the project folder |
| The jury's video has no zone | Loitering, running, fall and near miss need no zone. For a zone: `python zone_picker.py --video X` (click 4 corners, then Enter) |
| A crash | `PS07_DEBUG=1 python run.py ...` prints the traceback. Meanwhile, show the pre-computed report |
| The UI does not start or a run fails | The page shows the last 20 lines of `run.py`. Open a finished folder with **Open existing results**, or use the command-line commands above |
| A run in the UI is taking too long | Change any setting, press Stop (top right) or close the tab: `run.py` and ffmpeg are stopped. Then open a finished folder with **Open existing results** |
| The wrong preset was run (YOLO started) | Stop it as above and pick the preset named in the lightbulb line. Old detections are kept in `outputs/_ui/cache/`, so the fast path is not lost |
| The page shows old results for a new choice | A blue note says the results belong to the earlier run. Press **Run analysis** for the new choice |

## Likely jury questions
- **"Why not train a model?"** Rules on top of tracks are explainable and need no labelled data. Every flag comes with the measured numbers. With one night and no dataset, a trained action model would be a black box that is never validated.
- **"What if IDs swap when people cross?"** ByteTrack keeps lost tracks for 60 processed frames. Swaps can still happen in crowds (see SCOPE.md). Each event names an ID plus a snapshot, so a human can check.
- **"How do you decide normal vs abnormal?"** Two ways:
  - Explicit rules with thresholds, all in `config.yaml`.
  - A **scene baseline**: anyone far outside that video's own typical speed or dwell time is marked unusual (robust z-score).
- **"Why body-heights?"** A person far away looks small. Dividing by their box size makes "running" mean the same thing near and far, and on the jury's camera too.
- **"Is near-miss distance real-world?"** No, it is measured in the image. Two objects that overlap in 2D can be at different depths. For two people we compare their sizes (the depth guard). Real metres would need a ground-plane calibration (4 floor points), which is the next step.
- **"How accurate is it?"** Show the evaluation table. Be clear about which numbers come from synthetic ground truth and which from real clips.

---

## Navigation prototype: a 5-walk recording protocol (to measure false alarms)
Fix a phone to your chest (portrait or landscape, kept still) and record at 30 fps. Then run each walk with
`python run.py --video walkN.mp4 --scenario navigation`.

| Walk | What to do | Expected |
|---|---|---|
| 1 | Walk down a corridor while people walk past you **beside** you, in the next lane | no alert |
| 2 | Walk towards a person **standing still in your path** | one alert about 2 s before you would reach them |
| 3 | A friend walks **straight at you**, then stops 1 m away | one alert, from 12 o'clock |
| 4 | A friend walks at you, then **sidesteps** at 3 m | ideally no alert, or a late one; note what happens |
| 5 | Walk through a **busy** area | count the alerts; every alert should be someone actually in your path |

The false-alarm rate is (alerts in walks 1, 4 and 5 that were not real threats) divided by (all alerts). Put the number in SCOPE.md.
