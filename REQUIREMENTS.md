# How this project meets the HackNex requirements

## Problem statement HNX26PSI07
| Requirement | Where it is met |
|---|---|
| Find people / objects | YOLO11n detector (`tracker.py`). Pool: stock person model now; a 5-class swimmer model is trained by `training/` |
| Track the same one across the video | ByteTrack IDs (`bytetrack_custom.yaml`, 60-frame buffer); ID-switch guard in `pool_behaviour.py` |
| Understand what they are doing (not just list detections) | Stage 2 signals per person every 0.5 s: posture, movement, displacement, progress ratio, arm motion, head, activity (swimming / treading / floundering / floating) in `stage2_behaviour.jsonl` |
| Normal vs unusual | Explainable rules plus a risk state machine (`distress.py`). Look-alikes stay normal: resting at the wall, treading water, diving, playing. Scene baseline flags unusual movers (`behaviors.py`) |
| Meaningful events, with WHO and WHEN | Events with entity ID, start time, alert time, location, risk %, an evidence list, a snapshot and a plot (`events.json`, `stage3_events.json`, `report.html`) |
| Collect videos for the scenario | Intel sample clips, Pexels pool clips (`samples/pexels`), synthetic staged pool story (`tests/synthetic_pool.py`), team's staged clips (protocol in POOL.md) |
| Advanced: several behaviours, general anomaly | Pool: distress and submersion. The engine also covers loitering, zone intrusion, running, fall, crowding, near miss, approaching, and a scene-baseline anomaly score |
| Events detected at the right time | `evaluate.py`: synthetic pool story gives precision 1.00, recall 1.00, timing error 1.4 s; workplace clip gives precision 1.00, recall 1.00, timing error under 0.2 s |
| Scenario | The PS lists workplace safety, retail, factory, campus, traffic and crowds. Ours, a pool, is public-space safety, and the same engine also runs `campus`, `workplace`, `traffic` and `public_safety` presets (campus tested on real Intel footage) |

## Submission guidelines
| Item | Where |
|---|---|
| Working system | `run.py` (CLI) and `app.py` (web UI with the lifeguard dashboard and spoken alert) |
| Source code + README | this repository; README.md, POOL.md (pool pipeline), DESIGN.md (contract) |
| Data pipeline | POOL.md "common pipeline" (stage 1 → 4 files), README "Data Pipeline", `docs/diagrams/data-pipeline.html` |
| Core model / reasoning | `pool_behaviour.py` (stage 2), `distress.py` (stage 3: weighted risk, look-alike damping, state machine, hold 5 s); POOL.md "How the decision is made" |
| Evidence and explanation | Every alert: evidence checklist, risk %, timestamps, snapshot, risk-over-time plot, behaviour timeline, intermediate stage files, spoken announcement (`alerts/*.wav`) |
| Sample input and output | `examples/pool_synthetic/` (distress then submersion), `examples/campus_one-by-one/` (real footage), `examples/workplace_near-miss_fall/` |
| Scope note | SCOPE.md (MVP / stretch / not done / limitations) |
| Live demonstration | DEMO.md; `streamlit run app.py`, then "Open existing results" as the fallback |

## README must explain
| Item | README section |
|---|---|
| What the project does | top banner plus POOL.md |
| Technologies, libraries, models | "Declared resources" (Ultralytics YOLO11 + pose, ByteTrack, OpenCV, NumPy, PyYAML, Matplotlib, Streamlit, FFmpeg, datasets and videos with licences) |
| Install dependencies | "Quick start": venv, then `pip install -r requirements.txt`, then `python get_samples.py` |
| Configure and run | "CLI reference", "Configuration" (`config.yaml`), "Scenario presets", "Web UI" |
| Reproduce the results | "Sample input and output" commands, plus `evaluate.py` with the label files in `tests/` |

## Test results on real pool footage (Pexels, no distress in them)
| Clip | Result |
|---|---|
| People playing, drone view, 41 s, 8 people | 0 alerts after fixing the submersion rule (before the fix: 5 false "possible submersion") |
| Butterfly swimmer, 17.5 s | 0 alerts (strokes are swimming, not distress) |

**Honest limits:**
- Real distress was only tested on staged/synthetic data.
- The stock detector is not a swimmer model.
- Distances are 2D, in image space.
