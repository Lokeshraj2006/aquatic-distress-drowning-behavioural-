# Example outputs (committed sample input and output)

Open the `report.html` in each folder in a browser. It is a single file, with the images embedded.

| Folder | Input | Command | What it shows |
|---|---|---|---|
| `campus_one-by-one/` | Intel `one-by-one-person-detection.mp4` (real footage, CC BY 4.0) + `zones.json` | `python run.py --video samples/one-by-one-person-detection.mp4 --zones samples/one-by-one_zones.json` | 6 people, each loitering inside the table zone. Shows incident chains, evidence cards, the timeline and the highlight reel |
| `workplace_near-miss_fall/` | `samples/synthetic_safety.mp4` (made by `python tests/synthetic_safety.py`) | `python run.py --video samples/synthetic_safety.mp4 --scenario workplace --out outputs/synthetic_safety --reuse` | A near miss between a cyclist and a pedestrian, the cyclist speeding, and a worker who falls. `labels.csv` is the ground truth: precision 1.00, recall 1.00 |

Each folder contains:
- `summary.txt`: the guard summary
- `events.json`: every event with times, measured numbers, confidence, severity and incident chains
- `report.html`: the evidence cards
- `highlights.mp4`: only the incidents
- `timeline.png`
- `snapshots/` and `plots/`: a selection of evidence images

Full annotated videos are left out to keep the repository small. Re-run the command to make them.
