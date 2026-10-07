# Guide for AI coding assistants (PS07, HackNEX 2026)

1. Read `DESIGN.md` first. It is the binding contract: function names, signatures, dict keys, file names and CLI flags.
2. Core logic is NumPy only: `utils.py`, `features.py`, `behaviors.py`, `events.py`, `evaluate.py` must never import cv2, torch or ultralytics.
3. Only `tracker.py`, `pose_verify.py`, `render.py`, `zone_picker.py` and `run.py` may use cv2; only `tracker.py` and `pose_verify.py` may use ultralytics, and only inside functions (so `--reuse` works without it).
4. Never hardcode a threshold. Every number lives in `config.yaml`, read through `utils.load_config()`; new keys also go into `DEFAULT_CONFIG` in `utils.py`.
5. Distances are in body-heights (box height), times in seconds from the video's own FPS, boxes in processing pixels (frame resized to `video.resize_width`).
6. Do not change `DESIGN.md`, `config.yaml` or `utils.py` without saying so; other modules depend on them. Change the contract first, then the code.
7. Style: short docstring per function, readable comments, plain English evidence sentences with numbers. Students must be able to explain every line.
8. Keep console output plain ASCII (Windows consoles) and never crash the run for an optional step (pose, rendering).
9. Run the tests before and after a change: `python -B -m pytest -p no:cacheprovider -q tests`
10. Do not install or download anything unasked; run `python -B -m py_compile <file>` on every file you write.
11. The web UI (`app.py`, `ui_helpers.py`) only wraps `run.py` (subprocess) and reads its output files. It never imports torch or ultralytics, and `ui_helpers.py` uses cv2 only (lazily) to read the first frame for the preview. Keep Streamlit calls in `app.py`; plain logic goes in `ui_helpers.py` so it can be unit-tested. Tests: `tests/test_app.py` (AppTest) and `tests/test_ui_helpers.py`.
