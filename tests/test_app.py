"""Tests for the Streamlit web UI (app.py), run headless with streamlit.testing.v1.AppTest.

No browser and no YOLO needed: the Run test uses the synthetic safety clip with its cached detections
(`--reuse`), so it calls the real run.py but only runs the rules (about 10 s). Folders the UI creates during a
test are removed again, unless they existed before the test.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # also works without conftest.py
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_helpers  # noqa: E402
import utils  # noqa: E402

APP = ROOT / "app.py"
SAFETY_VIDEO = ROOT / "samples" / "synthetic_safety.mp4"
SAFETY_RESULTS = ROOT / "outputs" / "synthetic_safety"
EXAMPLE_RESULTS = ROOT / "examples" / "workplace_near-miss_fall"

needs_safety_clip = pytest.mark.skipif(
    not (SAFETY_VIDEO.exists() and (SAFETY_RESULTS / "detections.json").exists()),
    reason="run `python get_samples.py --no-intel` and the workplace demo first (clip + cached detections missing)")
needs_safety_results = pytest.mark.skipif(not (SAFETY_RESULTS / "events.json").exists(),
                                          reason="outputs/synthetic_safety has no events.json")


def load_app(timeout: int = 60) -> AppTest:
    """A fresh AppTest of app.py, already run once."""
    return AppTest.from_file(str(APP), default_timeout=timeout).run()


def texts(elements) -> list[str]:
    """The .value of a list of AppTest elements (markdown, caption, ...)."""
    return [str(e.value) for e in elements]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test starts clean: its own last_run.json (a page reload restores that file, so a shared one would
    leak the previous test's run into the next page) and an empty Streamlit data cache."""
    monkeypatch.setattr(ui_helpers, "LAST_RUN_FILE", tmp_path / "last_run.json")
    st.cache_data.clear()
    yield


@pytest.fixture
def clean_ui_files():
    """Remove what the UI writes during a test (results folders, zones, uploads, kept detections).

    Takes a snapshot of the folders the UI writes into; afterwards everything that was not there before is deleted,
    and a folder that did not exist before is removed whole.
    """
    watched = [ROOT / "outputs", ROOT / "outputs" / "_ui", ROOT / "outputs" / "_ui" / "cache",
               ROOT / "samples", ROOT / "samples" / "uploads"]
    before = {d: set(d.iterdir()) if d.is_dir() else None for d in watched}
    yield
    for folder in reversed(watched):                                      # deepest folder first
        if before[folder] is None:
            shutil.rmtree(folder, ignore_errors=True)
        elif folder.is_dir():
            for extra in set(folder.iterdir()) - before[folder]:
                shutil.rmtree(extra, ignore_errors=True) if extra.is_dir() else extra.unlink(missing_ok=True)


# --------------------------------------------------------------------------- page load

def test_app_loads_without_exception():
    at = load_app()
    assert not at.exception
    assert [t.value for t in at.title] == ["Aquatic Distress Behaviour Intelligence"]
    assert any("Early warning for lifeguards" in v for v in texts(at.markdown))  # the one-line pitch
    assert any(b.label == "Run analysis" for b in at.button)
    assert at.selectbox(key="open_results").value is None                         # nothing opened by default


def test_scenario_selectbox_lists_all_presets():
    at = load_app()
    box = at.selectbox(key="scenario")
    presets = utils.list_scenarios()
    assert len(box.options) == len(presets) >= 5
    for name in presets:
        title = utils.load_config(None, name)["scenario"]["title"]
        assert title in box.options                                               # shown by title, not by file name
    assert box.value == "pool"                                                    # the default preset (team scenario)
    # the sidebar says what the chosen preset tracks and which rules are on
    assert any(v.startswith("Tracks:") and "Rules:" in v for v in texts(at.sidebar.caption))


def test_choosing_a_preset_updates_the_rule_list():
    at = load_app()
    at.selectbox(key="scenario").select("workplace").run()
    assert not at.exception
    rules = [v for v in texts(at.sidebar.caption) if v.startswith("Tracks:")][0]
    assert "bicycle" in rules and "Near miss" in rules


# --------------------------------------------------------------------------- opening existing results

@needs_safety_results
def test_open_existing_results_shows_metrics_and_near_miss():
    at = load_app()
    options = at.selectbox(key="open_results").options
    assert "outputs/synthetic_safety" in options and "examples/campus_one-by-one" in options
    at.selectbox(key="open_results").select("outputs/synthetic_safety").run()
    assert not at.exception

    data = utils.read_json(SAFETY_RESULTS / "events.json")
    expected = ui_helpers.result_metrics(data)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Incidents"] == str(expected["incidents"])
    assert tiles["Events"] == str(len(data["events"]))
    assert tiles["Video length"] == expected["duration"]
    assert tiles["Scenario"] == "workplace"
    assert any(label.endswith("tracked") for label in tiles)

    assert [t.label for t in at.tabs] == ["Summary", "Highlight reel", "Incidents", "Timeline & heatmap",
                                          "Full report", "Annotated video", "Downloads"]
    labels = [e.label for e in at.expander]
    assert any("Near miss" in label for label in labels), labels                  # one expander per event
    near = next(e for e in at.expander if "Near miss" in e.label)
    assert "HIGH" in near.label
    shown = " ".join(texts(near.markdown))
    assert "came within" in shown                                                 # the evidence sentence


def test_opening_the_example_with_missing_files_does_not_crash():
    """examples/ has no heatmap.jpg and no annotated.mp4: the page says so instead of failing."""
    if not (EXAMPLE_RESULTS / "events.json").exists():
        pytest.skip("examples/workplace_near-miss_fall is missing")
    at = load_app()
    at.selectbox(key="open_results").select("examples/workplace_near-miss_fall").run()
    assert not at.exception
    notes = " ".join(texts(at.caption))
    if not (EXAMPLE_RESULTS / "heatmap.jpg").exists():
        assert "heatmap.jpg" in notes and "not in this folder" in notes
    assert len(at.metric) >= 7                                                    # 3 video facts + the result tiles


def test_a_folder_with_a_corrupt_events_json_shows_an_error(tmp_path):
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "events.json").write_text("{not json", encoding="utf-8")
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["last_run"] = {"ok": True, "code": 0, "out_dir": str(bad), "cmd": "", "interpreter": "",
                                    "lines": [], "seconds": 0.0, "video": ""}     # an absolute path replaces ROOT / ...
    at.run()
    assert not at.exception
    assert any("Cannot read" in v for v in texts(at.error))


# --------------------------------------------------------------------------- running run.py from the page

@needs_safety_clip
def test_run_with_cached_detections_produces_results(clean_ui_files):
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.selectbox(key="scenario").select("workplace").run()
    assert at.toggle(key="reuse").value is True                                   # default on
    at.button(key="run_button").click().run(timeout=180)
    assert not at.exception
    assert not at.error, texts(at.error)

    last = at.session_state["last_run"]
    assert last["ok"] and last["code"] == 0
    assert last["out_dir"] == "outputs/synthetic_safety_workplace"                # outputs/<stem>_<scenario>
    for flag in ("--video samples/synthetic_safety.mp4", "--scenario workplace", "--reuse"):
        assert flag in last["cmd"]
    assert "cached detections" in " ".join(last["lines"]).lower()                 # YOLO was skipped
    assert (ROOT / last["out_dir"] / "events.json").exists()

    assert any("Last run finished" in v for v in texts(at.success))
    assert {"Incidents", "Events"} <= {m.label for m in at.metric}
    assert any("Near miss" in e.label for e in at.expander)
    assert at.selectbox(key="open_results").value is None                         # results are of the new run
    assert "outputs/synthetic_safety_workplace" in at.selectbox(key="open_results").options


@needs_safety_clip
def test_rectangle_zone_is_written_in_original_pixels_and_passed_to_run(clean_ui_files):
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.selectbox(key="scenario").select("workplace").run()
    at.button_group(key="zone_mode").set_value("Rectangle").run()
    assert not at.exception
    assert len(at.sidebar.slider) == 4                                            # x1, y1, x2, y2
    at.toggle(key="skip_video").set_value(True).run()                             # faster: no videos
    at.button(key="run_button").click().run(timeout=180)
    assert not at.exception and not at.error, texts(at.error)

    last = at.session_state["last_run"]
    assert "--zones outputs/_ui/synthetic_safety_zones.json" in last["cmd"]
    assert "--no-video" in last["cmd"]
    zones = json.loads((ROOT / "outputs" / "_ui" / "synthetic_safety_zones.json").read_text(encoding="utf-8"))
    assert zones["image_size"] == [1280, 720]
    pts = zones["zones"][0]["points"]
    assert [pts[0], pts[2]] == [[320.0, 180.0], [960.0, 540.0]]                   # default: centred box, 25%..75%


def test_failed_run_shows_the_last_lines_and_does_not_crash(monkeypatch, clean_ui_files):
    def failing_command(*_args, **_kwargs):
        return [sys.executable, "-c", "import sys; print('step one'); print('boom: bad thing'); sys.exit(3)"]

    monkeypatch.setattr(ui_helpers, "build_command", failing_command)
    at = load_app()
    at.toggle(key="reuse").set_value(False).run()                                 # no copying of cached detections
    at.button(key="run_button").click().run(timeout=60)
    assert not at.exception
    assert at.error, "a failing run must be shown as an error"
    message = texts(at.error)[0]
    assert "exit code 3" in message and "boom: bad thing" in message
    assert at.session_state["last_run"]["ok"] is False
    assert not at.success                                                         # no 'finished' message
    assert any("Run log" == e.label for e in at.expander)


# --------------------------------------------------------------------------- uploads

@needs_safety_clip
def test_upload_with_spaces_and_odd_characters_is_saved_safely(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"                                                # not the real samples/uploads: that
    monkeypatch.setattr(ui_helpers, "UPLOAD_DIR", uploads)                        # may hold a clip of the same name
    at = load_app()
    at.button_group(key="source").set_value("Upload").run()
    assert not at.exception
    assert at.get("file_uploader")                                                # the uploader is shown
    at.file_uploader(key="upload").set_value(("My Clip (v2) #1.mp4", SAFETY_VIDEO.read_bytes(), "video/mp4")).run()
    assert not at.exception, at.exception
    saved = uploads / "My_Clip_v2_1.mp4"
    assert saved.exists() and saved.stat().st_size == SAFETY_VIDEO.stat().st_size
    assert any("My_Clip_v2_1.mp4" in v for v in texts(at.sidebar.caption))
    assert {"Duration", "Frame rate", "Size"} <= {m.label for m in at.metric}      # the video was readable
    assert any("outputs/My_Clip_v2_1" in v for v in texts(at.caption))             # the out dir uses the safe stem
    assert not at.button(key="run_button").disabled


def test_upload_that_is_not_a_video_is_rejected_without_crashing(clean_ui_files):
    at = load_app()
    at.button_group(key="source").set_value("Upload").run()
    at.file_uploader(key="upload").set_value(("broken.mp4", b"this is not a video", "video/mp4")).run()
    assert not at.exception
    assert any("cannot be opened as a video" in v for v in texts(at.sidebar.error))
    assert at.button(key="run_button").disabled                                   # nothing to run


# =========================================================================== adversarial / regression tests
# Each test below pins down something that went wrong (or could have gone wrong) in a live demo.

def selected(at: AppTest, label: str):
    """A sidebar widget by its label (the keys of some widgets change with the scenario)."""
    return next(w for w in list(at.sidebar.number_input) + list(at.sidebar.slider) if w.label == label)


def seed_last_run(at: AppTest, out_dir: str, **extra) -> None:
    """Pretend a finished run into `out_dir` exists (as if Run had been pressed earlier in this session)."""
    at.session_state["last_run"] = dict({"ok": True, "code": 0, "out_dir": out_dir, "cmd": "", "interpreter": "",
                                         "lines": [], "seconds": 1.0, "video": "samples/synthetic_safety.mp4",
                                         "scenario": "workplace"}, **extra)


def prepare_failing_run(monkeypatch, script: str) -> None:
    """Make the next Run start `python -c script` instead of run.py (to test how the page reports a failure)."""
    monkeypatch.setattr(ui_helpers, "build_command", lambda *a, **k: [sys.executable, "-c", script])


# --------------------------------------------------------------------------- presets and options

def test_every_preset_loads_and_the_stride_default_follows_the_preset():
    at = load_app()
    for name in utils.list_scenarios():
        at.selectbox(key="scenario").select(name).run()
        assert not at.exception, name
        assert selected(at, "Frame stride").value == ui_helpers.default_stride(name), name
    assert ui_helpers.default_stride("navigation") == 1 and ui_helpers.default_stride("campus") == 2


def test_navigation_preset_passes_no_stride_flag_when_the_box_is_untouched(monkeypatch, clean_ui_files):
    """navigation uses stride 1 on its own; sending the old base default (2) would silently change the analysis."""
    commands = []
    monkeypatch.setattr(ui_helpers, "build_command",
                        lambda *a, **k: commands.append(k) or [sys.executable, "-c", "print('x')"])
    at = load_app()
    at.selectbox(key="scenario").select("navigation").run()
    at.toggle(key="reuse").set_value(False).run()
    at.button(key="run_button").click().run(timeout=60)
    assert not at.exception
    assert commands and commands[0]["stride"] is None


# --------------------------------------------------------------------------- stale results, reload

@needs_safety_results
def test_changing_the_selection_after_a_run_warns_that_the_results_are_old():
    at = AppTest.from_file(str(APP), default_timeout=60)
    seed_last_run(at, "outputs/synthetic_safety")
    at.run()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.selectbox(key="scenario").select("workplace").run()               # -> outputs/synthetic_safety_workplace
    assert not at.exception
    banner = " ".join(texts(at.info))
    assert "belong to the last run" in banner and "outputs/synthetic_safety_workplace" in banner
    assert {"Incidents", "Events"} <= {m.label for m in at.metric}        # the old results stay visible, but labelled
    at.selectbox(key="scenario").select("campus").run()                  # -> outputs/synthetic_safety = the last run
    assert not any("belong to the last run" in v for v in texts(at.info))


@needs_safety_results
def test_page_reload_restores_the_last_run_and_the_sidebar():
    ui_helpers.save_last_run({"ok": True, "code": 0, "out_dir": "outputs/synthetic_safety", "cmd": "python run.py",
                              "interpreter": "python", "lines": ["done"], "seconds": 8.3,
                              "video": "samples/synthetic_safety.mp4", "scenario": "workplace"})
    at = load_app()                                                      # a brand new session = a browser reload
    assert not at.exception
    assert at.selectbox(key="sample").value == "synthetic_safety.mp4"
    assert at.selectbox(key="scenario").value == "workplace"
    assert any("Last run finished" in v and "outputs/synthetic_safety" in v for v in texts(at.success))
    assert {"Incidents", "Events"} <= {m.label for m in at.metric}
    assert not at.warning                                                # no "set both by key and by value" warning


@pytest.mark.parametrize("content", ["{not json", "[]", '{"ok": true}', '{"ok": true, "out_dir": "../../outside"}',
                                     '{"ok": true, "out_dir": "outputs/folder_that_is_gone"}'])
def test_a_broken_or_stale_last_run_file_is_ignored(content):
    ui_helpers.LAST_RUN_FILE.write_text(content, encoding="utf-8")
    at = load_app()
    assert not at.exception and not at.error
    assert not at.success
    assert any("Results appear here" in v for v in texts(at.caption))


# --------------------------------------------------------------------------- zones

@needs_safety_clip
def test_rectangle_with_no_area_blocks_run_and_swapped_corners_are_sorted():
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.button_group(key="zone_mode").set_value("Rectangle").run()
    for label, value in (("x1 (left)", 500), ("x2 (right)", 500)):
        selected(at, label).set_value(value)
    at.run()
    assert not at.exception
    assert any("no area" in v for v in texts(at.sidebar.error))
    assert at.button(key="run_button").disabled
    for label, value in (("x1 (left)", 900), ("x2 (right)", 100)):       # corners the wrong way round: still a zone
        selected(at, label).set_value(value)
    at.run()
    assert not at.exception and not at.sidebar.error
    assert not at.button(key="run_button").disabled


@needs_safety_clip
@pytest.mark.parametrize("raw", [b'{"zones": [{"name": "a", "points": [[0, 0], [1e999, 5], [3, 3]]}]}',     # infinity
                                 b'{"zones": [{"name": "a", "points": [[0, 0], [NaN, 5], [3, 3]]}]}',
                                 b'{"zones": [{"name": "a", "points": [[0, 0], [1e12, 5], [-1e12, 3]]}]}',
                                 b'{"zones": "abc"}', b'{"zones": null}', b'[1, 2, 3]', b'{"zones": []}', b'\x00\x01'])
def test_unusable_uploaded_zones_are_explained_not_crashed(raw, clean_ui_files):
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.button_group(key="zone_mode").set_value("Zones file").run()
    at.file_uploader(key="zone_upload").set_value(("z.json", raw, "application/json")).run()
    assert not at.exception, at.exception
    message = " ".join(texts(at.sidebar.error))
    assert "not usable" in message
    assert "object has no attribute" not in message and "NoneType" not in message      # no Python jargon
    assert at.button(key="run_button").disabled


@needs_safety_clip
def test_a_zone_outside_the_frame_warns_but_does_not_block(clean_ui_files):
    far = b'{"image_size": [1280, 720], "zones": [{"name": "far", "points": [[2000, 2000], [2100, 2000], [2100, 2100]]}]}'
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.button_group(key="zone_mode").set_value("Zones file").run()
    at.file_uploader(key="zone_upload").set_value(("far.json", far, "application/json")).run()
    assert not at.exception and not at.error
    assert any("completely outside" in v for v in texts(at.warning))
    assert not at.button(key="run_button").disabled


# --------------------------------------------------------------------------- uploads

@needs_safety_clip
def test_unicode_upload_names_are_made_safe(tmp_path, monkeypatch):
    """(Streamlit itself refuses a name without a video extension, so only the other odd names reach the app.)"""
    monkeypatch.setattr(ui_helpers, "UPLOAD_DIR", tmp_path / "uploads")
    at = load_app()
    at.button_group(key="source").set_value("Upload").run()
    odd = "".join(chr(c) for c in (0x4E2D, 0x6587)) + " clip (1).MP4"              # non-ASCII text, spaces, brackets
    at.file_uploader(key="upload").set_value((odd, SAFETY_VIDEO.read_bytes(), "video/mp4")).run()
    assert not at.exception, at.exception
    saved = list((tmp_path / "uploads").iterdir())
    assert len(saved) == 1 and saved[0].name.isascii() and saved[0].suffix == ".mp4", saved
    assert not at.button(key="run_button").disabled


def test_a_file_that_is_not_a_video_is_not_kept_in_samples_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_helpers, "UPLOAD_DIR", tmp_path / "uploads")
    at = load_app()
    at.button_group(key="source").set_value("Upload").run()
    at.file_uploader(key="upload").set_value(("fake.mp4", b"MZ this is an exe", "video/mp4")).run()
    assert not at.exception
    assert any("cannot be opened as a video" in v for v in texts(at.sidebar.error))
    assert not (tmp_path / "uploads" / "fake.mp4").exists()                         # nothing left to pick by mistake
    assert at.button(key="run_button").disabled


# --------------------------------------------------------------------------- failing and noisy runs

def test_run_py_exiting_with_code_2_shows_its_message(clean_ui_files):
    """A zones file that does not exist makes run.py stop at once with 'error: zones file not found' (exit code 2)."""
    original = ui_helpers.build_command

    def with_missing_zones(*args, **kwargs):
        return original(*args, **kwargs) + ["--zones", "outputs/_ui/no_such_zones.json"]

    at = AppTest.from_file(str(APP), default_timeout=60)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ui_helpers, "build_command", with_missing_zones)
        at.run()
        at.toggle(key="reuse").set_value(False).run()
        at.button(key="run_button").click().run(timeout=60)
    assert not at.exception
    message = texts(at.error)[0]
    assert "exit code 2" in message and "zones file not found" in message
    assert "Traceback" not in message


def test_very_long_output_is_cut_and_the_page_survives(monkeypatch, clean_ui_files):
    script = ("import sys\n"
              "for i in range(20000): print('progress line', i)\n"
              "print('x' * 300000)\n"
              "print('final problem', file=sys.stderr)\n"
              "sys.exit(4)")
    prepare_failing_run(monkeypatch, script)
    at = load_app()
    at.toggle(key="reuse").set_value(False).run()
    at.button(key="run_button").click().run(timeout=120)
    assert not at.exception
    message = texts(at.error)[0]
    assert "exit code 4" in message and "final problem" in message
    assert len(message) < 20000                                                      # the 300 000 character line is cut
    last = at.session_state["last_run"]
    assert len(last["lines"]) <= 400 and max(len(x) for x in last["lines"]) <= ui_helpers.MAX_LINE_CHARS


def test_a_failed_run_is_restored_after_a_reload_with_its_log(monkeypatch, clean_ui_files):
    prepare_failing_run(monkeypatch, "import sys; print('disk is on fire'); sys.exit(5)")
    at = load_app()
    at.toggle(key="reuse").set_value(False).run()
    at.button(key="run_button").click().run(timeout=60)
    reloaded = load_app()                                                            # new session
    assert not reloaded.exception
    assert "disk is on fire" in texts(reloaded.error)[0] and "exit code 5" in texts(reloaded.error)[0]


@needs_safety_clip
def test_running_twice_in_a_row_replaces_the_results(clean_ui_files):
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.selectbox(key="scenario").select("workplace").run()
    at.toggle(key="skip_video").set_value(True).run()
    for attempt in (1, 2):
        at.button(key="run_button").click().run(timeout=180)
        assert not at.exception and not at.error, (attempt, texts(at.error))
        assert at.session_state["last_run"]["ok"]
    assert "reusing cached detections" in " ".join(at.session_state["last_run"]["lines"])   # the 2nd run found its own
    assert len([m for m in at.metric if m.label == "Incidents"]) == 1                      # one results block, not two
    stored = ui_helpers.load_last_run()
    assert stored and stored["ok"] and stored["scenario"] == "workplace"                    # and it survives a reload


@needs_safety_clip
def test_a_video_name_with_spaces_goes_through_the_whole_chain(clean_ui_files):
    """The project path already has a space ('LOKESHRAJ M'); a space in the video name adds quoting in the command,
    the output folder name and the cached-detections lookup."""
    name = "clip with spaces.mp4"
    shutil.copyfile(SAFETY_VIDEO, ROOT / "samples" / name)
    header = utils.read_json(SAFETY_RESULTS / "detections.json")
    header["video"] = f"samples/{name}"                                              # detections are matched by file name
    utils.write_json(ROOT / "outputs" / "_spaces_cache" / "detections.json", header)
    at = load_app()
    at.selectbox(key="sample").select(name).run()
    at.selectbox(key="scenario").select("workplace").run()
    at.toggle(key="skip_video").set_value(True).run()
    assert any("Cached detections found" in v for v in texts(at.caption))
    at.button(key="run_button").click().run(timeout=180)
    assert not at.exception and not at.error, texts(at.error)
    last = at.session_state["last_run"]
    assert last["ok"] and last["out_dir"] == "outputs/clip with spaces_workplace"
    assert '"samples/clip with spaces.mp4"' in last["cmd"]                            # quoted for a person to copy
    assert (ROOT / last["out_dir"] / "events.json").exists()


@needs_safety_clip
def test_a_run_with_another_preset_keeps_the_old_detections(monkeypatch, clean_ui_files):
    """campus and workplace share outputs/synthetic_safety/; a campus run must not destroy the workplace cache."""
    prepare_failing_run(monkeypatch, "import sys; sys.exit(3)")
    before = (SAFETY_RESULTS / "detections.json").read_bytes()
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.selectbox(key="scenario").select("campus").run()                              # pool is now the default preset
    at.toggle(key="reuse").set_value(False).run()
    at.button(key="run_button").click().run(timeout=60)
    kept = ROOT / "outputs" / "_ui" / "cache" / "synthetic_safety.json"
    assert kept.is_file() and kept.read_bytes() == before
    assert any("kept the old detections" in line for line in at.session_state["last_run"]["lines"])


# --------------------------------------------------------------------------- malformed results never crash the page

GOOD_EVENT = {"event_id": 1, "behavior": "near_miss", "behavior_name": "Near miss", "entity_name": "Person #1",
              "start": "00:06", "end": "00:06", "duration_s": 0.4, "severity": "high", "confidence": 0.8,
              "confidence_parts": {"margin": 0.9, "duration": 0.5, "detection": 0.9}, "evidence": "came within 0.1"}


@pytest.mark.parametrize("name, patch", [
    ("events_are_strings", {"events": ["x", 3, None]}),
    ("confidence_is_text", {"events": [dict(GOOD_EVENT, confidence="abc")]}),
    ("parts_are_junk", {"events": [dict(GOOD_EVENT, confidence_parts={"margin": "x", "duration": None})]}),
    ("duration_is_text", {"events": [dict(GOOD_EVENT, duration_s="long")]}),
    ("pose_is_junk", {"events": [dict(GOOD_EVENT, verified=True, pose={"median_lean_deg": "x"})]}),
    ("baseline_is_text", {"events": [dict(GOOD_EVENT, baseline="oops")]}),
    ("snapshot_escapes_the_folder", {"events": [dict(GOOD_EVENT, snapshot="../../../Windows/win.ini", speed_plot="C:/x")]}),
    ("html_in_the_evidence", {"events": [dict(GOOD_EVENT, evidence="<script>alert(1)</script> **b** [x](http://e) $$ `")]}),
    ("incidents_are_strings", {"events": [GOOD_EVENT], "incidents": ["a", 1]}),
    ("incident_ids_are_junk", {"events": [GOOD_EVENT], "incidents": [{"event_ids": ["a", None], "severity": "high"}]}),
    ("incident_ids_not_a_list", {"events": [GOOD_EVENT], "incidents": [{"event_ids": 5}]}),
    ("meta_is_text", {"meta": "x"}),
    ("meta_duration_is_text", {"meta": {"duration_s": "abc", "processing_fps": "fast", "config": "cfg", "scenario": "s"}}),
    ("near_misses_are_strings", {"near_misses": ["a"], "unusual_tracks": [1, "a"]}),
    ("reel_is_a_list", {"highlight_reel": ["a"]}),
    ("everything_is_null", {"events": None, "incidents": None, "meta": None, "near_misses": None}),
])
def test_malformed_events_json_never_crashes_the_page(tmp_path, name, patch):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "events.json").write_text(json.dumps(patch), encoding="utf-8")
    at = AppTest.from_file(str(APP), default_timeout=60)
    seed_last_run(at, str(folder))
    at.run()
    assert not at.exception, (name, at.exception[0].value if at.exception else "")
    assert "Results" in " ".join(texts(at.subheader))


@pytest.mark.parametrize("content", ["[]", "null", '"text"', "12", ""])
def test_events_json_that_is_not_an_object_gives_a_readable_error(tmp_path, content):
    (tmp_path / "events.json").write_text(content, encoding="utf-8")
    at = AppTest.from_file(str(APP), default_timeout=60)
    seed_last_run(at, str(tmp_path))
    at.run()
    assert not at.exception
    assert any("Cannot read" in v for v in texts(at.error))


# --------------------------------------------------------------------------- performance: cached, not repeated

@needs_safety_results
@needs_safety_clip
def test_reruns_do_not_reopen_the_video_or_reread_the_results(monkeypatch):
    calls = {"frame": 0, "facts": 0, "events": 0}
    real_frame, real_facts, real_read = ui_helpers.first_frame, ui_helpers.video_facts, utils.read_json

    def counting_frame(*a, **k):
        calls["frame"] += 1
        return real_frame(*a, **k)

    def counting_facts(*a, **k):
        calls["facts"] += 1
        return real_facts(*a, **k)

    def counting_read(path, *a, **k):
        calls["events"] += str(path).endswith("events.json")
        return real_read(path, *a, **k)

    monkeypatch.setattr(ui_helpers, "first_frame", counting_frame)
    monkeypatch.setattr(ui_helpers, "video_facts", counting_facts)
    monkeypatch.setattr(utils, "read_json", counting_read)
    at = AppTest.from_file(str(APP), default_timeout=60)
    seed_last_run(at, "outputs/synthetic_safety")
    at.run()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    first = dict(calls)
    assert first["frame"] >= 1 and first["events"] == 1
    for value in (True, False, True):                                                # sidebar changes = full reruns
        at.toggle(key="privacy").set_value(value).run()
        assert not at.exception
    at.selectbox(key="scenario").select("workplace").run()
    at.selectbox(key="scenario").select("campus").run()
    assert calls == first, f"a rerun reopened the video or reread events.json: {first} -> {calls}"


# --------------------------------------------------------------------------- every results folder, odd videos, hints

def test_every_existing_results_folder_opens_without_a_crash():
    """examples/ and outputs/ hold folders of every shape (trimmed examples, with or without videos or a heatmap)."""
    at = load_app()
    folders = at.selectbox(key="open_results").options
    assert "examples/campus_one-by-one" in folders or "examples/workplace_near-miss_fall" in folders
    for folder in folders:
        at.selectbox(key="open_results").select(folder).run()
        assert not at.exception, (folder, at.exception[0].value if at.exception else "")
        assert not at.error, (folder, texts(at.error))
        assert {"Incidents", "Events"} <= {m.label for m in at.metric}, folder
        assert [t.label for t in at.tabs][:3] == ["Summary", "Highlight reel", "Incidents"], folder


def test_videos_that_are_cut_or_empty_do_not_crash_the_page(tmp_path, monkeypatch):
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not (ffmpeg and SAFETY_VIDEO.exists()):
        pytest.skip("needs ffmpeg and samples/synthetic_safety.mp4")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    whole = tmp_path / "whole.mp4"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(SAFETY_VIDEO), "-t", "6", "-c", "copy",
                    "-movflags", "+faststart", str(whole)], check=True)           # the index (moov) first, so a cut still opens
    data = whole.read_bytes()
    (uploads / "cut.mp4").write_bytes(data[: len(data) // 3])                       # opens, but most frames are missing
    (uploads / "header_only.mp4").write_bytes(data[:100])                           # cannot be opened
    (uploads / "zero.mp4").write_bytes(b"")
    monkeypatch.setattr(ui_helpers, "UPLOAD_DIR", uploads)
    at = load_app()
    for name in ("uploads/cut.mp4", "uploads/header_only.mp4", "uploads/zero.mp4"):
        at.selectbox(key="sample").select(name).run()
        assert not at.exception, (name, at.exception[0].value if at.exception else "")
        if name != "uploads/cut.mp4":
            assert any("cannot be opened as a video" in v for v in texts(at.sidebar.error)), name
            assert at.button(key="run_button").disabled, name


@needs_safety_clip
def test_the_page_points_to_the_preset_that_has_cached_detections():
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()                 # campus is the default preset
    captions = texts(at.caption)
    assert any("No cached detections" in c for c in captions)
    hint = [c for c in captions if "Cached detections exist for" in c]
    assert hint and "Industrial / workplace safety" in hint[0]                      # the preset the cache was made with
    at.selectbox(key="scenario").select("workplace").run()
    assert any("Cached detections found" in c for c in texts(at.caption))
    assert not any("Cached detections exist for" in c for c in texts(at.caption))


@needs_safety_clip
def test_max_seconds_that_the_cache_does_not_cover_means_yolo_will_run():
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()
    at.selectbox(key="scenario").select("workplace").run()
    selected(at, "Max seconds (0 = all)").set_value(5).run()                        # the cached run covers all 30 s
    assert any("No cached detections" in c for c in texts(at.caption))
    at.toggle(key="reuse").set_value(False).run()
    assert any("Reuse is off" in c for c in texts(at.caption))


@needs_safety_clip
def test_a_missing_yolo_weights_file_is_flagged_before_the_run(monkeypatch):
    monkeypatch.setattr(ui_helpers, "weights_present", lambda name: False)
    at = load_app()
    at.selectbox(key="sample").select("synthetic_safety.mp4").run()                 # campus: no cache, YOLO would run
    assert any("is not in the project folder" in c for c in texts(at.caption))
    at.selectbox(key="scenario").select("workplace").run()                          # cached detections: YOLO is skipped
    assert not any("is not in the project folder" in c for c in texts(at.caption))


def test_a_long_event_list_is_cut_until_show_all_is_chosen(tmp_path):
    events = [dict(GOOD_EVENT, event_id=i, start=f"00:{i % 60:02d}", end=f"00:{i % 60:02d}") for i in range(1, 121)]
    (tmp_path / "events.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    at = AppTest.from_file(str(APP), default_timeout=60)
    seed_last_run(at, str(tmp_path))
    at.run()
    assert not at.exception
    cards = [e for e in at.expander if e.label.startswith("#")]
    assert len(cards) == 30                                                          # not 120 cards with 240 pictures
    assert {m.label: m.value for m in at.metric}["Events"] == "120"                  # the numbers still count them all
    at.toggle(key="all_events").set_value(True).run()
    assert len([e for e in at.expander if e.label.startswith("#")]) == 120


ONE_BY_ONE = ROOT / "samples" / "one-by-one-person-detection.mp4"
needs_campus_demo = pytest.mark.skipif(
    not (ONE_BY_ONE.exists() and (ROOT / "outputs" / "one-by-one" / "detections.json").exists()
         and (ROOT / "samples" / "one-by-one_zones.json").exists()),
    reason="demo step 3 needs samples/one-by-one-person-detection.mp4, its zones file and cached detections")


@needs_campus_demo
def test_demo_step_3_real_footage_with_the_preselected_zones_file(clean_ui_files):
    """DEMO.md step 3: the zones file is preselected, the cached detections are found, the zone event appears."""
    at = load_app()
    at.selectbox(key="sample").select("one-by-one-person-detection.mp4").run()
    at.selectbox(key="scenario").select("campus").run()                              # this demo step uses campus
    at.button_group(key="zone_mode").set_value("Zones file").run()
    picker = next(b for b in at.sidebar.selectbox if b.key and b.key.startswith("zfile_"))
    assert ui_helpers.rel_path(picker.value) == "samples/one-by-one_zones.json"        # chosen by name, not by luck
    assert any("Cached detections found" in c for c in texts(at.caption))
    at.toggle(key="skip_video").set_value(True).run()
    at.button(key="run_button").click().run(timeout=180)
    assert not at.exception and not at.error, texts(at.error)
    last = at.session_state["last_run"]
    assert last["ok"] and "--zones samples/one-by-one_zones.json" in last["cmd"]
    summary = (ROOT / last["out_dir"] / "summary.txt").read_text(encoding="utf-8")
    assert "zone" in summary.lower()
