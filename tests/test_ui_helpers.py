"""Unit tests for ui_helpers.py (the plain-Python half of the Streamlit UI). No browser, no YOLO, no Streamlit."""
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # also works without conftest.py
    sys.path.insert(0, str(ROOT))

import ui_helpers as ui  # noqa: E402
import utils  # noqa: E402


# --------------------------------------------------------------------------- file names and discovery

@pytest.mark.parametrize("raw, expected", [
    ("clip.mp4", "clip.mp4"),
    ("My Clip (final) #2.MP4", "My_Clip_final_2.mp4"),            # spaces, brackets, hash, upper-case extension
    ("../../etc/passwd", "passwd.mp4"),                           # folders are dropped, no video extension -> .mp4
    ("..\\..\\win\\evil.mov", "evil.mov"),                        # Windows separators too
    ("a b.c.mkv", "a_b.c.mkv"),
    ("CON.mp4", "_CON.mp4"),                                      # reserved Windows device name
    ("x.exe", "x.mp4"),                                           # never keep an executable extension
])
def test_safe_filename(raw, expected):
    assert ui.safe_filename(raw) == expected


def test_safe_filename_for_names_made_only_of_odd_characters():
    japanese = "".join(chr(c) for c in (0x65E5, 0x672C, 0x8A9E)) + ".mp4"           # non-ASCII names
    chinese = "".join(chr(c) for c in (0x4E2D, 0x6587)) + ".mp4"
    a, b = ui.safe_filename(japanese), ui.safe_filename(chinese)
    assert a != b and a.startswith("video_") and a.endswith(".mp4")        # a hash keeps two uploads apart
    assert a.isascii()
    assert ui.safe_filename("").endswith(".mp4")
    assert len(ui.safe_filename("x" * 500 + ".mp4")) <= 90


def test_save_upload_writes_into_the_upload_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UPLOAD_DIR", tmp_path / "uploads")
    path = ui.save_upload("Odd name (1).mp4", memoryview(b"abc"))
    assert path == tmp_path / "uploads" / "Odd_name_1.mp4"
    assert path.read_bytes() == b"abc"


def test_list_videos_and_result_dirs_and_zone_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(ui, "UPLOAD_DIR", tmp_path / "samples" / "uploads")
    (tmp_path / "samples" / "uploads").mkdir(parents=True)
    (tmp_path / "samples" / "b.mp4").write_bytes(b"x")
    (tmp_path / "samples" / "a.MOV").write_bytes(b"x")
    (tmp_path / "samples" / "notes.txt").write_text("no")
    (tmp_path / "samples" / "uploads" / "u.mp4").write_bytes(b"x")
    assert [label for label, _ in ui.list_videos()] == ["a.MOV", "b.mp4", "uploads/u.mp4"]

    (tmp_path / "outputs" / "done").mkdir(parents=True)
    (tmp_path / "outputs" / "done" / "events.json").write_text("{}")
    (tmp_path / "outputs" / "unfinished").mkdir()
    (tmp_path / "outputs" / "_ui").mkdir()
    (tmp_path / "outputs" / "_ui" / "events.json").write_text("{}")             # hidden: starts with "_"
    (tmp_path / "examples" / "demo").mkdir(parents=True)
    (tmp_path / "examples" / "demo" / "events.json").write_text("{}")
    assert [ui.rel_path(d) for d in ui.list_result_dirs()] == ["outputs/done", "examples/demo"]

    (tmp_path / "samples" / "x_zones.json").write_text("{}")
    (tmp_path / "examples" / "demo" / "zones.json").write_text("{}")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "other.json").write_text("{}")                         # does not match *zones*
    assert [ui.rel_path(p) for p in ui.list_zone_files()] == ["examples/demo/zones.json", "samples/x_zones.json"]


def test_suggest_zone_file_needs_a_name_match():
    files = [Path("samples/one-by-one_zones.json"), Path("tests/synthetic_zones.json"), Path("examples/x/zones.json")]
    assert ui.suggest_zone_file(Path("one-by-one-person-detection.mp4"), files) == files[0]
    assert ui.suggest_zone_file(Path("synthetic.mp4"), files) == files[1]
    assert ui.suggest_zone_file(Path("synthetic_safety.mp4"), files) is None      # not a zones file for this clip
    assert ui.suggest_zone_file(Path("zones.mp4"), files) is None
    assert ui.suggest_zone_file(None, files) is None


def test_safe_child_stays_inside_the_folder(tmp_path):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "e1.jpg").write_bytes(b"x")
    (tmp_path.parent / "secret.txt").write_text("no")
    assert ui.safe_child(tmp_path, "snapshots/e1.jpg") == (tmp_path / "snapshots" / "e1.jpg").resolve()
    assert ui.safe_child(tmp_path, "snapshots/missing.jpg") is None
    assert ui.safe_child(tmp_path, "../secret.txt") is None                       # events.json is data, not trusted
    assert ui.safe_child(tmp_path, None) is None
    assert ui.safe_child(tmp_path, "") is None


def test_zip_folder_bytes_holds_every_file_under_the_folder_name(tmp_path):
    folder = tmp_path / "res"
    (folder / "plots").mkdir(parents=True)
    (folder / "events.json").write_text("{}")
    (folder / "plots" / "e1.png").write_bytes(b"png")
    with zipfile.ZipFile(io.BytesIO(ui.zip_folder_bytes(folder))) as zf:
        assert sorted(zf.namelist()) == ["res/events.json", "res/plots/e1.png"]
        assert zf.read("res/plots/e1.png") == b"png"


# --------------------------------------------------------------------------- presets, command line

def test_scenario_summary_has_title_rules_and_classes():
    info = ui.scenario_summary("workplace")
    assert info["title"] == utils.load_config(None, "workplace")["scenario"]["title"]
    assert "Near miss" in info["rules"] and "bicycle" in info["classes"]
    broken = ui.scenario_summary("no_such_preset")                                 # never raises
    assert "cannot be loaded" in broken["title"] and broken["rules"] == []
    assert ui.preset_classes("campus") == [0] and ui.preset_classes("no_such_preset") == [0]


def test_default_out_dir_follows_run_py():
    import run
    video = Path("samples/clip one.mp4")
    for scenario in ("campus", "workplace", "traffic"):
        assert ui.default_out_dir(video, scenario) == run.default_out_dir(video, scenario).as_posix()


def test_build_command_minimal_and_full():
    video = ui.SAMPLES_DIR / "my clip.mp4"                                        # a space: still ONE argument
    cmd = ui.build_command(video, "campus", "outputs/my clip", reuse=False)
    assert cmd == [sys.executable, "run.py", "--video", "samples/my clip.mp4", "--scenario", "campus",
                   "--out", "outputs/my clip"]
    full = ui.build_command(video, "workplace", "outputs/x", zones=ROOT / "tests" / "synthetic_zones.json",
                            privacy=True, device="cpu", stride=3, max_seconds=30.0, reuse=True, no_video=True,
                            pose=True)
    for flag in (["--zones", "tests/synthetic_zones.json"], ["--device", "cpu"], ["--stride", "3"],
                 ["--max-seconds", "30"]):
        assert full[full.index(flag[0]):full.index(flag[0]) + 2] == flag
    for switch in ("--privacy", "--pose", "--no-video", "--reuse"):
        assert switch in full
    # run.py's own defaults are not sent
    default = ui.build_command(video, "campus", "outputs/x", device="auto", stride=None, max_seconds=0)
    assert "--device" not in default and "--stride" not in default and "--max-seconds" not in default


def test_every_built_command_is_accepted_by_run_py_parser():
    import run
    args = ui.build_command(ui.SAMPLES_DIR / "a.mp4", "workplace", "outputs/a_workplace", zones="z.json", privacy=True,
                            device="0", stride=4, max_seconds=12.5, reuse=True, no_video=True, pose=True)[2:]
    parsed = run.build_parser().parse_args(args)                                   # would exit(2) on an unknown flag
    assert parsed.scenario == "workplace" and parsed.stride == 4 and parsed.max_seconds == 12.5
    assert parsed.privacy and parsed.reuse and parsed.no_video and parsed.pose and parsed.device == "0"


def test_display_command_shows_python_and_quotes_spaces():
    text = ui.display_command([sys.executable, "run.py", "--video", "samples/a b.mp4"])
    assert text == 'python run.py --video "samples/a b.mp4"'


# --------------------------------------------------------------------------- zones and preview

def test_rect_zone_orders_corners_and_detects_empty():
    zone = ui.rect_zone(960, 540, 320, 180)                                        # opposite order
    assert zone[0]["points"] == [(320.0, 180.0), (960.0, 180.0), (960.0, 540.0), (320.0, 540.0)]
    assert not ui.rect_is_empty(zone)
    assert ui.rect_is_empty(ui.rect_zone(100, 100, 100, 300))
    assert ui.rect_is_empty(ui.rect_zone(100, 100, 300, 101))


def test_write_zone_file_round_trips_through_run_py_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UI_DIR", tmp_path / "_ui")
    path = ui.write_zone_file("My Clip", ui.rect_zone(10, 20, 110, 220), (200, 400))
    assert path == tmp_path / "_ui" / "My_Clip_zones.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["image_size"] == [200, 400] and data["zones"][0]["name"] == "restricted"
    scaled = utils.load_zones(path, (100, 200))                                    # what run.py does with the file
    assert scaled[0]["points"][0] == (5.0, 10.0) and scaled[0]["points"][2] == (55.0, 110.0)


def test_save_uploaded_zones_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UI_DIR", tmp_path / "_ui")
    good = json.dumps({"image_size": [100, 100], "zones": [{"name": "a", "points": [[0, 0], [10, 0], [10, 10]]}]})
    path, err = ui.save_uploaded_zones("clip", good.encode())
    assert err == "" and path.exists()
    assert ui.save_uploaded_zones("clip", b"{nope")[0] is None                      # not JSON
    path2, err2 = ui.save_uploaded_zones("clip2", json.dumps({"zones": [{"name": "a", "points": [[0, 0]]}]}).encode())
    assert path2 is None and "no zone" in err2 and not (tmp_path / "_ui" / "clip2_uploaded_zones.json").exists()
    assert ui.save_uploaded_zones("clip3", b'{"zones": [{"name": "a", "points": [[0, 0, 1], [1, 1], [2, 2]]}]}')[0] is None
    assert ui.save_uploaded_zones("clip4", b"[1, 2, 3]")[0] is None                 # wrong shape


def test_zones_in_video_pixels_scales_like_run_py(tmp_path):
    sized = tmp_path / "sized.json"
    sized.write_text(json.dumps({"image_size": [768, 432],
                                 "zones": [{"name": "t", "points": [[350, 300], [570, 300], [570, 432]]}]}))
    zones, err = ui.zones_in_video_pixels(sized, (1536, 864))                        # video twice as large
    assert err == "" and zones[0]["points"][0] == (700.0, 600.0)
    bare = tmp_path / "bare.json"                                                    # points in 640 px processing space
    bare.write_text(json.dumps([[320, 100], [640, 100], [640, 200]]))
    zones, err = ui.zones_in_video_pixels(bare, (1280, 720))
    assert err == "" and zones[0]["points"][0] == (640.0, 200.0)
    zones, err = ui.zones_in_video_pixels(tmp_path / "missing.json", (100, 100))
    assert zones == [] and "cannot read" in err


def test_draw_zones_returns_an_rgb_copy():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    zones = [{"name": "z", "points": [(20, 20), (120, 20), (120, 80), (20, 80)]}]
    out = ui.draw_zones(frame, zones, scale=1.0)
    assert out.shape == frame.shape and frame.sum() == 0                             # the input is untouched
    assert out[20, 60, 0] > 200 and out[20, 60, 2] < 60                              # red outline in RGB order
    assert out[50, 60, 0] > 0                                                        # translucent fill
    half = ui.draw_zones(frame, zones, scale=0.5)                                    # points scale with the picture
    assert half[10, 30, 0] > 200
    assert ui.draw_zones(frame, [], 1.0).sum() == 0


def make_video(path: Path, frames: int = 10, fps: float = 10.0, size=(64, 48)):
    """A tiny mp4 of a moving white square; skips the test when this OpenCV build has no mp4 writer."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 here")
    for i in range(frames):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[10:20, i:i + 10] = 255
        writer.write(img)
    writer.release()
    return path


def test_video_facts_and_first_frame(tmp_path):
    video = make_video(tmp_path / "tiny.mp4", frames=20, fps=10.0, size=(64, 48))
    facts = ui.video_facts(video)
    assert (facts["width"], facts["height"]) == (64, 48)
    assert facts["fps"] == pytest.approx(10.0) and facts["duration_s"] == pytest.approx(2.0, abs=0.2)
    frame, scale = ui.first_frame(video, max_width=32)
    assert scale == 0.5 and frame.shape[:2] == (24, 32)
    frame, scale = ui.first_frame(video, max_width=960)                              # never enlarged
    assert scale == 1.0 and frame.shape[:2] == (48, 64)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert ui.video_facts(junk) is None
    assert ui.first_frame(junk) == (None, 1.0)
    assert ui.video_facts(tmp_path / "missing.mp4") is None


# --------------------------------------------------------------------------- cached detections (--reuse)

def header(**over):
    base = {"video": "samples\\clip.mp4", "stride": 2, "classes": [0, 1], "orig_size": [1280, 720], "detections": []}
    base.update(over)
    return base


def test_find_cached_detections_checks_name_stride_classes_and_size(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "OUTPUTS_DIR", tmp_path / "outputs")
    (tmp_path / "outputs" / "clip").mkdir(parents=True)
    src = tmp_path / "outputs" / "clip" / "detections.json"
    src.write_text(json.dumps(header()))
    video = Path("samples/clip.mp4")
    found = ui.find_cached_detections(video, "outputs/clip_traffic", [1, 0], 2, (1280, 720))
    assert found == str(src)                                                          # class order does not matter
    for kwargs in ({"classes": [0]}, {"stride": 3}, {"size": (640, 360)}):
        args = {"classes": [0, 1], "stride": 2, "size": (1280, 720)} | kwargs
        assert ui.find_cached_detections(video, "outputs/clip_traffic", args["classes"], args["stride"], args["size"]) is None
    assert ui.find_cached_detections(Path("samples/other.mp4"), "outputs/o", [0, 1], 2, (1280, 720)) is None
    src.write_text("{broken")
    assert ui.find_cached_detections(video, "outputs/clip_traffic", [0, 1], 2, (1280, 720)) is None


def test_seed_detections_copies_into_the_output_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    (tmp_path / "outputs" / "clip").mkdir(parents=True)
    src = tmp_path / "outputs" / "clip" / "detections.json"
    src.write_text(json.dumps(header()))
    note = ui.seed_detections(str(src), "outputs/clip_workplace")
    assert (tmp_path / "outputs" / "clip_workplace" / "detections.json").read_text() == src.read_text()
    assert "copied cached detections" in note
    assert ui.seed_detections(None, "outputs/x") == ""
    assert "reusing cached detections" in ui.seed_detections(str(src), "outputs/clip")      # already in place


def test_detections_signature_changes_with_the_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "OUTPUTS_DIR", tmp_path)
    assert ui.detections_signature() == ()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "detections.json").write_text("{}")
    assert len(ui.detections_signature()) == 1


# --------------------------------------------------------------------------- running run.py

def test_progress_follows_step_lines_and_tracking_percent():
    prog = ui.Progress()
    assert prog.feed("PS07 behaviour analysis | video: x") is None
    assert prog.feed("[1/8] Detecting and tracking people (YOLO + ByteTrack) ...") == 0.0
    assert prog.label == "[1/8] Detecting and tracking people (YOLO + ByteTrack)"
    assert prog.feed("  tracking  50%  (frame 10/20, 5.0 processed fps, ETA 0:02)") == pytest.approx(0.5 / 8)
    assert prog.feed("      done in 4.0 s") is None
    assert prog.feed("[2/8] Building per-person tracks (smoothing, speed) ...") == pytest.approx(1 / 8)
    assert prog.feed("[8/8] Writing events.json, summary.txt, report.html ...") == pytest.approx(7 / 8)
    assert 0.0 <= prog.fraction <= 1.0


def test_run_pipeline_streams_lines_and_reports_the_exit_code():
    seen = []
    code, lines = ui.run_pipeline([sys.executable, "-c", "print('one'); print('two')"], seen.append)
    assert code == 0 and lines == ["one", "two"] and seen == lines
    code, lines = ui.run_pipeline([sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"])
    assert code == 7 and lines == ["bad"]


def test_run_pipeline_merges_stderr_and_survives_odd_bytes():
    code, lines = ui.run_pipeline([sys.executable, "-c",
                                   "import sys; sys.stderr.write('warn\\n'); sys.stdout.buffer.write(b'caf\\xe9\\n')"])
    assert code == 0 and lines[0] == "warn" and lines[1].startswith("caf")           # bad byte replaced, no crash


def test_run_pipeline_does_not_raise_when_the_program_cannot_start():
    code, lines = ui.run_pipeline(["definitely-not-a-program-xyz"])
    assert code == 1 and "could not start" in lines[0]


def test_run_pipeline_kills_the_child_when_the_callback_raises():
    """A page rerun interrupts the script with an exception inside on_line: the child must not keep running."""
    def stop(_line):
        raise KeyboardInterrupt

    script = "import time\nprint('go', flush=True)\ntime.sleep(60)"
    started = time.time()
    with pytest.raises(KeyboardInterrupt):
        ui.run_pipeline([sys.executable, "-c", script], stop)
    assert time.time() - started < 30                                  # did not wait for the sleep


def test_last_lines_skips_blank_lines():
    assert ui.last_lines(["a", "", "  ", "b", "c"], 2) == "b\nc"
    assert ui.last_lines([], 20) == ""


# --------------------------------------------------------------------------- reading results

CHAINED = {"incidents": [{"chain_id": 1, "event_ids": [1, 2], "severity": "high"}],
           "events": [{"event_id": 1, "severity": "high"}, {"event_id": 2, "severity": "medium"},
                      {"event_id": 3, "severity": "low"}, {"event_id": 4, "severity": "high"}]}


def test_incident_count_matches_the_highlight_reel():
    assert ui.incident_count(CHAINED) == 3                    # one chain + events 3 and 4 outside any chain
    assert ui.high_incident_count(CHAINED) == 2               # the chain and event 4
    assert ui.incident_count({"events": [{"event_id": 1}], "incidents": []}) == 1
    assert ui.incident_count({}) == 0


def test_result_metrics_handles_sparse_files():
    m = ui.result_metrics({})
    assert m["incidents"] == 0 and m["tracked"] is None and m["duration"] == "?" and m["processing_fps"] is None
    data = {"events": [{"event_id": 1}], "meta": {"n_tracks": 5, "duration_s": 75.0, "processing_fps": 5.67,
                                                  "scenario": {"name": "traffic", "title": "Traffic"}, "privacy": True,
                                                  "config": {"scenario": {"entity_word": "Person"},
                                                             "model": {"classes": [0, 2]}}}}
    m = ui.result_metrics(data)
    assert (m["tracked"], m["duration"], m["processing_fps"], m["scenario"]) == (5, "01:15", "5.7", "traffic")
    assert m["tracked_label"] == "Objects" and m["privacy"] is True


def test_tracked_label_follows_the_scenario():
    assert ui.tracked_label(None) == "People"
    assert ui.tracked_label({"scenario": {"entity_word": "Animal"}, "model": {"classes": [17, 18]}}) == "Animals"
    assert ui.tracked_label({"scenario": {"entity_word": "Person"}, "model": {"classes": [0]}}) == "People"
    assert ui.tracked_label({"scenario": {"entity_word": "Person"}, "model": {"classes": [0, 2]}}) == "Objects"


def test_confidence_line_shows_the_three_parts():
    event = {"confidence": 0.78, "confidence_parts": {"margin": 0.81, "duration": 0.62, "detection": 0.89}}
    weights = ui.confidence_weights({})
    assert weights == {"margin": 0.4, "duration": 0.3, "detection": 0.3}
    assert ui.confidence_line(event, weights) == "confidence 0.78 = 0.4 x 0.81 + 0.3 x 0.62 + 0.3 x 0.89"
    assert ui.confidence_line({"confidence": 0.5}, weights) == "confidence 0.50"
    custom = {"meta": {"config": {"events": {"confidence_weights": {"margin": 0.5, "duration": 0.25, "detection": 0.25}}}}}
    assert ui.confidence_weights(custom)["margin"] == 0.5


def test_severity_color_and_md_escape_and_pose_text():
    assert [ui.severity_color(s) for s in ("high", "MEDIUM", "low", None, "weird")] == \
        ["red", "orange", "blue", "gray", "gray"]
    assert ui.md_escape("Rule: >= 0.6 *fast* _x_ [a] $5 `c` | ~") == \
        "Rule: \\>= 0.6 \\*fast\\* \\_x\\_ \\[a\\] \\$5 \\`c\\` \\| \\~"
    assert ui.pose_text({"verified": None}) is None
    assert ui.pose_text({"verified": True, "pose": {"median_lean_deg": 12.34, "usable_frames": 8, "frames_checked": 12}}) == \
        ("green", "Pose check agrees (median lean 12.3 deg, 8/12 usable frames)")
    assert ui.pose_text({"verified": False})[0] == "red"
    assert ui.pose_text({"verified": "uncertain"}) == ("gray", "Pose check uncertain")


# =========================================================================== adversarial / regression tests

# --------------------------------------------------------------------------- uploads

def test_save_upload_same_name_different_video_gets_its_own_file(tmp_path, monkeypatch):
    """Cached detections are matched by file name: a second, different clip must not reuse the first one's name."""
    monkeypatch.setattr(ui, "UPLOAD_DIR", tmp_path / "uploads")
    first = ui.save_upload("clip.mp4", b"video one")
    again = ui.save_upload("clip.mp4", b"video one")                              # same bytes: nothing is rewritten
    other = ui.save_upload("clip.mp4", memoryview(b"video two"))
    assert first == again == tmp_path / "uploads" / "clip.mp4"
    assert other != first and other.name.startswith("clip_") and other.suffix == ".mp4"
    assert first.read_bytes() == b"video one" and other.read_bytes() == b"video two"
    assert ui.save_upload("clip.mp4", b"video two") == other                      # and it is stable
    assert len(list((tmp_path / "uploads").iterdir())) == 2


def test_discard_upload_only_deletes_inside_the_upload_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UPLOAD_DIR", tmp_path / "uploads")
    mine = ui.save_upload("junk.mp4", b"x")
    outside = tmp_path / "keep.mp4"
    outside.write_bytes(b"precious")
    assert ui.discard_upload(outside) is False and outside.exists()
    assert ui.discard_upload(tmp_path / "uploads" / ".." / "keep.mp4") is False and outside.exists()
    assert ui.discard_upload(mine) is True and not mine.exists()
    assert ui.discard_upload(mine) is False                                       # already gone: no error


# --------------------------------------------------------------------------- zones

def test_zones_problem_and_friendly_errors():
    ok = [{"name": "a", "points": [(0, 0), (10, 0), (10, 10)]}]
    assert ui.zones_problem(ok) == ""
    assert "finite" in ui.zones_problem([{"name": "a", "points": [(0, 0), (float("inf"), 1), (2, 2)]}])
    assert "finite" in ui.zones_problem([{"name": "a", "points": [(0, 0), (float("nan"), 1), (2, 2)]}])
    assert "further than" in ui.zones_problem([{"name": "a", "points": [(0, 0), (1e9, 1), (2, 2)]}])
    assert "layout" in ui.friendly_zone_error(AttributeError("'str' object has no attribute 'get'"))
    assert "layout" in ui.friendly_zone_error(TypeError("'NoneType' object is not iterable"))
    assert "not a number" in ui.friendly_zone_error(ValueError("could not convert string to float: 'a'"))
    assert "too large" in ui.friendly_zone_error(OverflowError("int too large to convert to float"))
    assert ui.friendly_zone_error(ValueError("zone 1: every point must be [x, y], got 1")).startswith("zone 1")


def test_save_uploaded_zones_rejects_infinity_and_huge_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UI_DIR", tmp_path / "_ui")
    for raw in (b'{"zones": [{"name": "a", "points": [[0, 0], [1e999, 5], [3, 3]]}]}',
                b'{"zones": [{"name": "a", "points": [[0, 0], [NaN, 5], [3, 3]]}]}',
                b'{"zones": [{"name": "a", "points": [[0, 0], [1e12, 5], [-1e12, 3]]}]}',
                b'{"image_size": [0, 0], "zones": [{"name": "a", "points": [[0, 0], [1, 1], [2, 0]]}]}',
                b'{"image_size": "big", "zones": [{"name": "a", "points": [[0, 0], [1, 1], [2, 0]]}]}',
                b'{"zones": "abc"}', b'{"zones": null}', b'{"zones": [5]}'):
        path, err = ui.save_uploaded_zones("clip", raw)
        assert path is None and err, raw
        assert "object has no attribute" not in err and "NoneType" not in err and "str' object" not in err
    assert not list((tmp_path / "_ui").glob("*.json"))                             # nothing unusable is kept
    path, err = ui.save_uploaded_zones("clip", b'{"zones": [{"name": 123, "points": [[0, 0], [5, 0], [5, 5]]}]}')
    assert path is not None and err == ""                                          # a numeric name is still fine


def test_zones_in_video_pixels_refuses_non_finite_points(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"image_size": [100, 100], "zones": [{"name": "a", "points": [[0, 0], [1e999, 5], [3, 3]]}]}')
    zones, err = ui.zones_in_video_pixels(bad, (200, 200))
    assert zones == [] and "finite" in err


def test_draw_zones_survives_nan_infinity_and_huge_points():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    for bad in (float("inf"), float("nan"), 1e12, -1e12):
        zones = [{"name": "z", "points": [(5, 5), (bad, 10), (40, 50)]}]
        out = ui.draw_zones(frame, zones, scale=0.5)
        assert out.shape == frame.shape                                            # no OverflowError, no ValueError


def test_zones_outside_frame():
    inside = [{"name": "a", "points": [(10, 10), (50, 10), (50, 50)]}]
    partly = [{"name": "a", "points": [(-50, -50), (20, -50), (20, 20)]}]
    far = [{"name": "a", "points": [(2000, 2000), (2100, 2000), (2100, 2100)]}]
    assert not ui.zones_outside_frame(inside, (100, 100))
    assert not ui.zones_outside_frame(partly, (100, 100))                          # touching the frame is enough
    assert ui.zones_outside_frame(far, (100, 100))
    assert not ui.zones_outside_frame([], (100, 100))                              # no zone: nothing to warn about
    assert not ui.zones_outside_frame(far + inside, (100, 100))                    # one zone inside is enough


# --------------------------------------------------------------------------- presets and cached detections

def test_scenario_needs_and_default_stride_follow_the_preset():
    campus, navigation = ui.scenario_needs("campus"), ui.scenario_needs("navigation")
    assert campus["classes"] == [0] and campus["resize_width"] == 640 and campus["weights"].endswith(".pt")
    assert campus["stride"] == 2 and navigation["stride"] == 1                      # navigation overrides the stride
    assert ui.default_stride("navigation") == 1 and ui.default_stride() == 2
    assert ui.scenario_needs("no_such_preset")["classes"] == [0]                    # a broken name does not raise


def full_header(**over):
    """A detections.json header with everything run.py checks."""
    base = {"video": "samples/clip.mp4", "stride": 2, "classes": [0], "orig_size": [1280, 720], "proc_size": [640, 360],
            "fps": 25.0, "duration_s": 30.0, "frames_processed": 375, "n_frames_read": 750, "model": "yolo11n.pt"}
    base.update(over)
    return base


@pytest.mark.parametrize("over, kwargs, expected", [
    ({}, {}, True),
    ({}, {"resize_width": 640, "weights": "yolo11n.pt"}, True),
    ({"proc_size": [960, 540]}, {"resize_width": 640}, False),                      # another resize width
    ({"proc_size": [1280, 720]}, {"resize_width": 0}, True),                        # 0 = keep the original width
    ({"proc_size": [640, 360]}, {"resize_width": 0}, False),
    ({"model": "yolo11s.pt"}, {"weights": "yolo11n.pt"}, False),                    # another detector
    ({"duration_s": 10.0, "n_frames_read": 250}, {"n_frames": 750}, False),         # the cache covers a third of the clip
    ({"duration_s": 10.0, "n_frames_read": 250}, {"max_seconds": 10.0, "n_frames": 750}, True),
    ({}, {"max_seconds": 10.0, "n_frames": 750}, False),                            # the cache covers more than asked
    ({"duration_s": 5.0}, {"max_seconds": 10.0, "n_frames": 750}, False),           # ... or less than asked
    ({"fps": "fast"}, {}, False),                                                   # a header with odd values
])
def test_detections_match_mirrors_the_checks_of_run_py(over, kwargs, expected):
    header = full_header(**over)
    args = {"classes": [0], "stride": 2, "size": (1280, 720)} | kwargs
    assert ui._detections_match(header, Path("samples/clip.mp4"), args["classes"], args["stride"], args["size"],
                                args.get("resize_width"), args.get("weights", ""), args.get("max_seconds", 0.0),
                                args.get("n_frames", 0)) is expected


def test_a_detections_file_kept_by_preserve_detections_is_found_again(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(ui, "CACHE_DIR", tmp_path / "outputs" / "_ui" / "cache")
    own = tmp_path / "outputs" / "clip"
    own.mkdir(parents=True)
    (own / "detections.json").write_text(json.dumps(full_header(classes=[0, 1])))
    note = ui.preserve_detections("outputs/clip")
    assert "kept the old detections" in note and (ui.CACHE_DIR / "clip.json").is_file()
    (own / "detections.json").write_text(json.dumps(full_header(classes=[0])))       # a campus run replaces the file
    found = ui.find_cached_detections(Path("samples/clip.mp4"), "outputs/clip_workplace", [0, 1], 2, (1280, 720))
    assert found == str(ui.CACHE_DIR / "clip.json")                                  # the workplace cache survived
    assert ui.preserve_detections("outputs/none_here") == ""                         # nothing to keep
    assert ui.preserve_detections("outputs/clip", reuse_from=str(own / "detections.json")) == ""   # this run reuses it


def test_yolo_seconds_estimate():
    facts = {"frames": 750, "fps": 25.0}
    assert ui.yolo_seconds(facts, 2) == pytest.approx(750 / 2 / 5.5)
    assert ui.yolo_seconds(facts, 2, max_seconds=10) == pytest.approx(250 / 2 / 5.5)  # only the first 10 s
    assert ui.yolo_seconds({"frames": 0, "fps": 0}, 2) == 0.0


# --------------------------------------------------------------------------- running run.py

def pid_alive(pid: int) -> bool:
    """Is a process with this id still running? (tasklist on Windows: os.kill(pid, 0) means Ctrl+C there)."""
    import subprocess
    if sys.platform.startswith("win"):
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    import os
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_run_pipeline_kills_grandchildren_too():
    """run.py starts ffmpeg, and on Windows the venv python.exe starts the real interpreter: kill the whole tree."""
    parent = ("import subprocess, sys, time\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
              "print('child', child.pid, flush=True)\n"
              "time.sleep(120)\n")
    pids = []

    def stop_when_child_is_known(line):
        if line.startswith("child "):
            pids.append(int(line.split()[1]))
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ui.run_pipeline([sys.executable, "-c", parent], stop_when_child_is_known)
    assert pids
    deadline = time.time() + 15
    while pid_alive(pids[0]) and time.time() < deadline:
        time.sleep(0.2)
    assert not pid_alive(pids[0]), "the grandchild process was left running"


def test_run_pipeline_calls_on_idle_while_the_command_is_silent_and_can_stop_it():
    ticks = []

    def idle():
        ticks.append(time.time())
        if len(ticks) >= 3:
            raise KeyboardInterrupt                                                  # what Streamlit does on a rerun

    started = time.time()
    with pytest.raises(KeyboardInterrupt):
        ui.run_pipeline([sys.executable, "-c", "import time; time.sleep(60)"], on_idle=idle, idle_s=0.1)
    assert len(ticks) == 3 and time.time() - started < 30                           # woke up without any output


def test_run_pipeline_keeps_only_the_last_lines_and_cuts_long_ones():
    script = "for i in range(%d): print('line', i)\nprint('y' * 50000)" % (ui.MAX_LOG_LINES + 500)
    code, lines = ui.run_pipeline([sys.executable, "-c", script])
    assert code == 0 and len(lines) == ui.MAX_LOG_LINES
    assert lines[-1] == "y" * ui.MAX_LINE_CHARS and lines[0].startswith("line ")
    assert len(ui.last_lines(["z" * 5000], 20, max_chars=400)) == 400


# --------------------------------------------------------------------------- remembering the last run

def test_last_run_round_trip_and_rejections(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "LAST_RUN_FILE", tmp_path / "outputs" / "_ui" / "last_run.json")
    assert ui.load_last_run() is None                                                # nothing saved yet
    (tmp_path / "outputs" / "clip").mkdir(parents=True)
    (tmp_path / "outputs" / "clip" / "events.json").write_text("{}")
    info = {"ok": True, "code": 0, "out_dir": "outputs/clip", "cmd": "python run.py", "interpreter": "python",
            "lines": ["a", "b"], "seconds": 3.5, "video": "samples/clip.mp4", "scenario": "workplace"}
    ui.save_last_run(info)
    loaded = ui.load_last_run()
    assert loaded["out_dir"] == "outputs/clip" and loaded["scenario"] == "workplace" and loaded["lines"] == ["a", "b"]
    assert loaded["ok"] is True and loaded["seconds"] == 3.5 and loaded["when"]
    (tmp_path / "outputs" / "clip" / "events.json").unlink()
    assert ui.load_last_run() is None                                                # the results are gone
    ui.save_last_run(dict(info, ok=False, code=2, lines=["error: boom"]))
    failed = ui.load_last_run()
    assert failed["ok"] is False and failed["code"] == 2 and failed["lines"] == ["error: boom"]   # a failure is kept too
    ui.save_last_run(dict(info, out_dir="../../elsewhere"))
    assert ui.load_last_run() is None                                                # never outside the project
    ui.LAST_RUN_FILE.write_text("{not json", encoding="utf-8")
    assert ui.load_last_run() is None
    ui.LAST_RUN_FILE.write_text("[1, 2]", encoding="utf-8")
    assert ui.load_last_run() is None


def test_save_last_run_never_raises(tmp_path, monkeypatch):
    blocker = tmp_path / "file_not_folder"
    blocker.write_text("x")
    monkeypatch.setattr(ui, "LAST_RUN_FILE", blocker / "last_run.json")             # the parent is a file: cannot write
    ui.save_last_run({"ok": True})                                                   # must not raise


# --------------------------------------------------------------------------- sanitizing results

def test_sanitize_results_repairs_types_and_keeps_good_values():
    with pytest.raises(ValueError):
        ui.sanitize_results([])
    with pytest.raises(ValueError):
        ui.sanitize_results("text")
    clean = ui.sanitize_results({"events": ["x", {"event_id": 1, "confidence": "abc", "duration_s": "1.5",
                                                  "confidence_parts": {"margin": "0.5", "duration": None},
                                                  "baseline": "oops", "pose": {"median_lean_deg": "x"}, "zone": 7},
                                            None],
                                 "incidents": ["a", {"event_ids": ["1", "z", None, 2.0], "severity": "high"}],
                                 "near_misses": [1, {"behavior": "running"}], "unusual_tracks": "no",
                                 "meta": {"duration_s": "12", "processing_fps": "fast", "n_tracks": "4",
                                          "config": {"scenario": "bad", "model": {"classes": ["0", "x", 2]}}},
                                 "highlight_reel": 5})
    (event,) = clean["events"]
    assert event["confidence"] == 0.0 and event["duration_s"] == 1.5 and event["confidence_parts"] == {"margin": 0.5}
    assert event["baseline"] == {} and event["pose"]["median_lean_deg"] is None and event["zone"] == "7"
    assert clean["incidents"] == [{"event_ids": [1, 2], "severity": "high"}]
    assert clean["near_misses"] == [{"behavior": "running"}] and clean["unusual_tracks"] == []
    assert clean["meta"]["duration_s"] == 12.0 and clean["meta"]["processing_fps"] is None
    assert clean["meta"]["n_tracks"] == 4 and clean["meta"]["config"]["model"]["classes"] == [0, 2]
    assert clean["highlight_reel"] is None
    metrics = ui.result_metrics(clean)
    assert metrics["incidents"] == 1 and metrics["events"] == 1 and metrics["duration"] == "00:12"


def test_sanitize_results_leaves_a_real_events_json_working():
    path = ROOT / "examples" / "workplace_near-miss_fall" / "events.json"
    if not path.exists():
        pytest.skip("examples/workplace_near-miss_fall is missing")
    raw = utils.read_json(path)
    clean = ui.sanitize_results(raw)
    assert ui.result_metrics(clean) == ui.result_metrics(raw)                        # the tiles do not change
    assert len(clean["events"]) == len(raw["events"]) and len(clean["incidents"]) == len(raw.get("incidents", []))
    assert [e["confidence"] for e in clean["events"]] == [e["confidence"] for e in raw["events"]]


def test_helpers_survive_odd_values_without_sanitizing():
    assert ui.incident_count({"events": ["x", {"event_id": 1}], "incidents": [5, {"event_ids": 3}]}) == 2
    assert ui.confidence_line({"confidence": "abc", "confidence_parts": {"margin": "x"}}, ui.confidence_weights({})) \
        == "confidence 0.00"
    assert ui.pose_text({"verified": True, "pose": {"median_lean_deg": "x", "max_leg_spread_bh": None}}) == \
        ("green", "Pose check agrees")
    assert ui.result_metrics({"meta": "text", "events": "text"})["events"] == 0
    assert ui.confidence_weights({"meta": {"config": {"events": {"confidence_weights": {"margin": "x"}}}}})["margin"] == 0.4
    assert ui.tracked_label("not a dict") == "People"


# --------------------------------------------------------------------------- the UI agrees with run.py about --reuse

REAL_DETECTIONS = ROOT / "outputs" / "synthetic_safety" / "detections.json"


@pytest.mark.skipif(not REAL_DETECTIONS.exists(), reason="outputs/synthetic_safety/detections.json is missing")
@pytest.mark.parametrize("scenario, max_seconds, stride", [
    ("workplace", 0, 2), ("workplace", 5, 2), ("workplace", 30, 2), ("workplace", 12.5, 2), ("workplace", 40, 2),
    ("workplace", 0, 1), ("workplace", 0, 3), ("campus", 0, 2), ("traffic", 0, 2), ("navigation", 0, 1),
])
def test_ui_verdict_on_cached_detections_equals_run_py(tmp_path, scenario, max_seconds, stride):
    """The page promises 'YOLO will be skipped' only when run.py itself would reuse the file (same rule, same answer)."""
    import run                                                                       # run.py: light to import (YOLO is lazy)
    folder = tmp_path / "out"
    folder.mkdir()
    shutil.copyfile(REAL_DETECTIONS, folder / "detections.json")
    header = utils.read_json(REAL_DETECTIONS)
    video = ROOT / "samples" / "synthetic_safety.mp4"
    n_frames = 750
    cfg = utils.load_config(None, scenario)
    cfg["video"]["max_seconds"] = float(max_seconds)
    cfg["video"]["frame_stride"] = int(stride)
    reused, _reason = run.load_reusable_detections(folder / "detections.json", video, cfg, n_frames)
    need = ui.scenario_needs(scenario)
    mine = ui._detections_match(header, video, need["classes"], stride, (1280, 720), need["resize_width"],
                                need["weights"], float(max_seconds) or need["max_seconds"], n_frames)
    assert mine == (reused is not None), (scenario, max_seconds, stride, _reason)


# --------------------------------------------------------------------------- video facts

def test_video_facts_use_the_real_frame_size_when_the_header_disagrees(tmp_path, monkeypatch):
    """A rotated phone video can report one size in its header and decode to another: zones must use the real one."""
    import cv2
    video = make_video(tmp_path / "tiny.mp4", frames=5, fps=10.0, size=(64, 48))
    real_capture = cv2.VideoCapture

    class LyingCapture:
        """A VideoCapture whose width and height properties are swapped (as in a video with a rotation flag)."""

        def __init__(self, path):
            self.cap = real_capture(path)

        def get(self, prop):
            swapped = {cv2.CAP_PROP_FRAME_WIDTH: cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT: cv2.CAP_PROP_FRAME_WIDTH}
            return self.cap.get(swapped.get(prop, prop))

        def __getattr__(self, name):
            return getattr(self.cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", LyingCapture)
    facts = ui.video_facts(video)
    assert (facts["width"], facts["height"]) == (64, 48)
