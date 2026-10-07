"""Tests for the scenario presets (scenarios/*.yaml), the --scenario / --list-scenarios plumbing in run.py,
and the class id that tracker.py now writes into every detection row. No video, YOLO or GPU needed."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # also works without conftest.py
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402
import tracker  # noqa: E402
import utils  # noqa: E402

PRESETS = ["agriculture", "campus", "disaster", "elderly", "livestock", "navigation", "pool", "public_safety", "traffic", "workplace"]


def raw_preset(name):
    """The preset file exactly as written (a partial config), before merging over config.yaml."""
    with open(utils.SCENARIOS_DIR / f"{name}.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def unknown_keys(preset, default, path=""):
    """Keys in a preset that do not exist in the default config (typos would otherwise be ignored silently)."""
    bad = []
    for key, value in preset.items():
        if key == "behavior_names":                       # free-form display names, checked separately
            continue
        if key not in default:
            bad.append(f"{path}{key}")
        elif isinstance(value, dict) and isinstance(default[key], dict):
            bad += unknown_keys(value, default[key], f"{path}{key}.")
    return bad


# --------------------------------------------------------------------------- the five presets exist and load

def test_list_scenarios_returns_all_presets():
    assert utils.list_scenarios() == PRESETS


@pytest.mark.parametrize("name", PRESETS)
def test_preset_loads_and_has_a_title(name):
    cfg = utils.load_config(None, name)
    assert cfg["scenario"]["name"] == name
    assert isinstance(cfg["scenario"]["title"], str) and cfg["scenario"]["title"].strip()
    assert cfg["scenario"]["entity_word"] and cfg["scenario"]["unit_word"]


@pytest.mark.parametrize("name", PRESETS)
def test_preset_only_uses_keys_that_exist_in_the_default_config(name):
    assert unknown_keys(raw_preset(name), utils.DEFAULT_CONFIG) == []


@pytest.mark.parametrize("name", PRESETS)
def test_preset_behaviour_keys_are_valid(name):
    cfg = utils.load_config(None, name)
    names = cfg["scenario"]["behavior_names"]
    assert set(names) <= set(utils.BEHAVIORS)
    assert all(isinstance(v, str) and v.strip() for v in names.values())
    for key in utils.BEHAVIORS:                           # every rule keeps its section and can be named
        assert key in cfg["behaviors"]
        assert utils.behavior_name(cfg, key)


@pytest.mark.parametrize("name", PRESETS)
def test_preset_thresholds_are_sane(name):
    b = utils.load_config(None, name)["behaviors"]
    assert 0 < b["running"]["end_speed_bh_s"] < b["running"]["start_speed_bh_s"]     # hysteresis needs end < start
    assert b["running"]["min_duration_s"] >= 0
    assert b["loitering"]["radius_bh"] > 0 and b["loitering"]["min_duration_s"] > 0
    assert b["zone_intrusion"]["min_duration_s"] >= 0
    assert b["fall"]["down_min_aspect"] > b["fall"]["upright_max_aspect"] > 0
    assert b["fall"]["min_down_s"] > 0 and b["fall"]["transition_s"] > 0
    assert b["crowding"]["min_count"] >= 2 and b["crowding"]["min_duration_s"] > 0
    nm = b["near_miss"]
    assert nm["near_distance_bh"] > 0 and 0 < nm["contact_bh"] < nm["near_distance_bh"]
    assert nm["min_rel_speed_bh_s"] >= 0 and nm["min_closing_speed_bh_s"] >= 0 and nm["ttc_s"] > 0
    assert 0 < nm["max_scale_diff"] <= 1


@pytest.mark.parametrize("name", PRESETS)
def test_preset_classes_are_known_coco_ids(name):
    classes = utils.load_config(None, name)["model"]["classes"]
    assert classes and len(set(classes)) == len(classes)
    assert all(isinstance(c, int) and c in utils.CLASS_NAMES for c in classes)


@pytest.mark.parametrize("name", PRESETS)
def test_enabled_near_miss_can_actually_happen_with_the_tracked_classes(name):
    cfg = utils.load_config(None, name)
    nm, classes = cfg["behaviors"]["near_miss"], set(cfg["model"]["classes"])
    if nm["enabled"]:
        assert classes & set(nm["vulnerable_classes"])    # someone who can get hurt is tracked
        assert classes & set(nm["other_classes"])         # ...and something they can nearly hit


def test_campus_is_the_default_config():
    assert utils.load_config(None, "campus") == utils.load_config(None)


def test_unknown_scenario_lists_the_available_ones():
    with pytest.raises(FileNotFoundError) as err:
        utils.load_config(None, "no_such_preset")
    assert "traffic" in str(err.value) and "livestock" in str(err.value)


# --------------------------------------------------------------------------- what each preset is for

def test_workplace_and_traffic_make_near_miss_the_headline_with_vehicles_tracked():
    for name in ("workplace", "traffic"):
        cfg = utils.load_config(None, name)
        assert cfg["behaviors"]["near_miss"]["enabled"]
        assert cfg["behaviors"]["near_miss"]["vulnerable_classes"] == [0]
        assert {2, 7} <= set(cfg["model"]["classes"])     # car and truck must be tracked, or nothing can be hit
    assert utils.load_config(None, "workplace")["behaviors"]["loitering"]["min_duration_s"] == 20
    assert utils.load_config(None, "workplace")["scenario"]["title"] == "Industrial / workplace safety"


def test_elderly_preset_is_about_falls_and_stillness():
    cfg = utils.load_config(None, "elderly")
    b = cfg["behaviors"]
    assert b["fall"]["enabled"] and not b["running"]["enabled"]
    assert not b["crowding"]["enabled"] and not b["near_miss"]["enabled"]
    assert (b["loitering"]["min_duration_s"], b["loitering"]["radius_bh"]) == (60, 0.3)
    assert utils.behavior_name(cfg, "loitering") == "No movement"
    assert utils.behavior_name(cfg, "zone_intrusion") == "Left through the exit"
    assert cfg["scenario"]["title"] == "Elderly care"


def test_livestock_preset_tracks_animals_and_turns_the_person_down_rule_off():
    cfg = utils.load_config(None, "livestock")
    assert cfg["model"]["classes"] == [17, 18, 19, 16]
    assert not cfg["behaviors"]["fall"]["enabled"] and not cfg["behaviors"]["near_miss"]["enabled"]
    assert (cfg["behaviors"]["running"]["start_speed_bh_s"], cfg["behaviors"]["running"]["end_speed_bh_s"]) == (2.5, 2.0)
    assert cfg["behaviors"]["crowding"]["min_count"] == 8
    assert cfg["scenario"]["entity_word"] == "Animal" and utils.unit_name(cfg) == "body-lengths"
    assert utils.behavior_name(cfg, "running") == "Stampede / distress"
    assert utils.behavior_name(cfg, "crowding") == "Bunching"
    assert utils.entity_name(cfg, 4, 18) == "Sheep #4"      # several classes -> the class name is used


def test_traffic_preset_wording_and_thresholds():
    cfg = utils.load_config(None, "traffic")
    b = cfg["behaviors"]
    assert cfg["model"]["classes"] == [0, 1, 2, 3, 5, 7]
    assert b["near_miss"]["other_classes"] == [1, 2, 3, 5, 7]
    assert (b["running"]["start_speed_bh_s"], b["running"]["end_speed_bh_s"]) == (3.0, 2.4)
    assert (b["loitering"]["min_duration_s"], b["loitering"]["radius_bh"]) == (30, 0.3)
    assert not b["fall"]["enabled"] and b["crowding"]["min_count"] == 8
    assert utils.behavior_name(cfg, "running") == "Speeding"
    assert utils.behavior_name(cfg, "zone_intrusion") == "Entered restricted lane"
    assert utils.behavior_name(cfg, "crowding") == "Congestion"
    assert utils.entity_name(cfg, 7, 2) == "Car #7" and utils.entity_name(cfg, 3, 0) == "Person #3"


# --------------------------------------------------------------------------- run.py plumbing

def test_default_out_dir_gets_a_suffix_for_every_preset_except_campus():
    clip = Path("samples/street.mp4")
    assert run.default_out_dir(clip, None) == Path("outputs/street")
    assert run.default_out_dir(clip, "campus") == Path("outputs/street")
    assert run.default_out_dir(clip, "traffic") == Path("outputs/street_traffic")


def test_list_scenarios_flag_prints_every_preset_and_needs_no_video(capsys):
    assert run.main(["--list-scenarios"]) == 0
    out = capsys.readouterr().out
    for name in PRESETS:
        assert name in out
    assert "Traffic and pedestrians" in out and out.isascii()


def test_video_is_still_required_without_list_scenarios():
    with pytest.raises(SystemExit):
        run.main([])


def test_privacy_flag_sets_the_config():
    args = run.build_parser().parse_args(["--video", "x.mp4", "--privacy"])
    assert run.apply_overrides(utils.load_config(), args)["privacy"]["enabled"] is True
    args = run.build_parser().parse_args(["--video", "x.mp4"])
    assert run.apply_overrides(utils.load_config(), args)["privacy"]["enabled"] is False


def test_no_video_flag_also_skips_the_highlight_reel():
    args = run.build_parser().parse_args(["--video", "x.mp4", "--no-video"])
    cfg = run.apply_overrides(utils.load_config(), args)
    assert cfg["output"]["annotated_video"] is False and cfg["highlights"]["enabled"] is False


def test_events_table_has_a_severity_column_and_uses_scenario_wording():
    cfg = utils.load_config(None, "traffic")
    events = [{"event_id": 1, "entity_id": 7, "behavior": "running", "start": "00:05", "end": "00:09",
               "confidence": 0.8, "evidence": "Moved fast.", "severity": "medium"},
              {"event_id": 2, "entity_id": None, "behavior": "crowding", "start": "00:10", "end": "00:20",
               "confidence": 0.7, "evidence": "9 objects.", "severity": None, "entity_name": "Group of 9",
               "behavior_name": "Congestion"}]
    table = run.format_events_table(events, 100, cfg)
    assert "severity" in table.splitlines()[0]
    assert "Speeding" in table and "medium" in table          # built from cfg when the event has no behavior_name
    assert "Group of 9" in table and "Congestion" in table


# --------------------------------------------------------------------------- --reuse must notice a class change

def fake_detections(tmp_path, classes=None):
    res = {"video": "samples/clip.mp4", "fps": 25.0, "stride": 2, "orig_size": [1280, 720], "proc_size": [640, 360],
           "n_frames_read": 100, "duration_s": 4.0, "frames_processed": 50, "device": "cpu", "model": "yolo11n.pt",
           "detections": []}
    if classes is not None:
        res["classes"] = classes
    path = tmp_path / "detections.json"
    utils.write_json(path, res)
    return path


def test_reuse_compares_the_tracked_classes(tmp_path):
    video = Path("samples/clip.mp4")
    cfg_person, cfg_traffic = utils.load_config(), utils.load_config(None, "traffic")
    old_file = fake_detections(tmp_path)                       # written before presets existed: no 'classes' key
    assert run.load_reusable_detections(old_file, video, cfg_person)[0] is not None
    res, why = run.load_reusable_detections(old_file, video, cfg_traffic)
    assert res is None and "classes" in why
    traffic_file = fake_detections(tmp_path, [7, 0, 1, 2, 3, 5])      # order does not matter
    assert run.load_reusable_detections(traffic_file, video, cfg_traffic)[0] is not None
    assert run.load_reusable_detections(traffic_file, video, cfg_person)[0] is None


# --------------------------------------------------------------------------- tracker rows carry the class id

class FakeBoxes:
    def __init__(self, with_cls=True):
        self.id = np.array([4.0, 9.0])
        self.xyxy = np.array([[10.0, 20.0, 50.0, 120.0], [200.0, 80.0, 400.0, 200.0]])
        self.conf = np.array([0.91, 0.55])
        if with_cls:
            self.cls = np.array([0.0, 2.0])


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_tracked_rows_have_the_class_id_as_ninth_element():
    rows = tracker._tracked_rows(FakeResult(FakeBoxes()), frame_idx=6, t=0.24)
    assert [len(r) for r in rows] == [9, 9]
    assert [r[8] for r in rows] == [0, 2]
    assert rows[1][:3] == [6, 0.24, 9] and rows[1][7] == 0.55


def test_tracked_rows_default_to_person_when_the_model_gives_no_classes():
    rows = tracker._tracked_rows(FakeResult(FakeBoxes(with_cls=False)), frame_idx=0, t=0.0)
    assert [r[8] for r in rows] == [0, 0]


def test_tracked_rows_skip_boxes_without_a_track_id():
    boxes = FakeBoxes()
    boxes.id = None
    assert tracker._tracked_rows(FakeResult(boxes), 0, 0.0) == []
