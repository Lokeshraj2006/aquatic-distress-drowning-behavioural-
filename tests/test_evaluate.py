"""Tests for evaluate.py: label parsing, interval matching, tolerance, FP / FN, 'none' clips, CLI."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # also works without conftest.py
    sys.path.insert(0, str(ROOT))

import evaluate as ev  # noqa: E402
import utils  # noqa: E402


# --------------------------------------------------------------------------- tiny builders

def lab(behavior, start, end, clip="a.mp4", desc="person"):
    return {"clip": clip, "entity_description": desc, "behavior": behavior, "start_s": start, "end_s": end}


def pred(behavior, start, end, event_id=1, entity_id=1):
    return {"event_id": event_id, "entity_id": entity_id, "behavior": behavior, "start_s": start, "end_s": end}


def score(labels, preds_by_clip, tol=2.0):
    return ev.evaluate(labels, preds_by_clip, tol)


# --------------------------------------------------------------------------- parsing

def test_parse_time_formats():
    assert ev.parse_time("12.5") == 12.5
    assert ev.parse_time("0:12") == 12.0
    assert ev.parse_time("1:05") == 65.0
    assert ev.parse_time("0:01:05.5") == 65.5
    assert ev.parse_time(" ") is None and ev.parse_time(None) is None
    assert ev.parse_time(7) == 7.0


@pytest.mark.parametrize("bad", ["abc", "1:2:3:4", "-3", "nan"])
def test_parse_time_rejects_garbage(bad):
    with pytest.raises(ValueError):
        ev.parse_time(bad)


def test_clip_key_strips_folders_and_video_extension():
    assert ev.clip_key("samples/x.mp4") == "x"
    assert ev.clip_key("X.MP4") == "X"
    assert ev.clip_key("x") == "x"
    assert ev.clip_key("clip.v2.mp4") == "clip.v2"
    assert ev.clip_key("clip.v2") == "clip.v2"            # '.v2' is not a video extension


def test_normalise_behavior_aliases():
    assert ev.normalise_behavior("Zone Intrusion") == "zone_intrusion"
    assert ev.normalise_behavior("run") == "running"
    assert ev.normalise_behavior("NONE") == "none"
    with pytest.raises(ValueError):
        ev.normalise_behavior("fighting")


def write_csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_labels_basic_with_comments_bom_and_none(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_bytes(
        "﻿clip,entity_description,behavior,start_s,end_s\n"
        "# a comment line\n"
        "\n"
        'a.mp4,"man, red jacket",loitering,12,0:34\n'
        "b.mp4,,none,,\n".encode("utf-8"))
    rows = ev.load_labels(p)
    assert len(rows) == 2
    assert rows[0]["entity_description"] == "man, red jacket"
    assert (rows[0]["start_s"], rows[0]["end_s"]) == (12.0, 34.0)
    assert rows[1]["behavior"] == "none" and rows[1]["start_s"] is None


@pytest.mark.parametrize("body, needle", [
    ("a.mp4,x,fighting,1,2\n", "unknown behavior"),
    ("a.mp4,x,running,5,2\n", "before start_s"),
    ("a.mp4,x,running,,\n", "required"),
    ("a.mp4,x,running,1,zz\n", "not a time"),
])
def test_load_labels_errors_name_the_line(tmp_path, body, needle):
    p = write_csv(tmp_path / "l.csv", "clip,entity_description,behavior,start_s,end_s\n" + body)
    with pytest.raises(ev.LabelError) as exc:
        ev.load_labels(p)
    assert needle in str(exc.value) and "line 2" in str(exc.value)


def test_load_labels_missing_header_columns(tmp_path):
    p = write_csv(tmp_path / "l.csv", "clip,behavior\na.mp4,running\n")
    with pytest.raises(ev.LabelError):
        ev.load_labels(p)


# --------------------------------------------------------------------------- matching and tolerance

def test_exact_match_is_one_tp():
    r = score([lab("running", 10, 15)], {"a": [pred("running", 10, 15)]})
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (1, 0, 0)
    assert r["overall"]["precision"] == 1.0 and r["overall"]["recall"] == 1.0 and r["overall"]["f1"] == 1.0
    assert r["per_behavior"]["running"]["tp"] == 1


def test_tolerance_widens_the_label_on_both_sides():
    labels = [lab("loitering", 10, 12)]
    late = {"a": [pred("loitering", 13.5, 15)]}           # starts 1.5 s after the label ends
    early = {"a": [pred("loitering", 5, 8.5)]}            # ends 1.5 s before the label starts
    for preds in (late, early):
        assert score(labels, preds, tol=2.0)["overall"]["tp"] == 1
        miss = score(labels, preds, tol=1.0)["overall"]
        assert (miss["tp"], miss["fp"], miss["fn"]) == (0, 1, 1)


def test_touching_after_widening_counts_as_overlap():
    r = score([lab("running", 10, 12)], {"a": [pred("running", 14, 16)]}, tol=2.0)   # label becomes [8, 14]
    assert r["overall"]["tp"] == 1


def test_zero_tolerance_requires_real_overlap():
    r = score([lab("running", 10, 12)], {"a": [pred("running", 12.5, 14)]}, tol=0.0)
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (0, 1, 1)


def test_false_positive_and_false_negative_counts():
    labels = [lab("running", 10, 12), lab("running", 40, 45)]
    preds = {"a": [pred("running", 10, 12, 1), pred("running", 70, 75, 2)]}
    r = score(labels, preds)
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (1, 1, 1)
    assert r["overall"]["precision"] == 0.5 and r["overall"]["recall"] == 0.5 and r["overall"]["f1"] == 0.5
    assert r["missed"][0]["start_s"] == 40 and r["false_positives"][0]["event_id"] == 2


def test_behaviour_must_agree():
    r = score([lab("loitering", 10, 25)], {"a": [pred("running", 10, 25)]})
    assert r["per_behavior"]["loitering"]["fn"] == 1
    assert r["per_behavior"]["running"]["fp"] == 1
    assert r["overall"]["tp"] == 0


def test_greedy_one_to_one_takes_largest_overlap_first():
    # Two predictions overlap one label; the one with the bigger overlap wins, the other is a false positive.
    labels = [lab("zone_intrusion", 20, 30)]
    preds = {"a": [pred("zone_intrusion", 19, 21, 1), pred("zone_intrusion", 22, 29, 2)]}
    r = score(labels, preds, tol=0.0)
    assert r["overall"]["tp"] == 1 and r["overall"]["fp"] == 1
    assert r["matches"][0]["event_id"] == 2 and r["false_positives"][0]["event_id"] == 1


def test_one_prediction_cannot_match_two_labels():
    labels = [lab("loitering", 10, 20), lab("loitering", 21, 30)]
    r = score(labels, {"a": [pred("loitering", 12, 29)]}, tol=2.0)
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (1, 0, 1)


def test_match_intervals_returns_pairs_and_leftovers():
    labels = [lab("running", 0, 5), lab("running", 50, 55)]
    preds = [pred("running", 52, 56), pred("running", 100, 101)]
    pairs, left_labels, left_preds = ev.match_intervals(labels, preds, tolerance=1.0)
    assert [(i, j) for i, j, _ in pairs] == [(1, 0)]
    assert left_labels == [0] and left_preds == [1]


def test_mean_start_and_end_errors_are_absolute():
    labels = [lab("running", 10, 20, desc="a"), lab("running", 50, 60, desc="b")]
    preds = {"a": [pred("running", 11, 19, 1), pred("running", 48.5, 62, 2)]}
    r = score(labels, preds)
    run = r["per_behavior"]["running"]
    assert run["tp"] == 2
    assert run["mean_abs_start_error_s"] == pytest.approx((1.0 + 1.5) / 2)
    assert run["mean_abs_end_error_s"] == pytest.approx((1.0 + 2.0) / 2)
    signed = {m["event_id"]: (m["start_error_s"], m["end_error_s"]) for m in r["matches"]}
    assert signed[1] == (1.0, -1.0) and signed[2] == (-1.5, 2.0)


def test_prf_edge_cases_are_none_not_zero_division():
    assert ev.prf(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}
    only_misses = ev.prf(0, 0, 3)
    assert only_misses["recall"] == 0.0 and only_misses["precision"] is None and only_misses["f1"] == 0.0
    assert ev.prf(2, 1, 1)["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_no_labels_and_no_predictions_is_all_undefined():
    r = score([lab("none", None, None)], {"a": []})
    assert r["overall"]["precision"] is None and r["overall"]["recall"] is None
    assert r["overall"]["mean_abs_start_error_s"] is None


# --------------------------------------------------------------------------- 'none' clips

def test_none_clip_with_events_counts_false_alarms():
    labels = [lab("none", None, None, clip="quiet.mp4")]
    preds = {"quiet": [pred("running", 3, 6, 1), pred("loitering", 8, 30, 2)]}
    r = score(labels, preds)
    assert r["none_clips"]["n_clips"] == 1
    assert r["none_clips"]["false_alarms"] == 2 and r["none_clips"]["clips_with_false_alarms"] == 1
    assert r["overall"]["fp"] == 2 and r["overall"]["tp"] == 0
    assert {d["behavior"] for d in r["none_clips"]["details"]} == {"running", "loitering"}


def test_none_clip_without_events_is_clean():
    r = score([lab("none", None, None, clip="quiet.mp4")], {"quiet": []})
    assert r["none_clips"] == {"n_clips": 1, "clips_with_false_alarms": 0, "false_alarms": 0, "details": []}
    assert r["overall"]["fp"] == 0 and r["overall"]["fn"] == 0


def test_none_row_next_to_real_labels_is_ignored_with_warning():
    labels = [lab("none", None, None), lab("running", 5, 8)]
    r = score(labels, {"a": [pred("running", 5, 8)]})
    assert r["warnings"] and r["none_clips"]["n_clips"] == 0 and r["overall"]["tp"] == 1


def test_false_alarms_only_counted_on_none_clips_in_the_none_section():
    labels = [lab("none", None, None, clip="quiet.mp4"), lab("running", 5, 8, clip="busy.mp4")]
    preds = {"quiet": [], "busy": [pred("running", 5, 8), pred("loitering", 20, 40, 2)]}
    r = score(labels, preds)
    assert r["none_clips"]["false_alarms"] == 0            # the extra loitering is on a labelled clip
    assert r["overall"]["fp"] == 1


# --------------------------------------------------------------------------- multi-clip + files + CLI

def write_events(outputs, stem, events, video=None):
    folder = outputs / stem
    folder.mkdir(parents=True)
    utils.write_json(folder / "events.json",
                     {"video": video or f"samples/{stem}.mp4", "meta": {}, "events": events,
                      "near_misses": [], "unusual_tracks": [], "baseline": {}, "tracks": []})


def make_dataset(tmp_path):
    labels = write_csv(tmp_path / "labels.csv",
                       "clip,entity_description,behavior,start_s,end_s\n"
                       "corridor.mp4,man in red,loitering,12,34\n"
                       "corridor.mp4,woman with bag,running,0:41,0:48\n"
                       "quiet.mp4,nobody,none,,\n"
                       "notrun.mp4,someone,running,1,3\n")
    outputs = tmp_path / "outputs"
    write_events(outputs, "corridor", [
        {"event_id": 1, "entity_id": 1, "behavior": "loitering", "start_s": 13.0, "end_s": 33.0},
        {"event_id": 2, "entity_id": 2, "behavior": "running", "start_s": 41.5, "end_s": 47.0},
        {"event_id": 3, "entity_id": 4, "behavior": "zone_intrusion", "start_s": 60.0, "end_s": 64.0}])
    write_events(outputs, "quiet", [
        {"event_id": 1, "entity_id": 7, "behavior": "running", "start_s": 5.0, "end_s": 7.0}])
    return labels, outputs


def test_evaluate_many_clips_skips_clip_without_events_json(tmp_path):
    labels_path, outputs = make_dataset(tmp_path)
    labels = ev.load_labels(labels_path)

    class A:                       # minimal stand-in for argparse's namespace
        events = None
        clip = None
    A.outputs = str(outputs)
    A.labels = str(labels_path)
    preds = ev.collect_predictions(labels, A)
    assert set(preds) == {"corridor", "quiet"}             # 'notrun' has no events.json

    r = ev.evaluate(labels, preds, 2.0)
    assert r["clips_evaluated"] == 2
    assert r["clips_skipped_no_events_json"] == ["notrun.mp4"]
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (2, 2, 0)   # zone + quiet running are FPs
    assert r["none_clips"]["false_alarms"] == 1
    assert r["per_behavior"]["zone_intrusion"]["fp"] == 1
    text = ev.format_report(r, "labels.csv")
    assert "OVERALL" in text and "SKIPPED" in text and "notrun.mp4" in text


def test_cli_outputs_mode_writes_reports(tmp_path, capsys):
    labels, outputs = make_dataset(tmp_path)
    code = ev.main(["--labels", str(labels), "--outputs", str(outputs)])
    out = capsys.readouterr().out
    assert code == 0 and "OVERALL" in out
    assert (tmp_path / "eval_report.txt").exists()
    data = utils.read_json(tmp_path / "eval_report.json")
    assert data["overall"]["tp"] == 2 and data["tolerance_s"] == 2.0
    assert out.isascii()                                   # safe for Windows consoles


def test_cli_events_mode_infers_clip_and_honours_report_dir_and_tolerance(tmp_path):
    labels, outputs = make_dataset(tmp_path)
    report_dir = tmp_path / "reports"
    code = ev.main(["--labels", str(labels), "--events", str(outputs / "corridor" / "events.json"),
                    "--tolerance", "0", "--report-dir", str(report_dir)])
    assert code == 0
    data = utils.read_json(report_dir / "eval_report.json")
    assert data["tolerance_s"] == 0.0
    assert data["clips_evaluated"] == 1 and data["overall"]["tp"] == 2


def test_cli_events_mode_with_explicit_clip(tmp_path):
    labels, outputs = make_dataset(tmp_path)
    code = ev.main(["--labels", str(labels), "--events", str(outputs / "quiet" / "events.json"),
                    "--clip", "quiet.mp4", "--report-dir", str(tmp_path / "r")])
    assert code == 0
    data = utils.read_json(tmp_path / "r" / "eval_report.json")
    assert data["none_clips"]["false_alarms"] == 1


def test_cli_bad_inputs_return_error_code(tmp_path, capsys):
    assert ev.main(["--labels", str(tmp_path / "nope.csv"), "--outputs", str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err
    labels, outputs = make_dataset(tmp_path)
    # events file for a clip that has no labels
    write_events(outputs, "stranger", [])
    assert ev.main(["--labels", str(labels), "--events", str(outputs / "stranger" / "events.json")]) == 2


def test_cli_nothing_to_score_exits_nonzero(tmp_path):
    labels = write_csv(tmp_path / "labels.csv", "clip,entity_description,behavior,start_s,end_s\nx.mp4,p,running,1,2\n")
    empty = tmp_path / "outputs"
    empty.mkdir()
    assert ev.main(["--labels", str(labels), "--outputs", str(empty)]) == 1


def test_template_labels_file_parses():
    rows = ev.load_labels(ROOT / "labels_template.csv")
    assert len(rows) == 3 and {r["behavior"] for r in rows} <= set(utils.BEHAVIORS) | {"none"}


# --------------------------------------------------------------------------- the three behaviours added in round 2

def test_all_six_behaviours_and_aliases_are_accepted():
    for key in utils.BEHAVIORS:
        assert ev.normalise_behavior(key) == key
    assert ev.normalise_behavior("Near miss") == "near_miss"
    assert ev.normalise_behavior("near-miss") == "near_miss"
    assert ev.normalise_behavior("NearMiss") == "near_miss"
    assert ev.normalise_behavior("fell") == "fall"
    assert ev.normalise_behavior("crowd") == "crowding"


def test_near_miss_fall_and_crowding_are_scored_like_the_others():
    labels = [lab("near_miss", 10, 12), lab("fall", 20, 30), lab("crowding", 40, 60)]
    preds = [pred("near_miss", 10.5, 11.0, 1), pred("fall", 21, 29, 2), pred("crowding", 80, 90, 3, entity_id=None)]
    r = score(labels, {"a": preds})
    assert (r["per_behavior"]["near_miss"]["tp"], r["per_behavior"]["fall"]["tp"]) == (1, 1)
    assert r["per_behavior"]["crowding"]["fp"] == 1 and r["per_behavior"]["crowding"]["fn"] == 1
    text = ev.format_report(r, "labels.csv")
    assert "near_miss" in text and "a group" in text            # a crowding event has no single entity id


def test_report_hides_behaviours_with_nothing_to_score():
    r = score([lab("running", 1, 3)], {"a": [pred("running", 1, 3)]})
    text = ev.format_report(r, "labels.csv")
    assert "running" in text
    assert "no labels and no events for:" in text and "near_miss" in text.split("no labels and no events for:")[1]
    assert set(r["per_behavior"]) == set(utils.BEHAVIORS)       # the json still lists all six


def test_outputs_folder_with_a_scenario_suffix_is_found(tmp_path):
    outputs = tmp_path / "outputs"
    write_events(outputs, "street_traffic", [{"event_id": 1, "entity_id": 3, "behavior": "near_miss",
                                              "start_s": 4.0, "end_s": 5.0}])
    assert ev.find_events_file(outputs, "street") == outputs / "street_traffic" / "events.json"
    assert ev.find_events_file(outputs, "street_traffic") == outputs / "street_traffic" / "events.json"
    assert ev.find_events_file(outputs, "other") is None
    write_events(outputs, "street", [])                          # the plain (campus) folder wins when both exist
    assert ev.find_events_file(outputs, "street") == outputs / "street" / "events.json"
