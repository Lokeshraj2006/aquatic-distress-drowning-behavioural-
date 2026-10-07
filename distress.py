"""Stage 3 of the pool pipeline: is this behaviour actually BECOMING dangerous? (temporal reasoning)

Input: the stage-2 behaviour samples of every swimmer (pool_behaviour.py). Output: events in the
team's stage-3 data contract, plus a risk curve and a state timeline per swimmer for the dashboard:

    {"person_id": 3, "event": "high_risk_aquatic_distress", "start_time": 79.2, "alert_time": 84.2,
     "risk_score": 0.92, "location": "Deep End",
     "evidence": ["vertical_posture", "low_displacement", "repeated_arm_motion", "transition_from_swimming"]}

How it decides (no single frame and no single signal decides anything):

1. Every sample gets a RISK = sum of the warning signs present (weights in config pool.weights, max 1.0):
   vertical posture, no forward progress, repeated arm motion, unstable / submerged head, and
   "transition" (the same swimmer was swimming normally within the last pool.lookback_s seconds).
2. Look-alikes are damped: at the wall with calm arms = resting (risk x pool.resting_factor);
   upright with calm arms and a steady head = treading water (risk capped at pool.treading_cap);
   anyone outside the water zone is ignored.
3. A per-swimmer STATE MACHINE turns the risk curve into states:
   NORMAL -> WATCH (risk >= watch_risk) -> WARNING (>= warning_risk) -> DISTRESS when risk stays
   >= alert_risk for hold_s seconds (it only ends when risk falls below release_risk: hysteresis).
   The alert is raised hold_s after the distress run began.
4. SUBMERSION: a swimmer in WARNING or DISTRESS who disappears for submersion_lost_s seconds (or whose
   head is "submerged" for submersion_head_s) -> "possible_submersion". A dive from normal swimming
   does not count, because no warning came first.

NumPy-free pure Python. Run on its own: python distress.py --behaviour stage2_behaviour.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

import utils

STATES = ("NORMAL", "WATCH", "WARNING", "DISTRESS", "SUBMERSION")
SIGNAL_NOTES = {
    "slowing": "Movement slowing",
    "vertical_posture": "Vertical posture",
    "repeated_arm_motion": "Repeated arm movement",
    "low_displacement": "Very low displacement",
    "unstable_head": "Head position unstable",
    "submerged_head": "Head going under",
}


def sample_signals(s: dict, pcfg: dict) -> dict:
    """Which warning signs are present in one behaviour sample, and is it one of the normal look-alikes."""
    low_disp = float(s["displacement"]) <= float(pcfg["displacement_low_bh_s"])
    # "diagonal" (tilted torso) counts as upright only when the swimmer is not getting anywhere
    vertical = s["posture"] in ("vertical", "head_only", "underwater") or (s["posture"] == "diagonal" and low_disp)
    # Arm strokes are normal while swimming; the distress "climbing the ladder" motion happens UPRIGHT.
    arms = s["arm_motion"] == "repeated" and vertical
    head_bad = s["head"] in ("unstable", "submerged")
    swimming = (s["posture"] == "horizontal" and s["movement"] in ("medium", "high") and not low_disp)
    resting = bool(s.get("near_wall")) and not arms
    treading = vertical and s["arm_motion"] == "calm" and s["head"] == "stable"
    return {"vertical_posture": vertical, "low_displacement": low_disp, "repeated_arm_motion": arms,
            "unstable_head": head_bad, "submerged_head": s["head"] == "submerged",
            "swimming_normally": swimming, "resting_at_wall": resting, "treading_water": treading,
            "slowing": s["posture"] == "horizontal" and s["movement"] == "low",
            "in_water": bool(s.get("in_water", True))}


def sample_risk(sig: dict, transition: bool, pcfg: dict) -> float:
    """Risk of one sample in [0, 1]: weighted warning signs, then the look-alike damping."""
    if not sig["in_water"]:
        return 0.0
    w = pcfg["weights"]
    r = (w["vertical"] * sig["vertical_posture"] + w["low_displacement"] * sig["low_displacement"]
         + w["repeated_arms"] * sig["repeated_arm_motion"] + w["unstable_head"] * sig["unstable_head"]
         + w["transition"] * (transition and sig["vertical_posture"]))
    if sig["resting_at_wall"]:
        r *= float(pcfg["resting_factor"])
    if sig["treading_water"]:
        r = min(r, float(pcfg["treading_cap"]))
    return round(min(1.0, r), 3)


def reason(samples: list[dict], cfg: dict, last_seen: dict | None = None, video_end: float | None = None) -> dict:
    """Run the state machine for every swimmer.

    `last_seen`: {person_id: last time the detector saw them}; `video_end`: video length in seconds
    (both used for "disappeared while in distress" = possible submersion).
    Returns {"events": [...], "risk": {pid: [[t, risk], ...]}, "timeline": {pid: [...]}, "status": {pid: ...}}.
    """
    pcfg = cfg["pool"]
    by_person: dict[int, list[dict]] = {}
    for s in samples:
        by_person.setdefault(int(s["person_id"]), []).append(s)
    events, risk_curves, timelines, status = [], {}, {}, {}
    for pid in sorted(by_person):
        seq = sorted(by_person[pid], key=lambda s: s["timestamp"])
        res = _reason_one(pid, seq, pcfg, (last_seen or {}).get(pid), video_end)
        events += res["events"]
        risk_curves[pid] = res["risk"]
        timelines[pid] = res["timeline"]
        status[pid] = res["final_state"]
    events.sort(key=lambda e: (e["start_time"], e["person_id"]))
    return {"events": events, "risk": risk_curves, "timeline": timelines, "status": status}


def _reason_one(pid: int, seq: list[dict], pcfg: dict, last_seen, video_end) -> dict:
    """State machine for one swimmer over its time-ordered samples."""
    hold, look = float(pcfg["hold_s"]), float(pcfg["lookback_s"])
    last_normal = None                  # last time this swimmer was swimming normally
    state, run_start, run_risks, run_evidence = "NORMAL", None, [], set()
    head_sub_since = None
    events, risk_curve, timeline, seen_notes = [], [], [], set()
    distress_event = None
    for s in seq:
        t = float(s["timestamp"])
        if s.get("track_valid") is False:          # ID switch suspected: this window mixes two people
            continue
        sig = sample_signals(s, pcfg)
        if sig["swimming_normally"]:
            last_normal = t
        transition = last_normal is not None and (t - last_normal) <= look
        r = sample_risk(sig, transition, pcfg)
        risk_curve.append([round(t, 2), r])

        # timeline notes: the first time each warning sign shows up after normal swimming
        if sig["swimming_normally"] and "swimming" not in seen_notes:
            timeline.append({"t": round(t, 2), "state": "NORMAL", "note": "Normal swimming"})
            seen_notes.add("swimming")
        # (warning signs are only logged once the risk is rising, so a lap swimmer's turn is not "news")
        if sig["in_water"] and not sig["resting_at_wall"] and not sig["treading_water"] and                 (r >= float(pcfg["watch_risk"]) or sig["slowing"]):
            for key, note in SIGNAL_NOTES.items():
                if sig.get(key) and key not in seen_notes and (key != "slowing" or "swimming" in seen_notes):
                    timeline.append({"t": round(t, 2), "state": _state_for(r, pcfg), "note": note})
                    seen_notes.add(key)

        # head under water for long enough while already in trouble = submersion
        if sig["submerged_head"]:
            head_sub_since = t if head_sub_since is None else head_sub_since
        else:
            head_sub_since = None

        if state == "DISTRESS":
            if r >= float(pcfg["release_risk"]):
                run_risks.append(r)
                run_evidence |= _evidence(sig, transition)
                distress_event["end_time"] = round(t, 2)
                distress_event["risk_score"] = _risk_score(run_risks, t - run_start, hold)
                distress_event["evidence"] = _ordered(run_evidence)
            else:
                state, run_start, run_risks, run_evidence = _state_for(r, pcfg), None, [], set()
                distress_event = None
        else:
            if r >= float(pcfg["alert_risk"]):
                if run_start is None:
                    run_start, run_risks, run_evidence = t, [], set()
                run_risks.append(r)
                run_evidence |= _evidence(sig, transition)
                if t - run_start >= hold - 1e-9:
                    state = "DISTRESS"
                    distress_event = {
                        "person_id": pid, "event": "high_risk_aquatic_distress",
                        "start_time": round(run_start, 2), "alert_time": round(t, 2), "end_time": round(t, 2),
                        "risk_score": _risk_score(run_risks, t - run_start, hold),
                        "evidence": _ordered(run_evidence), "location": s["location"]}
                    events.append(distress_event)
                    timeline.append({"t": round(t, 2), "state": "DISTRESS", "note": "HIGH-RISK AQUATIC DISTRESS"})
            else:
                run_start, run_risks, run_evidence = None, [], set()
                state = _state_for(r, pcfg)

        if (head_sub_since is not None and state == "DISTRESS"
                and t - head_sub_since >= float(pcfg["submersion_head_s"])):
            events.append(_submersion(pid, head_sub_since, s["location"], "head_submerged"))
            timeline.append({"t": round(head_sub_since, 2), "state": "SUBMERSION", "note": "Possible submersion"})
            state, head_sub_since = "SUBMERSION", None

    # Vanished while in trouble (the detector lost them in the water) = possible submersion.
    last_t = float(seq[-1]["timestamp"]) if seq else 0.0
    gone_at = float(last_seen) if last_seen is not None else last_t
    still_video = video_end is None or (float(video_end) - gone_at) >= float(pcfg["submersion_lost_s"])
    # Only a swimmer already in DISTRESS who then vanishes counts: people who were merely "warning" (playing,
    # treading) disappear all the time by swimming out of view or behind others (seen on real pool footage).
    in_trouble = state == "DISTRESS"
    if (state != "SUBMERSION" and seq and in_trouble and still_video and seq[-1].get("in_water", True)
            and not any(e["event"] == "possible_submersion" for e in events)):
        events.append(_submersion(pid, gone_at, seq[-1]["location"], "lost_from_view"))
        timeline.append({"t": round(gone_at, 2), "state": "SUBMERSION",
                         "note": "Possible submersion (lost from view)"})
        state = "SUBMERSION"
    return {"events": events, "risk": risk_curve, "timeline": timeline, "final_state": state}


def _state_for(r: float, pcfg: dict) -> str:
    if r >= float(pcfg["warning_risk"]):
        return "WARNING"
    if r >= float(pcfg["watch_risk"]):
        return "WATCH"
    return "NORMAL"


def _evidence(sig: dict, transition: bool) -> set:
    ev = {k for k in ("vertical_posture", "low_displacement", "repeated_arm_motion", "unstable_head") if sig[k]}
    if transition and sig["vertical_posture"]:
        ev.add("transition_from_swimming")
    return ev


EVIDENCE_ORDER = ("vertical_posture", "low_displacement", "repeated_arm_motion", "unstable_head",
                  "transition_from_swimming", "head_submerged", "lost_from_view")
EVIDENCE_TEXT = {"vertical_posture": "Vertical posture", "low_displacement": "Little or no forward progress",
                 "repeated_arm_motion": "Repeated distress-like arm movement", "unstable_head": "Unstable head position",
                 "transition_from_swimming": "Was swimming normally just before (distress transition)",
                 "head_submerged": "Head went under the water", "lost_from_view": "Lost from view in the water"}


def _ordered(ev: set) -> list[str]:
    return [k for k in EVIDENCE_ORDER if k in ev]


def _risk_score(risks: list[float], held_s: float, hold: float) -> float:
    """Mean risk of the run, nudged up the longer it lasts (max 1.0)."""
    mean = sum(risks) / max(1, len(risks))
    return round(min(1.0, mean * (0.9 + 0.1 * min(1.0, held_s / (2 * hold)))), 2)


def _submersion(pid: int, t: float, location: str, why: str) -> dict:
    return {"person_id": pid, "event": "possible_submersion", "start_time": round(float(t), 2),
            "alert_time": round(float(t), 2), "end_time": round(float(t), 2), "risk_score": 0.98,
            "evidence": [why], "location": location}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: distress reasoning from stage-2 behaviour samples")
    ap.add_argument("--behaviour", required=True, help="stage2_behaviour.jsonl")
    ap.add_argument("--scenario", default="pool")
    ap.add_argument("--out", default="stage3_events.json")
    args = ap.parse_args(argv)
    cfg = utils.load_config(scenario=args.scenario)
    with open(args.behaviour, encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]
    res = reason(samples, cfg)
    utils.write_json(args.out, {"events": res["events"], "timeline": res["timeline"]})
    for e in res["events"]:
        print(f"Person #{e['person_id']}: {e['event']} at {utils.fmt_time(e['alert_time'])} "
              f"(risk {e['risk_score']:.0%}) evidence: {', '.join(e['evidence'])}")
    print(f"{len(res['events'])} event(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
