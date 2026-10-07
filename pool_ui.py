"""Stage 4 of the pool pipeline: the lifeguard dashboard inside the Streamlit app (app.py calls this).

Shows, for a results folder that has pool_dashboard.json:
  * one ALERT card per distress / submersion alert: who + where + when + risk + evidence checklist,
    the snapshot and risk plot ("view evidence"), and the spoken announcement (alarm + voice),
    auto-played once in the browser so it comes out of the dashboard device's speaker;
  * a status board (green / yellow / red per swimmer) with a time slider to replay the scene;
  * the risk curve of every swimmer and each swimmer's behaviour timeline.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import distress
import utils

STATE_BADGE = {"NORMAL": ":green-badge[NORMAL]", "WATCH": ":blue-badge[WATCH]", "WARNING": ":orange-badge[WARNING]",
               "DISTRESS": ":red-badge[DISTRESS]", "SUBMERSION": ":red-badge[POSSIBLE SUBMERSION]"}
STATE_DOT = {"NORMAL": "🟢", "WATCH": "🔵", "WARNING": "🟡", "DISTRESS": "🔴", "SUBMERSION": "🆘"}


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:
    """pool_dashboard.json (cached per file version)."""
    return utils.read_json(path)


def _state_at(t: float, pid: int, dash: dict) -> tuple[str, float]:
    """Swimmer state and risk at video time t, from the risk curve and the alerts."""
    curve = dash["risk"].get(str(pid)) or []
    risk = 0.0
    for tt, r in curve:
        if tt > t:
            break
        risk = r
    th = dash["thresholds"]
    state = "WARNING" if risk >= th["warning"] else "WATCH" if risk >= th["watch"] else "NORMAL"
    for a in dash["alerts"]:
        if int(a["person_id"]) != pid or t < float(a["alert_time"]):
            continue
        if a["event"] == "possible_submersion":
            state = "SUBMERSION"
        elif t <= float(a.get("end_time", a["alert_time"])) + 1.0 and state != "SUBMERSION":
            state = "DISTRESS"
    seen = next((p for p in dash["people"] if p["person_id"] == pid), None)
    if seen and t > seen["last_seen"] + 0.5 and state not in ("SUBMERSION", "DISTRESS"):
        state = "NORMAL"
    return state, risk


def render_pool_panel(folder: Path, data: dict) -> None:
    """Draw the lifeguard view for one results folder (call it above the result tabs)."""
    path = folder / "pool_dashboard.json"
    dash = _load(str(path), path.stat().st_mtime)
    events = {(e["entity_id"], e["behavior"]): e for e in data.get("events") or []}
    st.subheader("Lifeguard dashboard")
    if dash.get("pose"):
        st.caption(f"Behaviour signals: {dash['pose']}")

    alerts = dash.get("alerts") or []
    if not alerts:
        st.success("No distress: every swimmer stayed within normal behaviour.", icon=":material/pool:")
    for i, a in enumerate(alerts):
        beh = "submersion" if a["event"] == "possible_submersion" else "aquatic_distress"
        ev = events.get((a["person_id"], beh)) or {}
        title = "POSSIBLE SUBMERSION" if beh == "submersion" else "HIGH-RISK AQUATIC DISTRESS"
        with st.container(border=True):
            st.markdown(f"### :red[🚨 {title}]")
            cols = st.columns([1, 2, 1, 1])                     # the location name needs the room
            cols[0].metric("Swimmer", f"#{a['person_id']}")
            cols[1].metric("Location", a.get("location") or "Pool")
            cols[2].metric("Time", utils.fmt_time(float(a["alert_time"])),
                           help=f"Behaviour began at {utils.fmt_time_precise(float(a['start_time']))}; "
                                f"alert raised at {utils.fmt_time_precise(float(a['alert_time']))}.")
            cols[3].metric("Risk", f"{float(a['risk_score']):.0%}")
            st.markdown("**Evidence**\n" + "\n".join(f"- ✓ {distress.EVIDENCE_TEXT.get(k, k)}" for k in a["evidence"]))
            ann = ev.get("announcement") or {}
            audio = folder / ann["audio"] if ann.get("audio") else None
            if audio is not None and audio.is_file():
                st.caption(f"🔊 Announcement: “{ann.get('text', '')}”")
                key = f"played::{folder}::{path.stat().st_mtime}::{i}"
                first = not st.session_state.get(key, False)
                st.audio(str(audio), format="audio/wav", autoplay=first and st.session_state.get("pool_autoplay", True))
                st.session_state[key] = True
            with st.expander("View evidence"):
                c1, c2 = st.columns(2)
                snap = folder / ev["snapshot"] if ev.get("snapshot") else None
                plot = folder / ev["speed_plot"] if ev.get("speed_plot") else None
                if snap is not None and snap.is_file():
                    c1.image(str(snap), caption="Snapshot at the alert")
                if plot is not None and plot.is_file():
                    c2.image(str(plot), caption="Distress risk over time")
                if ev.get("evidence"):
                    st.write(ev["evidence"])
    st.toggle("Auto-play spoken alerts", value=True, key="pool_autoplay",
              help="Plays the alarm and the spoken alert once when new results appear, on this device's speaker.")

    duration = float(dash.get("duration_s") or 0.0)
    if duration > 0 and dash.get("people"):
        def in_view(p, tt):
            return p["first_seen"] - 0.5 <= tt <= p["last_seen"] + 0.5

        if alerts:
            default_t = float(alerts[0]["alert_time"])
        else:                                   # no alert: open at the moment with the most swimmers in view
            grid = [i * 0.5 for i in range(int(duration / 0.5) + 1)]
            default_t = max(grid, key=lambda tt: sum(in_view(p, tt) for p in dash["people"])) if grid else 0.0
        t = st.slider("Replay the pool at time (s)", 0.0, duration, min(default_t, duration), step=0.5,
                      key=f"pool_t::{folder}")
        alert_ids = {int(a["person_id"]) for a in alerts}
        shown = [p for p in dash["people"] if in_view(p, t) or
                 (p["person_id"] in alert_ids and _state_at(t, p["person_id"], dash)[0] in ("DISTRESS", "SUBMERSION"))]
        st.markdown(f"**Status at {utils.fmt_time_precise(t)}**: {len(shown)} swimmer(s) in view")
        st.caption(f"Numbers are tracker IDs, not a count: every new track gets the next number "
                   f"({len(dash['people'])} IDs over the whole video). A swimmer who splashes or dives can come back "
                   f"with a new ID; a trained swimmer detector reduces this.")
        if not shown:
            st.info("Nobody is in view at this moment. Move the slider.")
        else:
            board = st.columns(min(6, len(shown)))
            for k, p in enumerate(shown):
                state, risk = _state_at(t, p["person_id"], dash)
                with board[k % len(board)]:
                    st.markdown(f"{STATE_DOT[state]} **{p['name']}**  \n{STATE_BADGE[state]}  \n"
                                + (f"risk {risk:.0%}" if in_view(p, t) else "_lost from view_"))

        try:
            import altair as alt
            import pandas as pd
            rows = [{"time (s)": tt, "risk": r, "swimmer": p["name"]}
                    for p in dash["people"] for tt, r in (dash["risk"].get(str(p["person_id"])) or [])]
            if rows:
                st.markdown("**Distress risk of every swimmer**")
                df = pd.DataFrame(rows)
                lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
                    x=alt.X("time (s):Q", scale=alt.Scale(domain=[0, duration])),
                    y=alt.Y("risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    color=alt.Color("swimmer:N", legend=alt.Legend(orient="bottom")))
                level = alt.Chart(pd.DataFrame({"risk": [dash["thresholds"]["alert"]]})).mark_rule(
                    strokeDash=[6, 4], color="#c0392b").encode(y="risk:Q")
                now = alt.Chart(pd.DataFrame({"time (s)": [t]})).mark_rule(color="#888").encode(x="time (s):Q")
                st.altair_chart(lines + level + now, height=240)
                st.caption(f"Dashed red line: alert level {dash['thresholds']['alert']:.0%}. Grey line: the replay time.")
        except ImportError:
            pass

    timelines = {k: v for k, v in (dash.get("timeline") or {}).items() if v}
    if timelines:
        with st.expander("Behaviour timelines (what the system noticed, in order)", expanded=bool(alerts)):
            for pid, items in timelines.items():
                st.markdown(f"**Swimmer #{pid}**  \n" + "  \n".join(
                    f"`{utils.fmt_time_precise(x['t'])}` {x['note']}" for x in items))
