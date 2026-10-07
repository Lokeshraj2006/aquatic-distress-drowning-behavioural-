"""PS07 Behaviour Intelligence - the web front end (Streamlit).

    streamlit run app.py

What it does: pick a video, a scenario preset and (optionally) a zone in the sidebar, press Run, and read the
results (incidents with evidence, highlight reel, timeline, full report) in tabs. It never re-implements the
analysis: the Run button starts `run.py` as a subprocess with exactly the flags you can also type yourself, and
the results are the files that run.py writes (events.json, report.html, highlights.mp4, ...). The heavy libraries
(torch, ultralytics) are never imported here, so the page stays fast. Any older results folder under outputs/ or
examples/ can be opened from the sidebar, which is the fallback when there is no time to run anything.

The last run is saved to outputs/_ui/last_run.json, so reloading the page (a new Streamlit session) shows it again.
Bad input (a broken zones file, a file that is not a video, a strange events.json) is explained on the page, never
a traceback. The plain logic (file discovery, command building, reading results) lives in ui_helpers.py.
"""
from __future__ import annotations

import functools
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:                      # work no matter where `streamlit run` was started from
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import ui_helpers as ui  # noqa: E402
import utils  # noqa: E402

PITCH = ("Early warning for lifeguards: it tracks every swimmer, watches how their behaviour changes over time, "
         "and raises an evidence-backed alert saying who, where, when and why, out loud.")
DEVICE_LABELS = {"auto": "auto (GPU when there is one)", "cpu": "cpu", "0": "0 (first GPU)"}
MAX_EVENT_CARDS = 30                              # event cards drawn before "Show all" (each one loads 2 pictures)
RESULT_TABS = ["Summary", "Highlight reel", "Incidents", "Timeline & heatmap", "Full report", "Annotated video",
               "Downloads"]

st.set_page_config(page_title="Aquatic Distress Intelligence", page_icon=":material/visibility:", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- cached readers (keyed by file mtime)

@st.cache_data(max_entries=16, show_spinner=False)
def cached_results(path: str, mtime: float) -> dict:
    """events.json read and made safe (ui.sanitize_results); `mtime` is only part of the cache key, so a file
    that is rewritten by a new run is read again, and an unchanged one is read once, not on every rerun."""
    return ui.sanitize_results(utils.read_json(path))


@st.cache_data(max_entries=16, show_spinner=False)
def cached_text(path: str, mtime: float) -> str:
    """Read a text file (summary.txt); cached by path and modification time."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


@st.cache_data(max_entries=8, show_spinner=False)
def cached_facts(path: str, mtime: float):
    """Duration, fps and size of a video (None when it cannot be opened); cached by path and mtime."""
    return ui.video_facts(path)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_preview(path: str, mtime: float) -> str | None:
    """Browser-playable (H.264) copy of the selected video, made once with ffmpeg; None if not possible."""
    clip = ui.playable_preview(path)
    return str(clip) if clip else None


@st.cache_data(max_entries=8, show_spinner=False)
def cached_frame(path: str, mtime: float, max_width: int = 960):
    """First frame of a video (BGR array, scale); cached by path and mtime."""
    return ui.first_frame(path, max_width)


@st.cache_data(ttl=300, show_spinner=False)
def cached_scenario(name: str) -> dict:
    """Title, rules and classes of a preset for the sidebar."""
    return ui.scenario_summary(name)


@st.cache_data(ttl=300, show_spinner=False)
def cached_needs(name: str) -> dict:
    """Classes, stride, resize width, weights and max seconds of a preset (config.yaml + the preset, read once)."""
    return ui.scenario_needs(name)


@st.cache_data(max_entries=64, show_spinner=False)
def cached_detections(video: str, out_dir: str, needs: tuple, stride: int, size: tuple, max_seconds: float,
                      n_frames: int, signature: tuple):
    """Which cached detections.json `--reuse` could use; `signature` changes when any detections file changes."""
    classes, resize_width, weights = needs
    return ui.find_cached_detections(video, out_dir, list(classes), stride, size, signature,
                                     resize_width=resize_width, weights=weights, max_seconds=max_seconds,
                                     n_frames=n_frames)


def find_cached(preset: str, video: Path, facts: dict, stride: int, max_seconds: float):
    """The cached detections.json that `run.py --reuse` could use for this video and preset (None when none)."""
    need = cached_needs(preset)
    # an explicit "max seconds" wins; otherwise the preset's own value (0 = whole video)
    seconds = float(max_seconds) or need["max_seconds"]
    return cached_detections(str(video), ui.default_out_dir(video, preset),
                             (tuple(need["classes"]), need["resize_width"], need["weights"]), int(stride),
                             (facts["width"], facts["height"]), seconds, int(facts["frames"] or 0),
                             ui.detections_signature())


def note_missing(name: str, why: str = "") -> None:
    """One quiet line for a file that is not in the results folder (the page never fails on a missing file)."""
    st.caption(f":material/info: `{name}` is not in this folder{(' (' + why + ')') if why else ''}.")


# --------------------------------------------------------------------------- result renderers

def show_chain(inc: dict) -> None:
    """One incident chain: severity badge, who, title, time span and the plain-English story."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(str(inc.get("severity") or "-").upper(), color=ui.severity_color(inc.get("severity")))
            st.markdown(f"**{ui.md_escape(inc.get('entity_name') or 'Group')}**: {ui.md_escape(inc.get('title', ''))}"
                        f"  ({inc.get('start', '?')} - {inc.get('end', '?')})")
        st.markdown(ui.md_escape(inc.get("story", "")))


def show_event(e: dict, folder: Path, weights: dict, expanded: bool) -> None:
    """One event as an expander: severity, who, behaviour, time, confidence and its parts, evidence, pictures."""
    sev = str(e.get("severity") or "-")
    who = e.get("entity_name") or f"Entity {e.get('entity_id')}"
    what = e.get("behavior_name") or str(e.get("behavior", "")).replace("_", " ").title()
    conf = float(e.get("confidence") or 0.0)
    label = (f"#{e.get('event_id')} | {sev.upper()} | {ui.md_escape(who)} | {ui.md_escape(what)} | "
             f"{e.get('start', '?')}-{e.get('end', '?')} | confidence {conf:.2f}")
    with st.expander(label, expanded=expanded):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(sev.upper(), color=ui.severity_color(sev))
            st.badge(f"confidence {conf:.2f}", icon=":material/verified:", color="gray")
            if e.get("zone"):
                st.badge(f"zone: {e['zone']}", icon=":material/crop_free:", color="violet")
        duration = e.get("duration_s")
        st.markdown(f"**Who:** {ui.md_escape(who)}  \n**Behaviour:** {ui.md_escape(what)}  \n"
                    f"**When:** {e.get('start', '?')} to {e.get('end', '?')}"
                    + (f" ({float(duration):.1f} s)" if duration is not None else ""))
        st.markdown("**Why it was flagged**")
        st.markdown(ui.md_escape(e.get("evidence", "")))
        st.caption(ui.confidence_line(e, weights))
        note = (e.get("baseline") or {}).get("note")
        if note:
            st.caption(f"Scene baseline: {note}")
        pose = ui.pose_text(e)
        if pose:
            st.badge(pose[1], icon=":material/accessibility_new:", color=pose[0])
        snapshot = ui.safe_child(folder, e.get("snapshot"))
        plot = ui.safe_child(folder, e.get("speed_plot"))
        if snapshot or plot:
            left, right = st.columns(2)
            if snapshot:
                left.image(snapshot, caption="Evidence snapshot", width="stretch",
                           alt=f"Snapshot of event {e.get('event_id')}: {what}")
            if plot:
                right.image(plot, caption="Measurement over time, with the rule threshold", width="stretch",
                            alt=f"Plot of the measurement behind event {e.get('event_id')}")
        else:
            note_missing(f"snapshots/e{e.get('event_id')}.jpg and its plot")


def show_near_and_unusual(data: dict, cfg: dict) -> None:
    """The 'almost flagged' cases and the unusual entities: what the system saw but did not report."""
    almost = data.get("near_misses") or []
    with st.expander(f"Almost flagged ({len(almost)})"):
        st.caption("Behaviour that reached 70 percent of a threshold but did not cross it, so a reader can judge "
                   "the thresholds.")
        if almost:
            rows = [{"Who": utils.entity_name(cfg, a.get("track_id")),
                     "Behaviour": utils.behavior_name(cfg, str(a.get("behavior", ""))),
                     "Reached": f"{a.get('value')} {a.get('unit', '')}".strip(),
                     "Rule": f"{a.get('threshold')} {a.get('unit', '')}".strip(),
                     "Why it was not flagged": a.get("note", "")} for a in almost]
            st.dataframe(rows, hide_index=True, alt="Cases that almost reached a rule")
        else:
            st.caption("None: no behaviour came close to a rule without crossing it.")
    unusual = data.get("unusual_tracks") or []
    with st.expander(f"Unusual compared with the scene ({len(unusual)})"):
        if unusual:
            rows = [{"Who": utils.entity_name(cfg, u.get("entity_id")), "Anomaly score": u.get("anomaly_score"),
                     "What is unusual": u.get("note", "")} for u in unusual]
            st.dataframe(rows, hide_index=True, alt="Entities unusual compared with the scene")
        else:
            st.caption("Nothing stood out from the scene's typical behaviour.")


@st.fragment
def show_results(folder_str: str, source: str) -> None:
    """Metric tiles and the result tabs for one results folder. A fragment: switching a tab does not rerun the page."""
    folder = Path(folder_str)
    events_path = folder / "events.json"
    try:
        data = cached_results(str(events_path), ui.mtime(events_path))
    except Exception as exc:                         # corrupt or half-written file: say so, do not crash
        st.error(f"Cannot read `{ui.rel_path(events_path)}`: {exc}")
        return
    try:
        render_results(folder, data, source)
    except Exception as exc:                         # last safety net: a surprise in one file never shows a traceback
        st.error(f"Could not show the results in `{ui.rel_path(folder)}` ({type(exc).__name__}: {exc}). "
                 "The files are still in the folder.")


def render_results(folder: Path, data: dict, source: str) -> None:
    """Draw the metric tiles and the tabs for the (already sanitized) events.json content of `folder`."""
    m = ui.result_metrics(data)
    cfg = (data.get("meta") or {}).get("config") or {}
    weights = ui.confidence_weights(data)

    st.subheader("Results")
    st.caption(f"{source}: `{ui.rel_path(folder)}`  |  {m['scenario_title'] or m['scenario']}"
               + ("  |  privacy mode: faces blurred in the pictures and videos" if m["privacy"] else ""))
    with st.container(horizontal=True):
        st.metric("Incidents", m["incidents"], border=True,
                  delta=f"{m['high']} high severity" if m["high"] else None, delta_color="inverse",
                  help="Incident chains plus single events that are not part of a chain (what the highlight reel shows).")
        st.metric("Events", m["events"], border=True)
        st.metric(m["tracked_label"] + " tracked", m["tracked"] if m["tracked"] is not None else "?", border=True)
        st.metric("Video length", m["duration"], border=True)
        if m["processing_fps"]:
            st.metric("Processing speed", f"{m['processing_fps']} fps", border=True,
                      help="Frames per second of detection and tracking. Unknown when cached detections were reused.")
        st.metric("Scenario", m["scenario"], border=True, help=m["scenario_title"] or None)

    if (folder / "pool_dashboard.json").is_file():                   # pool scenario: lifeguard view first
        import pool_ui
        try:
            pool_ui.render_pool_panel(folder, data)
        except Exception as exc:                                     # never lose the normal results over it
            st.warning(f"Lifeguard dashboard could not be drawn ({type(exc).__name__}: {exc}).")

    # Heavy tabs (the embedded report, the long annotated video) only run while they are the open tab.
    tabs = st.tabs(RESULT_TABS, on_change="rerun", key="result_tab")

    with tabs[0]:                                                    # Summary
        summary = folder / "summary.txt"
        if summary.is_file():
            st.code(cached_text(str(summary), ui.mtime(summary)), language="text", wrap_lines=True)
        else:
            note_missing("summary.txt")
        st.markdown("**Incident chains**")
        chains = data.get("incidents") or []
        if chains:
            for inc in chains:
                show_chain(inc)
        else:
            st.caption("No incident chains. A chain links events of one entity that follow each other within 15 s; "
                       "the single events are listed in the Incidents tab.")

    with tabs[1]:                                                    # Highlight reel
        reel = ui.safe_child(folder, data.get("highlight_reel") or "highlights.mp4")
        if reel:
            st.video(reel, alt="Highlight reel: only the incidents, each one captioned")
            st.caption("Only the incidents, with a caption bar. If your browser cannot play it, use the Downloads tab.")
        else:
            note_missing("highlights.mp4", "the run used 'skip videos', or this is a trimmed example")

    with tabs[2]:                                                    # Incidents (one expander per event)
        events = data.get("events") or []
        if not events:
            st.success("No incidents flagged in this video.")
        shown = events
        if len(events) > MAX_EVENT_CARDS and not st.toggle(
                f"Show all {len(events)} events (the first {MAX_EVENT_CARDS} are shown)", key="all_events",
                help="Every event card loads two pictures; a long video can have hundreds of events."):
            shown = events[:MAX_EVENT_CARDS]
        for i, ev in enumerate(shown):
            show_event(ev, folder, weights, expanded=(i == 0))
        show_near_and_unusual(data, cfg)

    with tabs[3]:                                                    # Timeline & heatmap
        timeline = folder / "timeline.png"
        heatmap = folder / "heatmap.jpg"
        if timeline.is_file():
            st.image(timeline, caption="Who was visible when (grey), with the events (colour)", width="stretch",
                     alt="Timeline with one bar per tracked entity and coloured blocks for events")
        else:
            note_missing("timeline.png")
        if heatmap.is_file():
            st.image(heatmap, caption="Where activity was seen", width="stretch",
                     alt="Heatmap of activity blended over the first frame")
        else:
            note_missing("heatmap.jpg")

    if tabs[4].open:                                                 # Full report
        with tabs[4]:
            report = folder / "report.html"
            if report.is_file():
                # report.html is written by this project's own report.py (every value is HTML-escaped there).
                st.iframe(report, height=1000, alt="Full evidence report")
            else:
                note_missing("report.html")

    if tabs[5].open:                                                 # Annotated video
        with tabs[5]:
            annotated = folder / "annotated.mp4"
            if annotated.is_file():
                st.video(annotated, alt="Annotated video with boxes, IDs, trails and event labels")
            else:
                note_missing("annotated.mp4", "the run used 'skip videos', or full videos were left out of an example")

    with tabs[6]:                                                    # Downloads
        st.caption("Files are read when you click, so nothing is loaded into memory before that.")
        with st.container(horizontal=True):
            for name, mime, icon in (("events.json", "application/json", ":material/data_object:"),
                                     ("summary.txt", "text/plain", ":material/description:"),
                                     ("report.html", "text/html", ":material/article:"),
                                     ("highlights.mp4", "video/mp4", ":material/movie:")):
                path = folder / name
                if path.is_file():
                    st.download_button(name, data=path.read_bytes, file_name=name, mime=mime, icon=icon,
                                       on_click="ignore", key=f"dl_{name}")
                else:
                    note_missing(name)
            st.download_button("Whole folder (.zip)", data=functools.partial(ui.zip_folder_bytes, folder),
                               file_name=f"{folder.name}.zip", mime="application/zip", type="primary",
                               icon=":material/folder_zip:", on_click="ignore", key="dl_zip")


# --------------------------------------------------------------------------- state

if "last_run" not in st.session_state:             # dict about the latest Run (command, log, out dir, ok)
    st.session_state["last_run"] = ui.load_last_run()   # a browser reload is a NEW session: take it from the disk
    restored = st.session_state["last_run"]
    if restored:                                    # put the sidebar back as it was, so a reload is not a different job
        label = next((lbl for lbl, p in ui.list_videos() if ui.rel_path(p) == restored["video"]), None)
        if label:
            st.session_state["sample"] = label
        if restored.get("scenario") in utils.list_scenarios():
            st.session_state["scenario"] = restored["scenario"]
st.session_state.setdefault("_uploads", {})         # uploaded file id -> saved path ("" = rejected), no rewrite per rerun
if st.session_state.pop("_clear_open_results", False):
    st.session_state["open_results"] = None         # a new run's results replace any folder opened earlier


# --------------------------------------------------------------------------- sidebar

video: Path | None = None
facts = None
zone_mode = "None"
zone_path: Path | None = None                       # a ready zones.json (file or upload); the rectangle is written at Run
rect_zones: list[dict] = []
zone_problem = ""

with st.sidebar:
    st.subheader("1. Video")
    source = st.segmented_control("Video source", ["Sample", "Upload"], default="Sample", key="source",
                                  label_visibility="collapsed") or "Sample"
    if source == "Upload":
        uploaded = st.file_uploader("Upload a video", type=[e.lstrip(".") for e in ui.VIDEO_EXTS], key="upload",
                                    help="Saved to samples/uploads/ so it can be picked again later (limit 500 MB).")
        if uploaded is not None:
            saved = st.session_state["_uploads"].get(uploaded.file_id)
            if saved is None or (saved and not Path(saved).exists()):
                try:
                    path = ui.save_upload(uploaded.name, uploaded.getbuffer())
                    if ui.video_facts(path) is None:             # not a video: do not keep it in samples/uploads/
                        ui.discard_upload(path)
                        path = None
                    saved = str(path) if path else ""
                    st.session_state["_uploads"][uploaded.file_id] = saved
                except OSError as exc:
                    saved = None
                    st.error(f"Could not save the upload: {exc}")
            if saved:
                video = Path(saved)
                st.caption(f"Saved as `{ui.rel_path(video)}`")
            elif saved == "":
                st.error(f"`{ui.safe_filename(uploaded.name)}` cannot be opened as a video. "
                         "Try an mp4 (H.264), mov, avi, mkv or webm file.")
        else:
            st.caption("Choose a video file, or switch back to Sample.")
    else:
        choices = dict(ui.list_videos())
        if choices:
            label = st.selectbox("Sample video", list(choices), key="sample")
            video = choices.get(label)
        else:
            st.info("No videos in samples/. Run `python get_samples.py`, or upload one.")

    if video is not None:
        facts = cached_facts(str(video), ui.mtime(video))
        if facts is None:
            st.error("This file cannot be opened as a video.")

    st.subheader("2. Scenario")
    names = utils.list_scenarios()
    infos = {n: cached_scenario(n) for n in names}
    scenario = st.selectbox("Scenario preset", names, key="scenario", index=names.index("pool") if "pool" in names else 0,
                            format_func=lambda n: infos[n]["title"]) if names else None
    if scenario:
        info = infos[scenario]
        st.caption(f"Tracks: {', '.join(info['classes']) or '-'}.  \nRules: {', '.join(info['rules']) or '-'}.")

    st.subheader("3. Zone (optional)")
    zone_mode = st.segmented_control("Zone type", ["None", "Rectangle", "Zones file"], default="None", key="zone_mode",
                                     label_visibility="collapsed") or "None"
    if zone_mode != "None" and facts is None:
        st.caption("Pick a readable video first.")
    elif zone_mode == "Rectangle":
        width, height = facts["width"], facts["height"]
        vkey = f"{video.name}_{width}x{height}"       # new sliders (and defaults) for every video
        st.caption(f"Corners in original video pixels ({width} x {height}).")
        c1, c2 = st.columns(2)
        x1 = c1.slider("x1 (left)", 0, width, int(width * 0.25), key=f"zx1_{vkey}")
        y1 = c1.slider("y1 (top)", 0, height, int(height * 0.25), key=f"zy1_{vkey}")
        x2 = c2.slider("x2 (right)", 0, width, int(width * 0.75), key=f"zx2_{vkey}")
        y2 = c2.slider("y2 (bottom)", 0, height, int(height * 0.75), key=f"zy2_{vkey}")
        rect_zones = ui.rect_zone(x1, y1, x2, y2)
        if ui.rect_is_empty(rect_zones):
            zone_problem = "The rectangle has no area: move x1 and x2 (or y1 and y2) apart."
            st.error(zone_problem)
    elif zone_mode == "Zones file":
        zone_files = ui.list_zone_files()
        suggested = ui.suggest_zone_file(video, zone_files)
        if zone_files:
            picked = st.selectbox("Zones file found in the project", zone_files, key=f"zfile_{video.name if video else ''}",
                                  index=zone_files.index(suggested) if suggested in zone_files else 0,
                                  format_func=ui.rel_path)
            zone_path = picked
        zone_upload = st.file_uploader("...or upload a zones.json", type=["json"], key="zone_upload")
        if zone_upload is not None:
            zone_path, err = ui.save_uploaded_zones(video.stem if video else "zones", zone_upload.getvalue())
            if err:
                zone_problem = f"The uploaded zones file is not usable: {err}"
                st.error(zone_problem)
            else:
                st.caption("Using the uploaded zones file.")
        if zone_path is None and not zone_problem:           # "Zones file" is chosen but there is no file to use
            zone_problem = "No zones file chosen: upload a zones.json, draw a rectangle, or set Zone type to None."
            st.warning(zone_problem)

    st.subheader("4. Options")
    privacy = st.toggle("Privacy mode (blur faces)", key="privacy",
                        help="Blurs the top of every person box in all pictures and videos. The analysis is unchanged.")
    with st.expander("Advanced options"):
        device = st.selectbox("Device", list(DEVICE_LABELS), key="device", format_func=DEVICE_LABELS.get)
        config_stride = cached_needs(scenario)["stride"] if scenario else 2      # the preset's own stride (navigation: 1)
        stride = st.number_input("Frame stride", min_value=1, max_value=30, value=config_stride, step=1,
                                 key=f"stride_{config_stride}",              # a preset with another default gets a fresh box
                                 help="Process every Nth frame. Higher is faster but loses fine detail. "
                                      "The default comes from the scenario.")
        max_seconds = st.number_input("Max seconds (0 = all)", min_value=0, max_value=36000, value=0, step=5,
                                      key="max_seconds", help="Only analyse the first S seconds of the video.")
        start_time = st.text_input("Video start time (optional)", key="start_time", placeholder="e.g. 22:30",
                                   help="Clock time of the first frame. Turns on time-of-day rules (active_hours, "
                                        "e.g. the elderly preset's night-time exit) and adds clock times to events.")
        start_time_ok = None
        if start_time.strip():
            try:
                utils.parse_clock(start_time)
                start_time_ok = start_time.strip()
            except ValueError as exc:
                st.warning(f"Start time ignored: {exc}")
        reuse = st.toggle("Reuse cached detections if available", value=True, key="reuse",
                          help="Skips YOLO when the same video was analysed before with the same stride and classes: "
                               "seconds instead of minutes.")
        skip_video = st.toggle("Skip videos (faster)", key="skip_video",
                               help="Do not write annotated.mp4 and highlights.mp4.")
        pose = st.toggle("Pose check", key="pose", help="Second opinion from a pose model for running and loitering events.")
    if pose:
        st.caption(":material/warning: Downloads a 6 MB model the first time it is used (needs internet).")

    st.subheader("Open existing results")
    result_dirs = ui.list_result_dirs()
    st.selectbox("Open existing results", [ui.rel_path(d) for d in result_dirs], index=None, key="open_results",
                 placeholder="Pick an outputs/ or examples/ folder", label_visibility="collapsed",
                 help="Shows finished results without running anything. Good when there is no time to run.")


# --------------------------------------------------------------------------- header and preview

st.title("Aquatic Distress Behaviour Intelligence")
st.caption("HackNex HNX26PSI07 · Autonomous Vision & Behaviour Understanding · the same engine also runs campus, workplace, traffic and crowd presets")
st.markdown(f"**{PITCH}**")

ready = video is not None and facts is not None
out_dir = ui.default_out_dir(video, scenario) if ready and scenario else None
zones_preview: list[dict] = []
if ready and zone_mode == "Rectangle" and not zone_problem:
    zones_preview = rect_zones
elif ready and zone_mode == "Zones file" and zone_path is not None and not zone_problem:
    zones_preview, err = ui.zones_in_video_pixels(zone_path, (facts["width"], facts["height"]))
    if err:
        zone_problem = err
        st.error(err)

preview_col, facts_col = st.columns([3, 2], vertical_alignment="top")
with preview_col:
    if ready:
        frame, scale = cached_frame(str(video), ui.mtime(video))
        if frame is not None:
            try:
                picture = ui.draw_zones(frame, zones_preview, scale)
            except Exception as exc:                  # a strange zone must not take the page down: show the plain frame
                picture, zones_preview = ui.draw_zones(frame, [], scale), []
                st.warning(f"The zone could not be drawn ({type(exc).__name__}: {exc}). The analysis can still run.")
            with st.spinner("Preparing the video player (first time only)..."):
                clip = cached_preview(str(video), ui.mtime(video))
            if clip:
                st.video(clip)
                st.caption("Selected video" + (" · the zone is shown on the first frame below" if zones_preview else ""))
                if zones_preview:
                    with st.expander("Zone on the first frame", expanded=False):
                        st.image(picture, width="stretch", caption="First frame with the zone in red",
                                 alt="First frame of the selected video with the zone outlined in red")
            else:                                   # no ffmpeg / not decodable: show the first frame as before
                st.image(picture, width="stretch",
                         caption="First frame" + (" with the zone in red" if zones_preview else " (no zone)"),
                         alt="First frame of the selected video" + (" with the zone outlined in red" if zones_preview else ""))
            if ui.zones_outside_frame(zones_preview, (facts["width"], facts["height"])):
                st.warning("The zone lies completely outside the video frame, so nothing can ever enter it. "
                           "Check the zone's image_size or move the rectangle.")
        else:
            st.warning("The first frame could not be decoded. run.py will probably fail on this file.")
    else:
        st.info("Pick a sample video or upload one in the sidebar. To look at finished results instead, use "
                "'Open existing results' at the bottom of the sidebar.")
with facts_col:
    with st.container(border=True):
        st.markdown("**Video facts**")
        if ready:
            with st.container(horizontal=True):
                st.metric("Duration", utils.fmt_time(facts["duration_s"]) if facts["duration_s"] else "?")
                st.metric("Frame rate", f"{facts['fps']:.1f} fps" if facts["fps"] else "?")
                st.metric("Size", f"{facts['width']} x {facts['height']}")
            st.caption(f"`{ui.rel_path(video)}`")
            found = find_cached(scenario, video, facts, stride, max_seconds) if reuse and scenario else None
            if found:
                st.caption(f":material/bolt: Cached detections found in `{ui.rel_path(found)}`: YOLO will be skipped.")
            elif scenario:
                est = ui.yolo_seconds(facts, stride, max_seconds)
                st.caption(":material/hourglass_top: " + ("No cached detections" if reuse else "Reuse is off") + ": YOLO will run"
                           + (f" (roughly {est / 60:.1f} min on a laptop CPU, faster on a GPU)." if est > 20 else "."))
                weights = cached_needs(scenario)["weights"]
                if not ui.weights_present(weights):
                    st.caption(f":material/warning: `{weights}` is not in the project folder: Ultralytics would try to "
                               "download it, which needs internet.")
                others =[n for n in names if n != scenario and find_cached(n, video, facts, stride, max_seconds)] if reuse else []
                if others:                            # the quickest way out of a long wait: another preset has a cache
                    st.caption(":material/lightbulb: Cached detections exist for: "
                               + ", ".join(f"**{infos[n]['title']}**" for n in others[:3])
                               + ". Pick one in step 2 to get results in seconds.")
        else:
            st.caption("No video selected.")


# --------------------------------------------------------------------------- run

can_run = ready and bool(scenario) and not zone_problem
run_clicked = st.button("Run analysis", type="primary", icon=":material/play_arrow:", width="stretch",
                        disabled=not can_run, key="run_button")
if can_run:
    st.caption(f"Results will be written to `{out_dir}`.")
elif ready and zone_problem:
    st.caption("Fix the zone in the sidebar to enable Run.")

if run_clicked and can_run:
    run_error = None
    started = time.time()
    try:
        if zone_mode == "Rectangle":
            zone_arg = ui.write_zone_file(video.stem, rect_zones, (facts["width"], facts["height"]))
        else:
            zone_arg = zone_path if zone_mode == "Zones file" else None
        found = find_cached(scenario, video, facts, stride, max_seconds) if reuse else None
        notes = [ui.preserve_detections(out_dir, found),             # never lose expensive detections to another preset
                 ui.seed_detections(found, out_dir)]
        notes = [n for n in notes if n]                              # log lines written by the app itself
        cmd = ui.build_command(video, scenario, out_dir, zones=zone_arg, privacy=privacy, device=device,
                               stride=int(stride) if int(stride) != config_stride else None,
                               max_seconds=float(max_seconds), reuse=reuse, no_video=skip_video, pose=pose,
                               start_time=start_time_ok)
        log = deque(notes, maxlen=12)                                  # only the newest lines are shown live
        with st.status("Running run.py ...", expanded=True) as status:
            st.caption("To cancel, change any setting or press Stop at the top right: the analysis is stopped.")
            bar = st.progress(0.0, text="Starting run.py")
            tail = st.empty()
            steps = ui.Progress()
            drawn = [0.0, 0]                           # when the log box was last redrawn, and how many lines it shows
            count = [len(log)]                         # lines received so far

            def draw_tail(force: bool = False) -> None:
                """Show the newest log lines; at most 5 times a second, so a chatty run cannot flood the browser."""
                if count[0] != drawn[1] and (force or time.time() - drawn[0] > 0.2):
                    drawn[0], drawn[1] = time.time(), count[0]
                    tail.code("\n".join(x[:300] for x in log), language="text")

            def show_label() -> None:
                """While run.py is silent: status title with the current step and the seconds since Run was pressed."""
                status.update(label=f"Running run.py: {steps.label} ({time.time() - started:.0f} s)", expanded=True)
                draw_tail(force=True)                  # also catches lines that arrived inside the 0.2 s window

            def on_line(line: str) -> None:
                """Called for every output line of run.py: move the bar and show the newest lines."""
                log.append(line)
                count[0] += 1
                fraction = steps.feed(line)
                if fraction is not None:
                    bar.progress(fraction, text=steps.label)
                    show_label()
                draw_tail()

            code, lines = ui.run_pipeline(cmd, on_line, on_idle=show_label)
            events_file = ROOT / out_dir / "events.json"
            ok = code == 0 and events_file.is_file() and ui.mtime(events_file) >= started - 2.0
            elapsed = time.time() - started
            if ok:
                bar.progress(1.0, text="Done")
                status.update(label=f"Finished in {elapsed:.1f} s", state="complete", expanded=False)
            else:
                status.update(label=f"run.py failed (exit code {code})", state="error", expanded=True)
        full_log = notes + lines
        st.session_state["last_run"] = {"ok": ok, "code": code, "out_dir": out_dir, "cmd": ui.display_command(cmd),
                                        "interpreter": cmd[0], "lines": full_log[-400:], "seconds": elapsed,
                                        "video": ui.rel_path(video), "scenario": scenario,
                                        "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        st.session_state["_clear_open_results"] = ok
    except Exception as exc:                          # never crash the app because a run failed
        run_error = f"{type(exc).__name__}: {exc}"
        st.session_state["last_run"] = {"ok": False, "code": -1, "out_dir": out_dir, "cmd": "", "interpreter": "",
                                        "lines": [run_error], "seconds": time.time() - started,
                                        "video": ui.rel_path(video), "scenario": scenario,
                                        "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    ui.save_last_run(st.session_state["last_run"])     # so a page reload (a new session) still shows this run
    st.rerun()                                         # redraw with the new results folder and an updated sidebar list

last = st.session_state.get("last_run")
if last:
    if last["ok"]:
        when = f" (at {last['when']})" if last.get("when") else ""
        st.success(f"Last run finished in {last['seconds']:.1f} s{when}: `{last['out_dir']}`", icon=":material/check_circle:")
    else:
        tail_text = ui.last_lines(last["lines"], 20).replace("```", "'''")
        st.error(f"run.py failed (exit code {last['code']}). Last 20 lines:\n```text\n{tail_text}\n```")
    with st.expander("Command used"):
        if last["cmd"]:
            st.code(last["cmd"], language="bash", wrap_lines=True)
            st.caption(f"Run from the project folder with the same Python as this app (`{last['interpreter']}`).")
    with st.expander("Run log"):
        st.code("\n".join(x[:500] for x in last["lines"]) or "(no output)", language="text", wrap_lines=True)


# --------------------------------------------------------------------------- results

picked = st.session_state.get("open_results")
if picked:
    show_results(str(ROOT / picked), "Opened folder")
elif last and last["ok"] and (ROOT / last["out_dir"] / "events.json").is_file():
    if out_dir and last["out_dir"] != out_dir:      # the sidebar moved on: do not let old results pass for the new choice
        st.info(f"The results below belong to the last run (`{last['out_dir']}`), not to the video and scenario "
                f"selected now (`{out_dir}`). Press Run analysis to analyse the current selection.",
                icon=":material/history:")
    show_results(str(ROOT / last["out_dir"]), "Last run")
else:
    st.caption("Results appear here after a run, or open a finished folder from the sidebar.")
