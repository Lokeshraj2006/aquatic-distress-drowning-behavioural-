"""Spoken alerts: "Attention lifeguard. Swimmer #3 may be in distress at the Deep End. Risk 92 percent."

For every alert event (pool distress / submersion by default, see config `announce`), this module
  1. writes the message text,
  2. makes an audio file alerts/alert_e<id>.wav = a short alarm tone + the spoken message
     (Windows: the built-in System.Speech voices, offline; macOS: `say`; Linux: `espeak`; if no voice
     is available the file is the alarm tone only),
  3. and, when asked (run.py --announce), plays it on this computer's speaker `repeat` times.
The web UI plays the same file in the browser, so the alert comes out of the dashboard device's speaker.

No extra Python packages: NumPy + the standard library (wave, subprocess, winsound on Windows).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import utils

RATE = 22050


def alert_events(final: dict, cfg: dict) -> list[dict]:
    """The events that deserve a spoken alert (behaviour listed in announce.behaviors, risk >= min_risk)."""
    acfg = cfg.get("announce") or {}
    wanted = set(acfg.get("behaviors") or [])
    min_risk = float(acfg.get("min_risk", 0.7))
    out = []
    for ev in final.get("events") or []:
        risk = float((ev.get("metrics") or {}).get("risk_score", ev.get("confidence") or 0.0))
        if ev.get("behavior") in wanted and risk >= min_risk:
            out.append(ev)
    return out


def message_for(ev: dict, cfg: dict) -> str:
    """The sentence to speak for one event (templates in config announce.message / submersion_message)."""
    acfg = cfg.get("announce") or {}
    m = ev.get("metrics") or {}
    who = (ev.get("entity_name") or utils.entity_name(cfg, ev.get("entity_id"))).replace("#", "number ")
    risk = int(round(100 * float(m.get("risk_score", ev.get("confidence") or 0.0))))
    template = acfg.get("submersion_message") if ev.get("behavior") == "submersion" else acfg.get("message")
    template = template or "Attention. {who} may be in danger at the {location}."
    return template.format(who=who, location=m.get("location") or ev.get("zone") or "pool", risk=risk)


# --------------------------------------------------------------------------- audio

def tone(seconds: float = 1.2) -> np.ndarray:
    """Two-tone alarm (int16 samples): 880 Hz / 660 Hz, three times."""
    t = np.arange(int(RATE * seconds / 6)) / RATE
    beep = lambda f: (0.45 * np.sin(2 * np.pi * f * t) * np.minimum(1, np.minimum(t, t[::-1]) * 40)).astype(np.float32)
    pattern = np.concatenate([beep(880), beep(660)] * 3)
    return (pattern * 32767).astype(np.int16)


def _read_wav(path) -> np.ndarray | None:
    """int16 mono samples of a WAV at RATE, or None if the format differs."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != RATE or w.getsampwidth() != 2:
            return None
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        return data


def _write_wav(path, samples: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.astype(np.int16).tobytes())


def synthesize(text: str, path, cfg: dict | None = None) -> str:
    """Speak `text` into a WAV file at RATE. Returns the method used ('windows', 'say', 'espeak') or '' if none."""
    acfg = (cfg or {}).get("announce") or {}
    path = str(path)
    system = platform.system()
    try:
        if system == "Windows":
            env = dict(os.environ, PS07_TTS_TEXT=text, PS07_TTS_PATH=path,
                       PS07_TTS_VOICE=str(acfg.get("voice") or ""), PS07_TTS_RATE=str(int(acfg.get("rate", 0))))
            script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "if ($env:PS07_TTS_VOICE) { try { $s.SelectVoice($env:PS07_TTS_VOICE) } catch {} }; "
                      "$s.Rate = [int]$env:PS07_TTS_RATE; "
                      "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, "
                      "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
                      "$s.SetOutputToWaveFile($env:PS07_TTS_PATH, $f); $s.Speak($env:PS07_TTS_TEXT); $s.Dispose()")
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], env=env,
                           capture_output=True, timeout=60, check=True)
            return "windows" if Path(path).exists() and Path(path).stat().st_size > 1000 else ""
        if system == "Darwin" and shutil.which("say"):
            subprocess.run(["say", "--data-format=LEI16@22050", "-o", path, text], capture_output=True, timeout=60, check=True)
            return "say"
        if shutil.which("espeak"):
            subprocess.run(["espeak", "-w", path, text], capture_output=True, timeout=60, check=True)
            return "espeak"
    except Exception:
        return ""
    return ""


def make_alert_audio(text: str, path, cfg: dict | None = None, speech: bool = True) -> str:
    """alarm tone + spoken message -> one WAV. Returns the method ('tone' when no voice was available)."""
    parts = [tone(), np.zeros(int(RATE * 0.25), dtype=np.int16)]
    method = "tone"
    if speech:
        with tempfile.TemporaryDirectory() as tmp:
            spoken = Path(tmp) / "speech.wav"
            used = synthesize(text, spoken, cfg)
            if used and spoken.exists():
                data = _read_wav(spoken)
                if data is not None and len(data):
                    parts.append(data)
                    method = used
    _write_wav(path, np.concatenate(parts))
    return method


def prepare(final: dict, cfg: dict, out_dir, speech: bool = True) -> list[dict]:
    """Write alerts/alert_e<id>.wav + announcements.json for every alert event; attach `announcement` to it."""
    out_dir = Path(out_dir)
    records = []
    for ev in alert_events(final, cfg):
        text = message_for(ev, cfg)
        rel = f"alerts/alert_e{ev['event_id']}.wav"
        method = make_alert_audio(text, out_dir / rel, cfg, speech=speech)
        ev["announcement"] = {"text": text, "audio": rel, "voice": method}
        records.append({"event_id": ev["event_id"], "entity_id": ev["entity_id"], "behavior": ev["behavior"],
                        "time": ev.get("start"), "text": text, "audio": rel, "voice": method})
    utils.write_json(out_dir / "announcements.json", records)
    return records


def play(path, repeat: int = 1) -> bool:
    """Play a WAV on this computer's speaker (blocking). Returns False if no player was found."""
    path = str(path)
    system = platform.system()
    for _ in range(max(1, int(repeat))):
        try:
            if system == "Windows":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME)
            elif system == "Darwin" and shutil.which("afplay"):
                subprocess.run(["afplay", path], check=True)
            elif shutil.which("aplay"):
                subprocess.run(["aplay", "-q", path], check=True)
            else:
                return False
        except Exception:
            return False
    return True


def speak_all(records: list[dict], cfg: dict, out_dir) -> int:
    """Play every prepared announcement (most urgent first: submersion, then by time). Returns how many played."""
    repeat = int((cfg.get("announce") or {}).get("repeat", 2))
    order = sorted(records, key=lambda r: (r["behavior"] != "submersion", r["event_id"]))
    played = 0
    for r in order:
        print(f"      ANNOUNCING: {r['text']}")
        if play(Path(out_dir) / r["audio"], repeat):
            played += 1
    return played
