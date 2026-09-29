"""
voice_commands.py
------------------
Voice recognition for EVA (ported from the voice demo, engine reworked).

Accuracy / performance changes vs the demo:
  * ONE persistent microphone stream (no re-opening the device per attempt).
  * Listening and recognizing run in separate threads, so speech spoken while
    a previous phrase is still being transcribed is not lost.
  * Google returns several alternative transcripts; ALL are matched, not just
    the first.
  * Stricter matching (exact word -> alias -> fuzzy with 0.75 cutoff).
"""

import difflib
import os
import queue
import threading

import speech_recognition as sr

COMMANDS = {
    "browser": "BROWSER", "media": "MEDIA", "files": "FILES", "file": "FILES",
    "notes": "NOTES", "note": "NOTES", "camera": "CAMERA",
}

ALIASES = {
    "browser": ["browsers", "brouser", "browse"],
    "media": ["medium", "medi", "video", "music"],
    "files": ["fells", "fills", "phials"],
    "notes": ["nodes", "knotes", "nots"],
    "camera": ["cameras", "camara", "cam", "kamera"],
}
_ALIAS_TO_CMD = {a: COMMANDS[k] for k, v in ALIASES.items() for a in v}
_FUZZY_TARGETS = ["browser", "media", "files", "notes", "camera"]
DEFAULT_LANGUAGE = os.environ.get("EVA_VOICE_LANG", "en-IN")


def command_to_id(text):
    """Map a transcript to BROWSER/MEDIA/FILES/NOTES/CAMERA, BACK (-> gaze mode)
    or VOICE_MODE (-> voice mode); None if nothing matches.
    "camera" and "open camera" both work (same for every button)."""
    if not text:
        return None
    words = " ".join(text.lower().replace("-", " ").split()).split()
    if not words:
        return None
    joined = " ".join(words)
    if len(words) <= 4:                   # mode switches: short utterances only
        if "voice" in words:
            return "VOICE_MODE"
        if "go back" in joined or "gaze" in words or words == ["back"]:
            return "BACK"
    for w in words:                      # 1. exact command word
        if w in COMMANDS:
            return COMMANDS[w]
    for w in words:                      # 2. known mishearing
        if w in _ALIAS_TO_CMD:
            return _ALIAS_TO_CMD[w]
    for w in words:                      # 3. fuzzy (e.g. "browsr")
        if len(w) < 3:
            continue
        m = difflib.get_close_matches(w, _FUZZY_TARGETS, n=1, cutoff=0.75)
        if m:
            return COMMANDS[m[0]]
    return None


class VoiceEngine:
    """
    start() returns immediately; mic calibration happens in the background.
    on_event(kind, data):
        "status":  data = "starting" | "listening" | "error"  (+ message via "error")
        "command": data = (command_id, heard_text)
        "heard":   data = text that matched no command
        "error":   data = message
    """

    def __init__(self, on_event, language=DEFAULT_LANGUAGE, microphone_index=None):
        self.on_event = on_event
        self.language = language
        self.microphone_index = microphone_index
        self._stop = threading.Event()
        self._audio_q = queue.Queue(maxsize=4)
        self._threads = []
        self.running = False

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self.on_event("status", "starting")
        for target in (self._listen_loop, self._recognize_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        self.running = False
        try:
            self._audio_q.put_nowait(None)
        except queue.Full:
            pass
        self._threads = []

    # -- listener thread: keeps the mic open, cuts phrases ------------------
    def _listen_loop(self):
        r = sr.Recognizer()
        r.dynamic_energy_threshold = True
        r.energy_threshold = 300
        r.pause_threshold = 0.6
        r.phrase_threshold = 0.1
        r.non_speaking_duration = 0.3
        try:
            with sr.Microphone(device_index=self.microphone_index) as source:
                r.adjust_for_ambient_noise(source, duration=1.0)
                r.energy_threshold = min(max(r.energy_threshold, 200), 900)
                self.on_event("status", "listening")
                while not self._stop.is_set():
                    try:
                        audio = r.listen(source, timeout=1, phrase_time_limit=3)
                    except sr.WaitTimeoutError:
                        continue
                    try:
                        self._audio_q.put_nowait(audio)
                    except queue.Full:
                        pass          # drop oldest-unprocessed burst rather than lag
        except Exception as exc:      # no mic / PyAudio missing
            self.running = False
            self.on_event("error", f"Microphone error: {exc}")

    # -- recognizer thread: network call, off the audio path ----------------
    def _recognize_loop(self):
        r = sr.Recognizer()
        while not self._stop.is_set():
            try:
                audio = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if audio is None or self._stop.is_set():
                break
            try:
                result = r.recognize_google(audio, language=self.language, show_all=True)
            except sr.RequestError as exc:
                self.on_event("error", f"Speech service/network error: {exc}")
                continue
            except Exception:
                continue

            alts = result.get("alternative", []) if isinstance(result, dict) else []
            texts = [a.get("transcript", "").lower().strip() for a in alts if a.get("transcript")]
            if not texts:
                continue
            for text in texts:                       # best alternative first
                cmd = command_to_id(text)
                if cmd:
                    self.on_event("command", (cmd, text))
                    break
            else:
                self.on_event("heard", texts[0])
