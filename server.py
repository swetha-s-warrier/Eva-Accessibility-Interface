import os
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from backend.action_mapper import ActionDispatcher
import time

app = Flask(__name__, static_folder=".")
CORS(app)

dispatcher = ActionDispatcher(cooldown_seconds=2.0)

# --- live gaze state, shared between the gaze module (main.py) and
# the browser frontend polling for it ---------------------------------
_gaze_lock = threading.Lock()
_gaze_state = {
    "detected": False,
    "x": None,
    "y": None,
    "target": None,
    "progress": 0.0,
    "selected_target": None,   # one-shot: cleared once the frontend reads it
}


# --- APP STATE MACHINE:  CALIBRATION_IDLE -> CALIBRATING -> ACTIVE ----------
# The server is the single source of truth. Gaze selections are honoured
# ONLY while state == ACTIVE. `epoch` increments on every transition so
# main.py can drop any stale dwell/gaze data from a previous state.
_state_lock = threading.Lock()
_app = {"state": "CALIBRATION_IDLE", "epoch": 0}


def _set_state(new_state):
    with _state_lock:
        if _app["state"] != new_state:
            _app["state"] = new_state
            _app["epoch"] += 1
    with _gaze_lock:  # clear stale gaze/dwell/selection left over from before
        _gaze_state.update({"detected": False, "x": None, "y": None, "target": None,
                            "progress": 0.0, "selected_target": None})


# --- INPUT CHANNEL: exactly one of "gaze" / "voice" is live at a time -------
_mode = {"value": "gaze"}
_voice_lock = threading.Lock()
_voice = {"status": "off", "heard": "", "message": "", "event": None}
_voice_engine = {"obj": None}


def _bump_epoch():
    with _state_lock:
        _app["epoch"] += 1
    with _gaze_lock:
        _gaze_state.update({"detected": False, "x": None, "y": None, "target": None,
                            "progress": 0.0, "selected_target": None})


def _switch_mode(mode):
    """Single place where the input channel changes (button, gaze dwell or voice)."""
    _mode["value"] = mode
    _bump_epoch()                       # drop stale gaze/dwell from the other channel
    with _voice_lock:
        _voice["event"] = None


def _ensure_voice_engine():
    """Mic stays open in BOTH modes: in gaze mode it only listens for
    'voice mode'; in voice mode it accepts the button commands + 'go back'."""
    try:
        from voice.voice_commands import VoiceEngine
        if _voice_engine["obj"] is None:
            _voice_engine["obj"] = VoiceEngine(_on_voice_event)
        if not _voice_engine["obj"].running:
            with _voice_lock:
                _voice.update({"status": "starting", "message": ""})
            _voice_engine["obj"].start()
    except ImportError as exc:
        with _voice_lock:
            _voice.update({"status": "error",
                           "message": f"Voice needs SpeechRecognition + PyAudio ({exc})"})


def _on_voice_event(kind, data):
    with _voice_lock:
        if kind == "status":
            _voice["status"] = data
            _voice["message"] = ""
            return
        if kind == "error":
            _voice["status"], _voice["message"] = "error", str(data)
            return
        if kind == "heard":
            _voice["heard"] = data
            return
        if kind != "command":
            return
        cmd, heard = data
        _voice["heard"] = heard

    if get_app_state()[0] == "CALIBRATING":      # calibration is gaze-only
        return

    if _mode["value"] == "gaze":
        if cmd != "VOICE_MODE":                  # gaze mode: only the wake command counts
            return
        _switch_mode("voice")
        print("\n[VOICE] 'voice mode' -> switched to VOICE")
        result = {"status": "success", "action": "VOICE_MODE", "message": "Voice mode on"}
    else:
        if cmd == "VOICE_MODE":
            return
        if cmd == "BACK":
            _switch_mode("gaze")
            print("\n[VOICE] 'go back' -> switched to GAZE")
            result = {"status": "success", "action": "BACK", "message": "Gaze mode on"}
        else:
            print(f"\n[VOICE] '{heard}' -> {cmd}")
            result = dispatcher.execute_action(cmd)
            print(f"[ACTION RESULT] {result}")
    with _voice_lock:
        _voice["event"] = {"command": cmd, "heard": heard, "result": result}


@app.route("/api/mode", methods=["POST"])
def set_mode():
    mode = (request.get_json(silent=True) or {}).get("mode")
    if mode not in ("gaze", "voice"):
        return jsonify({"status": "error", "message": "mode must be gaze or voice"}), 400
    if get_app_state()[0] == "CALIBRATING":
        return jsonify({"status": "ignored", "mode": _mode["value"]}), 409
    _switch_mode(mode)
    _ensure_voice_engine()
    return jsonify({"status": "ok", "mode": mode})


@app.route("/api/voice", methods=["GET"])
def get_voice():
    with _voice_lock:
        out = {"status": _voice["status"], "heard": _voice["heard"],
               "message": _voice["message"], "event": _voice["event"]}
        _voice["event"] = None          # one-shot
    out["mode"] = _mode["value"]
    return jsonify(out)


def get_app_state():
    with _state_lock:
        return _app["state"], _app["epoch"]


@app.route("/api/state", methods=["GET"])
def api_state():
    st, ep = get_app_state()
    return jsonify({"state": st, "epoch": ep, "mode": _mode["value"]})


# Real on-screen tile rectangles, reported by the page (normalized to the
# same box the calibration dots use). main.py hit-tests against these.
_layout = {"tiles": None}


@app.route("/api/layout", methods=["POST"])
def set_layout():
    data = request.get_json(silent=True) or {}
    tiles = data.get("tiles")
    if isinstance(tiles, dict) and tiles:
        _layout["tiles"] = tiles
    return jsonify({"status": "ok"})


# Serve the HTML frontend
@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")


# Manual trigger endpoint — used by the browser UI's own mouse/keyboard
# fallback input (Simulated gaze / Keyboard only modes).
@app.route("/api/trigger", methods=["POST"])
def trigger_action():
    data = request.get_json(silent=True) or {}
    target = data.get("target", "").upper()

    if get_app_state()[0] == "CALIBRATING":
        return jsonify({"status": "ignored", "message": "Calibration in progress"}), 409

    print(f"\n[HTTP REQUEST] Received trigger for: {target}")
    result = dispatcher.execute_action(target)
    print(f"[ACTION RESULT] {result}")

    status_code = 400 if result.get("status") == "error" else 200
    return jsonify(result), status_code


# Called continuously by main.py (the real gaze pipeline) with the
# latest coordinates + dwell progress. When a dwell completes here,
# the action is executed immediately — this is the live-gaze path.
@app.route("/api/gaze", methods=["POST"])
def update_gaze():
    data = request.get_json(silent=True) or {}
    target = data.get("target")
    selected = bool(data.get("selected", False))

    st, ep = get_app_state()
    # Ignore anything not ACTIVE, or stamped with an old epoch (stale event).
    if st != "ACTIVE" or _mode["value"] != "gaze" or data.get("epoch") != ep:
        return jsonify({"status": "ignored", "state": st, "action_result": None})

    action_result = None
    if selected and target and str(target).upper() == "MODE_VOICE":
        print("\n[GAZE SELECT] Dwell completed on: Voice mode button")
        _switch_mode("voice")
        _ensure_voice_engine()
        return jsonify({"status": "ok", "action_result": {"status": "success", "action": "MODE_VOICE"}})
    if selected and target:
        print(f"\n[GAZE SELECT] Dwell completed on: {target}")
        action_result = dispatcher.execute_action(str(target).upper())
        print(f"[ACTION RESULT] {action_result}")

    with _gaze_lock:
        _gaze_state["detected"] = bool(data.get("detected", False))
        _gaze_state["x"] = data.get("x")
        _gaze_state["y"] = data.get("y")
        _gaze_state["target"] = target
        _gaze_state["progress"] = data.get("progress", 0.0)
        if selected:
            _gaze_state["selected_target"] = target

    return jsonify({"status": "ok", "action_result": action_result})


# Polled by frontend.html a few times a second to render the live
# gaze cursor + dwell ring + trigger the "selected" flash animation.
@app.route("/api/gaze", methods=["GET"])
def get_gaze():
    with _gaze_lock:
        state = dict(_gaze_state)
        state["justSelected"] = _gaze_state["selected_target"]
        _gaze_state["selected_target"] = None  # one-shot, so it fires once
    state["app_state"], state["epoch"] = get_app_state()
    if state["app_state"] != "ACTIVE" or _mode["value"] != "gaze":
        state.update({"target": None, "progress": 0.0, "justSelected": None})
    return jsonify(state)


# --- in-browser calibration, driven by the "Calibration" tab -----------
# Fixed 5 points, matching the frontend's calib-dot positions exactly
# (top-left, top-right, center, bottom-left, bottom-right) so the
# sample -> target pairing lines up with what's on screen.
CALIB_POINTS = [(0.08, 0.12), (0.92, 0.12), (0.50, 0.50), (0.08, 0.88), (0.92, 0.88)]

_calib_lock = threading.Lock()
_calib_state = {
    "active": False,
    "point_index": None,
    "samples": {i: [] for i in range(len(CALIB_POINTS))},
    "just_finished": False,   # one-shot: main.py polls this to know when to hot-reload
    "last_result": None,
}


@app.route("/api/calibration/start", methods=["POST"])
def calibration_start():
    with _calib_lock:
        _calib_state["active"] = True
        _calib_state["point_index"] = None
        _calib_state["samples"] = {i: [] for i in range(len(CALIB_POINTS))}
        _calib_state["just_finished"] = False
        _calib_state["last_result"] = None
    _set_state("CALIBRATING")
    return jsonify({"status": "ok", "points": CALIB_POINTS})

# --- browser-closed detection --------------------------------------
# frontend.html pings this every ~2s while the tab is open. main.py
# watches for the gap between pings growing too large (tab closed)
# and stops itself automatically instead of needing Ctrl+C. A quick
# page refresh only causes a brief gap, well under the timeout, so it
# won't falsely trigger a stop.
_heartbeat_lock = threading.Lock()
_last_heartbeat = {"time": time.time()}

@app.route("/api/heartbeat", methods=["POST"])
def heartbeat():
    with _heartbeat_lock:
        _last_heartbeat["time"] = time.time()
    return jsonify({"status": "ok"})


def seconds_since_heartbeat():
    with _heartbeat_lock:
        return time.time() - _last_heartbeat["time"]


# Called by the frontend as each dot lights up, so main.py knows which
# known screen point the samples it's collecting right now belong to.
@app.route("/api/calibration/point", methods=["POST"])
def calibration_point():
    data = request.get_json(silent=True) or {}
    index = data.get("index")
    with _calib_lock:
        if _calib_state["active"] and isinstance(index, int) and 0 <= index < len(CALIB_POINTS):
            _calib_state["point_index"] = index
    return jsonify({"status": "ok"})


# Polled by main.py every frame during calibration: which point (if
# any) it should be collecting raw gaze samples for right now.
@app.route("/api/calibration/status", methods=["GET"])
def calibration_status():
    with _calib_lock:
        idx = _calib_state["point_index"]
        state = {
            "active": _calib_state["active"],
            "point_index": idx,
            "target": list(CALIB_POINTS[idx]) if (_calib_state["active"] and idx is not None) else None,
            "sample_count": len(_calib_state["samples"].get(idx, [])) if idx is not None else 0,
            "just_finished": _calib_state["just_finished"],
            "last_result": _calib_state["last_result"],
        }
        _calib_state["just_finished"] = False  # one-shot
    state["app_state"], state["epoch"] = get_app_state()
    state["mode"] = _mode["value"]
    state["layout"] = _layout["tiles"]
    return jsonify(state)


# main.py posts one raw (base-model, pre-correction) gaze sample here
# per frame while a calibration point is active.
@app.route("/api/calibration/sample", methods=["POST"])
def calibration_sample():
    data = request.get_json(silent=True) or {}
    raw_x, raw_y = data.get("raw_x"), data.get("raw_y")
    with _calib_lock:
        idx = _calib_state["point_index"]
        if _calib_state["active"] and idx is not None and raw_x is not None and raw_y is not None:
            _calib_state["samples"][idx].append((float(raw_x), float(raw_y)))
    return jsonify({"status": "ok"})


# Called by the frontend once all 5 dots are done: fits the linear
# calibration correction on the collected samples and saves it, so
# main.py's next /api/calibration/status poll can tell it to reload —
# no restart needed.
@app.route("/api/calibration/finish", methods=["POST"])
def calibration_finish():
    import numpy as np
    from gaze.calibration import CalibrationManager

    with _calib_lock:
        samples = _calib_state["samples"]
        raw_points, target_points = [], []
        for i, pts in samples.items():
            for (rx, ry) in pts:
                raw_points.append((rx, ry))
                target_points.append(CALIB_POINTS[i])
        _calib_state["active"] = False
        _calib_state["point_index"] = None

    if len(raw_points) < 5:
        result = {"status": "error",
                   "message": "Not enough samples collected — hold still on each dot and try again."}
        with _calib_lock:
            _calib_state["last_result"] = result
            _calib_state["just_finished"] = True
        _set_state("CALIBRATION_IDLE")
        return jsonify(result), 400

    calib = CalibrationManager()
    calib.fit(raw_points, target_points)

    preds = np.array([calib.predict(rx, ry) for rx, ry in raw_points])
    mae = float(np.mean(np.abs(preds - np.array(target_points))))

    os.makedirs("models", exist_ok=True)
    calib.save("models/calibration_model.pkl")

    result = {"status": "ok", "samples": len(raw_points), "mae": round(mae, 4)}
    with _calib_lock:
        _calib_state["last_result"] = result
        _calib_state["just_finished"] = True
    _set_state("ACTIVE")   # only after the new calibration is saved
    return jsonify(result)


if __name__ == "__main__":
    print("\n=======================================================")
    print("  EVA System Server Running")
    print("  Open your browser to: http://127.0.0.1:5000")
    print("  Then set 'Input: Live gaze (webcam)' and run main.py")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
