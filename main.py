
"""
main.py
--------
SINGLE-COMMAND FULL MERGE. This one script:

  1. Starts the Flask backend (server.py) in a background thread.
  2. Opens your browser to the EVA website, already switched into
     "Live gaze (webcam)" mode — nothing to click.
  3. Runs the real webcam gaze pipeline (dataset-trained base model +
     your per-session calibration correction), streaming live
     coordinates + dwell progress to that same backend, which renders
     them on the page and fires the actual action when a dwell
     completes.
  4. Also listens for the website's own "Calibration" tab: clicking
     "Start calibration" there collects real webcam samples through
     this script and fits + saves a new calibration correction —
     calibrate.py is no longer a required separate step.

You only ever run:

    python main.py

A default calibration is shipped (models/calibration_model.pkl), so
this works out of the box; use the website's Calibration tab any time
to redo it for your seating position — it hot-reloads automatically,
no restart needed. (calibrate.py still exists as a standalone
alternative if you'd rather calibrate from the terminal.)

There is no separate debug window — the browser tab is the whole
interface. Stop this script with Ctrl+C in the terminal when you're
done.
"""

import threading
import time
import webbrowser

import cv2
import requests

import server  # the Flask app + /api/gaze + /api/trigger routes
from gaze.landmark_detection import FaceMeshDetector
from gaze.feature_extraction import extract_features
from gaze.gaze_model import GazeModel
from gaze.smoothing import GazeSmoother
from gaze.dwell import DwellDetector

HEARTBEAT_TIMEOUT = 6.0  # seconds of silence from the browser before we stop
DWELL_SECONDS = 2.0
BASE_MODEL_PATH = "models/base_gaze_model.pkl"
CALIBRATION_PATH = "models/calibration_model.pkl"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
BACKEND_GAZE_URL = f"{BACKEND_URL}/api/gaze"
BACKEND_CALIB_STATUS_URL = f"{BACKEND_URL}/api/calibration/status"
BACKEND_CALIB_SAMPLE_URL = f"{BACKEND_URL}/api/calibration/sample"

# Same 5 regions as the website's tile grid (rough approximation —
# good enough to tell which tile is being looked at).
BUTTONS = {
    "BROWSER": (0.05, 0.10, 0.47, 0.45),
    "MEDIA":   (0.53, 0.10, 0.95, 0.45),
    "FILES":   (0.05, 0.50, 0.47, 0.85),
    "NOTES":   (0.53, 0.50, 0.95, 0.85),
    "CAMERA":  (0.30, 0.87, 0.70, 0.97),
}

DEBUG = True            # prints: Raw | Calibrated | Smoothed | Target | Dwell | State
DEBUG_EVERY = 0.25      # seconds between debug lines

session = requests.Session()


LAYOUT = {}   # real tile rects from the page; refreshed every status poll


def which_button(x, y):
    rects = LAYOUT or BUTTONS
    # tiles first; the top-right Voice-mode button only wins where no tile is
    for name in [n for n in rects if n != "MODE_VOICE"] + (["MODE_VOICE"] if "MODE_VOICE" in rects else []):
        x1, y1, x2, y2 = rects[name]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


def send_gaze_update(detected, x=None, y=None, target=None, progress=0.0, selected=False, epoch=None):
    """Streams the current frame's gaze state to the backend so the
    website can render it live. Best-effort — never blocks/crashes
    the CV loop if server.py isn't running yet."""
    try:
        session.post(BACKEND_GAZE_URL, json={
            "detected": detected, "x": x, "y": y,
            "target": target, "progress": progress, "selected": selected,
            "epoch": epoch,
        }, timeout=0.3)
    except requests.exceptions.RequestException:
        pass


def poll_calibration_status():
    """Asks the backend whether the "Start calibration" flow on the
    website is currently in progress and, if so, which known point
    it's on. Best-effort — returns None if unreachable."""
    try:
        resp = session.get(BACKEND_CALIB_STATUS_URL, timeout=0.3)
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def send_calibration_sample(raw_x, raw_y):
    """Reports one raw (base-model, pre-correction) gaze sample for
    whichever calibration point is currently active on the backend."""
    try:
        session.post(BACKEND_CALIB_SAMPLE_URL, json={"raw_x": raw_x, "raw_y": raw_y}, timeout=0.3)
    except requests.exceptions.RequestException:
        pass


def start_backend_in_background():
    """Runs server.py's Flask app in a daemon thread so this one
    script is the only thing you have to launch, then waits until
    it's actually answering requests before continuing."""
    thread = threading.Thread(
        target=server.app.run,
        kwargs=dict(host=BACKEND_HOST, port=BACKEND_PORT, debug=False,
                    threaded=True, use_reloader=False),
        daemon=True,
    )
    thread.start()

    for _ in range(50):  # up to ~5s
        try:
            requests.get(BACKEND_URL, timeout=0.2)
            return
        except requests.exceptions.RequestException:
            time.sleep(0.1)
    print("[WARN] Backend didn't respond in time — continuing anyway.")


def main():
    print("Starting EVA backend + website...")
    start_backend_in_background()

    # Opens the site pre-switched into live-gaze mode via ?mode=live
    # (frontend.html reads this on load) — nothing to click.
    webbrowser.open(f"{BACKEND_URL}/?mode=live")

    detector = FaceMeshDetector()
    gaze_model = GazeModel(BASE_MODEL_PATH, CALIBRATION_PATH)
    smoother = GazeSmoother()  # alpha=0.35, max_jump=0.35 — see smoothing.py
    dwell = DwellDetector(dwell_time=DWELL_SECONDS)

    cap = cv2.VideoCapture(1)  # 0=default webcam, 1=next, etc. — change if needed
    # cv2.namedWindow("EVA - Gaze Debug Feed", cv2.WINDOW_NORMAL)

    print("Gaze module running — the website should have opened in your browser.")
    print("Use the website's 'Calibration' tab any time to (re)calibrate live —")
    print("no need to run a separate script.")
    print("Press Ctrl+C in this terminal to stop.")

    state, epoch = "CALIBRATION_IDLE", -1
    last_dbg = 0.0

    def reset_gaze_pipeline():
        smoother.reset()
        dwell.reset()

    try:
        while server.seconds_since_heartbeat() < HEARTBEAT_TIMEOUT:
            ok, frame = cap.read()
            if not ok:
                break

            # ---- app state (server is the source of truth) --------------
            calib_status = poll_calibration_status() or {}
            if calib_status.get("layout"):
                LAYOUT.clear()
                LAYOUT.update({k.upper(): tuple(v) for k, v in calib_status["layout"].items()})
            new_state = calib_status.get("app_state", state)
            new_epoch = calib_status.get("epoch", epoch)
            if new_state != state or new_epoch != epoch:
                # ANY state change wipes target/dwell/smoothing so nothing
                # stale from before can fire after it.
                reset_gaze_pipeline()
                state, epoch = new_state, new_epoch
                print(f"[STATE] -> {state}")

            # calibration finished: hot-reload BEFORE any ACTIVE processing
            if calib_status.get("just_finished"):
                result = calib_status.get("last_result") or {}
                if result.get("status") == "ok":
                    gaze_model.reload_calibration()
                    reset_gaze_pipeline()
                    print(f"[CALIBRATION] Reloaded — fit error {result.get('mae')} "
                          f"from {result.get('samples')} samples.")
                else:
                    print(f"[CALIBRATION] Failed: {result.get('message')}")

            # ---- voice mode: gaze pipeline fully OFF -------------------------
            if calib_status.get("mode") == "voice" and state != "CALIBRATING":
                continue

            # ---- CALIBRATION_IDLE: gaze completely OFF ------------------
            if state == "CALIBRATION_IDLE":
                continue

            landmarks = detector.get_landmarks(frame)
            features = extract_features(landmarks) if landmarks is not None else None

            # ---- CALIBRATING: only collect raw samples for the lit dot ---
            if state == "CALIBRATING":
                if (features is not None and calib_status.get("active")
                        and calib_status.get("target") is not None):
                    raw_x, raw_y = gaze_model.predict_raw(features)
                    send_calibration_sample(raw_x, raw_y)
                continue   # no smoothing, no target, no dwell, no actions

            # ---- ACTIVE: raw -> calibrated -> smooth -> valid -> target -> dwell
            if features is None:
                # lost face: treated as an invalid frame (dwell holds briefly, then drops)
                stable, progress, selected = dwell.update(None, valid=False)
                send_gaze_update(False, epoch=epoch)
                continue

            raw_x, raw_y = gaze_model.predict_raw(features)
            cal_x, cal_y = gaze_model.calibrate(raw_x, raw_y)
            x, y, valid = smoother.update(cal_x, cal_y)

            if x is None:
                send_gaze_update(False, epoch=epoch)
                continue

            raw_target = which_button(x, y)
            stable, progress, selected = dwell.update(raw_target, valid)

            send_gaze_update(True, x, y, stable, progress, selected, epoch=epoch)
            if selected:
                print(f"[SELECTED] {stable}")

            if DEBUG and time.time() - last_dbg > DEBUG_EVERY:
                last_dbg = time.time()
                print(f"Raw({raw_x:.2f},{raw_y:.2f}) | Cal({cal_x:.2f},{cal_y:.2f}) | "
                      f"Smooth({x:.2f},{y:.2f}) | Tgt {raw_target}->{stable} | "
                      f"Dwell {progress:.0%} | {state} | {'ok' if valid else 'REJECTED'}")
    except KeyboardInterrupt:
        print("\nStopping gaze module...")

    if server.seconds_since_heartbeat() >= HEARTBEAT_TIMEOUT:
        print("\nBrowser tab closed — stopping gaze module...")

    cap.release()
    detector.close()


if __name__ == "__main__":
    main()