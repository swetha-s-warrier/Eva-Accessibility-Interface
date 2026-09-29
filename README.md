# EVA — Full Merge, Single Command (dataset-trained gaze)

`main.py` does everything: starts the backend, opens the website
already in live-gaze mode, and runs the real webcam gaze pipeline —
one script, one terminal.

```
python main.py
  ├─ starts server.py's Flask app in a background thread
  ├─ opens http://127.0.0.1:5000/?mode=live in your browser
  └─ runs the webcam gaze loop:
       webcam -> MediaPipe -> features
       -> BASE MODEL (MLP trained on your dataset)  -> rough (x, y)
       -> CALIBRATION (5-point linear correction)    -> corrected (x, y)
       -> smoothing -> which tile -> dwell -> POST /api/gaze
            |
            ▼
     server.py stores the state + fires the real action on dwell-complete
            |
            ▼
     frontend.html polls /api/gaze every 100ms and renders the live
     gaze cursor + dwell ring + selection flash
```

## What's new vs the previous merge
- **In-browser calibration**: the website's "Calibration" tab (5 dots)
  is now wired to the real pipeline. Clicking "Start calibration"
  collects actual webcam samples through `main.py` for each dot,
  fits + saves a new correction, and `main.py` hot-reloads it — no
  restart, and `calibrate.py` is no longer a required step.
- **Two-stage gaze model**: `gaze/base_gaze_model.py` wraps an MLP
  trained on your dataset (`dataset/prepare_dataset.py` ->
  `dataset/train_base_model.py` -> `models/base_gaze_model.pkl`).
  `gaze/calibration.py` is a small linear correction on top of that
  base model's output.
- Backend/frontend action-trigger plumbing and the single-command /
  live-gaze-mode wiring are unchanged from the previous merge.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows   |   source venv/bin/activate  (Mac/Linux)
pip install -r requirements.txt
```

## Run the demo (1 terminal)
```bash
python main.py
```
That's the only command you need. Your browser opens automatically
straight into the **Calibration** screen (nothing runs until you
click **Start calibration** yourself, so you have time to get ready).
Look at each of the 5 dots as it lights up (~2s each) — `main.py`
collects real samples for each one, fits a new correction, and
reloads it live. On success it shows the fit error and automatically
returns you to the Home screen after a couple seconds. Then just look
at a tile; the gaze dot follows your eyes, the ring fills over ~2s,
and on completion the tile flashes and the real action fires.

You can jump back to Calibration any time via the tab, and switch to
"Simulated gaze (hover)" / "Keyboard only" via the bottom-center
input-mode button for testing without a webcam.

`calibrate.py` still exists as a standalone terminal alternative if
you'd rather calibrate outside the browser (same underlying fit
logic, its own 5-point sequence and its own cv2 window).

## If you need to retrain the base model
```bash
# edit DATASET_DIR at the top of dataset/prepare_dataset.py first
python dataset/prepare_dataset.py     # builds dataset/cache/*.npy
python dataset/train_base_model.py    # trains + saves models/base_gaze_model.pkl
python calibrate.py                   # per-session correction on top of it
```
`dataset/cache/features.npy` / `targets.npy` (already built) and
`models/base_gaze_model.pkl` / `calibration_model.pkl` (already
trained) are included, so you don't have to redo this unless you want
to retrain on more data.

## Known gaps (be upfront with your professor)
- Tile-region mapping in `main.py` (`BUTTONS` dict) approximates the
  browser's tile grid by fixed fractions of the screen — not
  pixel-calibrated to the actual rendered tile positions yet.
- Voice pathway (`voice/`) is not implemented — teammate 4's part.
- Gaze accuracy/response-time evaluation numbers aren't collected yet
  (`benchmark_test.py` only covers backend action-dispatch latency).

## App states & input channels (latest)
- **States:** `CALIBRATION_IDLE -> CALIBRATING -> ACTIVE`. Gaze selection/dwell is
  OFF until calibration finishes; stale events are dropped via an epoch counter.
- **Gaze pipeline:** raw -> calibration -> EMA smoothing + outlier rejection ->
  stable target (hysteresis) -> 2 s dwell -> select. Tile hit-boxes come from the
  real page layout (`/api/layout`).
- **Voice mode:** one input is live at a time. The mic stays on in BOTH modes.
  - Gaze mode: say **"voice mode"** (or "switch to voice") to switch — works even before
    calibration. You can also dwell your gaze on the top-right **Voice mode** button (~2 s).
  - Voice mode: say **"open camera" / "camera"**, and likewise **browser, media, files, notes**.
    Say **"go back"** (or "gaze mode") to return to gaze mode.
  - Calibration is gaze-only; voice switching is ignored while calibrating.
  - Code: `voice/voice_commands.py` (persistent mic stream, listen/recognize threads, all Google
    alternatives matched). Needs internet + microphone. Language `en-IN` by default; override with
    env var `EVA_VOICE_LANG`. Note: in gaze mode the mic listens continuously for the wake phrase.
- Run with `python main.py` or double-click `START_EVA.bat`.
