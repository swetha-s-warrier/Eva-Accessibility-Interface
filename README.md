# EVA — Eye & Voice Accessibility Interface

A hands-free desktop launcher controlled by **eye gaze** (webcam) or **voice**.
Look at a tile for ~2 seconds, or say its name, to open **Browser, Media, Files, Notes** or **Camera**.

Built for people who can't comfortably use a mouse or keyboard. Runs locally with a standard webcam and microphone; no special eye-tracking hardware.

---

## Features

- **Webcam gaze control** — MediaPipe Face Mesh + iris landmarks → MLP base model (trained on MPIIGaze) → per-user 5-point calibration.
- **Stable selection** — outlier rejection, EMA smoothing, target hysteresis and a 2-second dwell, so a single noisy frame never triggers a button.
- **Safe startup** — gaze is completely **off until calibration finishes** (`CALIBRATION_IDLE → CALIBRATING → ACTIVE`).
- **Voice control** — say `open camera` or `camera` (same for browser, media, files, notes).
- **Hands-free mode switching** — say **"voice mode"** to switch to voice, **"go back"** to return to gaze, or dwell on the top-right *Voice mode* button.
- **One input at a time** — Gaze *or* Voice is live, never both, to prevent accidental triggers.
- **Single command launch** — `python main.py` starts the backend, opens the UI and runs the CV loop.

## How it works

```
Webcam ─► MediaPipe Face Mesh ─► feature vector (8)
                                     │
                              Base gaze model (MLP, MPIIGaze)
                                     │  raw (x, y)
                              Calibration (5-point linear fix)
                                     │  calibrated (x, y)
                        Smoothing + validity/outlier rejection
                                     │
                     Stable target (hysteresis) ─► Dwell (2 s)
                                     │
                          POST /api/gaze  ──►  Flask backend ──► launches app
                                                    ▲
Microphone ─► SpeechRecognition (Google) ───────────┘  (voice commands / mode switch)
                                                    │
                                   frontend.html polls & renders UI
```

### App states

| State | Gaze selection | Notes |
|---|---|---|
| `CALIBRATION_IDLE` | OFF | Startup. Nothing can trigger. Voice-mode switch still works. |
| `CALIBRATING` | Dots only | Collects samples; no buttons, no voice switching. |
| `ACTIVE` | ON | Target detection + dwell enabled; state/dwell reset on entry, stale events dropped. |

## Project structure

```
EVA_Final/
├── main.py                  # Entry point: backend + browser + gaze loop
├── server.py                # Flask API, app state machine, voice engine control
├── frontend.html            # Web UI (tiles, calibration, mode toggle)
├── calibrate.py             # Optional terminal-based calibration
├── benchmark_test.py        # Action dispatcher test
├── START_EVA.bat            # Windows launcher (installs deps, runs main.py)
├── requirements.txt
├── backend/
│   ├── action_mapper.py     # Target → action, with cooldown
│   └── actions.py           # Launches browser/media/files/notes/camera
├── gaze/
│   ├── landmark_detection.py
│   ├── feature_extraction.py   # Single source of truth for train + live features
│   ├── base_gaze_model.py      # Dataset-trained MLP wrapper
│   ├── calibration.py          # Per-user linear correction
│   ├── gaze_model.py           # base + calibration
│   ├── smoothing.py            # EMA + outlier rejection
│   └── dwell.py                # Stable target + dwell + cooldown
├── voice/
│   └── voice_commands.py       # Mic engine + command matching
├── dataset/
│   ├── prepare_dataset.py      # MPIIGaze → features.npy / targets.npy
│   ├── train_base_model.py     # Trains models/base_gaze_model.pkl
│   └── cache/
└── models/
    ├── base_gaze_model.pkl
    └── calibration_model.pkl
```

## Requirements

- Python **3.10**
- Webcam and microphone
- Internet connection (voice uses Google's speech service)
- Windows is the primary target (app launchers in `backend/actions.py` also have macOS/Linux fallbacks)

Key packages are pinned in `requirements.txt` (`scikit-learn==1.7.2` must match the saved `.pkl` models).

## Installation

```bash
git clone <your-repo-url>
cd EVA_Final

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

> **PyAudio on Windows:** if `pip install PyAudio` fails, use `pip install pipwin && pipwin install pyaudio`, or download a matching wheel.

## Usage

```bash
python main.py
```
or double-click **`START_EVA.bat`** on Windows.

Your browser opens on the **Calibration** screen.

1. **Calibrate** — click *Start calibration* and look at each of the 5 dots as it lights up (~2 s each). Sit at your normal distance and keep your head fairly still. On success you're taken to the tiles.
2. **Gaze** — look at a tile; it highlights, a ring fills for ~2 s, then the app opens.
3. **Voice** — say *"voice mode"* (or dwell on the top-right *Voice mode* button), then:

| Say | Action |
|---|---|
| `open browser` / `browser` | Opens the browser |
| `open media` / `media` | Opens media |
| `open files` / `files` | Opens file explorer |
| `open notes` / `notes` | Opens notes |
| `open camera` / `camera` | Opens camera |
| `go back` / `gaze mode` | Returns to gaze mode |

Stop with `Ctrl+C`, or just close the browser tab (the script exits automatically).

## Configuration

| What | Where | Default |
|---|---|---|
| Dwell time | `main.py` → `DWELL_SECONDS` | `2.0` s |
| Smoothing / outlier limits | `gaze/smoothing.py` → `alpha`, `max_jump`, `accept_after` | `0.30`, `0.25`, `4` |
| Target acquire / release / cooldown | `gaze/dwell.py` | `0.25` s / `0.35` s / `1.0` s |
| Tile hit tolerance | `frontend.html` → `HIT_PAD` | `12` px |
| Debug log (`Raw │ Cal │ Smooth │ Target │ Dwell │ State`) | `main.py` → `DEBUG` | `True` |
| Voice language | env var `EVA_VOICE_LANG` | `en-IN` |

## Retraining the base model

1. Download [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) and set `DATASET_DIR` in `dataset/prepare_dataset.py`.
2. From the project root:
   ```bash
   python dataset/prepare_dataset.py
   python dataset/train_base_model.py
   ```
3. Recalibrate in the app (the calibration is fitted on top of the base model).

Training and live inference share `gaze/feature_extraction.py`, so features stay identical.

## Troubleshooting

| Problem | Fix |
|---|---|
| Nothing happens before calibration | Expected — gaze is disabled until calibration completes. |
| Wrong tile selected / offset gaze | Recalibrate; keep lighting even and your face centered in the webcam. |
| Voice status shows an error | Check the mic, `PyAudio` installation and internet connection. |
| Commands not recognized | Speak clearly at normal volume; try `EVA_VOICE_LANG=en-US`. |
| Page can't reach backend | Make sure `python main.py` is running on port `5000`. |

## Known limitations

- Webcam gaze accuracy is limited (roughly tile-sized); that's why tiles are large and dwell-based.
- `face_width` / `face_center_y` features are in pixels, so they depend on camera resolution. Normalizing them by frame size and retraining would improve cross-camera consistency.
- Voice requires internet, and in gaze mode the mic listens continuously for the wake phrase.
- Head movement after calibration degrades accuracy; recalibrate if you shift position.

## Tech stack

Python · OpenCV · MediaPipe · scikit-learn · NumPy/SciPy · Flask · SpeechRecognition · HTML/CSS/JS

## Acknowledgements

- [MPIIGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) (Zhang et al.)
- [MediaPipe](https://developers.google.com/mediapipe)
- [SpeechRecognition](https://github.com/Uberi/speech_recognition)

## License

Add a license of your choice (e.g. MIT) before publishing.
