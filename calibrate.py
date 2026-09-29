"""
calibrate.py
-------------
Run this AFTER training the base model (dataset/prepare_dataset.py
then dataset/train_base_model.py). It shows 5 points (4 corners +
center) instead of a full 3x3 grid, because the base model already
carries most of the mapping - this step only learns a small linear
correction for your specific camera/screen setup.

Usage:
    python calibrate.py
"""

import os
import time
import cv2
import numpy as np

from gaze.landmark_detection import FaceMeshDetector
from gaze.feature_extraction import extract_features
from gaze.base_gaze_model import BaseGazeModel
from gaze.calibration import CalibrationManager

WINDOW_NAME = "EVA Calibration"
SCREEN_W, SCREEN_H = 1280, 720
SAMPLES_PER_POINT = 30
BASE_MODEL_PATH = "models/base_gaze_model.pkl"
CALIBRATION_SAVE_PATH = "models/calibration_model.pkl"

# 4 corners + center is enough for a linear correction
POINTS = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.5)]


def collect_calibration_data(base_model):
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    raw_points = []
    target_points = []

    for point_index, (gx, gy) in enumerate(POINTS):
        px, py = int(gx * SCREEN_W), int(gy * SCREEN_H)
        collected = 0

        while collected < SAMPLES_PER_POINT:
            ok, frame = cap.read()
            if not ok:
                continue

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (px, py), 18, (0, 255, 0), -1)
            cv2.putText(canvas, "Look at the green dot and hold still.",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(canvas, f"Point {point_index + 1}/{len(POINTS)}   "
                                 f"samples {collected}/{SAMPLES_PER_POINT}   (q = cancel)",
                        (30, SCREEN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                raise SystemExit("Calibration cancelled.")

            landmarks = detector.get_landmarks(frame)
            feats = extract_features(landmarks)
            if feats is not None:
                raw_x, raw_y = base_model.predict(feats)
                raw_points.append((raw_x, raw_y))
                target_points.append((gx, gy))
                collected += 1
                time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return raw_points, target_points


if __name__ == "__main__":
    if not os.path.exists(BASE_MODEL_PATH):
        raise SystemExit(
            f"Missing {BASE_MODEL_PATH}. Train it first: "
            f"python dataset/prepare_dataset.py then python dataset/train_base_model.py"
        )

    base_model = BaseGazeModel(BASE_MODEL_PATH)

    print("Starting calibration. A window will open - look at each green dot.")
    raw_points, target_points = collect_calibration_data(base_model)

    calib = CalibrationManager()
    calib.fit(raw_points, target_points)

    os.makedirs("models", exist_ok=True)
    calib.save(CALIBRATION_SAVE_PATH)
    print(f"Done. Saved calibration correction to {CALIBRATION_SAVE_PATH}")
