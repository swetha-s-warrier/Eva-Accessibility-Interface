"""
prepare_dataset.py
--------------------
Turns MPIIGaze (Data/Original) into (feature_vector, screen_xy) training
pairs, using the SAME feature_extraction.py as the live pipeline.

Annotation line format (confirmed from your file):
  <image>  gaze_x_px gaze_y_px  <12 landmark numbers>  <3 head rot> <3 head trans>
           <3 face center> <3 gaze target 3D>  left|right
We only use the first two numbers (gaze_x_px, gaze_y_px) as the target.

SET THESE FIRST:
  DATASET_DIR   -> the "Original" folder (contains p00 ... p14)
  ONLY_SUBJECTS -> ["p00"] for a first test, None for everything
  STRIDE        -> 1 = every image, 5 = every 5th image (much faster)
"""

import glob
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from scipy.io import loadmat
from tqdm import tqdm

from gaze.landmark_detection import FaceMeshDetector
from gaze.feature_extraction import extract_features

DATASET_DIR = "c:\Datasets\Data"   # <-- CHANGE THIS
OUTPUT_DIR = "dataset/cache"
ONLY_SUBJECTS = None     # <-- test on p00 first, then set to None
STRIDE = 10                  # <-- use every 5th image for speed; set 1 for all

GAZE_X_COL = 0   # first number after the image name
GAZE_Y_COL = 1   # second number after the image name


def load_screen_size(subject_dir):
    mat_path = os.path.join(subject_dir, "Calibration", "screenSize.mat")
    mat = loadmat(mat_path)
    try:
        return int(mat["width_pixel"][0][0]), int(mat["height_pixel"][0][0])
    except KeyError:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"screenSize.mat keys are {keys} - expected width_pixel/height_pixel")


def find_annotation_files(subject_dir):
    files = []
    for root, _, names in os.walk(subject_dir):
        if "Calibration" in root:
            continue
        for n in names:
            if n.lower().endswith(".txt"):
                files.append(os.path.join(root, n))
    return sorted(files)


def resolve_image(subject_dir, ann_path, image_rel):
    candidates = [
        os.path.join(subject_dir, image_rel),
        os.path.join(os.path.dirname(ann_path), image_rel),
        os.path.join(os.path.dirname(ann_path), os.path.basename(image_rel)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def process_subject(subject_dir, detector, features_out, targets_out, log):
    try:
        width_px, height_px = load_screen_size(subject_dir)
    except Exception as e:
        print(f"  Could not read screen size for {subject_dir}: {e}")
        log["skipped_subjects"] += 1
        return

    ann_files = find_annotation_files(subject_dir)
    if not ann_files:
        print(f"  No annotation .txt files found in {subject_dir}")
        log["skipped_subjects"] += 1
        return

    for ann_path in ann_files:
        with open(ann_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
        lines = lines[::STRIDE]

        for line in tqdm(lines, desc=os.path.basename(ann_path), leave=False):
            parts = line.split()
            try:
                image_rel = parts[0]
                gaze_x_px = float(parts[1 + GAZE_X_COL])
                gaze_y_px = float(parts[1 + GAZE_Y_COL])
            except (IndexError, ValueError):
                continue

            image_path = resolve_image(subject_dir, ann_path, image_rel)
            if image_path is None:
                log["missing_images"] += 1
                continue
            frame = cv2.imread(image_path)
            if frame is None:
                log["missing_images"] += 1
                continue

            log["total_images"] += 1
            feats = extract_features(detector.get_landmarks(frame))
            if feats is None:
                log["detection_failed"] += 1
                continue

            tx, ty = gaze_x_px / width_px, gaze_y_px / height_px
            if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
                log["out_of_range"] += 1
                continue

            features_out.append(feats)
            targets_out.append((tx, ty))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = FaceMeshDetector()
    features_out, targets_out = [], []
    log = {"total_images": 0, "detection_failed": 0, "missing_images": 0,
           "out_of_range": 0, "skipped_subjects": 0}

    subject_dirs = sorted(d for d in glob.glob(os.path.join(DATASET_DIR, "p*")) if os.path.isdir(d))
    if ONLY_SUBJECTS:
        subject_dirs = [d for d in subject_dirs if os.path.basename(d) in ONLY_SUBJECTS]
    if not subject_dirs:
        raise SystemExit(f"No subject folders found under {DATASET_DIR}. Check DATASET_DIR / ONLY_SUBJECTS.")

    for d in subject_dirs:
        print("Processing", os.path.basename(d))
        process_subject(d, detector, features_out, targets_out, log)
    detector.close()

    np.save(os.path.join(OUTPUT_DIR, "features.npy"), np.array(features_out, dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "targets.npy"), np.array(targets_out, dtype=np.float32))

    print("\n--- Dataset prep summary ---")
    print(f"Images read:             {log['total_images']}")
    print(f"MediaPipe detect fails:  {log['detection_failed']}")
    print(f"Missing image files:     {log['missing_images']}")
    print(f"Out-of-range targets:    {log['out_of_range']}")
    print(f"Skipped subjects:        {log['skipped_subjects']}")
    print(f"Usable training pairs:   {len(features_out)}")
    if log["total_images"] > 0:
        rate = log["detection_failed"] / log["total_images"]
        print(f"Detection failure rate:  {rate:.1%}")


if __name__ == "__main__":
    main()