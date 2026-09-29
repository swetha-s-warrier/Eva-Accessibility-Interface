"""
feature_extraction.py
----------------------
Turns raw MediaPipe landmarks into a small, fixed-length feature
vector describing "where inside the eye is the iris pointing" plus a
rough head-position proxy.

IMPORTANT: whatever feature vector shape you use here for LIVE
inference must be exactly what you train/calibrate the gaze model on
(this is the "critical training/inference rule" in the project doc).
Keep this function as the single source of truth for both.
"""

import numpy as np

# MediaPipe Face Mesh landmark indices (with refine_landmarks=True)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS = [468, 469, 470, 471]

RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS = [473, 474, 475, 476]

NOSE_TIP = 1
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454

FEATURE_LENGTH = 8


def _iris_ratio(points, outer_idx, inner_idx, top_idx, bottom_idx, iris_idx):
    outer = np.array(points[outer_idx])
    inner = np.array(points[inner_idx])
    top = np.array(points[top_idx])
    bottom = np.array(points[bottom_idx])
    iris = np.mean([points[i] for i in iris_idx], axis=0)

    eye_width = np.linalg.norm(outer - inner) + 1e-6
    eye_height = np.linalg.norm(top - bottom) + 1e-6

    # Horizontal position of the iris between the two eye corners (0 = inner, 1 = outer)
    h_ratio = np.dot(iris - inner, outer - inner) / (eye_width ** 2)
    # Vertical position of the iris between top and bottom lid (0 = top, 1 = bottom)
    v_ratio = np.dot(iris - top, bottom - top) / (eye_height ** 2)

    return float(h_ratio), float(v_ratio)


def extract_features(points):
    """
    points: list of (x, y) pixel coordinates, as returned by
            FaceMeshDetector.get_landmarks()

    Returns a numpy array of length FEATURE_LENGTH, or None if the
    landmark list looks wrong (e.g. no face detected).
    """
    if points is None or len(points) < 478:
        return None

    left_h, left_v = _iris_ratio(points, LEFT_EYE_OUTER, LEFT_EYE_INNER,
                                  LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS)
    right_h, right_v = _iris_ratio(points, RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
                                    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS)

    # A rough head-position/pose proxy, so the model has *some* idea
    # of whether the user shifted in their chair rather than only
    # moved their eyes. This is not a real head-pose estimate, just
    # enough signal for the calibration regression to use.
    left_edge = np.array(points[LEFT_FACE_EDGE])
    right_edge = np.array(points[RIGHT_FACE_EDGE])
    nose = np.array(points[NOSE_TIP])
    face_width = np.linalg.norm(right_edge - left_edge) + 1e-6
    face_center_x = (nose[0] - left_edge[0]) / face_width
    face_center_y = nose[1] / face_width
    tilt = float(np.arctan2(right_edge[1] - left_edge[1], right_edge[0] - left_edge[0]))

    features = np.array([
        left_h, left_v, right_h, right_v,
        face_width, face_center_x, face_center_y, tilt
    ], dtype=np.float32)

    return features
