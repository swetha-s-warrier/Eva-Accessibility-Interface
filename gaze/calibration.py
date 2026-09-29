"""
calibration.py
---------------
This is now a SMALL correction step, not the whole gaze mapping.

The base model (base_gaze_model.py, trained on a public dataset)
already gives a reasonable, dataset-informed guess at gaze position.
But it was trained on a different camera/screen setup, so for your
specific webcam position and screen it will usually come out shifted
and/or scaled wrong.

CalibrationManager fits a simple linear correction:
    true_x = a*raw_x + b*raw_y + c
    true_y = d*raw_x + e*raw_y + f
using only a handful of "look at this known point" samples (5 points
is enough - far fewer than the 9+ needed when calibration had to learn
the ENTIRE mapping by itself with no dataset backing it).
"""

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


class CalibrationManager:
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, raw_points, target_points):
        """
        raw_points:    list of (x_raw, y_raw) from BaseGazeModel.predict()
        target_points: list of (x_true, y_true) normalized screen targets, each in [0, 1]
        """
        X = np.array(raw_points)
        y = np.array(target_points)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, raw_x, raw_y):
        if not self.is_fitted:
            raise RuntimeError("CalibrationManager: call fit() (or load()) before predict().")
        pred = self.model.predict(np.array([[raw_x, raw_y]]))[0]
        x, y = float(pred[0]), float(pred[1])
        # keep predictions inside the screen/UI area
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        return x, y

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.is_fitted = True
        return obj
