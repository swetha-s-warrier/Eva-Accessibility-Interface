"""
base_gaze_model.py
--------------------
Wraps a model trained OFFLINE on a public dataset (default: MPIIFaceGaze)
that maps our MediaPipe feature vector -> a rough normalized on-screen
gaze position (x, y), learned across many people/head positions instead
of just one calibration session.

This is still a "lightweight" model (an MLPRegressor from scikit-learn,
not a deep CNN) - consistent with the project doc's rule against big
from-scratch deep-learning models. Train it with:
    dataset/prepare_dataset.py   (builds features.npy / targets.npy)
    dataset/train_base_model.py  (trains + saves models/base_gaze_model.pkl)

NOTE: this model was trained on a DIFFERENT camera/screen setup than
yours, so its raw output will usually be offset/scaled wrong for your
setup. gaze_model.py applies a small per-user calibration correction on
top of this (see calibration.py) - that step is still required, but is
now a tiny linear fix instead of the whole mapping, so it needs far
fewer calibration points and behaves more stably.
"""

import pickle


class BaseGazeModel:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data.get("scaler")  # StandardScaler used at train time

    def predict(self, features):
        X = features.reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        pred = self.model.predict(X)[0]
        x, y = float(pred[0]), float(pred[1])
        return x, y
