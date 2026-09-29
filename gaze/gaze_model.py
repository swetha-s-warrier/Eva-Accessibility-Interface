"""
gaze_model.py
--------------
Combines the dataset-trained BaseGazeModel with the small per-user
CalibrationManager correction on top of it. Everything downstream
only ever calls GazeModel.predict(features) -> (x, y), same as
before - the two-stage design underneath is invisible to main.py.
"""

from gaze.base_gaze_model import BaseGazeModel
from gaze.calibration import CalibrationManager


class GazeModel:
    def __init__(self, base_model_path, calibration_path):
        self._base = BaseGazeModel(base_model_path)
        self._calibration_path = calibration_path
        self._calibration = CalibrationManager.load(calibration_path)

    def predict(self, features):
        """
        features: numpy array from feature_extraction.extract_features()
        returns: (x, y) normalized in [0, 1], (0, 0) = top-left of the UI
        """
        raw_x, raw_y = self._base.predict(features)
        return self._calibration.predict(raw_x, raw_y)

    def calibrate(self, raw_x, raw_y):
        """Apply the per-user linear correction to an already-computed raw
        base prediction (lets main.py log raw and calibrated separately
        without running the base model twice)."""
        return self._calibration.predict(raw_x, raw_y)

    def predict_raw(self, features):
        """Base-model-only prediction, bypassing the per-user calibration
        correction. Used while COLLECTING new calibration samples — the
        correction we're about to refit shouldn't be applied to the data
        used to fit it."""
        return self._base.predict(features)

    def reload_calibration(self, calibration_path=None):
        """Hot-reloads the calibration correction after the in-browser
        calibration flow (frontend.html's "Start calibration") has fit
        and saved a new one — no need to restart main.py."""
        path = calibration_path or self._calibration_path
        self._calibration = CalibrationManager.load(path)
