"""
landmark_detection.py
----------------------
Wraps MediaPipe Face Mesh (with iris landmarks) so the rest of the
gaze pipeline can just ask "give me this frame's face landmarks"
without knowing anything about MediaPipe's API.
"""

import cv2
import mediapipe as mp


class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # turns on the iris landmarks (indices 468-477)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def get_landmarks(self, frame_bgr):
        """
        Returns a list of (x, y) landmark points in PIXEL coordinates
        for the first detected face, or None if no face was found.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        points = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
        return points

    def close(self):
        self.face_mesh.close()
