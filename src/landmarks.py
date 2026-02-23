"""
MediaPipe Pose landmark extraction. Produces a flat vector per frame for the LSTM.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# MediaPipe Pose has 33 landmarks, each (x, y, z, visibility) -> 132 dims per frame
NUM_LANDMARKS = 33
DIMS_PER_LANDMARK = 4
LANDMARK_VEC_SIZE = NUM_LANDMARKS * DIMS_PER_LANDMARK  # 132


class LandmarkExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.pose.close()

    def process_frame(self, frame: "cv2.Mat") -> Optional[np.ndarray]:
        """
        Run MediaPipe Pose on a BGR frame. Returns a flat vector of shape (132,)
        or None if no pose detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        vec = []
        for lm in results.pose_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(vec, dtype=np.float32)

    def process_and_draw(self, frame: "cv2.Mat") -> tuple[Optional[np.ndarray], "cv2.Mat"]:
        """Process frame, draw landmarks on a copy, return (vector, drawn_frame)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        out = frame.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                out,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        vec = None
        if results.pose_landmarks:
            vec = np.array(
                [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)],
                dtype=np.float32,
            )
        return vec, out

    def draw_landmarks(self, frame: "cv2.Mat", results) -> "cv2.Mat":
        """Draw pose landmarks on frame (for visualization)."""
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame
