"""
MediaPipe landmark extraction. Produces flat vectors per frame for the LSTM.

Includes Pose, Hand, and Holistic landmark extractors.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# MediaPipe Pose has 33 landmarks, each (x, y, z, visibility) -> 132 dims per frame
NUM_LANDMARKS = 33
DIMS_PER_LANDMARK = 4
LANDMARK_VEC_SIZE = NUM_LANDMARKS * DIMS_PER_LANDMARK  # 132

# MediaPipe Hands: 21 landmarks per hand, (x, y, z) each, 2 hands -> 126 dims
HAND_NUM_LANDMARKS = 21
HAND_DIMS_PER_LANDMARK = 3
HAND_LANDMARK_VEC_SIZE = 2 * HAND_NUM_LANDMARKS * HAND_DIMS_PER_LANDMARK  # 126

# Holistic: hands (126) + selected upper-body pose landmarks (7 * 3 = 21) -> 147 dims
# Pose indices: nose(0), L/R shoulder(11,12), L/R elbow(13,14), L/R wrist(15,16)
POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16]
NUM_POSE_SELECTED = len(POSE_LANDMARK_INDICES)
HOLISTIC_VEC_SIZE = (
    2 * HAND_NUM_LANDMARKS * HAND_DIMS_PER_LANDMARK  # 126 (hands)
    + NUM_POSE_SELECTED * HAND_DIMS_PER_LANDMARK  # 21 (upper body)
)  # 147


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

    def process_and_draw(
        self, frame: "cv2.Mat"
    ) -> tuple[Optional[np.ndarray], "cv2.Mat"]:
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
                [
                    v
                    for lm in results.pose_landmarks.landmark
                    for v in (lm.x, lm.y, lm.z, lm.visibility)
                ],
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


class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe Hands.

    Produces a 126-d vector per frame (2 hands x 21 landmarks x 3 dims).
    Landmarks are normalized relative to each hand's wrist for position/scale
    invariance.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.hands.close()

    def _normalize_hand(self, hand_landmarks) -> np.ndarray:
        """Normalize 21 landmarks relative to wrist, scaled by wrist-to-middle-MCP distance."""
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )  # (21, 3)

        wrist = raw[0]  # landmark 0
        centered = raw - wrist

        # Scale by distance from wrist to middle finger MCP (landmark 9)
        scale = np.linalg.norm(centered[9])
        if scale > 1e-6:
            centered = centered / scale

        return centered.flatten()  # (63,)

    def _extract_hands(self, results) -> Optional[np.ndarray]:
        """Build 126-d vector from detection results. Returns None if no hands."""
        if not results.multi_hand_landmarks:
            return None

        left = np.zeros(HAND_NUM_LANDMARKS * HAND_DIMS_PER_LANDMARK, dtype=np.float32)
        right = np.zeros(HAND_NUM_LANDMARKS * HAND_DIMS_PER_LANDMARK, dtype=np.float32)

        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness_info.classification[0].label  # "Left" or "Right"
            normalized = self._normalize_hand(hand_landmarks)
            # MediaPipe labels are mirrored (camera view), so "Left" = user's right
            if label == "Left":
                right = normalized
            else:
                left = normalized

        return np.concatenate([left, right])  # (126,)

    def process_frame(self, frame: "cv2.Mat") -> Optional[np.ndarray]:
        """Run MediaPipe Hands on a BGR frame.

        Returns a flat vector of shape (126,) or None if no hands detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return self._extract_hands(results)

    def process_and_draw(
        self, frame: "cv2.Mat"
    ) -> tuple[Optional[np.ndarray], "cv2.Mat"]:
        """Process frame, draw hand landmarks on a copy, return (vector, drawn_frame)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        out = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    out,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        vec = self._extract_hands(results)
        return vec, out


class HolisticLandmarkExtractor:
    """Extract hand + upper-body pose landmarks using MediaPipe Holistic.

    Produces a 147-d vector per frame:
    - Left hand: 21 landmarks x 3 dims = 63 (normalized to wrist)
    - Right hand: 21 landmarks x 3 dims = 63 (normalized to wrist)
    - Upper body pose: 7 landmarks x 3 dims = 21 (normalized to shoulder midpoint)

    The pose landmarks give spatial context (where hands are relative to body),
    which is critical for signs like "hello" (hand near forehead) vs "thank you"
    (hand near chin).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands_module = mp.solutions.hands

    def __enter__(self) -> "HolisticLandmarkExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.holistic.close()

    def _normalize_hand(self, hand_landmarks) -> np.ndarray:
        """Normalize 21 hand landmarks relative to wrist, scaled by wrist-to-middle-MCP."""
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )  # (21, 3)

        wrist = raw[0]
        centered = raw - wrist

        scale = np.linalg.norm(centered[9])  # wrist to middle finger MCP
        if scale > 1e-6:
            centered = centered / scale

        return centered.flatten()  # (63,)

    def _extract_pose(self, pose_landmarks) -> np.ndarray:
        """Extract and normalize selected upper-body pose landmarks."""
        all_landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark],
            dtype=np.float32,
        )

        selected = all_landmarks[POSE_LANDMARK_INDICES]  # (7, 3)

        # Normalize relative to shoulder midpoint, scale by shoulder width
        left_shoulder = all_landmarks[11]
        right_shoulder = all_landmarks[12]
        midpoint = (left_shoulder + right_shoulder) / 2.0
        centered = selected - midpoint

        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width > 1e-6:
            centered = centered / shoulder_width

        return centered.flatten()  # (21,)

    def _extract(self, results) -> Optional[np.ndarray]:
        """Build 147-d vector from holistic results. Returns None if no hands detected."""
        # Require at least one hand
        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            return None

        # Hands (zero-padded if one hand missing)
        zeros_hand = np.zeros(
            HAND_NUM_LANDMARKS * HAND_DIMS_PER_LANDMARK, dtype=np.float32
        )

        if results.left_hand_landmarks is not None:
            left = self._normalize_hand(results.left_hand_landmarks)
        else:
            left = zeros_hand

        if results.right_hand_landmarks is not None:
            right = self._normalize_hand(results.right_hand_landmarks)
        else:
            right = zeros_hand

        if results.pose_landmarks is not None:
            pose = self._extract_pose(results.pose_landmarks)
        else:
            pose = np.zeros(
                NUM_POSE_SELECTED * HAND_DIMS_PER_LANDMARK, dtype=np.float32
            )

        return np.concatenate([left, right, pose])  # (147,)

    def process_frame(self, frame: "cv2.Mat") -> Optional[np.ndarray]:
        """Run MediaPipe Holistic on a BGR frame.

        Returns a flat vector of shape (147,) or None if no hands detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        return self._extract(results)

    def process_and_draw(
        self, frame: "cv2.Mat"
    ) -> tuple[Optional[np.ndarray], "cv2.Mat"]:
        """Process frame, draw landmarks on a copy, return (vector, drawn_frame)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        out = frame.copy()

        # Draw pose skeleton
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                out,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                out,
                results.left_hand_landmarks,
                self.mp_hands_module.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                out,
                results.right_hand_landmarks,
                self.mp_hands_module.HAND_CONNECTIONS,
            )

        vec = self._extract(results)
        return vec, out
