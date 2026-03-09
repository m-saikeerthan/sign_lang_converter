"""
Optimized MediaPipe hand tracker for ISL detection.
Extracts normalized 42-feature landmark vectors from webcam frames.
"""

import numpy as np
import mediapipe as mp
import cv2

from config import (
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    NUM_LANDMARKS, COLOR_LANDMARK, COLOR_CONNECTION
)


class HandTracker:
    """Wraps MediaPipe Hands for fast, ISL-tuned hand tracking."""

    def __init__(self, static_mode=False):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )

        # Custom drawing specs
        self._landmark_style = self.mp_draw.DrawingSpec(
            color=COLOR_LANDMARK, thickness=2, circle_radius=3
        )
        self._connection_style = self.mp_draw.DrawingSpec(
            color=COLOR_CONNECTION, thickness=1, circle_radius=1
        )

    def process(self, frame):
        """
        Process a BGR frame and return (landmarks_array, hand_landmarks_raw).
        
        Returns:
            landmarks: np.array of shape (42,) with wrist-centered x,y coords,
                       or None if no hand detected.
            raw_hand:  MediaPipe hand landmarks object (for drawing), or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]

        # Normalize: center on wrist (landmark 0)
        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y

        landmarks = []
        for lm in hand.landmark:
            landmarks.append(lm.x - base_x)
            landmarks.append(lm.y - base_y)

        return np.array(landmarks, dtype=np.float32), hand

    def draw(self, frame, raw_hand):
        """Draw hand skeleton on frame with styled landmarks."""
        if raw_hand is None:
            return frame
        self.mp_draw.draw_landmarks(
            frame, raw_hand, self.mp_hands.HAND_CONNECTIONS,
            self._landmark_style, self._connection_style
        )
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
