"""
Optimized MediaPipe hand tracker for ISL detection.
Supports 1 or 2 hands. Returns normalized landmark vectors.
"""

import numpy as np
import mediapipe as mp
import cv2

from config import (
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    NUM_LANDMARKS, NUM_FEATURES, NUM_TWO_HAND_RAW,
    COLOR_LANDMARK, COLOR_CONNECTION
)


class HandTracker:
    """Wraps MediaPipe Hands for fast, ISL-tuned hand tracking (1 or 2 hands)."""

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

        self._landmark_style = self.mp_draw.DrawingSpec(
            color=COLOR_LANDMARK, thickness=2, circle_radius=3
        )
        self._connection_style = self.mp_draw.DrawingSpec(
            color=COLOR_CONNECTION, thickness=1, circle_radius=1
        )

    def _extract_one_hand(self, hand):
        """Extract 42-dim wrist-centered landmarks from a single hand."""
        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y
        landmarks = []
        for lm in hand.landmark:
            landmarks.append(lm.x - base_x)
            landmarks.append(lm.y - base_y)
        return np.array(landmarks, dtype=np.float32)

    def process(self, frame):
        """
        Process a BGR frame. Returns combined two-hand landmarks.

        Returns:
            landmarks: np.array of shape (84,) — two hands concatenated,
                       left hand first (sorted by wrist x). Second hand
                       zero-padded if only 1 detected. None if no hand.
            raw_hands: list of MediaPipe hand landmark objects for drawing,
                       or empty list.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, []

        detected = results.multi_hand_landmarks
        raw_hands = list(detected)

        # Extract landmark arrays for each hand
        hand_data = []
        for hand in detected:
            lm = self._extract_one_hand(hand)
            wrist_x = hand.landmark[0].x  # absolute x for sorting
            hand_data.append((wrist_x, lm, hand))

        # Sort left-to-right by wrist x-position
        hand_data.sort(key=lambda h: h[0])

        # Build combined 84-dim vector
        if len(hand_data) >= 2:
            combined = np.concatenate([hand_data[0][1], hand_data[1][1]])
            raw_hands = [hand_data[0][2], hand_data[1][2]]
        else:
            # Single hand: pad second hand with zeros
            zeros = np.zeros(NUM_FEATURES, dtype=np.float32)
            combined = np.concatenate([hand_data[0][1], zeros])
            raw_hands = [hand_data[0][2]]

        return combined, raw_hands

    def draw(self, frame, raw_hands):
        """Draw hand skeleton(s) on frame with styled landmarks."""
        if not raw_hands:
            return frame
        for hand in raw_hands:
            self.mp_draw.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS,
                self._landmark_style, self._connection_style
            )
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
