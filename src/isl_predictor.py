"""
ISL Predictor — Indian Sign Language Real-Time Detection
A polished, fast, and smooth sign language translator with live UI.

Controls:
    q — Quit
    c — Clear sentence
    
    r — Reset gesture state
"""

import cv2
import time
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, NO_HAND_TIMEOUT,
    COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE, COLOR_BLACK,
    COLOR_GRAY, COLOR_DARK_BG, COLOR_ACCENT, COLOR_SENTENCE_BG,
    FONT, FONT_SMALL, FONT_MEDIUM, FONT_LARGE, FONT_THICKNESS,
    GESTURE_HOLD_TIME
)
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor


class ISLPredictor:
    """Main application class for real-time ISL detection."""

    def __init__(self):
        print("[ISL] Initializing Indian Sign Language Detector...")

        # Core modules
        self.tracker = HandTracker(static_mode=False)
        self.classifier = GestureClassifier()
        self.sentence_processor = SentenceProcessor()

        # State
        self.sentence = []
        self.last_confirmed = ""
        self.fps = 0.0
        self._prev_time = time.time()
        self._fps_samples = []
        self._no_hand_since = None
        self._NO_HAND_TIMEOUT = NO_HAND_TIMEOUT  # 5 seconds

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[ISL] ERROR: Cannot open camera!")
            sys.exit(1)

        print(f"[ISL] Ready! {len(self.classifier.labels)} gestures loaded.")
        print("[ISL] Controls: 'q' quit | 'c' clear | 'r' reset")

    def _update_fps(self):
        """Calculate smoothed FPS."""
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps_samples.append(1.0 / dt)
            if len(self._fps_samples) > 20:
                self._fps_samples.pop(0)
            self.fps = sum(self._fps_samples) / len(self._fps_samples)

    def _draw_top_bar(self, frame, result):
        """Draw the top info bar with FPS and status."""
        h, w = frame.shape[:2]

        # Dark overlay bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # FPS
        fps_color = COLOR_GREEN if self.fps > 20 else COLOR_YELLOW if self.fps > 10 else COLOR_RED
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (15, 40),
                    FONT, FONT_MEDIUM, fps_color, FONT_THICKNESS)

        # Title
        cv2.putText(frame, "ISL Detector", (w // 2 - 70, 40),
                    FONT, FONT_MEDIUM, COLOR_WHITE, FONT_THICKNESS)

        # Status indicator
        status = result.get('status', 'detecting') if result else 'no hand'
        status_map = {
            'no hand': ('No Hand', COLOR_GRAY),
            'detecting': ('Detecting...', COLOR_YELLOW),
            'holding': ('Hold Steady...', COLOR_ACCENT),
            'confirmed': ('Confirmed!', COLOR_GREEN),
        }
        text, color = status_map.get(status, ('...', COLOR_GRAY))
        cv2.putText(frame, text, (w - 180, 40),
                    FONT, FONT_SMALL, color, FONT_THICKNESS)

    def _draw_prediction(self, frame, result):
        """Draw the predicted word and confidence bar."""
        h, w = frame.shape[:2]

        if not result or not result['word']:
            # Show "Show hand gesture" message
            cv2.putText(frame, "Show a hand gesture", (w // 2 - 140, h // 2),
                        FONT, FONT_MEDIUM, COLOR_GRAY, 1)
            return

        word = result['word']
        confidence = result['confidence']
        hold_progress = result['hold_progress']
        status = result['status']

        # Color based on confidence
        if confidence > 0.90:
            word_color = COLOR_GREEN
        elif confidence > 0.80:
            word_color = COLOR_YELLOW
        else:
            word_color = COLOR_RED

        # ─── Big word display ─────────────────────────────
        # Background box for word
        text_size = cv2.getTextSize(word, FONT, FONT_LARGE, FONT_THICKNESS + 1)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 120

        # Semi-transparent background
        pad = 15
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (text_x - pad, text_y - text_size[1] - pad),
                      (text_x + text_size[0] + pad, text_y + pad),
                      COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, word, (text_x, text_y),
                    FONT, FONT_LARGE, word_color, FONT_THICKNESS + 1)

        # ─── Confidence bar ───────────────────────────────
        bar_x, bar_y = 20, h - 80
        bar_w, bar_h = w - 40, 12
        fill_w = int(bar_w * confidence)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), COLOR_GRAY, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), word_color, -1)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (bar_x, bar_y - 6),
                    FONT, FONT_SMALL, COLOR_WHITE, 1)

        # ─── Hold progress ring ──────────────────────────
        if status == 'holding' and hold_progress > 0:
            center = (w - 50, 120)
            radius = 28
            angle = int(360 * hold_progress)

            cv2.ellipse(frame, center, (radius, radius), -90, 0, 360,
                        COLOR_GRAY, 3)
            cv2.ellipse(frame, center, (radius, radius), -90, 0, angle,
                        COLOR_GREEN, 3)

            pct_text = f"{int(hold_progress * 100)}%"
            ts = cv2.getTextSize(pct_text, FONT, 0.4, 1)[0]
            cv2.putText(frame, pct_text, 
                        (center[0] - ts[0] // 2, center[1] + ts[1] // 2),
                        FONT, 0.4, COLOR_WHITE, 1)

    def _draw_sentence(self, frame):
        """Draw the sentence bar at the bottom."""
        h, w = frame.shape[:2]

        # Dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), COLOR_SENTENCE_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if self.sentence:
            sentence_text = " ".join(self.sentence)
            # Truncate if too long
            max_chars = w // 12
            if len(sentence_text) > max_chars:
                sentence_text = "..." + sentence_text[-(max_chars - 3):]
            cv2.putText(frame, sentence_text, (15, h - 18),
                        FONT, FONT_MEDIUM, COLOR_WHITE, FONT_THICKNESS)
        else:
            cv2.putText(frame, "Sentence will appear here...", (15, h - 18),
                        FONT, FONT_SMALL, COLOR_GRAY, 1)

    def run(self):
        """Main loop."""
        print("[ISL] Starting live detection... Press 'q' to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ISL] Camera read failed!")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            self._update_fps()

            # ─── Process ──────────────────────────────────
            landmarks, raw_hands = self.tracker.process(frame)

            result = None
            if landmarks is not None:
                self._no_hand_since = None  # hand detected, reset timer
                result = self.classifier.classify(landmarks)

                # Add confirmed word to sentence (no consecutive duplicates)
                if result['status'] == 'confirmed':
                    word = result['word']
                    if not self.sentence or self.sentence[-1] != word:
                        self.sentence.append(word)
            else:
                # No hand detected — start/check timeout
                if self._no_hand_since is None:
                    self._no_hand_since = time.time()
                elif time.time() - self._no_hand_since >= self._NO_HAND_TIMEOUT:
                    if self.sentence:  # only clear if there's something
                        self.sentence_processor.process(self.sentence)
                        self.sentence = []
                        self.last_confirmed = ""
                        self.classifier.reset()
                    self._no_hand_since = None  # reset so it doesn't keep firing

            # ─── Draw ────────────────────────────────────
            self.tracker.draw(frame, raw_hands)
            self._draw_top_bar(frame, result)
            self._draw_prediction(frame, result)
            self._draw_sentence(frame)

            # ─── Display ─────────────────────────────────
            cv2.imshow("ISL - Indian Sign Language Detector", frame)

            # ─── Controls ────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence = []
                self.last_confirmed = ""
            elif key == ord('r'):
                self.classifier.reset()

        self._cleanup()

    def _cleanup(self):
        """Release all resources."""
        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()
        print("[ISL] Shut down cleanly.")


if __name__ == "__main__":
    app = ISLPredictor()
    app.run()
