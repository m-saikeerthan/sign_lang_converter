"""
Smart Capture — Capture training images for ISL gestures.
Supports two-hand gesture capture with live hand overlay and progress bar.

Usage:
    python src/smart_capture.py WORD_NAME
    python src/smart_capture.py WORD_NAME --two-hands
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, WORD_FRAMES_DIR,
    CAPTURE_IMAGES_PER_WORD, CAPTURE_IMAGES_PER_PHASE, CAPTURE_INTERVAL,
    MP_MAX_HANDS, COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_WHITE,
    FONT, FONT_SMALL, FONT_MEDIUM, FONT_THICKNESS
)
from hand_tracker import HandTracker


def capture_word(word, two_hands=False):
    """Capture training images for a word."""
    word = word.upper()
    save_dir = os.path.join(WORD_FRAMES_DIR, word)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    print(f"\n[ISL] Capturing: {word}")
    print(f"[ISL] Mode: {'Two-Hand' if two_hands else 'Single-Hand'}")
    print(f"[ISL] Existing images: {existing}")
    print(f"[ISL] Target: {CAPTURE_IMAGES_PER_WORD} total")
    print(f"[ISL] Press SPACE to start/pause, Q to quit")

    tracker = HandTracker(static_mode=False)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ISL] ERROR: Cannot open camera!")
        return

    capturing = False
    captured = existing
    last_capture_time = 0
    target = CAPTURE_IMAGES_PER_WORD

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process hand tracking
        landmarks, raw_hands = tracker.process(frame)

        # Draw landmarks
        tracker.draw(frame, raw_hands)

        # Status info
        num_hands = len(raw_hands)
        min_hands = 2 if two_hands else 1
        hand_ok = num_hands >= min_hands

        # Hand detection status
        if hand_ok:
            status_text = f"{num_hands} hand{'s' if num_hands > 1 else ''} detected"
            status_color = COLOR_GREEN
        else:
            status_text = f"Need {min_hands} hand{'s' if min_hands > 1 else ''} (got {num_hands})"
            status_color = COLOR_RED

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(frame, f"Capturing: {word}", (15, 30),
                    FONT, FONT_MEDIUM, COLOR_WHITE, FONT_THICKNESS)
        cv2.putText(frame, status_text, (15, 55),
                    FONT, FONT_SMALL, status_color, 1)
        cv2.putText(frame, f"{captured}/{target}", (w - 100, 30),
                    FONT, FONT_MEDIUM, COLOR_WHITE, FONT_THICKNESS)

        # Progress bar
        progress = min(captured / target, 1.0) if target > 0 else 0
        bar_w = w - 40
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 18), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + int(bar_w * progress), h - 18), COLOR_GREEN, -1)

        # Capture mode indicator
        if capturing:
            cv2.putText(frame, "● RECORDING", (w - 180, 55),
                        FONT, FONT_SMALL, COLOR_RED, FONT_THICKNESS)

            # Auto-capture
            now = time.time()
            if hand_ok and (now - last_capture_time) >= CAPTURE_INTERVAL:
                filename = f"{word}_{captured:04d}.jpg"
                cv2.imwrite(os.path.join(save_dir, filename), frame)
                captured += 1
                last_capture_time = now

                if captured >= target:
                    print(f"\n[ISL] Done! Captured {captured} images for {word}")
                    break
        else:
            cv2.putText(frame, "SPACE to start", (w - 180, 55),
                        FONT, FONT_SMALL, COLOR_YELLOW, 1)

        cv2.imshow(f"ISL Capture - {word}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            capturing = not capturing
            if capturing:
                print(f"[ISL] Recording started...")
            else:
                print(f"[ISL] Paused ({captured} captured)")

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"[ISL] Final: {captured} images saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture ISL gesture images")
    parser.add_argument("word", help="Word/gesture name to capture")
    parser.add_argument("--two-hands", action="store_true", help="Require two hands")
    args = parser.parse_args()
    capture_word(args.word, args.two_hands)
