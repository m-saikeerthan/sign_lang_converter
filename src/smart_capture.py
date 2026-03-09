"""
Smart Capture — Capture ISL training images with live hand overlay.
Usage: python src/smart_capture.py WORD_NAME
"""

import cv2
import os
import sys
import time
import mediapipe as mp

# Add src to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, CAMERA_INDEX,
    CAPTURE_IMAGES_PER_WORD, CAPTURE_IMAGES_PER_PHASE, CAPTURE_INTERVAL,
    COLOR_GREEN, COLOR_YELLOW, COLOR_ACCENT, COLOR_WHITE, FONT
)

# ─── Get word from CLI argument or default ────────────────────────────
if len(sys.argv) > 1:
    word = sys.argv[1].upper()
else:
    word = input("Enter the ISL word to capture: ").strip().upper()
    if not word:
        print("No word given. Exiting.")
        sys.exit(1)

save_dir = os.path.join(DATA_DIR, "word_frames", word)
os.makedirs(save_dir, exist_ok=True)

existing_files = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
count = existing_files

phases = [
    "Phase 1: Keep hand CLOSE to camera",
    "Phase 2: Keep hand FAR from camera",
    "Phase 3: Tilt hand slightly LEFT",
    "Phase 4: Tilt hand slightly RIGHT",
    "Phase 5: Small NATURAL movement"
]

print(f"[ISL] Smart capture for word: {word}")
print(f"[ISL] Existing images: {existing_files}")
print(f"[ISL] Will capture {CAPTURE_IMAGES_PER_WORD} images across {len(phases)} phases")
print("[ISL] Press SPACE to start auto-capture, Q to quit")

# ─── Setup ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7)

current_phase = 0
phase_count = 0
auto_capture = False
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ─── Live hand detection overlay ──────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ─── UI elements ─────────────────────────────────────
    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, f"Word: {word}", (15, 30),
                FONT, 0.7, COLOR_GREEN, 2)
    cv2.putText(frame, phases[current_phase], (15, 60),
                FONT, 0.55, COLOR_ACCENT, 2)

    # Progress bar
    progress = count / CAPTURE_IMAGES_PER_WORD
    bar_w = w - 40
    cv2.rectangle(frame, (20, 78), (20 + bar_w, 92), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 78), (20 + int(bar_w * progress), 92), COLOR_GREEN, -1)
    cv2.putText(frame, f"{count}/{CAPTURE_IMAGES_PER_WORD}", (w - 100, 90),
                FONT, 0.45, COLOR_WHITE, 1)

    # Hand status
    status_text = "Hand OK" if hand_detected else "No Hand!"
    status_color = COLOR_GREEN if hand_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (w - 120, 30),
                FONT, 0.55, status_color, 2)

    if not auto_capture:
        cv2.putText(frame, "Press SPACE to start", (w // 2 - 100, h // 2),
                    FONT, 0.6, COLOR_YELLOW, 2)

    cv2.imshow("ISL Smart Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        auto_capture = True
    if key == ord('q'):
        break

    # ─── Auto capture ────────────────────────────────────
    if auto_capture and hand_detected:
        current_time = time.time()
        if current_time - last_capture_time > CAPTURE_INTERVAL:
            # Save the un-annotated frame (flip the original capture)
            ret2, raw_frame = cap.read()
            if ret2:
                raw_frame = cv2.flip(raw_frame, 1)
                filename = os.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(filename, raw_frame)
                count += 1
                phase_count += 1
                last_capture_time = current_time

                if phase_count >= CAPTURE_IMAGES_PER_PHASE:
                    current_phase += 1
                    phase_count = 0
                    if current_phase >= len(phases):
                        print(f"[ISL] {CAPTURE_IMAGES_PER_WORD} images captured!")
                        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print(f"[ISL] Capture complete. Total images: {count}")
