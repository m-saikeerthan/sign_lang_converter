"""
Extract hand landmarks from captured word frames using MediaPipe.
Outputs .npy landmark files for training.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# Add src to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, NUM_FEATURES

INPUT_DIR = os.path.join(DATA_DIR, "word_frames")
OUTPUT_DIR = os.path.join(DATA_DIR, "landmarks")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[ISL] Extracting landmarks from word frames...")
print(f"  Input:  {INPUT_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print("-" * 50)

summary = []

for word in sorted(os.listdir(INPUT_DIR)):
    word_path = os.path.join(INPUT_DIR, word)
    if not os.path.isdir(word_path):
        continue

    data = []
    total_images = 0
    detected = 0

    for img_name in sorted(os.listdir(word_path)):
        img_path = os.path.join(word_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        total_images += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            base_x = hand.landmark[0].x
            base_y = hand.landmark[0].y

            landmark_list = []
            for lm in hand.landmark:
                landmark_list.append(lm.x - base_x)
                landmark_list.append(lm.y - base_y)

            if len(landmark_list) == NUM_FEATURES:
                data.append(landmark_list)
                detected += 1

    data = np.array(data, dtype=np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{word}.npy"), data)

    rate = (detected / total_images * 100) if total_images > 0 else 0
    summary.append((word, total_images, detected, rate))
    print(f"  {word:20s} | Images: {total_images:4d} | Detected: {detected:4d} | Rate: {rate:.1f}%")

hands.close()

# Summary table
print("\n" + "=" * 50)
print(f"  {'SUMMARY':^46s}")
print("=" * 50)
total_imgs = sum(s[1] for s in summary)
total_det = sum(s[2] for s in summary)
avg_rate = (total_det / total_imgs * 100) if total_imgs > 0 else 0
print(f"  Words:        {len(summary)}")
print(f"  Total images: {total_imgs}")
print(f"  Detected:     {total_det}")
print(f"  Avg rate:     {avg_rate:.1f}%")
print("=" * 50)
print("[ISL] Landmark extraction complete!")