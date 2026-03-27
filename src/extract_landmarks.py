"""
Extract landmarks from captured word frames.
Produces .npy files in data/landmarks/ with 84-feature vectors (two-hand).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WORD_FRAMES_DIR, LANDMARKS_DIR, NUM_TWO_HAND_RAW
from hand_tracker import HandTracker


def extract():
    print("=" * 50)
    print("  ISL Landmark Extraction (Two-Hand)")
    print("=" * 50)

    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    tracker = HandTracker(static_mode=True)

    words = sorted([d for d in os.listdir(WORD_FRAMES_DIR)
                    if os.path.isdir(os.path.join(WORD_FRAMES_DIR, d))])

    if not words:
        print("[ISL] No word folders found in", WORD_FRAMES_DIR)
        return

    total_samples = 0
    for word in words:
        word_dir = os.path.join(WORD_FRAMES_DIR, word)
        images = sorted([f for f in os.listdir(word_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not images:
            print(f"  {word}: 0 images (skipped)")
            continue

        landmarks_list = []
        for img_file in images:
            import cv2
            frame = cv2.imread(os.path.join(word_dir, img_file))
            if frame is None:
                continue

            landmarks, _ = tracker.process(frame)
            if landmarks is not None:
                # landmarks is now 84-dim (two-hand)
                landmarks_list.append(landmarks)

        if landmarks_list:
            data = np.array(landmarks_list, dtype=np.float32)
            save_path = os.path.join(LANDMARKS_DIR, f"{word}.npy")
            np.save(save_path, data)
            total_samples += len(data)
            print(f"  {word}: {len(data)}/{len(images)} frames → {data.shape}")
        else:
            print(f"  {word}: No hands detected (skipped)")

    tracker.close()

    print(f"\n{'=' * 50}")
    print(f"  Done! {total_samples} total samples from {len(words)} words")
    print(f"  Feature size: {NUM_TWO_HAND_RAW} per sample")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    extract()