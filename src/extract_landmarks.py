import cv2
import mediapipe as mp
import numpy as np
import os

INPUT_DIR = "../data/word_frames"
OUTPUT_DIR = "../data/landmarks"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for word in os.listdir(INPUT_DIR):

    word_path = os.path.join(INPUT_DIR, word)
    if not os.path.isdir(word_path):
        continue

    print(f"\nProcessing word: {word}")

    data = []
    total_images = 0
    detected = 0

    for img_name in os.listdir(word_path):

        img_path = os.path.join(word_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        total_images += 1

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                data.append(landmark_list)
                detected += 1

    data = np.array(data)
    np.save(os.path.join(OUTPUT_DIR, f"{word}.npy"), data)

    print(f"Total images: {total_images}")
    print(f"Hand detected: {detected}")
    print(f"Detection rate: {detected/total_images*100:.2f}%")

print("\nLandmark extraction completed.")
