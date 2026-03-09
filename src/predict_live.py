import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
from tensorflow.keras.models import load_model

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineer import compute_extended_features_single

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..")

# Load model
model = load_model(os.path.join(OUTPUT_DIR, "model.h5"))
labels = np.load(os.path.join(OUTPUT_DIR, "labels.npy"))
mean = np.load(os.path.join(OUTPUT_DIR, "mean.npy"))
std = np.load(os.path.join(OUTPUT_DIR, "std.npy"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

sentence = []
last_added_word = ""
current_word = ""
gesture_start_time = None

THRESHOLD = 0.90
DIFF_THRESHOLD = 0.25
GESTURE_HOLD_TIME = 1  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    prediction_text = "No clear gesture"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Same preprocessing as training
        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y

        data = []
        for lm in hand.landmark:
            data.append(lm.x - base_x)
            data.append(lm.y - base_y)

        input_data = np.array(data).astype(np.float32)
        input_data = compute_extended_features_single(input_data)
        input_data = (input_data.reshape(1, -1) - mean) / std

        pred = model.predict(input_data, verbose=0)[0]

        sorted_probs = np.sort(pred)
        top1 = sorted_probs[-1]
        top2 = sorted_probs[-2]

        if top1 > THRESHOLD and (top1 - top2) > DIFF_THRESHOLD:
            word = labels[np.argmax(pred)]
            prediction_text = word

            if word == current_word:
                # If same word, check duration
                if gesture_start_time is not None:
                    elapsed = time.time() - gesture_start_time

                    if elapsed >= GESTURE_HOLD_TIME:
                        if word != last_added_word:
                            sentence.append(word)
                            last_added_word = word
                        gesture_start_time = None
            else:
                # New gesture detected
                current_word = word
                gesture_start_time = time.time()
        else:
            current_word = ""
            gesture_start_time = None

    else:
        current_word = ""
        gesture_start_time = None

    cv2.putText(frame, "Word: " + prediction_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Sentence: " + " ".join(sentence), (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("AI Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        sentence = []
        last_added_word = ""

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()