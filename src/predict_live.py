
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("../model.h5")
labels = [
    'ALL',
    'ANGRY',
    'COLD',
    'CONGRATULATIONS',
    'HEAR',
    'HEART',
    'LIKE',
    'NO',
    'PHONE',
    'PLEASE',
    'SORRY',
    'THIRSTY',
    'WHERE',
    'YES',
    'YOU'
]



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# 🧠 Sentence builder variables
sentence = []
last_word = ""
word_counter = 0
threshold = 0.80   # same as your confidence
stable_frames = 12 # how many frames word must repeat

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    word = "No clear gesture"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        data = []
        for lm in hand.landmark:
            data.append(lm.x)
            data.append(lm.y)

        input_data = np.array(data).reshape(1, 42)

        pred = model.predict(input_data, verbose=0)[0]
        confidence = np.max(pred)
        label_index = np.argmax(pred)

        if confidence > threshold:
            word = labels[label_index]

            # 🧠 Sentence logic
            if word == last_word:
                word_counter += 1
            else:
                word_counter = 0

            last_word = word

            if word_counter == stable_frames:
                sentence.append(word)

    # Show current word
    cv2.putText(frame, "Word: " + word, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Show sentence
    cv2.putText(frame, "Sentence: " + " ".join(sentence), (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):   # Clear sentence
        sentence = []

    if key == ord('q'):   # Quit
        break

cap.release()
cv2.destroyAllWindows()

