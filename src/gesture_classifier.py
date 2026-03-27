"""
Fast gesture classifier with TFLite inference and temporal smoothing.
Supports two-hand (120-dim) input with fallback to Keras .h5.
"""

import os
import time
import numpy as np

from config import (
    MODEL_TFLITE_PATH, MODEL_H5_PATH, LABELS_PATH, MEAN_PATH, STD_PATH,
    CONFIDENCE_THRESHOLD, DIFF_THRESHOLD, GESTURE_HOLD_TIME,
    SMOOTHING_ALPHA, NUM_TWO_HAND_EXTENDED
)
from feature_engineer import compute_two_hand_features_single


class GestureClassifier:
    """
    Classifies two-hand landmarks into ISL gesture words.
    Uses EMA smoothing + gesture hold timer for stable predictions.
    """

    def __init__(self):
        self.labels = np.load(LABELS_PATH)
        self.mean = np.load(MEAN_PATH)
        self.std = np.load(STD_PATH)
        self.num_classes = len(self.labels)

        # Try TFLite first, fall back to Keras
        self._use_tflite = False
        if os.path.exists(MODEL_TFLITE_PATH):
            self._load_tflite()
        else:
            self._load_keras()

        # Smoothing state
        self._smoothed_probs = np.zeros(self.num_classes, dtype=np.float32)
        self._current_word = ""
        self._gesture_start_time = None
        self._last_confirmed_word = ""

    def _load_tflite(self):
        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._use_tflite = True
            print(f"[ISL] Loaded TFLite model ({self.num_classes} classes)")
        except Exception as e:
            print(f"[ISL] TFLite failed ({e}), falling back to Keras")
            self._load_keras()

    def _load_keras(self):
        from tensorflow.keras.models import load_model
        self._keras_model = load_model(MODEL_H5_PATH)
        self._use_tflite = False
        print(f"[ISL] Loaded Keras model ({self.num_classes} classes)")

    def _predict_raw(self, input_data):
        if self._use_tflite:
            self._interpreter.set_tensor(
                self._input_details[0]['index'],
                input_data.astype(np.float32)
            )
            self._interpreter.invoke()
            return self._interpreter.get_tensor(
                self._output_details[0]['index']
            )[0]
        else:
            return self._keras_model.predict(input_data, verbose=0)[0]

    def classify(self, landmarks_84):
        """
        Classify a two-hand 84-feature landmark vector.

        Args:
            landmarks_84: np.array shape (84,) — two-hand wrist-centered landmarks.

        Returns:
            dict: word, confidence, status, hold_progress, all_probs
        """
        # Feature engineering: 84 raw → 120 extended
        extended = compute_two_hand_features_single(landmarks_84)

        # Standardize
        num_feats = len(self.mean)
        input_data = (extended[:num_feats].reshape(1, num_feats) - self.mean) / self.std

        # Raw prediction
        raw_probs = self._predict_raw(input_data)

        # EMA smoothing
        self._smoothed_probs = (
            SMOOTHING_ALPHA * raw_probs +
            (1 - SMOOTHING_ALPHA) * self._smoothed_probs
        )

        probs = self._smoothed_probs
        sorted_probs = np.sort(probs)
        top1 = sorted_probs[-1]
        top2 = sorted_probs[-2]

        result = {
            'word': '',
            'confidence': float(top1),
            'status': 'detecting',
            'hold_progress': 0.0,
            'all_probs': probs.copy()
        }

        if top1 > CONFIDENCE_THRESHOLD and (top1 - top2) > DIFF_THRESHOLD:
            word = self.labels[np.argmax(probs)]
            result['word'] = word

            if word == self._current_word:
                if self._gesture_start_time is not None:
                    elapsed = time.time() - self._gesture_start_time
                    progress = min(elapsed / GESTURE_HOLD_TIME, 1.0)
                    result['hold_progress'] = progress

                    if elapsed >= GESTURE_HOLD_TIME:
                        result['status'] = 'confirmed'
                        if word != self._last_confirmed_word:
                            self._last_confirmed_word = word
                        self._gesture_start_time = None
                    else:
                        result['status'] = 'holding'
            else:
                self._current_word = word
                self._gesture_start_time = time.time()
                result['status'] = 'holding'
                result['hold_progress'] = 0.0
        else:
            self._current_word = ""
            self._gesture_start_time = None

        return result

    def reset(self):
        self._smoothed_probs = np.zeros(self.num_classes, dtype=np.float32)
        self._current_word = ""
        self._gesture_start_time = None
        self._last_confirmed_word = ""
