"""
Centralized configuration for the ISL Sign Language Detector.
All paths, thresholds, and UI settings in one place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")
WORD_FRAMES_DIR = os.path.join(DATA_DIR, "word_frames")

MODEL_H5_PATH = os.path.join(PROJECT_ROOT, "model.h5")
MODEL_TFLITE_PATH = os.path.join(PROJECT_ROOT, "model.tflite")
LABELS_PATH = os.path.join(PROJECT_ROOT, "labels.npy")
MEAN_PATH = os.path.join(PROJECT_ROOT, "mean.npy")
STD_PATH = os.path.join(PROJECT_ROOT, "std.npy")

# ─── MediaPipe Hand Tracking ─────────────────────────────────────────
MP_MAX_HANDS = 2
MP_MIN_DETECTION_CONF = 0.7
MP_MIN_TRACKING_CONF = 0.7
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 2        # x, y per landmark per hand = 42
NUM_TWO_HAND_RAW = NUM_FEATURES * 2     # 84 (42 per hand × 2 hands)
NUM_EXTENDED_FEATURES = 60              # 42 raw + 18 engineered per hand
NUM_TWO_HAND_EXTENDED = NUM_EXTENDED_FEATURES * 2  # 120 (60 per hand × 2)

# ─── Prediction ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80
DIFF_THRESHOLD = 0.20
GESTURE_HOLD_TIME = 1.0
SMOOTHING_ALPHA = 0.55
NO_HAND_TIMEOUT = 5.0   # seconds before finalizing sentence

# ─── Camera ──────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ─── UI Colors (BGR for OpenCV) ──────────────────────────────────────
COLOR_GREEN = (0, 220, 80)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED = (0, 60, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (100, 100, 100)
COLOR_DARK_BG = (30, 30, 30)
COLOR_ACCENT = (255, 180, 0)
COLOR_SENTENCE_BG = (40, 40, 40)
COLOR_LANDMARK = (0, 255, 200)
COLOR_CONNECTION = (200, 200, 200)

# ─── UI Layout ───────────────────────────────────────────────────────
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.5
FONT_MEDIUM = 0.7
FONT_LARGE = 1.2
FONT_THICKNESS = 2

# ─── Training ────────────────────────────────────────────────────────
MIN_SAMPLES_PER_CLASS = 140
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 10
CAPTURE_IMAGES_PER_WORD = 150
CAPTURE_IMAGES_PER_PHASE = 30
CAPTURE_INTERVAL = 0.4
