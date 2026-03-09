"""
Feature engineering for ISL hand landmarks.
Computes extended features (distances + angles) from raw 42-coord landmarks.
Shared by both training and inference pipelines.
"""

import numpy as np


# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIP_JOINTS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
MCP_JOINTS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
DIP_JOINTS = [THUMB_MCP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]


def _get_point(landmarks, idx):
    """Extract (x, y) for landmark index from flat array."""
    return landmarks[idx * 2], landmarks[idx * 2 + 1]


def _distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle_at_joint(a, b, c):
    """Angle (radians) at point b, formed by points a-b-c."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = np.sqrt(ba[0] ** 2 + ba[1] ** 2) + 1e-8
    mag_bc = np.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-8
    cos_angle = np.clip(dot / (mag_ba * mag_bc), -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_extended_features_single(landmarks):
    """
    Compute extended features for a single 42-element landmark vector.

    Returns: np.array of shape (60,) — 42 raw + 18 engineered features.
    """
    features = list(landmarks)  # start with raw 42

    wrist = _get_point(landmarks, WRIST)

    # --- 5 fingertip-to-wrist distances ---
    for tip in FINGERTIPS:
        features.append(_distance(_get_point(landmarks, tip), wrist))

    # --- 4 adjacent fingertip distances ---
    for i in range(len(FINGERTIPS) - 1):
        p1 = _get_point(landmarks, FINGERTIPS[i])
        p2 = _get_point(landmarks, FINGERTIPS[i + 1])
        features.append(_distance(p1, p2))

    # --- 4 thumb-to-other-fingertip distances ---
    thumb = _get_point(landmarks, THUMB_TIP)
    for tip in FINGERTIPS[1:]:  # skip thumb itself
        features.append(_distance(thumb, _get_point(landmarks, tip)))

    # --- 5 finger curl angles (at PIP/IP joint) ---
    for i in range(5):
        a = _get_point(landmarks, MCP_JOINTS[i])
        b = _get_point(landmarks, PIP_JOINTS[i])
        c = _get_point(landmarks, DIP_JOINTS[i])
        features.append(_angle_at_joint(a, b, c))

    return np.array(features, dtype=np.float32)


def compute_extended_features(X):
    """
    Compute extended features for a batch of landmark vectors.

    Args:
        X: np.array of shape (N, 42) — raw wrist-centered landmarks.

    Returns:
        np.array of shape (N, 60) — extended feature vectors.
    """
    return np.array(
        [compute_extended_features_single(row) for row in X],
        dtype=np.float32
    )


# Number of extended features produced
NUM_EXTENDED = 60
