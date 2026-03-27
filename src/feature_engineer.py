"""
Feature engineering for ISL hand landmarks.
Computes extended features (distances + angles) from raw landmarks.
Supports single-hand (42→60) and two-hand (84→120).
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


def compute_extended_features_single(landmarks_42):
    """
    Compute extended features for a single 42-element landmark vector.
    Returns: np.array of shape (60,) — 42 raw + 18 engineered features.
    """
    features = list(landmarks_42)

    wrist = _get_point(landmarks_42, WRIST)

    # 5 fingertip-to-wrist distances
    for tip in FINGERTIPS:
        features.append(_distance(_get_point(landmarks_42, tip), wrist))

    # 4 adjacent fingertip distances
    for i in range(len(FINGERTIPS) - 1):
        p1 = _get_point(landmarks_42, FINGERTIPS[i])
        p2 = _get_point(landmarks_42, FINGERTIPS[i + 1])
        features.append(_distance(p1, p2))

    # 4 thumb-to-other-fingertip distances
    thumb = _get_point(landmarks_42, THUMB_TIP)
    for tip in FINGERTIPS[1:]:
        features.append(_distance(thumb, _get_point(landmarks_42, tip)))

    # 5 finger curl angles
    for i in range(5):
        a = _get_point(landmarks_42, MCP_JOINTS[i])
        b = _get_point(landmarks_42, PIP_JOINTS[i])
        c = _get_point(landmarks_42, DIP_JOINTS[i])
        features.append(_angle_at_joint(a, b, c))

    return np.array(features, dtype=np.float32)


def compute_extended_features(X):
    """Batch version for single-hand: (N, 42) → (N, 60)."""
    return np.array(
        [compute_extended_features_single(row) for row in X],
        dtype=np.float32
    )


def compute_two_hand_features_single(landmarks_84):
    """
    Compute extended features for a two-hand 84-element landmark vector.
    Splits into two 42-vectors, engineers each, concatenates.
    Returns: np.array of shape (120,) — 60 per hand × 2.
    """
    hand1 = landmarks_84[:42]
    hand2 = landmarks_84[42:]
    feat1 = compute_extended_features_single(hand1)
    feat2 = compute_extended_features_single(hand2)
    return np.concatenate([feat1, feat2])


def compute_two_hand_features(X):
    """Batch version for two-hand: (N, 84) → (N, 120)."""
    return np.array(
        [compute_two_hand_features_single(row) for row in X],
        dtype=np.float32
    )


# Feature counts
NUM_EXTENDED = 60
NUM_TWO_HAND_EXT = 120
