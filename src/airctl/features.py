"""Landmark normalization and per-frame derived features (design doc §4.1).

Raw MediaPipe coordinates vary with hand position, distance, and rotation, so
classification runs on a normalized copy:

1. translate so the wrist is the origin,
2. scale by the wrist -> middle-MCP distance,
3. rotate in the image plane so wrist -> middle-MCP points "up".

All thresholds below are expressed in these hand-relative units, where the
wrist -> middle-MCP distance is exactly 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Landmark indices (Appendix A of the design doc).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# (PIP, TIP) per non-thumb finger, in (index, middle, ring, pinky) order.
_FINGER_JOINTS = ((INDEX_PIP, INDEX_TIP), (MIDDLE_PIP, MIDDLE_TIP),
                  (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP))

# A finger counts as extended when its tip is this much farther from the
# wrist than its PIP joint (hand-relative units).
EXTENSION_MARGIN = 0.10
# Thumb is "extended" when its tip is at least this far from the index MCP
# (lateral test — the radial-distance test misfires for thumbs).
THUMB_EXTENSION_DIST = 0.70


@dataclass(frozen=True)
class HandFeatures:
    """Everything the classifier needs about one hand in one frame."""

    landmarks: np.ndarray      # (21, 3) image-normalized, filtered
    norm: np.ndarray           # (21, 3) translated/scaled/rotated
    handedness: str            # "Left" or "Right"
    finger_states: tuple[bool, bool, bool, bool, bool]  # thumb..pinky
    pinch_index: float         # thumb-tip <-> index-tip distance (hand units)
    pinch_middle: float        # thumb-tip <-> middle-tip distance (hand units)
    palm_facing: bool          # palm normal points at the camera
    centroid: np.ndarray       # (2,) mean landmark position, image coords
    index_dir_y: float         # index tip-to-MCP direction, image y (neg = up)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate/scale/rotate landmarks into the hand-relative frame."""
    lm = np.asarray(landmarks, dtype=np.float64)
    centered = lm - lm[WRIST]

    ref = centered[MIDDLE_MCP, :2]
    scale = float(np.linalg.norm(ref))
    if scale < 1e-6:
        return centered  # degenerate detection; caller's thresholds won't match
    out = centered / scale

    # Rotate the image plane so wrist -> middle-MCP points to (0, -1)
    # ("up" with image y pointing down).
    angle = np.arctan2(out[MIDDLE_MCP, 0], -out[MIDDLE_MCP, 1])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    out[:, :2] = out[:, :2] @ rot.T
    return out


def _finger_extensions(norm: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    dists = np.linalg.norm(norm[:, :2], axis=1)  # wrist is the origin
    fingers = tuple(
        bool(dists[tip] > dists[pip] + EXTENSION_MARGIN)
        for pip, tip in _FINGER_JOINTS
    )
    thumb = bool(
        np.linalg.norm(norm[THUMB_TIP, :2] - norm[INDEX_MCP, :2])
        > THUMB_EXTENSION_DIST
    )
    return (thumb, *fingers)


def _palm_facing(norm: np.ndarray, handedness: str) -> bool:
    # Normal of the wrist / index-MCP / pinky-MCP triangle. The z-sign
    # convention was fixed empirically for the mirrored (selfie-view) frame;
    # v1 computes this for debugging but no pose gates on it (see classifier).
    v1 = norm[INDEX_MCP] - norm[WRIST]
    v2 = norm[PINKY_MCP] - norm[WRIST]
    nz = float(np.cross(v1, v2)[2])
    return nz < 0 if handedness == "Right" else nz > 0


def extract_features(landmarks: np.ndarray, handedness: str) -> HandFeatures:
    lm = np.asarray(landmarks, dtype=np.float64)
    norm = normalize_landmarks(lm)
    index_dir = lm[INDEX_TIP] - lm[INDEX_MCP]
    return HandFeatures(
        landmarks=lm,
        norm=norm,
        handedness=handedness,
        finger_states=_finger_extensions(norm),
        pinch_index=float(np.linalg.norm(norm[THUMB_TIP] - norm[INDEX_TIP])),
        pinch_middle=float(np.linalg.norm(norm[THUMB_TIP] - norm[MIDDLE_TIP])),
        palm_facing=_palm_facing(norm, handedness),
        centroid=lm[:, :2].mean(axis=0),
        index_dir_y=float(index_dir[1]),
    )
