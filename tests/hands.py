"""Synthetic hand-landmark builders for camera-free tests.

Hands are constructed in "hand units" (wrist at origin, wrist->middle-MCP
distance = 1.0, +y pointing up) and then mapped into image coordinates
(+y pointing down) with an arbitrary scale, position, and in-plane rotation —
exactly the variation the normalization in features.py must cancel out.
"""

from __future__ import annotations

import numpy as np

# Per-finger joint positions in hand units, (extended, curled) variants.
_FINGER_BASES = {
    "index": np.array([-0.35, 0.95]),
    "middle": np.array([0.0, 1.0]),
    "ring": np.array([0.3, 0.95]),
    "pinky": np.array([0.6, 0.85]),
}
_EXTENDED_OFFSETS = (np.array([0.0, 0.4]), np.array([0.0, 0.65]),
                     np.array([0.0, 0.85]))  # PIP, DIP, TIP from MCP
_CURLED_OFFSETS = (np.array([0.0, 0.25]), np.array([0.05, 0.05]),
                   np.array([0.05, -0.3]))

_THUMB_EXTENDED = (np.array([-0.3, 0.25]), np.array([-0.6, 0.45]),
                   np.array([-0.9, 0.5]), np.array([-1.2, 0.55]))
_THUMB_CURLED = (np.array([-0.3, 0.25]), np.array([-0.35, 0.55]),
                 np.array([-0.25, 0.75]), np.array([-0.15, 0.9]))


def make_hand(
    thumb: bool = True,
    index: bool = True,
    middle: bool = True,
    ring: bool = True,
    pinky: bool = True,
    pinch_dist: float | None = None,
    scale: float = 0.12,
    center: tuple[float, float] = (0.5, 0.5),
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Build a (21, 3) landmark array in image coordinates.

    ``pinch_dist`` (hand units) overrides the thumb so its tip sits that far
    from the index fingertip — for pinch-hysteresis tests.
    """
    pts = np.zeros((21, 2))
    # Thumb: landmarks 1-4.
    for i, p in enumerate(_THUMB_EXTENDED if thumb else _THUMB_CURLED):
        pts[1 + i] = p
    # Fingers: 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
    for base_idx, (name, extended) in zip(
        (5, 9, 13, 17),
        (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)),
    ):
        mcp = _FINGER_BASES[name]
        pts[base_idx] = mcp
        offsets = _EXTENDED_OFFSETS if extended else _CURLED_OFFSETS
        for j, off in enumerate(offsets):
            pts[base_idx + 1 + j] = mcp + off

    if pinch_dist is not None:
        index_tip = pts[8]
        pts[4] = index_tip + np.array([pinch_dist, 0.0])
        pts[3] = index_tip + np.array([pinch_dist + 0.25, -0.15])

    # Hand units -> image: rotate, scale, flip y (image y points down).
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ rot.T
    img = np.empty((21, 3))
    img[:, 0] = center[0] + scale * pts[:, 0]
    img[:, 1] = center[1] - scale * pts[:, 1]
    img[:, 2] = 0.0
    return img


# Convenience pose builders -------------------------------------------------

def open_palm(**kw) -> np.ndarray:
    return make_hand(True, True, True, True, True, **kw)


def fist(**kw) -> np.ndarray:
    return make_hand(False, False, False, False, False, **kw)


def point(**kw) -> np.ndarray:
    return make_hand(False, True, False, False, False, **kw)


def peace(**kw) -> np.ndarray:
    return make_hand(False, True, True, False, False, **kw)


def three(**kw) -> np.ndarray:
    return make_hand(False, True, True, True, False, **kw)


def shaka(**kw) -> np.ndarray:
    return make_hand(True, False, False, False, True, **kw)


def pinch(dist: float = 0.05, **kw) -> np.ndarray:
    return make_hand(True, True, True, True, True, pinch_dist=dist, **kw)
