"""Tier-1 rule-based pose classification + velocity swipe detection (§4.3).

Static poses come from the finger-extension bitmask; pinches use hysteresis so
the pose doesn't flicker at the threshold boundary. Swipes are detected from
the smoothed hand-centroid velocity, gated on a pose so incidental hand motion
doesn't trigger them:

  open palm + fast horizontal motion  -> swipe_left / swipe_right
  peace sign + fast horizontal motion -> swipe2_left / swipe2_right

Note on palm orientation: the design doc gates open_palm on the palm facing
the camera, but the z-sign convention is hardware/lighting dependent enough
that v1 does not gate on it (a wrong sign would silently break arming). The
value is computed and shown in --debug-poses for future tuning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import INDEX_TIP, MIDDLE_TIP, HandFeatures

# Finger bitmask -> pose name, order (thumb, index, middle, ring, pinky).
# None entries mean "don't care" for that finger.
_POSE_PATTERNS: tuple[tuple[tuple[bool | None, ...], str], ...] = (
    ((True, True, True, True, True), "open_palm"),
    ((False, False, False, False, False), "fist"),
    ((True, False, False, False, True), "shaka"),
    ((None, True, True, True, False), "three"),
    ((None, True, True, False, False), "peace"),
    ((None, True, False, False, False), "point"),
)

# Index direction (image y, normalized by hand presence in frame) beyond which
# a "point" splits into point_up / point_down.
_POINT_DIR_THRESHOLD = 0.0


@dataclass(frozen=True)
class ClassifiedFrame:
    """Classifier output for one frame."""

    t: float
    pose: str | None              # stable-pose name or None (no hand / unknown)
    swipe: str | None             # one-shot swipe event name or None
    speed: float                  # centroid speed, image-widths per second
    features: HandFeatures | None


@dataclass
class ClassifierConfig:
    pinch_enter: float = 0.30     # hand-relative units (wrist->middle MCP = 1)
    pinch_exit: float = 0.45
    # A pinch only counts while the pinching fingertip reaches at least this
    # far from the wrist — in a fist the curled thumb sits close to the curled
    # fingertips, which would otherwise read as a pinch.
    pinch_min_reach: float = 1.0
    swipe_velocity: float = 1.2   # image-widths per second
    swipe_refractory_s: float = 0.5
    velocity_window_s: float = 0.15


@dataclass
class _PinchState:
    active: bool = False

    def update(self, dist: float, enter: float, exit_: float) -> bool:
        if self.active:
            if dist > exit_:
                self.active = False
        elif dist < enter:
            self.active = True
        return self.active


class PoseClassifier:
    """Stateful per-frame classifier (pinch hysteresis + velocity history)."""

    def __init__(self, cfg: ClassifierConfig | None = None) -> None:
        self.cfg = cfg or ClassifierConfig()
        self._pinch_index = _PinchState()
        self._pinch_middle = _PinchState()
        self._history: deque[tuple[float, np.ndarray]] = deque(maxlen=30)
        self._last_swipe_t = -1e9

    def reset(self) -> None:
        self._pinch_index.active = False
        self._pinch_middle.active = False
        self._history.clear()

    def classify(self, t: float, features: HandFeatures | None) -> ClassifiedFrame:
        if features is None:
            self.reset()
            return ClassifiedFrame(t, None, None, 0.0, None)

        velocity = self._update_velocity(t, features.centroid)
        speed = float(np.linalg.norm(velocity))
        pose = self._classify_pose(features)
        swipe = self._detect_swipe(t, pose, velocity)
        return ClassifiedFrame(t, pose, swipe, speed, features)

    # -- internals ---------------------------------------------------------

    def _classify_pose(self, f: HandFeatures) -> str | None:
        cfg = self.cfg
        inf = float("inf")
        index_reach = float(np.linalg.norm(f.norm[INDEX_TIP, :2]))
        middle_reach = float(np.linalg.norm(f.norm[MIDDLE_TIP, :2]))
        index_pinched = self._pinch_index.update(
            f.pinch_index if index_reach > cfg.pinch_min_reach else inf,
            cfg.pinch_enter, cfg.pinch_exit)
        middle_pinched = self._pinch_middle.update(
            f.pinch_middle if middle_reach > cfg.pinch_min_reach else inf,
            cfg.pinch_enter, cfg.pinch_exit)

        # Pinches take precedence over bitmask poses. Thumb-middle ("pinch2",
        # the window switcher) only counts when the index is clearly NOT
        # pinched, since a middle pinch drags the thumb near the index too.
        if index_pinched:
            return "pinch"
        if middle_pinched:
            return "pinch2"

        states = f.finger_states
        for pattern, name in _POSE_PATTERNS:
            if all(p is None or p == s for p, s in zip(pattern, states)):
                if name == "point":
                    return "point_up" if f.index_dir_y < _POINT_DIR_THRESHOLD \
                        else "point_down"
                return name
        return None

    def _update_velocity(self, t: float, centroid: np.ndarray) -> np.ndarray:
        self._history.append((t, centroid.copy()))
        window = self.cfg.velocity_window_s
        # Compare against the oldest sample still inside the window.
        oldest_t, oldest_c = self._history[0]
        for ht, hc in self._history:
            if t - ht <= window:
                oldest_t, oldest_c = ht, hc
                break
        dt = t - oldest_t
        if dt <= 1e-6:
            return np.zeros(2)
        return (centroid - oldest_c) / dt

    def _detect_swipe(
        self, t: float, pose: str | None, velocity: np.ndarray
    ) -> str | None:
        cfg = self.cfg
        if pose == "open_palm":
            prefix = "swipe"
        elif pose == "peace":
            prefix = "swipe2"
        else:
            return None
        if t - self._last_swipe_t < cfg.swipe_refractory_s:
            return None
        vx, vy = float(velocity[0]), float(velocity[1])
        if abs(vx) < cfg.swipe_velocity or abs(vx) < 1.5 * abs(vy):
            return None
        self._last_swipe_t = t
        return f"{prefix}_right" if vx > 0 else f"{prefix}_left"
