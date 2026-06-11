"""Gesture lifecycle FSM (§4.4) and the wake-gesture arming layer (§4.5).

    IDLE -> CANDIDATE -(stable confirm_frames)-> ACTIVE -> ... -> COOLDOWN

The FSM turns noisy per-frame classifications into a small stream of events:

  ArmedChanged       arming state flipped (drive the tray icon)
  GestureActivated   a confirmed gesture fired (discrete actions run here)
  GestureHeld        emitted every frame while the gesture is held
  GestureReleased    the gesture ended (continuous actions finalize here)

Swipes arrive pre-debounced from the classifier and emit a bare
GestureActivated. The system is DISARMED by default: open palm held
``arm_hold_s`` arms it for ``arm_timeout_s`` (refreshed by activity), a held
fist disarms. While disarmed the FSM still tracks pose lifecycle but emits no
gesture events — only ArmedChanged.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .classifier import ClassifiedFrame
from .features import HandFeatures

ARM_POSE = "open_palm"
DISARM_POSE = "fist"


# -- events ----------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    t: float


@dataclass(frozen=True)
class ArmedChanged(Event):
    armed: bool


@dataclass(frozen=True)
class GestureActivated(Event):
    name: str
    features: HandFeatures | None


@dataclass(frozen=True)
class GestureHeld(Event):
    name: str
    features: HandFeatures | None
    held_s: float


@dataclass(frozen=True)
class GestureReleased(Event):
    name: str
    features: HandFeatures | None


# -- config / state ---------------------------------------------------------

@dataclass
class FSMConfig:
    confirm_frames: int = 5
    cooldown_ms: int = 400
    arm_hold_s: float = 1.0
    disarm_hold_s: float = 1.0
    arm_timeout_s: float = 10.0
    # A pose only confirms while the hand is reasonably still; this kills
    # false activations from a hand moving through a pose-like shape.
    max_activation_speed: float = 0.8


class _Phase(enum.Enum):
    IDLE = enum.auto()
    CANDIDATE = enum.auto()
    ACTIVE = enum.auto()
    COOLDOWN = enum.auto()


class GestureFSM:
    def __init__(self, cfg: FSMConfig | None = None) -> None:
        self.cfg = cfg or FSMConfig()
        self.armed = False
        self._armed_until = 0.0
        self._arm_hold_start: float | None = None
        self._disarm_hold_start: float | None = None

        self._phase = _Phase.IDLE
        self._pose: str | None = None        # candidate or active pose
        self._stable_count = 0
        self._activated_at = 0.0
        self._active_emitted = False         # was Activated emitted (armed)?
        self._cooldown_until = 0.0

    def update(self, cf: ClassifiedFrame) -> list[Event]:
        events: list[Event] = []
        self._update_arming(cf, events)
        self._update_pose(cf, events)
        self._update_swipe(cf, events)
        return events

    # -- arming -------------------------------------------------------------

    def _set_armed(self, t: float, armed: bool, events: list[Event]) -> None:
        if armed != self.armed:
            self.armed = armed
            events.append(ArmedChanged(t, armed))
        if armed:
            self._armed_until = t + self.cfg.arm_timeout_s

    def _update_arming(self, cf: ClassifiedFrame, events: list[Event]) -> None:
        t = cf.t
        if self.armed and t >= self._armed_until:
            self._set_armed(t, False, events)

        if cf.pose == ARM_POSE:
            if self._arm_hold_start is None:
                self._arm_hold_start = t
            elif t - self._arm_hold_start >= self.cfg.arm_hold_s:
                self._set_armed(t, True, events)  # also refreshes the timeout
        else:
            self._arm_hold_start = None

        if cf.pose == DISARM_POSE and self.armed:
            if self._disarm_hold_start is None:
                self._disarm_hold_start = t
            elif t - self._disarm_hold_start >= self.cfg.disarm_hold_s:
                self._set_armed(t, False, events)
                self._disarm_hold_start = None
        elif cf.pose != DISARM_POSE:
            self._disarm_hold_start = None

    # -- pose lifecycle -------------------------------------------------------

    def _release_active(self, t: float, features, events: list[Event]) -> None:
        # Cooldown only applies when an action actually fired; an unarmed or
        # unmapped hold (e.g. the arming palm itself) shouldn't delay the
        # next gesture.
        if self._active_emitted and self._pose is not None:
            events.append(GestureReleased(t, self._pose, features))
            self._phase = _Phase.COOLDOWN
            self._cooldown_until = t + self.cfg.cooldown_ms / 1000.0
        else:
            self._phase = _Phase.IDLE
        self._pose = None
        self._active_emitted = False

    def _update_pose(self, cf: ClassifiedFrame, events: list[Event]) -> None:
        t, pose = cf.t, cf.pose
        cfg = self.cfg

        if self._phase == _Phase.COOLDOWN:
            if t < self._cooldown_until:
                return
            self._phase = _Phase.IDLE

        if self._phase == _Phase.IDLE:
            if pose is not None and cf.speed <= cfg.max_activation_speed:
                self._phase = _Phase.CANDIDATE
                self._pose = pose
                self._stable_count = 1
            return

        if self._phase == _Phase.CANDIDATE:
            if pose != self._pose or cf.speed > cfg.max_activation_speed:
                self._phase = _Phase.IDLE
                self._pose = None
                self._stable_count = 0
                # Re-enter CANDIDATE for the new pose on this same frame.
                if pose is not None and cf.speed <= cfg.max_activation_speed:
                    self._phase = _Phase.CANDIDATE
                    self._pose = pose
                    self._stable_count = 1
                return
            self._stable_count += 1
            if self._stable_count >= cfg.confirm_frames:
                self._phase = _Phase.ACTIVE
                self._activated_at = t
                self._active_emitted = self.armed
                if self.armed:
                    events.append(GestureActivated(t, self._pose, cf.features))
                    self._armed_until = t + cfg.arm_timeout_s
            return

        if self._phase == _Phase.ACTIVE:
            if pose == self._pose:
                if self._active_emitted:
                    events.append(GestureHeld(
                        t, self._pose, cf.features, t - self._activated_at))
                    self._armed_until = t + cfg.arm_timeout_s
            else:
                self._release_active(t, cf.features, events)

    # -- swipes ---------------------------------------------------------------

    def _update_swipe(self, cf: ClassifiedFrame, events: list[Event]) -> None:
        # Swipes are debounced in the classifier (refractory window); they
        # bypass CANDIDATE because a swipe is over before confirm_frames.
        if cf.swipe is not None and self.armed:
            events.append(GestureActivated(cf.t, cf.swipe, cf.features))
            self._armed_until = cf.t + self.cfg.arm_timeout_s
