"""Gesture -> action binding (§5.5, §6.2).

Config maps a gesture name to an action spec table, e.g.::

    [modes.default]
    "peace"     = { action = "media.play_pause" }
    "point_up"  = { action = "audio.volume_step", args = "+0.05", repeat_ms = 250 }
    "pinch"     = { action = "audio.volume_pinch", sensitivity = 1.5 }
    "pinch2"    = { action = "window.switcher" }
    "three"     = { action = "key", chord = "win+shift+s" }
    "swipe_right" = { action = "media.next" }

Discrete actions fire on activation; repeat actions re-fire while held;
continuous actions (pinch volume, the window switcher) consume the whole
activate/hold/release lifecycle.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .backends.base import Backends
from .fsm import GestureActivated, GestureHeld, GestureReleased

log = logging.getLogger(__name__)

# Volume updates are rate-limited to ~30/s — not for performance (COM calls
# are sub-ms) but to avoid visual flicker in the system volume OSD (§3.2).
VOLUME_UPDATE_HZ = 30.0


class Action(Protocol):
    def on_activate(self, ev: GestureActivated, b: Backends) -> None: ...
    def on_hold(self, ev: GestureHeld, b: Backends) -> None: ...
    def on_release(self, ev: GestureReleased, b: Backends) -> None: ...


class _DiscreteAction:
    """Fires once on activation; optionally re-fires every repeat_ms held."""

    def __init__(self, repeat_ms: int | None = None) -> None:
        self._repeat_s = repeat_ms / 1000.0 if repeat_ms else None
        self._last_fire = 0.0

    def fire(self, b: Backends) -> None:  # overridden
        raise NotImplementedError

    def on_activate(self, ev: GestureActivated, b: Backends) -> None:
        self._last_fire = ev.t
        self.fire(b)

    def on_hold(self, ev: GestureHeld, b: Backends) -> None:
        if self._repeat_s and ev.t - self._last_fire >= self._repeat_s:
            self._last_fire = ev.t
            self.fire(b)

    def on_release(self, ev: GestureReleased, b: Backends) -> None:
        pass


class KeyChordAction(_DiscreteAction):
    def __init__(self, chord: str, repeat_ms: int | None = None) -> None:
        super().__init__(repeat_ms)
        self.chord = chord

    def fire(self, b: Backends) -> None:
        b.input.tap_chord(self.chord)


class VolumeStepAction(_DiscreteAction):
    def __init__(self, delta: float, repeat_ms: int | None = None) -> None:
        super().__init__(repeat_ms)
        self.delta = delta

    def fire(self, b: Backends) -> None:
        b.audio.step_volume(self.delta)


class MuteToggleAction(_DiscreteAction):
    def fire(self, b: Backends) -> None:
        b.audio.mute_toggle()


class MediaAction(_DiscreteAction):
    def __init__(self, kind: str, repeat_ms: int | None = None) -> None:
        super().__init__(repeat_ms)
        self.kind = kind

    def fire(self, b: Backends) -> None:
        {"play_pause": b.media.play_pause,
         "next": b.media.next_track,
         "prev": b.media.prev_track}[self.kind]()


class ShellAction(_DiscreteAction):
    def __init__(self, cmd: str) -> None:
        super().__init__()
        self.cmd = cmd

    def fire(self, b: Backends) -> None:
        b.custom.run_shell(self.cmd)


class LaunchAction(_DiscreteAction):
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def fire(self, b: Backends) -> None:
        b.custom.launch(self.target)


class VolumePinchAction:
    """Continuous relative volume: pinch, then drag up/down.

    The volume at engage time is the baseline, so there is no jump when the
    pinch starts (the "relative" choice from design-doc open question #2).
    ``sensitivity`` is the full-volume-range fraction per image-height of
    hand travel; image y points down, so moving the hand up raises volume.
    """

    def __init__(self, sensitivity: float = 1.5) -> None:
        self.sensitivity = sensitivity
        self._base_y = 0.0
        self._base_vol = 0.0
        self._last_update = 0.0

    def on_activate(self, ev: GestureActivated, b: Backends) -> None:
        if ev.features is None:
            return
        self._base_y = float(ev.features.centroid[1])
        self._base_vol = b.audio.get_volume()
        self._last_update = 0.0

    def on_hold(self, ev: GestureHeld, b: Backends) -> None:
        if ev.features is None:
            return
        if ev.t - self._last_update < 1.0 / VOLUME_UPDATE_HZ:
            return
        self._last_update = ev.t
        dy = self._base_y - float(ev.features.centroid[1])  # up = positive
        b.audio.set_volume(self._base_vol + self.sensitivity * dy)

    def on_release(self, ev: GestureReleased, b: Backends) -> None:
        pass


class WindowSwitcherAction:
    """Pinch-hold Alt+Tab (§5.3 tier 1): activation opens the native
    switcher, horizontal hand travel steps through it, release selects.

    Position-driven rather than velocity-driven: every ``step_dist`` of
    horizontal travel from the last step point advances one window, so the
    hand works like a carousel you slide.
    """

    def __init__(self, step_dist: float = 0.10) -> None:
        self.step_dist = step_dist
        self._anchor_x = 0.0

    def on_activate(self, ev: GestureActivated, b: Backends) -> None:
        if ev.features is None:
            return
        self._anchor_x = float(ev.features.centroid[0])
        b.window.switcher_begin()

    def on_hold(self, ev: GestureHeld, b: Backends) -> None:
        if ev.features is None:
            return
        x = float(ev.features.centroid[0])
        while x - self._anchor_x >= self.step_dist:
            b.window.switcher_step(forward=True)
            self._anchor_x += self.step_dist
        while self._anchor_x - x >= self.step_dist:
            b.window.switcher_step(forward=False)
            self._anchor_x -= self.step_dist

    def on_release(self, ev: GestureReleased, b: Backends) -> None:
        b.window.switcher_end()


class DesktopSwitchAction(_DiscreteAction):
    def __init__(self, direction: str) -> None:
        super().__init__()
        self.direction = direction

    def fire(self, b: Backends) -> None:
        b.window.desktop_switch(self.direction)


class WindowCycleAction:
    """Flick to the next/previous window via direct Win32 targeting (§5.3
    tier 2). Each swipe brings one adjacent window straight to the front —
    no Alt+Tab overlay.

    Cycling needs a stable ordering, but every focus change reshuffles the
    Z-order, so we snapshot the window list at the start of a "session" and
    walk an index through it. Swipes within ``session_gap_s`` of each other
    continue the same session (so three quick flicks step 1-2-3 through the
    snapshot); a longer pause starts fresh from the current foreground window.
    """

    def __init__(self, direction: str, session_gap_s: float = 2.0) -> None:
        if direction not in ("next", "prev"):
            raise ValueError(f"bad window direction {direction!r}")
        self.direction = direction
        self.session_gap_s = session_gap_s
        self._windows: list[int] = []
        self._cursor = 0
        self._last_t = float("-inf")

    def on_activate(self, ev: GestureActivated, b: Backends) -> None:
        windows = b.window.list_windows()
        if not windows:
            return
        fresh = (ev.t - self._last_t > self.session_gap_s
                 or not self._windows)
        self._last_t = ev.t
        if fresh:
            self._windows = windows
            fg = b.window.get_foreground()
            self._cursor = windows.index(fg) if fg in windows else 0
        step = 1 if self.direction == "next" else -1
        self._cursor = (self._cursor + step) % len(self._windows)
        b.window.focus_window(self._windows[self._cursor])

    def on_hold(self, ev: GestureHeld, b: Backends) -> None:
        pass

    def on_release(self, ev: GestureReleased, b: Backends) -> None:
        pass


def build_action(spec: dict) -> Action:
    """Build one Action from a config spec table. Raises ValueError on a bad
    spec so config reload can reject the file without crashing."""
    kind = spec.get("action")
    repeat_ms = spec.get("repeat_ms")
    match kind:
        case "media.play_pause" | "media.next" | "media.prev":
            return MediaAction(kind.removeprefix("media."), repeat_ms)
        case "audio.volume_step":
            return VolumeStepAction(float(spec.get("args", "+0.05")), repeat_ms)
        case "audio.mute_toggle":
            return MuteToggleAction(repeat_ms)
        case "audio.volume_pinch":
            return VolumePinchAction(float(spec.get("sensitivity", 1.5)))
        case "window.switcher":
            return WindowSwitcherAction(float(spec.get("step_dist", 0.10)))
        case "window.next" | "window.prev":
            return WindowCycleAction(kind.removeprefix("window."),
                                     float(spec.get("session_gap_s", 2.0)))
        case "desktop.left" | "desktop.right":
            return DesktopSwitchAction(kind.removeprefix("desktop."))
        case "key":
            return KeyChordAction(str(spec["chord"]), repeat_ms)
        case "shell":
            return ShellAction(str(spec["cmd"]))
        case "launch":
            return LaunchAction(str(spec["target"]))
        case _:
            raise ValueError(f"unknown action {kind!r}")


def build_action_map(mapping: dict[str, dict]) -> dict[str, Action]:
    """gesture name -> Action, for one mode's config table."""
    actions: dict[str, Action] = {}
    for gesture, spec in mapping.items():
        try:
            actions[gesture] = build_action(spec)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"bad action spec for {gesture!r}: {e}") from e
    return actions
