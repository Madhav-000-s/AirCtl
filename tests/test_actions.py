from types import SimpleNamespace

import numpy as np
import pytest

from airctl.actions import build_action_map
from airctl.backends.fakes import fake_backends
from airctl.dispatcher import Dispatcher
from airctl.fsm import GestureActivated, GestureHeld, GestureReleased

DOC_MAPPING = {
    "peace": {"action": "media.play_pause"},
    "point_up": {"action": "audio.volume_step", "args": "+0.05", "repeat_ms": 250},
    "point_down": {"action": "audio.volume_step", "args": "-0.05", "repeat_ms": 250},
    "shaka": {"action": "audio.mute_toggle"},
    "swipe_right": {"action": "media.next"},
    "swipe_left": {"action": "media.prev"},
    "three": {"action": "key", "chord": "win+shift+s"},
    "pinch": {"action": "audio.volume_pinch", "sensitivity": 1.5},
    "pinch2": {"action": "window.switcher", "step_dist": 0.1},
    "swipe2_right": {"action": "desktop.right"},
    "swipe3_right": {"action": "window.next"},
    "swipe3_left": {"action": "window.prev"},
}


def feat(x=0.5, y=0.5):
    return SimpleNamespace(centroid=np.array([x, y]))


def activate(actions, b, name, t=0.0, features=None):
    actions[name].on_activate(GestureActivated(t, name, features), b)


def test_doc_default_mapping_builds():
    actions = build_action_map(DOC_MAPPING)
    assert set(actions) == set(DOC_MAPPING)


def test_unknown_action_rejected():
    with pytest.raises(ValueError, match="frobnicate"):
        build_action_map({"fist": {"action": "frobnicate"}})


def test_missing_chord_rejected():
    with pytest.raises(ValueError, match="fist"):
        build_action_map({"fist": {"action": "key"}})


def test_discrete_actions_fire():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    activate(actions, b, "peace")
    activate(actions, b, "shaka")
    activate(actions, b, "swipe_right")
    activate(actions, b, "three")
    assert calls == [
        ("media.play_pause",),
        ("audio.mute_toggle", True),
        ("media.next",),
        ("input.tap_chord", "win+shift+s"),
    ]


def test_repeat_action_refires_while_held():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    a = actions["point_up"]
    a.on_activate(GestureActivated(0.0, "point_up", None), b)
    a.on_hold(GestureHeld(0.1, "point_up", None, 0.1), b)    # < 250 ms: no fire
    a.on_hold(GestureHeld(0.3, "point_up", None, 0.3), b)    # fires
    a.on_hold(GestureHeld(0.4, "point_up", None, 0.4), b)    # no fire
    a.on_hold(GestureHeld(0.6, "point_up", None, 0.6), b)    # fires
    assert calls == [("audio.set_volume", 0.55),
                     ("audio.set_volume", 0.6),
                     ("audio.set_volume", 0.65)]


def test_pinch_volume_is_relative_to_engage_point():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    b.audio.volume = 0.5
    a = actions["pinch"]
    a.on_activate(GestureActivated(0.0, "pinch", feat(y=0.6)), b)
    # Hand moves up 0.2 image-heights: 0.5 + 1.5 * 0.2 = 0.8.
    a.on_hold(GestureHeld(0.1, "pinch", feat(y=0.4), 0.1), b)
    assert calls[-1] == ("audio.set_volume", 0.8)
    # Down past the baseline: 0.5 + 1.5 * (-0.2) = 0.2.
    a.on_hold(GestureHeld(0.2, "pinch", feat(y=0.8), 0.2), b)
    assert calls[-1] == ("audio.set_volume", 0.2)


def test_pinch_volume_rate_limited():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    a = actions["pinch"]
    a.on_activate(GestureActivated(0.0, "pinch", feat(y=0.5)), b)
    for i in range(1, 91):  # 90 frames at 90 fps = 1 second
        a.on_hold(GestureHeld(i / 90, "pinch", feat(y=0.5 - i / 900), i / 90), b)
    assert len(calls) <= 33  # ~30 Hz cap


def test_window_switcher_lifecycle():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    a = actions["pinch2"]
    a.on_activate(GestureActivated(0.0, "pinch2", feat(x=0.5)), b)
    a.on_hold(GestureHeld(0.1, "pinch2", feat(x=0.55), 0.1), b)  # < step_dist
    a.on_hold(GestureHeld(0.2, "pinch2", feat(x=0.62), 0.2), b)  # 1 forward
    a.on_hold(GestureHeld(0.3, "pinch2", feat(x=0.83), 0.3), b)  # 2 forward
    a.on_hold(GestureHeld(0.4, "pinch2", feat(x=0.69), 0.4), b)  # 1 back
    a.on_release(GestureReleased(0.5, "pinch2", None), b)
    assert calls == [
        ("window.switcher_begin",),
        ("window.switcher_step", True),
        ("window.switcher_step", True),
        ("window.switcher_step", True),
        ("window.switcher_step", False),
        ("window.switcher_end",),
    ]


def test_window_cycle_steps_through_snapshot():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()  # FakeWindow desktop: [10, 20, 30], fg=10
    a = actions["swipe3_right"]
    # Three quick flicks step forward 10 -> 20 -> 30 -> 10 (wraps).
    a.on_activate(GestureActivated(0.0, "swipe3_right", None), b)
    a.on_activate(GestureActivated(0.3, "swipe3_right", None), b)
    a.on_activate(GestureActivated(0.6, "swipe3_right", None), b)
    assert [c for c in calls if c[0] == "window.focus_window"] == [
        ("window.focus_window", 20),
        ("window.focus_window", 30),
        ("window.focus_window", 10),
    ]


def test_window_cycle_prev_goes_backward():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    a = actions["swipe3_left"]
    a.on_activate(GestureActivated(0.0, "swipe3_left", None), b)  # 10 -> 30 (wrap)
    assert calls[-1] == ("window.focus_window", 30)


def test_window_cycle_pause_restarts_session_from_foreground():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    a = actions["swipe3_right"]
    a.on_activate(GestureActivated(0.0, "swipe3_right", None), b)  # 10 -> 20
    assert b.window.get_foreground() == 20
    # Long pause (> session_gap_s): new session starts from current fg (20).
    a.on_activate(GestureActivated(10.0, "swipe3_right", None), b)  # 20 -> 30
    assert calls[-1] == ("window.focus_window", 30)


def test_window_cycle_no_windows_is_noop():
    actions = build_action_map(DOC_MAPPING)
    b, calls = fake_backends()
    b.window.windows = []
    a = actions["swipe3_right"]
    a.on_activate(GestureActivated(0.0, "swipe3_right", None), b)
    assert calls == []


def test_dispatcher_routes_events_to_actions():
    b, calls = fake_backends()
    actions = build_action_map({"peace": {"action": "media.play_pause"}})
    d = Dispatcher(make_backends=lambda: b, action_map=actions)
    d.start()
    d.submit(GestureActivated(0.0, "peace", None))
    d.submit(GestureActivated(0.1, "unmapped_pose", None))  # silently ignored
    d.stop()
    d.join(timeout=5)
    assert not d.is_alive()
    assert calls == [("media.play_pause",)]
