"""Recording fakes so the full pipeline is testable with no audio device,
display, or input permissions (§8)."""

from __future__ import annotations

from .base import Backends


class _Recorder:
    def __init__(self, calls: list) -> None:
        self.calls = calls


class FakeAudio(_Recorder):
    def __init__(self, calls: list, volume: float = 0.5) -> None:
        super().__init__(calls)
        self.volume = volume
        self.muted = False

    def get_volume(self) -> float:
        return self.volume

    def set_volume(self, level: float) -> None:
        self.volume = min(1.0, max(0.0, level))
        self.calls.append(("audio.set_volume", round(self.volume, 4)))

    def step_volume(self, delta: float) -> None:
        self.set_volume(self.volume + delta)

    def mute_toggle(self) -> None:
        self.muted = not self.muted
        self.calls.append(("audio.mute_toggle", self.muted))


class FakeInput(_Recorder):
    def tap_chord(self, chord: str) -> None:
        self.calls.append(("input.tap_chord", chord))

    def tap_vk(self, vk: int) -> None:
        self.calls.append(("input.tap_vk", vk))

    def scan_key_down(self, vk: int) -> None:
        self.calls.append(("input.scan_key_down", vk))

    def scan_key_up(self, vk: int) -> None:
        self.calls.append(("input.scan_key_up", vk))

    def scan_tap(self, vk: int, with_shift: bool = False) -> None:
        self.calls.append(("input.scan_tap", vk, with_shift))

    def scroll(self, delta: int) -> None:
        self.calls.append(("input.scroll", delta))


class FakeWindow(_Recorder):
    def __init__(self, calls: list, windows: list[int] | None = None,
                 foreground: int | None = None) -> None:
        super().__init__(calls)
        # A small fake desktop: window handles in Z-order, frontmost first.
        self.windows = windows if windows is not None else [10, 20, 30]
        self.foreground = (foreground if foreground is not None
                           else (self.windows[0] if self.windows else 0))

    def switcher_begin(self) -> None:
        self.calls.append(("window.switcher_begin",))

    def switcher_step(self, forward: bool) -> None:
        self.calls.append(("window.switcher_step", forward))

    def switcher_end(self) -> None:
        self.calls.append(("window.switcher_end",))

    def desktop_switch(self, direction: str) -> None:
        self.calls.append(("window.desktop_switch", direction))

    def list_windows(self) -> list[int]:
        return list(self.windows)

    def get_foreground(self) -> int:
        return self.foreground

    def window_title(self, hwnd: int) -> str:
        return f"window-{hwnd}"

    def focus_window(self, hwnd: int) -> None:
        self.foreground = hwnd
        self.calls.append(("window.focus_window", hwnd))


class FakeMedia(_Recorder):
    def play_pause(self) -> None:
        self.calls.append(("media.play_pause",))

    def next_track(self) -> None:
        self.calls.append(("media.next",))

    def prev_track(self) -> None:
        self.calls.append(("media.prev",))


class FakeCustom(_Recorder):
    def run_shell(self, cmd: str) -> None:
        self.calls.append(("custom.run_shell", cmd))

    def launch(self, target: str) -> None:
        self.calls.append(("custom.launch", target))


def fake_backends() -> tuple[Backends, list]:
    """One shared call log across all fakes, in dispatch order."""
    calls: list = []
    return Backends(
        audio=FakeAudio(calls),
        input=FakeInput(calls),
        window=FakeWindow(calls),
        media=FakeMedia(calls),
        custom=FakeCustom(calls),
    ), calls
