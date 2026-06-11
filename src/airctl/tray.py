"""Tray icon (§6.1): the v1 HUD. Icon color shows state at a glance —
green = armed, gray = disarmed, red = camera paused (released, LED off).
"""

from __future__ import annotations

import logging
from typing import Callable

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

_COLORS = {
    "armed": (52, 199, 89),     # green
    "disarmed": (142, 142, 147),  # gray
    "paused": (255, 69, 58),    # red
}


def _icon_image(state: str, size: int = 64) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = _COLORS[state]
    pad = size // 8
    d.ellipse((pad, pad, size - pad, size - pad), fill=color + (255,))
    # A small "hand" cue: three finger bars in the circle.
    bar_w = size // 10
    for i, h in enumerate((0.45, 0.55, 0.45)):
        x = size * (0.32 + 0.18 * i)
        d.rounded_rectangle(
            (x, size * (0.5 - h / 2), x + bar_w, size * (0.5 + h / 2)),
            radius=bar_w // 2, fill=(255, 255, 255, 230))
    return img


class Tray:
    def __init__(self, on_pause_toggle: Callable[[bool], None],
                 on_quit: Callable[[], None]) -> None:
        import pystray

        self._pystray = pystray
        self._on_pause_toggle = on_pause_toggle
        self._on_quit = on_quit
        self._armed = False
        self._paused = False

        menu = pystray.Menu(
            pystray.MenuItem(
                "Pause camera",
                self._toggle_pause,
                checked=lambda item: self._paused,
            ),
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon = pystray.Icon(
            "airctl", _icon_image("disarmed"), "AirCtl — disarmed", menu)

    # pystray invokes menu handlers on its own thread.
    def _toggle_pause(self, icon, item) -> None:
        self._paused = not self._paused
        self._on_pause_toggle(self._paused)
        self._refresh()

    def _quit(self, icon, item) -> None:
        self._icon.stop()
        self._on_quit()

    def _refresh(self) -> None:
        state = "paused" if self._paused else (
            "armed" if self._armed else "disarmed")
        self._icon.icon = _icon_image(state)
        self._icon.title = f"AirCtl — {state}"

    def set_armed(self, armed: bool) -> None:
        """Thread-safe enough: assigns then redraws via pystray setters."""
        self._armed = armed
        self._refresh()

    def run(self) -> None:
        """Blocks; call on the main thread."""
        self._icon.run()

    def stop(self) -> None:
        self._icon.stop()
