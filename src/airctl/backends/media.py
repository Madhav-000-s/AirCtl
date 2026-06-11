"""Media control via virtual media keys (§5.4 v1).

Spotify, browsers, VLC, and the system media overlay all honor these.
WinRT SMTC (per-player targeting, track metadata) is the v2 upgrade.
"""

from __future__ import annotations

from .base import InputBackend

VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3


class MediaKeyBackend:
    def __init__(self, input_backend: InputBackend) -> None:
        self._input = input_backend

    def play_pause(self) -> None:
        self._input.tap_vk(VK_MEDIA_PLAY_PAUSE)

    def next_track(self) -> None:
        self._input.tap_vk(VK_MEDIA_NEXT_TRACK)

    def prev_track(self) -> None:
        self._input.tap_vk(VK_MEDIA_PREV_TRACK)
