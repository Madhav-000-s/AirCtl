"""Window switching (§5.3, tier 1 + 3).

Tier 1: drive Windows' own Alt+Tab switcher by holding Alt down and tapping
Tab — scan-code injection, which the switcher handles more reliably than
virtual-key-only events. Tier 3: virtual desktop switching is just a
Ctrl+Win+arrow chord.

Tier 2 (direct SetForegroundWindow targeting with the focus-lock dance) is
deliberately deferred to v2.
"""

from __future__ import annotations

import logging

from .base import InputBackend
from .input import VK_ALT, VK_TAB

log = logging.getLogger(__name__)


class Win32WindowBackend:
    def __init__(self, input_backend: InputBackend) -> None:
        self._input = input_backend
        self._switcher_open = False

    def switcher_begin(self) -> None:
        if self._switcher_open:
            return
        self._input.scan_key_down(VK_ALT)
        # First Tab opens the switcher UI on the most-recent window.
        self._input.scan_tap(VK_TAB)
        self._switcher_open = True

    def switcher_step(self, forward: bool) -> None:
        if not self._switcher_open:
            return
        self._input.scan_tap(VK_TAB, with_shift=not forward)

    def switcher_end(self) -> None:
        if not self._switcher_open:
            return
        # Releasing Alt selects the highlighted window.
        self._input.scan_key_up(VK_ALT)
        self._switcher_open = False

    def desktop_switch(self, direction: str) -> None:
        if direction not in ("left", "right"):
            raise ValueError(f"bad desktop direction {direction!r}")
        self._input.tap_chord(f"ctrl+win+{direction}")
