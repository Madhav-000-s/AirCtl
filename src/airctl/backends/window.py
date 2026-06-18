"""Window switching (§5.3).

Two independent mechanisms, both useful:

* **Alt+Tab switcher** (tier 1): hold Alt down and tap Tab via scan-code
  injection, driving Windows' own switcher UI. Good for browsing many windows
  with a live preview. Used by the pinch-hold gesture.
* **Direct targeting** (tier 2): enumerate top-level windows with
  ``EnumWindows`` and jump straight to one with ``SetForegroundWindow``. No
  switcher overlay — the chosen window comes to the front instantly. Used by
  the three-finger swipe for a quick "flick to the next window".

Direct targeting has to defeat Windows' foreground-lock protection, which
blocks ``SetForegroundWindow`` from a background process. We use the
documented ``AttachThreadInput`` workaround (attach our input queue to the
current foreground thread so the OS treats the call as user-initiated) with
an Alt-key tap as a fallback. Both paths are wrapped so a failure logs and
no-ops rather than crashing the dispatcher.

Tier 3, virtual desktop switching, is just a Ctrl+Win+arrow chord.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging

import win32api
import win32con
import win32gui
import win32process

from .base import InputBackend
from .input import VK_ALT, VK_TAB

log = logging.getLogger(__name__)

# DwmGetWindowAttribute(DWMWA_CLOAKED) is non-zero for UWP "ghost" windows
# that are technically visible but not really on screen (e.g. the suspended
# "Windows Input Experience"). Filtering them keeps the cycle list clean.
_DWMWA_CLOAKED = 14
_dwmapi = ctypes.WinDLL("dwmapi")


def _is_cloaked(hwnd: int) -> bool:
    cloaked = ctypes.c_int(0)
    res = _dwmapi.DwmGetWindowAttribute(
        ctypes.wintypes.HWND(hwnd), _DWMWA_CLOAKED,
        ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    return res == 0 and cloaked.value != 0


def _is_alt_tab_window(hwnd: int) -> bool:
    """Roughly the set of windows Alt+Tab shows: visible, titled, top-level
    (no owner), not a tool window, not cloaked."""
    if not win32gui.IsWindowVisible(hwnd):
        return False
    if win32gui.GetWindow(hwnd, win32con.GW_OWNER) != 0:
        return False
    if not win32gui.GetWindowText(hwnd):
        return False
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    if ex_style & win32con.WS_EX_TOOLWINDOW:
        return False
    return not _is_cloaked(hwnd)


class Win32WindowBackend:
    def __init__(self, input_backend: InputBackend) -> None:
        self._input = input_backend
        self._switcher_open = False

    # -- Alt+Tab switcher (tier 1) -------------------------------------------

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

    # -- direct targeting (tier 2) -------------------------------------------

    def list_windows(self) -> list[int]:
        """Top-level app windows in current Z-order (frontmost first)."""
        out: list[int] = []
        win32gui.EnumWindows(
            lambda h, _: out.append(h) if _is_alt_tab_window(h) else None, None)
        return out

    def get_foreground(self) -> int:
        return win32gui.GetForegroundWindow()

    def window_title(self, hwnd: int) -> str:
        return win32gui.GetWindowText(hwnd)

    def focus_window(self, hwnd: int) -> None:
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            if self._set_foreground(hwnd):
                return
            # Fallback: a no-op Alt tap satisfies the foreground-lock rules
            # for the SetForegroundWindow that follows.
            self._input.scan_tap(VK_ALT)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            log.exception("could not focus window %s", hwnd)

    def _set_foreground(self, hwnd: int) -> bool:
        """SetForegroundWindow defeating the foreground lock via
        AttachThreadInput. Returns True on success."""
        fg = win32gui.GetForegroundWindow()
        if fg == hwnd:
            return True
        cur_tid = win32api.GetCurrentThreadId()
        fg_tid = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0
        attached = False
        try:
            if fg_tid and fg_tid != cur_tid:
                win32process.AttachThreadInput(cur_tid, fg_tid, True)
                attached = True
            win32gui.SetForegroundWindow(hwnd)
            win32gui.BringWindowToTop(hwnd)
            return win32gui.GetForegroundWindow() == hwnd
        except Exception:
            return False
        finally:
            if attached:
                try:
                    win32process.AttachThreadInput(cur_tid, fg_tid, False)
                except Exception:
                    pass

    # -- virtual desktops (tier 3) -------------------------------------------

    def desktop_switch(self, direction: str) -> None:
        if direction not in ("left", "right"):
            raise ValueError(f"bad desktop direction {direction!r}")
        self._input.tap_chord(f"ctrl+win+{direction}")
