"""Synthetic input via SendInput (§5.2).

Two paths, per the design doc:

- **pynput** for key chords parsed from config strings ("win+shift+s").
- **raw ctypes SendInput** for the cases pynput abstracts poorly: scan-code
  key events (the Alt+Tab switcher — some apps and the switcher itself handle
  scan codes more reliably than virtual-key-only events), virtual media keys,
  and mouse-wheel injection for the v2 virtual knob.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes

from pynput.keyboard import Controller, Key

log = logging.getLogger(__name__)

# -- raw SendInput plumbing ---------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_WHEEL = 0x0800
MAPVK_VK_TO_VSC = 0

VK_SHIFT = 0x10
VK_ALT = 0x12
VK_TAB = 0x09

# Extended keys need KEYEVENTF_EXTENDEDKEY with scan codes (arrows, etc.).
_EXTENDED_VKS = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
                           0x2D, 0x2E, 0x5B, 0x5C})

_ULONG_PTR = ctypes.c_size_t


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR))


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT))


class _INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


_user32 = ctypes.WinDLL("user32", use_last_error=True)


def _send(*inputs: _INPUT) -> None:
    array = (_INPUT * len(inputs))(*inputs)
    sent = _user32.SendInput(len(inputs), array, ctypes.sizeof(_INPUT))
    if sent != len(inputs):
        log.warning("SendInput injected %d/%d events (error %d)",
                    sent, len(inputs), ctypes.get_last_error())


def _key_event(vk: int, up: bool, scancode: bool) -> _INPUT:
    flags = KEYEVENTF_KEYUP if up else 0
    scan = 0
    if scancode:
        scan = _user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
        flags |= KEYEVENTF_SCANCODE
        if vk in _EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
    ki = _KEYBDINPUT(wVk=0 if scancode else vk, wScan=scan,
                     dwFlags=flags, time=0, dwExtraInfo=0)
    return _INPUT(type=INPUT_KEYBOARD, union=_INPUTUNION(ki=ki))


# -- chord parsing (pynput path) ----------------------------------------------

_KEY_ALIASES: dict[str, Key | str] = {
    "ctrl": Key.ctrl, "control": Key.ctrl,
    "alt": Key.alt,
    "shift": Key.shift,
    "win": Key.cmd, "cmd": Key.cmd, "super": Key.cmd,
    "enter": Key.enter, "return": Key.enter,
    "esc": Key.esc, "escape": Key.esc,
    "space": Key.space,
    "tab": Key.tab,
    "backspace": Key.backspace,
    "delete": Key.delete, "del": Key.delete,
    "home": Key.home, "end": Key.end,
    "pgup": Key.page_up, "pageup": Key.page_up,
    "pgdn": Key.page_down, "pagedown": Key.page_down,
    "left": Key.left, "right": Key.right, "up": Key.up, "down": Key.down,
    "printscreen": Key.print_screen,
    **{f"f{i}": getattr(Key, f"f{i}") for i in range(1, 13)},
}


def parse_chord(chord: str) -> list[Key | str]:
    keys: list[Key | str] = []
    for part in chord.lower().split("+"):
        part = part.strip()
        if not part:
            raise ValueError(f"empty key in chord {chord!r}")
        if part in _KEY_ALIASES:
            keys.append(_KEY_ALIASES[part])
        elif len(part) == 1:
            keys.append(part)
        else:
            raise ValueError(f"unknown key {part!r} in chord {chord!r}")
    return keys


class SendInputBackend:
    """InputBackend implementation. Stateless; safe to call from the
    dispatcher thread only (single writer keeps modifier state sane)."""

    def __init__(self) -> None:
        self._kb = Controller()

    def tap_chord(self, chord: str) -> None:
        keys = parse_chord(chord)
        for k in keys:
            self._kb.press(k)
        for k in reversed(keys):
            self._kb.release(k)

    def tap_vk(self, vk: int) -> None:
        _send(_key_event(vk, up=False, scancode=False),
              _key_event(vk, up=True, scancode=False))

    def scan_key_down(self, vk: int) -> None:
        _send(_key_event(vk, up=False, scancode=True))

    def scan_key_up(self, vk: int) -> None:
        _send(_key_event(vk, up=True, scancode=True))

    def scan_tap(self, vk: int, with_shift: bool = False) -> None:
        events = []
        if with_shift:
            events.append(_key_event(VK_SHIFT, up=False, scancode=True))
        events.append(_key_event(vk, up=False, scancode=True))
        events.append(_key_event(vk, up=True, scancode=True))
        if with_shift:
            events.append(_key_event(VK_SHIFT, up=True, scancode=True))
        _send(*events)

    def scroll(self, delta: int) -> None:
        mi = _MOUSEINPUT(dx=0, dy=0, mouseData=ctypes.c_uint32(delta & 0xFFFFFFFF).value,
                         dwFlags=MOUSEEVENTF_WHEEL, time=0, dwExtraInfo=0)
        _send(_INPUT(type=INPUT_MOUSE, union=_INPUTUNION(mi=mi)))
