<div align="center">

# 🖐 AirCtl

**Control Windows with hand gestures — touchlessly, in real time, fully offline.**

Wave to switch windows, pinch-and-drag to set the volume, flash a peace sign to play/pause.
A webcam becomes a system input device through a multi-threaded computer-vision pipeline
that runs entirely on your machine.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6?logo=windows&logoColor=white)
![Tests](https://img.shields.io/badge/tests-60%20passing-2ea44f)
![Latency](https://img.shields.io/badge/latency-~15ms%20p50-blueviolet)
![Privacy](https://img.shields.io/badge/processing-100%25%20local-success)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## What it does

AirCtl watches your hand through the webcam, recognizes a small vocabulary of gestures, and
dispatches them as real OS actions — volume, media, window switching, virtual desktops, and a
fully configurable shortcut layer. The hard problem isn't recognizing a fist; it's doing so
**without firing on every incidental hand movement while you type, drink coffee, or talk.**
AirCtl solves that with a deliberate *wake gesture*, a debouncing state machine, and a
disarmed-by-default policy.

```
┌────────────┐  frames   ┌──────────────┐ landmarks ┌──────────────┐
│  Capture   │ ────────▶ │  MediaPipe   │ ────────▶ │   Features   │
│  (OpenCV)  │ latest-win│  HandLandmkr │           │  + One Euro  │
└────────────┘   buffer  └──────────────┘           └──────┬───────┘
                                                            │ 21 landmarks
┌────────────┐  actions  ┌──────────────┐  events   ┌──────▼───────┐
│ Dispatcher │ ◀──────── │ Gesture FSM  │ ◀──────── │  Classifier  │
│ (owns COM) │   queue   │ debounce/arm │           │ rules+swipes │
└─────┬──────┘           └──────────────┘           └──────────────┘
      ├─▶ Audio    (Core Audio / pycaw — sub-ms volume)
      ├─▶ Input    (Win32 SendInput — scan codes, media keys, wheel)
      ├─▶ Window   (Alt+Tab injection + direct SetForegroundWindow targeting)
      └─▶ Custom   (key chords, shell commands, app launches — from config)
```

## Gesture vocabulary

> The system is **disarmed by default** and ignores everything until you arm it — so talking
> with your hands or typing never triggers an action.

| Gesture | Action |
|---|---|
| 🖐 Open palm, hold 1 s | **Arm** (10 s window, auto-extended by activity) |
| ✊ Fist, hold 1 s | **Disarm** |
| 🤟 **Three fingers, flick ← / →** | **Switch to previous / next window** (direct, instant) |
| 🤏 Thumb + middle pinch, hold + slide | Alt+Tab switcher (browse many windows with preview) |
| ✌️ Peace | Play / pause |
| 🤏 Pinch (thumb + index) + drag up/down | Continuous volume (relative to where you pinched) |
| ☝️ Point up / down | Volume step up / down (repeats while held) |
| 🤙 Shaka | Mute toggle |
| 🖐 Swipe ← / → | Previous / next track |
| ✌️ Swipe ← / → | Virtual desktop left / right |
| 🤟 Three fingers (held still) | Screenshot snip (`Win+Shift+S`) |

Every mapping lives in a TOML file and **hot-reloads while running** — no restart, no code changes.

## Quick start

```powershell
# Requires Windows 10/11 and uv  (https://docs.astral.sh/uv/)
uv sync                      # creates a Python 3.12 venv, installs everything

uv run airctl                # normal use: tray icon only
uv run airctl --preview      # camera window with landmarks drawn (q to quit)
uv run airctl --debug-poses  # print recognized poses + live latency stats
```

On first launch AirCtl downloads the MediaPipe hand model (~8 MB) and writes a default
config to `%APPDATA%\airctl\config.toml`. The tray icon shows state at a glance:
🟢 armed · ⚪ disarmed · 🔴 camera paused (the webcam is *released*, LED off).

## Engineering highlights

This is built as a **systems engineering problem**, not a CV demo:

- **Latency-critical threading.** Four threads — capture, inference, dispatch, UI — with a
  **latest-wins single-slot frame buffer** between capture and inference. A naive queue would
  accumulate backlog whenever inference hiccups and latency would grow unbounded; AirCtl always
  processes the newest frame and drops the rest. Measured **p50 ≈ 15 ms, p95 ≈ 23 ms** end of
  capture to action on a laptop CPU (budget was 100 ms).
- **A gesture lifecycle state machine.** `IDLE → CANDIDATE → ACTIVE → HELD → COOLDOWN`. A pose
  must hold stable for ~150 ms before it fires, with a post-action cooldown and a stillness gate
  — which is what kills false positives during normal desk activity.
- **Rotation/scale/translation-invariant features.** Landmarks are normalized into a hand-relative
  frame so a gesture reads the same near or far, left or right, tilted or straight. A **One Euro
  filter** smooths jitter without adding lag to fast motion.
- **Real Windows internals, not shell-outs.** Volume goes through a persistent Core Audio COM
  object (sub-millisecond, no subprocess). Input uses `SendInput` with scan codes. Window
  targeting enumerates top-level windows via `EnumWindows`, filters DWM-cloaked ghost windows,
  and defeats Windows' foreground-lock protection with the `AttachThreadInput` workaround to make
  `SetForegroundWindow` succeed from a background process.
- **Disciplined COM ownership.** All COM objects live on a single dispatcher thread that calls
  `CoInitialize` once — the one rule that prevents a whole class of intermittent COM crashes.
- **Testable without hardware.** The full pipeline runs against synthetic landmark geometry and
  recording fake backends, so **60 unit tests** cover classifier, FSM, action routing, and config
  on a machine with no camera, audio device, or display.

## Configuration

`%APPDATA%\airctl\config.toml` — edits take effect live; a file with a syntax error is rejected
and the previous config keeps running.

```toml
[modes.default]
"peace"        = { action = "media.play_pause" }
"swipe3_right" = { action = "window.next" }                  # built-in
"three"        = { action = "key", chord = "ctrl+shift+esc" } # any key chord
"shaka"        = { action = "shell", cmd = "code ." }         # any shell command
"fist"         = { action = "launch", target = "https://news.ycombinator.com" }
```

Recognition thresholds (`confirm_frames`, `cooldown_ms`, pinch hysteresis, swipe velocity) live
in `[recognition]` — run `--debug-poses` to watch live values while tuning.

## Project structure

```
src/airctl/
  capture.py       latest-wins camera thread (OpenCV, CAP_DSHOW)
  landmarker.py    MediaPipe HandLandmarker wrapper + model auto-download
  filters.py       One Euro filter
  features.py      landmark normalization + derived features
  classifier.py    rule-based pose classification + velocity swipes
  fsm.py           gesture lifecycle FSM + wake-gesture arming
  dispatcher.py    action queue, COM ownership
  actions.py       config action specs -> behavior
  backends/        audio · input · window · media · custom (+ fakes for tests)
  config.py        TOML load + watchdog hot-reload
  tray.py / app.py / __main__.py
tests/             60 unit tests, no hardware required
```

```powershell
uv run pytest                # full suite
uv run airctl --debug-poses -v
```

## Tech stack

Python 3.12 · MediaPipe Tasks (HandLandmarker) · OpenCV · NumPy · pycaw (Core Audio COM) ·
pywin32 · pynput + raw `ctypes` SendInput · pystray · watchdog · uv

## Windows notes (by design)

- **Elevated windows** can't receive synthetic input from a normal-privilege process (UIPI).
  Don't run AirCtl elevated to "fix" this — it's a security boundary.
- **Camera privacy:** if the camera won't open, allow desktop apps under *Settings → Privacy &
  security → Camera*. AirCtl retries every 5 s.
- **Single instance** is enforced via a named mutex; a second launch exits immediately.
- All processing is local — **frames are never written to disk and never leave the machine.**



## License

[MIT](LICENSE) © 2026 Madhav
