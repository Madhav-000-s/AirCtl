# AirCtl

Real-time gesture-controlled system interface for Windows. AirCtl turns a
webcam into a system input device: MediaPipe hand landmarks → rule-based
pose classifier → gesture lifecycle FSM → OS actions (volume, media,
window switching, custom shortcuts). All processing is fully local — frames
never leave the machine and are never written to disk.

Design: see [gesture-control-design-doc-windows.md](gesture-control-design-doc-windows.md).

## Quick start

```powershell
# Requires Windows 10/11 and uv (https://docs.astral.sh/uv/)
uv sync

# First run: prints each recognized pose + latency stats to the console
uv run airctl --debug-poses

# With a camera preview window (q to quit)
uv run airctl --preview

# Normal use: tray icon only
uv run airctl
```

On first run AirCtl downloads the MediaPipe hand model (~8 MB) to
`%APPDATA%\airctl\models\` and creates `%APPDATA%\airctl\config.toml` from
the bundled default.

## Using it

**The system is disarmed by default** — it ignores everything until you arm
it, so talking with your hands or typing never triggers actions.

| Gesture | Action (default config) |
|---|---|
| 🖐 Open palm, hold 1 s | **Arm** (10 s window, refreshed by activity) |
| ✊ Fist, hold 1 s | **Disarm** |
| ✌️ Peace | Play / pause |
| ☝️ Point up / down | Volume step up / down (repeats while held) |
| 🤙 Shaka | Mute toggle |
| 🤏 Pinch (thumb+index) + drag up/down | Continuous volume (relative to engage point) |
| Pinch (thumb+middle) + slide left/right | Alt+Tab window switcher; release to select |
| 🖐 Swipe left / right | Previous / next track |
| ✌️ Swipe left / right | Virtual desktop left / right |
| Three fingers | Screenshot snip (`Win+Shift+S`) |

The tray icon shows state: **green** = armed, **gray** = disarmed,
**red** = camera paused. "Pause camera" in the tray menu fully releases the
webcam (LED turns off).

## Configuration

Edit `%APPDATA%\airctl\config.toml` — changes hot-reload while AirCtl runs;
a file with errors is rejected and the previous config keeps working.
Any gesture can map to:

```toml
[modes.default]
"peace"  = { action = "media.play_pause" }
"three"  = { action = "key", chord = "ctrl+shift+esc" }
"shaka"  = { action = "shell", cmd = "explorer.exe" }
"fist"   = { action = "launch", target = "https://example.com" }
```

Recognition thresholds (`confirm_frames`, `cooldown_ms`, pinch hysteresis,
swipe velocity, …) live in `[recognition]` — run `--debug-poses` to see live
values while tuning.

## Development

```powershell
uv run pytest          # unit tests, no camera/audio needed
uv run airctl --debug-poses -v
```

Architecture (one process, four threads):

```
capture (OpenCV, DSHOW) ──latest-wins slot──▶ inference (MediaPipe → One Euro
filter → features → classifier → FSM) ──event queue──▶ dispatcher (COM owner:
pycaw volume, SendInput keys, Alt+Tab switcher) · main thread runs the tray
```

Key modules: [capture.py](src/airctl/capture.py) (latest-wins frame slot),
[features.py](src/airctl/features.py) (landmark normalization),
[classifier.py](src/airctl/classifier.py) (pose rules + swipes),
[fsm.py](src/airctl/fsm.py) (debounce/arm/cooldown),
[actions.py](src/airctl/actions.py) (config → behavior),
[backends/](src/airctl/backends/) (OS integration, with fakes for tests).

## Windows-specific limitations (by design)

1. **Elevated windows:** synthetic input from a normal process cannot reach
   windows running as Administrator (Task Manager, elevated terminals) — UIPI
   blocks it. Don't run AirCtl elevated to "fix" this.
2. **Camera privacy:** if the camera won't open, allow desktop apps under
   Settings → Privacy & security → Camera. AirCtl retries every 5 s.
3. **Antivirus:** global SendInput use can trip AV heuristics, especially in
   packaged builds. Running from source via `uv` is the documented-safe path.
4. **Single instance:** a second launch exits immediately (named mutex).

## Not yet built (v2 in the design doc)

Gesture recorder with DTW matching, mode layers, two-hand chords, per-app
volume, WinRT SMTC track HUD, radial menu, PyInstaller packaging + autostart.
