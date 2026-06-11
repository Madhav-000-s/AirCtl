"""CLI entry point: ``airctl`` / ``python -m airctl``."""

from __future__ import annotations

import argparse
import ctypes
import logging
import sys
from pathlib import Path

ERROR_ALREADY_EXISTS = 183


def _acquire_single_instance() -> bool:
    """Named mutex so a second launch exits instead of fighting over the
    camera (design-doc open question #5). The handle is held for the process
    lifetime; Windows cleans it up on exit."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, "Local\\airctl-singleton")
    return ctypes.get_last_error() != ERROR_ALREADY_EXISTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="airctl",
        description="Gesture-controlled system interface for Windows.")
    parser.add_argument("--config", type=Path, default=None,
                        help="config file (default: %%APPDATA%%\\airctl\\config.toml)")
    parser.add_argument("--preview", action="store_true",
                        help="show a camera window with landmarks drawn")
    parser.add_argument("--debug-poses", action="store_true",
                        help="print recognized poses and latency stats")
    parser.add_argument("--no-tray", action="store_true",
                        help="run without the tray icon (Ctrl+C to quit)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not _acquire_single_instance():
        print("AirCtl is already running (check the tray).", file=sys.stderr)
        return 1

    from .app import App

    app = App(
        config_path=args.config,
        preview=args.preview,
        debug_poses=args.debug_poses,
        use_tray=not args.no_tray,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
