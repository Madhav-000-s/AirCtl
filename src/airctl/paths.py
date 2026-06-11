"""Filesystem locations for user data (%APPDATA%\\airctl)."""

from __future__ import annotations

import os
from pathlib import Path


def appdata_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    d = root / "airctl"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = appdata_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_config_path() -> Path:
    return appdata_dir() / "config.toml"
