"""
Central place for every filesystem location the app needs.

Two kinds of paths, never mixed:

* ``get_resource_path``  - READ-ONLY files shipped with the app (DEM, horizon DB,
  shapefiles, ...). In a PyInstaller build they live under ``sys._MEIPASS``; in
  development they live under the project root.
* ``get_user_data_dir``  - WRITABLE per-user files (autosave, logs, generated maps).
  Never inside the install / bundle folder, which may be read-only or temporary.

This module imports only the standard library, so any module may import it
(including ``ui.state`` and ``desktop_app``) without circular-import risk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "LinkBudgetCalculator"

# core/app_paths.py -> core/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _base_dir() -> Path:
    """Bundle root when frozen by PyInstaller (sys._MEIPASS), else the project root."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return _PROJECT_ROOT


def get_resource_path(relative_path: str) -> str:
    """Absolute path to a bundled, read-only file or folder.

    Checks ``sys._MEIPASS`` first (frozen exe), then falls back to the project root.
    The result does NOT depend on the current working directory. The path is not
    checked for existence - callers decide how to handle a missing file.
    """
    return str(_base_dir() / relative_path)


def _user_data_root() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if not base:  # very rare, but keep it writable rather than crash
            base = str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    return Path.home() / ".link_budget_calculator"


def get_user_data_dir(subdir: str = "") -> Path:
    """Writable per-user data directory, created if missing.

    Windows:     %LOCALAPPDATA%/LinkBudgetCalculator[/subdir]
    Linux/macOS: ~/.link_budget_calculator[/subdir]
    """
    path = _user_data_root()
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path