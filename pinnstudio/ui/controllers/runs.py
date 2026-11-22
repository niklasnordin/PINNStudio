"""The Python kept ready for the next run (core/warm.py): the setting, and starting it.

A controller: the window asks it; it keeps the setting in QSettings and starts the spare in a thread of its
own, so the window never waits for PyTorch to load.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

from PySide6.QtCore import QObject, QSettings, QTimer

from pinnstudio.core import warm

KEY = "runs/keep_ready"


class RunsController(QObject):
    def __init__(self, config: Callable, parent=None):
        """`config()` gives the problem as the window has it now (its device setting decides the spare's)."""
        super().__init__(parent)
        self._config = config

    @staticmethod
    def _settings():
        return QSettings("PINNStudio", "PINNStudio")

    def keep_ready(self) -> bool:
        try:
            return str(self._settings().value(KEY, "true")).lower() in ("true", "1")
        except Exception:
            return True

    def set_keep_ready(self, on: bool):
        try:
            self._settings().setValue(KEY, bool(on))
        except Exception:
            pass
        warm.POOL.set_enabled(on)
        if on:
            self.prestart()

    def started(self, delay_ms: int = 3000):
        """The window is on screen: the spare starts a little later (the window first draws itself)."""
        warm.POOL.set_enabled(self.keep_ready())
        QTimer.singleShot(delay_ms, self.prestart)

    def prestart(self):
        """Start the spare, unless off, in tests (off screen, unless PINNSTUDIO_WARM is set) or in the Windows
        program's launch check."""
        if (os.environ.get("QT_QPA_PLATFORM") == "offscreen" and not os.environ.get("PINNSTUDIO_WARM")) \
                or os.environ.get("PINNSTUDIO_LAUNCH_CHECK") or not warm.POOL.enabled:
            return
        try:
            env = warm.run_env(self._config())
        except Exception:
            env = warm.run_env()
        threading.Thread(target=warm.POOL.prestart, args=(env,), name="pinnstudio-warm-start",
                         daemon=True).start()
