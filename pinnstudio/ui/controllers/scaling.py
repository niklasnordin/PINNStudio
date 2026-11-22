"""Scaling the network's inputs and outputs (Neural network page): suggested output scales.

A controller: the window asks it for the suggestion (core/scaling.py, from the reference formula or the
initial and boundary values) and puts the numbers into its rows.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject

from pinnstudio.core import scaling as S


class ScalingController(QObject):
    def __init__(self, config: Callable, parent=None):
        """`config()` gives the problem as the window has it now."""
        super().__init__(parent)
        self._config = config

    def suggest_outputs(self) -> dict:
        """{"scale", "shift", "source", "text"}: the suggestion, and a line saying where it came from."""
        try:
            s = S.suggest_output_scales(self._config())
        except Exception as exc:                     # an expression that does not evaluate
            return {"scale": [], "shift": [], "source": [], "text": f"No suggestion: {exc}"}
        parts = [f"output {i + 1}: ×{sc:g} + {sh:g} ({src})" for i, (sc, sh, src)
                 in enumerate(zip(s["scale"], s["shift"], s["source"]))]
        s["text"] = "Output scales from " + "; ".join(parts)
        return s
