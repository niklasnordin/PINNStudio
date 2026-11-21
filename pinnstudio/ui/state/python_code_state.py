"""The problem's Python code (the Python code page): the module, what it is used for, and the last Check.

`pde_mode` is "expressions" or "python" (the equations from pde() in the code); `conditions_mode` is "rows"
or "python" (the boundary and initial conditions from conditions(c)). `from_outside` is True while the code
came from a problem file or a .py file and has not been trusted yet: Solve asks before running it.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

# The methods that change the state: only controllers may call them (tests/test_mvc.py, rule R5).
MUTATORS = ("put_code", "put_modes", "put_result", "put_message")


class PythonCodeState(QObject):
    code_changed = Signal()          # the code (typed, written, loaded)
    modes_changed = Signal()         # pde_mode or conditions_mode
    checked = Signal(dict)           # the last Check: {"ok", ...} (pde_code.parse_dry_run), or {} while running
    message = Signal(str)            # a short note for the user

    def __init__(self, parent=None):
        super().__init__(parent)
        self._code = ""
        self._pde_mode = "expressions"
        self._conditions_mode = "rows"
        self._from_outside = False
        self._result = {}

    # ── queries ────────────────────────────────────────────────────
    @property
    def code(self) -> str:
        return self._code

    @property
    def pde_mode(self) -> str:
        return self._pde_mode

    @property
    def conditions_mode(self) -> str:
        return self._conditions_mode

    @property
    def from_outside(self) -> bool:
        return self._from_outside

    @property
    def result(self) -> dict:
        return dict(self._result)

    def uses_code(self) -> bool:
        return self._pde_mode == "python" or self._conditions_mode == "python"

    # ── mutators (controllers only) ────────────────────────────────
    def put_code(self, code: str, from_outside: bool | None = None):
        code = code or ""
        changed = code != self._code
        self._code = code
        if from_outside is not None:
            self._from_outside = bool(from_outside)
        if changed:
            self._result = {}
            self.code_changed.emit()

    def put_modes(self, pde_mode: str | None = None, conditions_mode: str | None = None):
        new_p = pde_mode if pde_mode in ("expressions", "python") else self._pde_mode
        new_c = conditions_mode if conditions_mode in ("rows", "python") else self._conditions_mode
        if (new_p, new_c) != (self._pde_mode, self._conditions_mode):
            self._pde_mode, self._conditions_mode = new_p, new_c
            self.modes_changed.emit()

    def put_result(self, result: dict):
        self._result = dict(result or {})
        self.checked.emit(dict(self._result))

    def put_message(self, text: str):
        self.message.emit(text or "")
