"""The Python code page's actions: the code typed, written from the expressions and rows, loaded, saved,
opened in the user's editor, checked by a dry run of the Solve script, and trusted before it runs.

A controller: it changes the state (`PythonCodeState`), asks the user only through the `Prompter`, and runs
the check in a second process (core/warm.py: the Python kept ready, or a new one), so the code never runs inside the window.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from typing import Callable

from PySide6.QtCore import QCoreApplication, QFileSystemWatcher, QObject, QTimer, Signal

from pinnstudio.core import pde_code as PC
from pinnstudio.core import warm
from pinnstudio.core.netdesign import trust as T
from pinnstudio.ui.controllers.prompter import Prompter
from pinnstudio.ui.state.python_code_state import PythonCodeState

CHECK_TIMEOUT_MS = 300_000
NO_START = "PINNSTUDIO_CHECK_NO_START "


class _CheckRun:
    """One running check: its process once started (in the reader thread)."""

    def __init__(self):
        self.proc = None
        self.cancelled = False

    def kill(self):
        self.cancelled = True
        try:
            if self.proc is not None:
                self.proc.kill()
        except Exception:
            pass


class PythonCodeController(QObject):
    _output = Signal(object, str)          # (process, its output): from the reader thread

    def __init__(self, state: PythonCodeState, prompter: Prompter, config: Callable, parent=None):
        """`config()` gives the problem as the window has it now (a PINNConfig), with this code in it."""
        super().__init__(parent)
        self.state, self.prompter, self._config = state, prompter, config
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._file_changed)
        self._watched = ""
        self._own = set()          # SHA-256 of code the user typed or wrote from the page in this session
        self._proc = None
        self._reader = None
        self._check_dir = ""
        self._output.connect(self._check_done)
        # () -> (key, [midpoint, ...]) or None: the Sketch and Geometry views' edges, which Check reports the
        # Python conditions on (set by the window)
        self.edges = None
        self._pending_edges = (None, [])
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._check_timed_out)

    # ── the code ───────────────────────────────────────────────────
    def load(self, pde_mode: str, conditions_mode: str, code: str):
        """The problem's own settings (a problem opened, an example, Undo): the code from a file asks
        before it runs, unless exactly this code was trusted before."""
        code = code or ""
        self._unwatch()                       # another problem: its editor file is not this one
        outside = bool(code.strip()) and not T.is_trusted(T.code_item(code)) and not _shipped(code)
        self.state.put_code(code, from_outside=outside)
        if not outside:
            self._own.add(T.sha256_text(code))
        self.state.put_modes(pde_mode or "expressions", conditions_mode or "rows")

    def follow(self, pde_mode: str, conditions_mode: str, code: str):
        """A settings Undo/Redo step: the code and modes as they were, keeping where the code came from
        (the user's own code stays the user's own; other code is trusted only by its hash)."""
        code = code or ""
        if code != self.state.code:
            own = T.sha256_text(code) in self._own or T.is_trusted(T.code_item(code)) or not code.strip()
            self.state.put_code(code, from_outside=not own)
        self.state.put_modes(pde_mode or "expressions", conditions_mode or "rows")

    def edit(self, code: str):
        """The user typed in the editor (in code of their own, it stays their own)."""
        self._mark_own(code)
        self.state.put_code(code)

    def _mark_own(self, code):
        if not self.state.from_outside:
            self._own.add(T.sha256_text(code or ""))

    def set_pde_mode(self, mode: str) -> bool:
        if mode == "python" and not PC.has_function(self.state.code, "pde"):
            if not self.write_equations(ask=False):
                return False
        self.state.put_modes(pde_mode=mode)
        return True

    def set_conditions_mode(self, mode: str) -> bool:
        if mode == "python" and not PC.has_function(self.state.code, "conditions"):
            if not self.write_conditions(ask=False):
                return False
        self.state.put_modes(conditions_mode=mode)
        return True

    def write_equations(self, ask: bool = True) -> bool:
        """pde() written from the Equations page's expressions (replacing the code's pde(), after asking)."""
        try:
            src = PC.from_expressions(self._config())
        except ValueError as exc:
            self.prompter.inform("Python code", str(exc))
            return False
        return self._put_function("pde", src, ask, "the equations")

    def write_conditions(self, ask: bool = True) -> bool:
        """conditions(c) written from the Boundary and Initial conditions rows."""
        try:
            src = PC.conditions_from_rows(self._config())
        except ValueError as exc:
            self.prompter.inform("Python code", str(exc))
            return False
        return self._put_function("conditions", src, ask, "the conditions")

    def write_exact_initial(self, ask: bool = True) -> bool:
        """output_transform(x, y) holding each output to its initial condition exactly, from the Initial
        conditions page's expressions."""
        try:
            src = PC.exact_initial_transform(self._config())
        except ValueError as exc:
            self.prompter.inform("Python code", f"Exact initial conditions: {exc}")
            return False
        return self._put_function("output_transform", src, ask, "exact initial conditions")

    def add_hook(self, name: str, ask: bool = True) -> bool:
        """A start for one of the hooks (output_transform, feature_transform, points, callbacks)."""
        return self._put_function(name, PC.HOOK_TEMPLATES[name], ask, "a template")

    def in_use(self) -> str:
        """The page's "In use:" line: what of the code the problem uses."""
        try:
            used = PC.in_use(self._config())
        except Exception:
            used = []
        return ", ".join(used) if used else "nothing yet (choose Python on the Equations or conditions pages, " \
                                              "or define a hook)"

    def _put_function(self, name, src, ask, what) -> bool:
        code = self.state.code
        if ask and PC.has_function(code, name) and not self.prompter.confirm(
                "Python code", f"Replace the code's {name}() with {what} as the page has them now?"):
            return False
        new = PC.replace_function(code, name, src)
        self._mark_own(new)
        self.state.put_code(new)
        self.state.put_message(f"{name}() written from {what}")
        return True

    def load_file(self) -> bool:
        path = self.prompter.ask_open_path("Load Python code", "Python (*.py);;All files (*)")
        if not path:
            return False
        try:
            with open(path, encoding="utf-8") as fh:
                code = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            self.prompter.inform("Python code", f"{path} cannot be read: {exc}")
            return False
        self.state.put_code(code, from_outside=not T.is_trusted(T.code_item(code)))
        self.state.put_message(f"Loaded {os.path.basename(path)}")
        return True

    def save_file(self, default_name: str = "problem.py") -> bool:
        path = self.prompter.ask_save_path("Save Python code", default_name, "Python (*.py)")
        if not path:
            return False
        try:
            self._write(path, self.state.code)
        except OSError as exc:
            self.prompter.inform("Python code", f"{path} cannot be written: {exc}")
            return False
        self.state.put_message(f"Saved {os.path.basename(path)}")
        return True

    @staticmethod
    def _write(path, text):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)

    def open_in_editor(self, folder: str = "", name: str = "problem") -> str:
        """Write the code to a .py file and open it in the user's editor; saving it there reloads the page.
        Returns the file's path ("" when it could not be written)."""
        folder = folder if folder and os.path.isdir(folder) else os.path.join(T.home(), "python_code")
        os.makedirs(folder, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or "problem")) or "problem"
        path = os.path.join(folder, f"{safe}_python.py")
        try:
            with open(path, encoding="utf-8") as fh:
                there = fh.read()
        except (OSError, UnicodeDecodeError):
            there = None
        if there is not None and there != self.state.code and not self.prompter.confirm(
                "Python code", f"{path} exists and differs from the page's code. Replace it with the page's code?"):
            return ""
        try:
            self._write(path, self.state.code)
        except OSError as exc:
            self.prompter.inform("Python code", f"{path} cannot be written: {exc}")
            return ""
        self._watch(path)
        if not self.prompter.open_external(path):
            self.state.put_message(f"Open {path} in your editor: saving it there updates this page")
        else:
            self.state.put_message(f"Editing {path}: saving it there updates this page")
        return path

    def _watch(self, path):
        self._unwatch()
        self._watched = path
        self._watcher.addPath(path)

    def _unwatch(self):
        if self._watched:
            self._watcher.removePath(self._watched)
            self._watched = ""

    def _file_changed(self, path):
        if path != self._watched:
            return
        if not os.path.exists(path):           # editors that save by replacing the file
            QTimer.singleShot(200, lambda: self._file_changed(path) if os.path.exists(path) else None)
            return
        if path not in self._watcher.files():
            self._watcher.addPath(path)
        try:
            with open(path, encoding="utf-8") as fh:
                code = fh.read()
        except (OSError, UnicodeDecodeError):
            return
        if code != self.state.code:
            self.state.put_code(code)
            self.state.put_message(f"Updated from {os.path.basename(path)}")

    # ── trust ──────────────────────────────────────────────────────
    def check_trust(self) -> bool:
        """True when the code may run: it is not used, it is the user's own, or the user trusts it now."""
        code = self.state.code
        if not self.state.uses_code() or not code.strip():
            return True
        item = T.code_item(code)
        if T.is_trusted(item):
            self.state.put_code(code, from_outside=False)
            return True
        if (not self.state.from_outside and T.sha256_text(code) in self._own) or self.prompter.confirm_trust([item]):
            T.remember([item])
            self.state.put_code(code, from_outside=False)
            return True
        self.state.put_message("The Python code stays inactive until you trust it")
        return False

    # ── Check: a dry run of the Solve script ───────────────────────
    def running(self) -> bool:
        return self._proc is not None

    def check(self, extra_env: dict | None = None) -> bool:
        """Start a dry run of the Solve script in a second process. `extra_env` adds to its environment."""
        if self._proc is not None:
            return False
        cfg = user_cfg = self._config()
        errors = PC.check_source(cfg.python_code, needs=(("pde",) if cfg.pde_mode == "python" else ())
                                 + (("conditions",) if cfg.conditions_mode == "python" else ()))
        if errors:
            self.state.put_result({"ok": False, "error": errors[0], "line": _line_of(errors[0])})
            return False
        if not self.check_trust():
            return False
        from pinnstudio.core import codegen as CG
        import dataclasses
        self._check_dir = tempfile.mkdtemp(prefix="pinnstudio_check_")
        cfg = dataclasses.replace(PC.dry_run_config(cfg), save_dir=os.path.join(self._check_dir, "run"))
        try:
            script = CG.generate_script(cfg)
        except Exception as exc:                           # a setting the generator refuses
            self._clean_check_dir()
            self.state.put_result({"ok": False, "error": f"{type(exc).__name__}: {exc}", "line": 0})
            return False
        path = os.path.join(self._check_dir, "check.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        extra = dict(extra_env or {})
        self._pending_edges = (None, [])
        edges = self._edges_now() if cfg.conditions_mode == "python" else None
        if edges and edges[1]:
            import numpy as np
            epath = os.path.join(self._check_dir, "edges.npy")
            np.save(epath, np.asarray(edges[1], dtype=float))
            extra["PINNSTUDIO_DRY_RUN_EDGES"] = epath
            self._pending_edges = edges
        # the device as Solve has it (DeepXDE picks it when imported), so Check and Solve share the Python kept
        # ready; a new Python runs the check on the CPU (the dry run's own setting)
        env = warm.run_env(user_cfg, extra=dict(PINNSTUDIO_DRY_RUN="1", MPLBACKEND="Agg", DDE_BACKEND="pytorch",
                                                **extra))
        run = _CheckRun()
        self._proc = run
        self.state.put_result({})
        self.state.put_message("Checking: the Solve script runs once, without training…")
        self._reader = threading.Thread(target=self._read, args=(run, path, env, self._check_dir),
                                        name="pinnstudio-check", daemon=True)
        self._reader.start()
        self._timer.start(CHECK_TIMEOUT_MS)
        return True

    def _read(self, run, path, env, cwd):
        """(A thread) start the check, in the Python kept ready when it fits (which may still be loading: the
        window does not wait for it), and hand its whole output to the window's thread when it ends."""
        try:
            run.proc = warm.POOL.popen(path, env, cwd=cwd)
        except OSError as exc:
            self._output.emit(run, NO_START + str(exc))
            return
        if run.cancelled:
            run.kill()
        try:
            out = run.proc.stdout.read()
        except Exception as exc:
            out = f"could not read the check's output: {exc}"
        try:
            run.proc.wait(10)
        except Exception:
            pass
        self._output.emit(run, out)

    def wait_for_check(self, ms: int = CHECK_TIMEOUT_MS) -> dict:
        """(Tests) wait for the running check and return its result."""
        if self._proc is not None and self._reader is not None:
            self._reader.join(ms / 1000)
            QCoreApplication.processEvents()
            QCoreApplication.sendPostedEvents()
            QCoreApplication.processEvents()
        return self.state.result

    def _check_done(self, run, out: str):
        if run is not self._proc:                  # a check that timed out, or was replaced
            return
        self._proc = None
        self._timer.stop()
        self._clean_check_dir()
        res = ({"ok": False, "error": "the check could not start Python", "line": 0} if out.startswith(NO_START)
               else PC.parse_dry_run(out))
        key, pts = self._pending_edges
        if res.get("edges") is not None and key is not None:
            self.state.put_result(res, edge_key=key, edge_points=pts)
        else:
            self.state.put_result(res, edge_key=None)
        self.state.put_message(_summary(res))

    def _edges_now(self):
        try:
            return self.edges() if self.edges else None
        except Exception:
            return None

    def edges_current(self, key) -> bool:
        """The last Check's edge result is for this key (the code, the shapes and the domain as they are now)."""
        return key is not None and self.state.edge_key == key and self.state.result.get("edges") is not None

    def _check_timed_out(self):
        run, self._proc = self._proc, None
        if run is not None:
            run.kill()
            self._clean_check_dir()
            res = {"ok": False, "error": f"the check took longer than {CHECK_TIMEOUT_MS // 1000} s", "line": 0}
            self.state.put_result(res)
            self.state.put_message(_summary(res))

    def _clean_check_dir(self):
        if self._check_dir:
            shutil.rmtree(self._check_dir, ignore_errors=True)
            self._check_dir = ""

    def names(self) -> str:
        try:
            return PC.available_names(self._config())
        except Exception:
            return ""


def _shipped(code: str) -> bool:
    """The code of an example that comes with PINNStudio (part of the program, so not asked about)."""
    from pinnstudio.core import pinnstudio_examples as PX
    return any(e.settings.get("python_code") == code for e in PX.EXAMPLES)


def _line_of(msg: str) -> int:
    import re
    m = re.search(r"line (\d+)", msg or "")
    return int(m.group(1)) if m else 0


def _summary(res: dict) -> str:
    if not res:
        return ""
    if res.get("ok"):
        r = res.get("residuals") or []
        conds = res.get("conditions") or []
        shapes = (f"{len(r)} residual{'s' if len(r) != 1 else ''} ({', '.join('×'.join(str(v) for v in s) for s in r)})"
                  if r else "the equations through their loss")
        hk = res.get("hooks") or []
        return (f"Check passed: {shapes}, {len(conds)} condition{'s' if len(conds) != 1 else ''}"
                + (f"; hooks used: {', '.join(hk)}" if hk else ""))
    where = f" (line {res['line']})" if res.get("line") else ""
    return f"Check failed{where}: {res.get('error', '')}"
