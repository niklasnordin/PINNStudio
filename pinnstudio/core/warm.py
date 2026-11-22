"""A Python process kept ready for the next run.

Every Solve, sweep run and Check used to start a new Python that first imports PyTorch and DeepXDE: a few
seconds on Linux, 7–9 s in the Windows program (benchmarks/platform/windows_speed.md). PINNStudio now keeps
one spare process that has done those imports already and waits for a script. A run takes it, and the next
spare starts in the background.

The spare runs one script and then ends, so nothing carries from one run to the next. Its environment is the
run's own, except for the variables that only count before PyTorch and DeepXDE are imported (DeepXDE picks the
GPU or the CPU when it is imported, and the thread settings are read then). A run whose values for those
differ from the spare's starts a new process, as before.

    serve()                   the spare: imports, says it is ready, runs the one script it is sent
    POOL.popen(script, env)   a started process running `script` (the spare when it fits, otherwise new)
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading

READY = "PINNSTUDIO_WARM_READY "
# variables read when PyTorch, DeepXDE or Python start, and not later
_BEFORE_IMPORT = ("OMP_", "MKL_", "KMP_", "OPENBLAS_", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
                  "HIP_VISIBLE_DEVICES", "DDE_", "PYTHONPATH", "PYTHONHOME", "PYTHONUTF8", "PYTHONIOENCODING",
                  "PYTORCH_ENABLE_MPS_FALLBACK", "TORCH_")


def _key(env: dict, cuda: bool = True) -> tuple:
    """The values that must match between the spare and a run."""
    names = sorted(k for k in env if k.startswith(_BEFORE_IMPORT))
    if not cuda:          # with no GPU, PyTorch sees none whatever CUDA_VISIBLE_DEVICES says
        names = [k for k in names if k != "CUDA_VISIBLE_DEVICES"]
    return tuple((k, env[k]) for k in names)


# ── the spare ─────────────────────────────────────────────────────────
def serve():
    """The spare process: import, write the ready line, then run the one script sent on stdin as a JSON line
    {"script": path, "env": {...}, "cwd": dir} as `python script` would. Ends when stdin closes."""
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    try:
        import numpy  # noqa: F401
        import matplotlib
        import torch
        import deepxde  # noqa: F401
        matplotlib.use(os.environ.get("MPLBACKEND") or "Agg")
        cuda = bool(torch.cuda.is_available())
    except BaseException as exc:                    # the run started instead shows the real error
        print(f"warm start failed: {exc!r}", file=sys.stderr, flush=True)
        raise SystemExit(3)
    sys.stdout.write(READY + json.dumps({"cuda": cuda}) + "\n")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line.strip():
        raise SystemExit(0)
    req = json.loads(line)
    os.environ.clear()
    os.environ.update(req.get("env") or {})
    if req.get("cwd"):
        os.chdir(req["cwd"])
    run_path(req["script"], req.get("args") or [])


def run_path(script: str, args=()):
    """Run `script` as __main__, with its output in UTF-8, line by line."""
    import runpy
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
        except Exception:
            pass
    sys.argv = [script, *args]
    sys.path[:] = [os.path.dirname(os.path.abspath(script))] + [p for p in sys.path if p not in ("", ".")]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    raise SystemExit(0)


# ── the pool (in PINNStudio) ──────────────────────────────────────────
def _spare_command():
    from pinnstudio.core import pyexe
    if pyexe.frozen():                       # the Windows program: PINNStudio.exe --warm
        return [sys.executable, "--warm"]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [pyexe.script_python(), "-c",
            f"import sys; sys.path.insert(0, {root!r}); from pinnstudio.core.warm import serve; serve()"]


class _Spare:
    def __init__(self, env: dict):
        from pinnstudio.core import pyexe
        self.env = dict(env)
        self.cuda = None
        self.proc = subprocess.Popen(_spare_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=self.env, creationflags=pyexe.NO_WINDOW,
                                     **pyexe.TEXT_UTF8)
        pyexe.end_with_pinnstudio(self.proc)
        self.ready = threading.Event()
        threading.Thread(target=self._wait_ready, name="pinnstudio-warm", daemon=True).start()

    def _wait_ready(self):
        try:
            for _ in range(200):          # what the imports print (DeepXDE: "Using backend: …") comes first
                line = self.proc.stdout.readline()
                if not line:
                    break
                if line.startswith(READY):
                    self.cuda = bool(json.loads(line[len(READY):]).get("cuda"))
                    break
        except Exception:
            pass
        self.ready.set()

    def usable(self, env: dict, wait: float) -> bool:
        if self.proc.poll() is not None:
            return False
        self.ready.wait(wait)
        if self.cuda is None or self.proc.poll() is not None:
            return False
        return _key(self.env, self.cuda) == _key(env, self.cuda)

    def kill(self):
        try:
            self.proc.kill()
            self.proc.wait(5)
        except Exception:
            pass


def _allowed() -> bool:
    """PINNSTUDIO_NO_WARM=1 turns the spare off for good (every run then starts a new Python)."""
    return os.environ.get("PINNSTUDIO_NO_WARM", "") in ("", "0")


class WarmPool:
    """One spare process. Thread-safe: sweeps run in a thread of their own."""

    def __init__(self):
        self.enabled = _allowed()
        self._spare = None
        self._lock = threading.Lock()
        self.wait = 120.0            # how long a run waits for a spare that is still importing

    def prestart(self, env: dict):
        """Start a spare for runs with this environment, unless one is ready already."""
        with self._lock:
            if not self.enabled:
                return
            s = self._spare
            if s is not None and s.proc.poll() is None and _key(s.env) == _key(env):
                return
            if s is not None:
                s.kill()
            try:
                self._spare = _Spare(env)
            except OSError:
                self._spare = None

    def take(self, env: dict):
        """The spare, when it fits a run with this environment (it is then no longer the spare), else None.
        A new spare starts for the next run either way."""
        with self._lock:
            s, self._spare = self._spare, None
        if s is not None and not s.usable(env, self.wait):
            s.kill()
            s = None
        if self.enabled:
            threading.Thread(target=self.prestart, args=(dict(env),), daemon=True).start()
        return s

    def popen(self, script: str, env: dict, cwd: str | None = None):
        """A process running `script` with `env`: the spare when it fits, otherwise a new Python.
        Its stdout carries stdout and stderr, as text."""
        from pinnstudio.core import pyexe
        s = self.take(env) if self.enabled else None
        if s is not None:
            try:
                s.proc.stdin.write(json.dumps({"script": script, "env": env, "cwd": cwd or os.getcwd()}) + "\n")
                s.proc.stdin.flush()
                s.proc.stdin.close()
                return s.proc
            except OSError:
                s.kill()
        proc = subprocess.Popen([pyexe.script_python(), script], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, cwd=cwd, creationflags=pyexe.NO_WINDOW,
                                **pyexe.TEXT_UTF8)
        pyexe.end_with_pinnstudio(proc)
        return proc

    def close(self):
        with self._lock:
            s, self._spare = self._spare, None
            self.enabled = False
        if s is not None:
            s.kill()

    def set_enabled(self, on: bool):
        if on:
            self.enabled = _allowed()
        else:
            self.close()


POOL = WarmPool()
atexit.register(POOL.close)


def run_env(config=None, extra: dict | None = None) -> dict:
    """The environment of a run of `config`: script_env(), with the CPU chosen before PyTorch is imported
    when the run trains on the CPU (DeepXDE picks the device when it is imported)."""
    from pinnstudio.core import pyexe
    env = pyexe.script_env()
    env.setdefault("DDE_BACKEND", "pytorch")
    if config is not None and getattr(config, "train_device", "auto") == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.update(extra or {})
    return env


if __name__ == "__main__":
    serve()
