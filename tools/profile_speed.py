"""Where PINNStudio's waiting goes: the window's start, the time until a Solve shows its first training line, and
a Check, each with and without what speeds it up (the equation image cache, the Python kept ready).

    python tools/profile_speed.py [--repeat 3]

Prints a Markdown table (median seconds). Runs off screen, with its own PINNSTUDIO_HOME, so nothing of the
user's is touched. benchmarks/performance.md has the numbers.
"""
import argparse
import dataclasses
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WINDOW = r"""
import os, sys, time
t0 = time.perf_counter()
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from pinnstudio.ui.main_window import MainWindow
w = MainWindow(); w.show()
from pinnstudio.ui import equation
for _ in range(40):                 # until the equation previews have drawn
    app.processEvents(); time.sleep(0.01)
print("WINDOW", time.perf_counter() - t0, "matplotlib" in sys.modules)
"""


def _window(home):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PINNSTUDIO_HOME=home, PYTHONPATH=ROOT,
               MPLCONFIGDIR=os.path.join(home, "mpl"))
    out = subprocess.run([sys.executable, "-c", WINDOW], capture_output=True, text=True, env=env, cwd=ROOT)
    for ln in out.stdout.splitlines():
        if ln.startswith("WINDOW"):
            return float(ln.split()[1])
    raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])


def _first_line(cfg, warm_on, marker):
    """Seconds from start until the script prints a line with `marker` (Solve: the first training line)."""
    from pinnstudio.core import codegen as CG
    from pinnstudio.core import warm
    pool = warm.WarmPool()
    pool.enabled = warm_on
    env = warm.run_env(cfg, extra=dict(PYTHONPATH=ROOT, MPLBACKEND="Agg", **(
        {"PINNSTUDIO_DRY_RUN": "1"} if marker == "PINNSTUDIO_DRY_RUN" else {})))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(CG.generate_script(dataclasses.replace(cfg, save_dir=os.path.join(d, "out"))))
        if warm_on:
            pool.prestart(env)
            pool._spare.ready.wait(300)
        t0 = time.perf_counter()
        proc = pool.popen(path, env, cwd=d)
        took = None
        for ln in proc.stdout:
            if marker in ln and took is None:
                took = time.perf_counter() - t0
                proc.kill()
                break
        proc.wait()
        pool.close()
    if took is None:
        raise RuntimeError(f"no line with {marker!r}")
    return took


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    from pinnstudio.core import pde_code as PC
    from pinnstudio.core import pinnstudio_examples as PX
    ex = next(e for e in PX.EXAMPLES if e.name == "Burgers equation (Python code)")
    cfg = PX.build_config(ex, os.path.join(ROOT, "reference_data"))
    cfg = dataclasses.replace(cfg, train_device="cpu")
    rows = []

    def med(f):
        return statistics.median(f() for _ in range(a.repeat))

    with tempfile.TemporaryDirectory() as home:
        first = _window(home)                     # an empty cache
        rows.append(("Window start, no cached equation images", first))
        rows.append(("Window start, equation images cached", med(lambda: _window(home))))
    full = dataclasses.replace(PC.dry_run_config(cfg), num_domain=cfg.num_domain, num_boundary=cfg.num_boundary,
                               num_initial=cfg.num_initial, num_test=cfg.num_test)
    rows.append(("Check, new Python, all points", med(lambda: _first_line(full, False, "PINNSTUDIO_DRY_RUN"))))
    rows.append(("Check, new Python, capped points",
                 med(lambda: _first_line(PC.dry_run_config(cfg), False, "PINNSTUDIO_DRY_RUN"))))
    rows.append(("Check, Python kept ready, capped points",
                 med(lambda: _first_line(PC.dry_run_config(cfg), True, "PINNSTUDIO_DRY_RUN"))))
    rows.append(("Solve to the first training line, new Python", med(lambda: _first_line(cfg, False, "Step"))))
    rows.append(("Solve to the first training line, Python kept ready", med(lambda: _first_line(cfg, True, "Step"))))
    print(f"| What ({sys.platform}, median of {a.repeat}) | Seconds |\n|---|---:|")
    for name, t in rows:
        print(f"| {name} | {t:.2f} |")


if __name__ == "__main__":
    main()
