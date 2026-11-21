"""Boundary and initial conditions written in Python (conditions(c), core/pde_code.py): the conditions
written from the rows train the same problem as the rows. Each example's Solve script runs as a dry run
(PINNSTUDIO_DRY_RUN) with its rows and with the equations and conditions written as Python; the number of
loss terms and every initial loss must agree. Also: any DeepXDE condition through c.add, the time windows,
and what Python conditions refuse."""
import dataclasses
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from pinnstudio.core import codegen as CG  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core import pde_code as PC  # noqa: E402
from pinnstudio.core import pinnstudio_examples as PX  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def _example(name):
    for mod in (DX, PX):
        for ex in mod.EXAMPLES:
            if ex.name == name:
                return mod.build_config(ex, REF)
    raise KeyError(name)


def _as_python(cfg, equations=True):
    code = (PC.from_expressions(cfg) + "\n\n" if equations else "") + PC.conditions_from_rows(cfg)
    return dataclasses.replace(cfg, pde_mode="python" if equations else "expressions", conditions_mode="python",
                               python_code=code)


def _env():
    return dict(os.environ, DDE_BACKEND="pytorch", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg",
                PYTHONPATH=ROOT, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))


def _dry(cfg, tmp_path, tag):
    cfg = dataclasses.replace(PC.dry_run_config(cfg), save_dir=str(tmp_path / tag))
    path = tmp_path / f"{tag}.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=900,
                       env=dict(_env(), PINNSTUDIO_DRY_RUN="1"), cwd=str(tmp_path))
    res = PC.parse_dry_run(r.stdout + r.stderr)
    assert res.get("ok"), (tag, res, (r.stdout + r.stderr)[-3000:])
    return res


def _same(name, tmp_path, equations=True):
    cfg = _example(name)
    a = _dry(cfg, tmp_path, "rows")
    b = _dry(_as_python(cfg, equations), tmp_path, "python")
    assert len(a["losses"]) == len(b["losses"]), (a, b)
    assert a["losses"] == pytest.approx(b["losses"], rel=1e-6, abs=1e-9), (a, b)
    return a, b


QUICK = ["Diffusion equation", "Kovasznay flow", "Second-order ODE"]
MORE = ["Beltrami flow", "3D Poisson (Cube with spherical cut-out)", "ODE system", "Inverse: diffusion equation",
        "Inverse: Lorenz system", "Inverse: Poisson with unknown forcing field", "Poisson 1D – Dirichlet/Neumann",
        "Euler beam", "Wave propagation – space-time multi-scale Fourier features", "Klein-Gordon equation",
        "Poisson 1D – Dirichlet/Periodic", "Schrödinger equation", "Poisson 1D – Dirichlet/PointSetOperator",
        "Poisson 1D – Dirichlet/Robin", "Helmholtz sound-hard scattering (absorbing BC)",
        "Laplace equation on a disk", "Poisson over an L-shaped domain", "Lotka-Volterra equation"]


@pytest.mark.parametrize("name", QUICK)
def test_the_rows_as_python_give_the_same_losses(name, tmp_path):
    _same(name, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("name", MORE)
def test_the_rows_as_python_give_the_same_losses_everywhere(name, tmp_path):
    _same(name, tmp_path)


def test_conditions_alone_in_python_with_the_expressions(tmp_path):
    _same("Kovasznay flow", tmp_path, equations=False)


def test_an_initial_condition_from_a_file(tmp_path):
    import numpy as np
    cfg = _example("Diffusion equation")
    xs = np.linspace(cfg.x_min, cfg.x_max, 41)
    data = np.column_stack([xs, np.full_like(xs, cfg.t_min), np.sin(np.pi * xs)])   # x, t, u
    path = tmp_path / "ic.txt"
    np.savetxt(path, data)
    cfg = dataclasses.replace(cfg, forward_ic_from_file=True, forward_ic_file=str(path))
    code = PC.conditions_from_rows(cfg)
    assert "c.initial_points(" in code
    a = _dry(cfg, tmp_path, "rows")
    b = _dry(dataclasses.replace(cfg, conditions_mode="python", python_code=code), tmp_path, "python")
    assert a["losses"] == pytest.approx(b["losses"], rel=1e-6)


def test_any_deepxde_condition_through_c_add(tmp_path):
    cfg = _example("Diffusion equation")
    code = PC.conditions_from_rows(cfg) + '''
    def flux(x, y, X):
        return c.d(x, y)("u", "x")
    c.add(dde.icbc.OperatorBC(c.geom, flux, lambda X, on: on and c.isclose(X[0], c.x_max)), weight=2.0,
          name="no flux on the right")
    c.neumann("u", where=lambda X, on: on and c.left(X), value=lambda X: 0 * X[:, 0:1], name="mine")
'''
    res = _dry(dataclasses.replace(cfg, conditions_mode="python", python_code=code), tmp_path, "python")
    assert res["conditions"][-3:-1] == ["no flux on the right", "mine"], res      # boundary terms, then the IC
    assert len(res["losses"]) == 1 + len(res["conditions"])


def test_a_bad_condition_says_its_line(tmp_path):
    cfg = _example("Diffusion equation")
    code = "def conditions(c):\n    c.dirichlet('q', where='left', value='0')\n"
    cfg = dataclasses.replace(PC.dry_run_config(dataclasses.replace(cfg, conditions_mode="python", python_code=code)),
                              save_dir=str(tmp_path / "run"))
    path = tmp_path / "bad.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=900,
                       env=dict(_env(), PINNSTUDIO_DRY_RUN="1"), cwd=str(tmp_path))
    res = PC.parse_dry_run(r.stdout + r.stderr)
    assert not res["ok"] and res["line"] == 2 and "no output 'q'" in res["error"], res


def test_python_conditions_refuse_exact_initial_conditions():
    cfg = PINNConfig(custom_bc_json="[]", conditions_mode="python",
                     python_code="def conditions(c):\n    c.initial('u', 0)\n", ic_hard="True")
    assert any("Exact initial conditions" in e for e in cfg.validate())
    cfg = dataclasses.replace(cfg, python_code="def conditions(c, extra):\n    pass\n", ic_hard="")
    assert any("conditions() takes 1 argument" in e for e in cfg.validate())


def test_the_export_is_the_solve_script_with_python_conditions():
    cfg = _as_python(_example("Kovasznay flow"))
    export = CG.generate_clean_script(cfg)
    assert export.startswith("# Exported from PINNStudio. This problem's boundary and initial conditions")
    assert "_run_conditions(geomtime)" in export
    import ast
    ast.parse(export)


def _train_lines(cfg, tmp_path, tag):
    cfg = dataclasses.replace(cfg, save_dir=str(tmp_path / tag), train_device="cpu")
    path = tmp_path / f"{tag}.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=1800, env=_env(),
                       cwd=str(tmp_path))
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    return [ln.split("]")[0] for ln in r.stdout.splitlines() if re.match(r"^\d+\s+\[", ln)]


@pytest.mark.slow
def test_time_windows_train_the_same_with_python_conditions(tmp_path):
    cfg = dataclasses.replace(_example("Diffusion equation"), time_adaptive=True, adapt_method="Time Adaptive",
                              iterations=40, optimizer2="none", loss_display_every=20, ta_num_steps=2)
    a = _train_lines(cfg, tmp_path, "rows")
    b = _train_lines(_as_python(cfg), tmp_path, "python")
    assert a and a == b


def _spacetime_l2(folder):
    import glob
    files = glob.glob(os.path.join(folder, "**", "error_spacetime*.txt"), recursive=True)
    assert files, f"no error_spacetime file under {folder}"
    with open(files[0], encoding="utf-8") as fh:
        return float(fh.read().splitlines()[1].split(",")[1])


def _train_l2(cfg, tmp_path, tag, iterations=2000):
    cfg = dataclasses.replace(cfg, iterations=iterations, optimizer2="none", save_dir=str(tmp_path / tag),
                              train_device="cpu")
    path = tmp_path / f"{tag}.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=3600, env=_env(),
                       cwd=str(tmp_path))
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    return _spacetime_l2(str(tmp_path / tag))


@pytest.mark.slow
def test_burgers_written_from_the_page_trains_exactly_as_the_expressions(tmp_path):
    """The code written from the expressions and rows trains the same network: the same error to the digit."""
    rows = dataclasses.replace(_example("Burgers equation (Python code)"), pde_mode="expressions",
                               conditions_mode="rows")
    assert _train_l2(rows, tmp_path, "rows") == pytest.approx(_train_l2(_as_python(rows), tmp_path, "python"),
                                                              rel=1e-6)


@pytest.mark.slow
def test_the_burgers_example_in_python_trains(tmp_path):
    """The shipped example, written by hand in conservation form (another floating-point path, so another
    training trajectory): 2000 Adam iterations reach the accuracy of the first stage of training."""
    assert _train_l2(_example("Burgers equation (Python code)"), tmp_path, "python") < 0.05
