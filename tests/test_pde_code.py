"""Equations written in Python (core/pde_code.py): the source checks, the start written from the expressions
(the same residuals as the expressions, on every shipped example), the Solve and Export scripts in Python
mode, and the check script."""
import ast
import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pinnstudio.core import codegen as CG  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core import pde_code as PC  # noqa: E402
from pinnstudio.core import pinnstudio_examples as PX  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def _example(name, inverse=False):
    for mod in (DX, PX):
        for ex in mod.EXAMPLES:
            if ex.name == name:
                return mod.build_config(ex, REF, inverse=inverse)
    raise KeyError(name)


def _python(cfg, code=None):
    return dataclasses.replace(cfg, pde_mode="python", python_code=code if code is not None else PC.from_expressions(cfg))


# ── source checks ───────────────────────────────────────────────────────
@pytest.mark.parametrize("code, says", [
    ("", "there is none"),
    ("def pde(x, y, d)\n    return [d.u]\n", "line 1"),
    ("def f(x, y, d):\n    return [d.u]\n", "define the function pde"),
    ("def pde(x, y):\n    return [y]\n", "3 arguments"),
    ("print('hi')\ndef pde(x, y, d):\n    return [d.u]\n", "line 1: only imports"),
    ("for i in range(3):\n    pass\ndef pde(x, y, d):\n    return [d.u]\n", "line 1: only imports"),
])
def test_check_source_says_what_is_wrong(code, says):
    errors = PC.check_source(code)
    assert errors and says in " ".join(errors), errors


def test_check_source_accepts_helpers_and_imports():
    code = ('"""Burgers."""\nimport math\nNU = 0.01 / math.pi\n\ndef flux(u):\n    return u * u / 2\n\n'
            'def pde(x, y, d):\n    return [d("u", "t") + d.diff(flux(d.u), "x") - NU * d("u", "x", "x")]\n')
    assert PC.check_source(code) == []


def test_config_validate_checks_the_code_and_skips_the_expression_checks():
    cfg = PINNConfig(pde_mode="python", python_code="def pde(x, y):\n    return [y]\n",
                     pde_expressions="du_xxxxxxx + frac(u)")
    errors = cfg.validate()
    assert any("3 arguments" in e for e in errors)
    assert not any("not available" in e or "frac" in e for e in errors), errors
    ok = PINNConfig(pde_mode="python", python_code=PC.template(PINNConfig()))
    assert not any("Python code" in e for e in ok.validate())


# ── the names a problem has ─────────────────────────────────────────────
@pytest.mark.parametrize("dim, steady, axes", [
    ("1D", False, ["x", "t"]), ("1D", True, ["x"]), ("2D", False, ["x", "y", "t"]), ("2D", True, ["x", "y"]),
    ("3D", False, ["x", "y", "z", "t"]), ("3D", True, ["x", "y", "z"]), ("ODE", False, ["t"]),
])
def test_axes_and_the_template_for_every_dimension(dim, steady, axes):
    cfg = PINNConfig(problem_dim=dim, steady_state=steady, pde_expressions="0")
    assert PC.axes(cfg) == axes
    code = PC.template(cfg)
    assert PC.check_source(code) == [], code
    assert PC.space_axes(cfg) == [a for a in axes if a != "t"]


def test_from_expressions_names_each_derivative_once():
    cfg = PINNConfig(problem_dim="2D", num_outputs=2, output_names="u,v", num_equations=2,
                     pde_expressions="du_t - nu*(du_xx + du_yy) + u*du_x|dv_t + sin(pi*x)*y")
    code = PC.from_expressions(cfg)
    assert 'du_xx = d("u", "x", "x")' in code and 'du_t = d("u", "t")' in code
    assert "u = d.u" in code and "x = d.x" in code and "y = d.y" in code
    assert "torch.sin(np.pi * x)" in code.replace("torch.sin(np.pi*x)", "torch.sin(np.pi * x)")
    assert code.count("du_x = ") == 1 and PC.check_source(code) == []


def test_from_expressions_refuses_integral_and_frac():
    with pytest.raises(ValueError):
        PC.from_expressions(PINNConfig(problem_dim="ODE", pde_expressions="du_t - integral(u)"))
    with pytest.raises(ValueError):
        PC.from_expressions(PINNConfig(pde_expressions="frac(u) - 1"))


# ── the same residuals as the expressions, on every shipped example ─────
def _all_examples():
    out = []
    for mod in (DX, PX):
        for ex in mod.EXAMPLES:
            out.append(ex.name)
    return out


def _residuals(block, cfg, x, net):
    import deepxde as dde
    import torch
    ns = {"dde": dde, "np": np, "torch": torch, "__name__": "t"}
    for name, init, _true in CG._parse_inverse_variables(cfg) if cfg.problem_type == "Inverse" else []:
        ns[name] = torch.tensor(float(init) + 0.25)
    exec(compile(block, "<block>", "exec"), ns)
    dde.grad.clear()
    r = ns["pde"](x, net(x))
    return [t.detach().numpy() for t in (r if isinstance(r, list) else [r])]


@pytest.mark.parametrize("name", _all_examples())
def test_from_expressions_gives_the_expressions_residuals(name):
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    import torch
    cfg = _example(name)
    try:
        code = PC.from_expressions(cfg)
    except ValueError:
        pytest.skip("integral or frac: no Python form")
    if cfg.network_type == "Designed" or getattr(cfg, "input_feature_formula", ""):
        pass                                     # the residuals do not depend on the network's insides
    py = _python(cfg, code)

    def block(c):
        consts = CG._pde_constant_dict(c) if c.pde_mode == "python" else {}
        eff = CG._effective_config(c)
        if c.pde_mode == "python":
            return PC.export_code(eff, consts), eff
        names = [n.strip() for n in eff.output_names.split(",")]
        while len(names) < eff.num_outputs:
            names.append(f"u{len(names)}")
        return CG._clean_pde_block(eff, names, eff.num_outputs, eff.problem_dim == "2D", eff.problem_dim == "3D",
                                   bool(eff.steady_state)), eff
    b_expr, eff = block(cfg)
    b_py, _ = block(py)
    n_in = len(PC.axes(eff))
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(n_in, 12), torch.nn.Tanh(), torch.nn.Linear(12, int(eff.num_outputs)))
    x = (torch.rand(40, n_in) * 0.8 + 0.1).requires_grad_(True)
    a = _residuals(b_expr, cfg, x, net)
    b = _residuals(b_py, cfg, x, net)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        np.testing.assert_allclose(rb, ra, rtol=1e-5, atol=1e-6)


# ── the scripts ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name, inverse, ta", [
    ("Diffusion equation", False, False), ("Diffusion equation", False, True),
    ("Inverse: diffusion equation", False, False), ("Kovasznay flow", False, False),
])
def test_scripts_parse_in_python_mode(name, inverse, ta):
    cfg = _python(_example(name, inverse))
    if ta:
        cfg = dataclasses.replace(cfg, time_adaptive=True)
    solve = CG.generate_script(cfg)
    ast.parse(solve)
    assert "_pde_user = _PDE_NS[\"pde\"]" in solve and "\npde = _pde_python\n" in solve
    export = CG.generate_clean_script(cfg)
    ast.parse(export)
    assert "def pde(x, y, d):" in export and "_pde_user = _PDE_NS[\"pde\"]" in export
    assert f"_PDE_SOURCE = {cfg.python_code!r}" in solve                   # embedded as written


def test_constants_reach_the_code_by_name():
    cfg = PINNConfig(constants=json.dumps([{"name": "nu", "value": "0.4"}]), pde_mode="python",
                     python_code="def pde(x, y, d):\n    return [d('u', 't') - nu * d('u', 'x', 'x') + 0 * d.nu]\n")
    assert PC.check_source(cfg.python_code) == []
    export = CG.generate_clean_script(cfg)
    assert "\nnu = 0.4\n" in export and "_PDE_CONSTANTS = {'nu': nu}" in export
    assert "'nu': 0.4" in CG.generate_script(cfg)
