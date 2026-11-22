"""The Python code's hooks besides pde and conditions (core/pde_code.py HOOKS): output_transform (exact
conditions), feature_transform (the first layer sized to it), points(geom, n), callbacks(), and c.loss in
conditions(c). Each is used when the code defines it, with the equations and conditions in either mode."""
import dataclasses
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import codegen as CG  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core import pde_code as PC  # noqa: E402
from pinnstudio.core import pinnstudio_examples as PX  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def _diffusion(**kw):
    cfg = DX.build_config(DX.BY_NAME["Diffusion equation"], REF)
    return dataclasses.replace(cfg, **kw)


def _burgers_python(**kw):
    ex = next(e for e in PX.EXAMPLES if e.name == "Burgers equation (Python code)")
    return dataclasses.replace(PX.build_config(ex, REF), **kw)


def _env(**kw):
    return dict(os.environ, DDE_BACKEND="pytorch", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg", PYTHONPATH=ROOT,
                OMP_NUM_THREADS="2", **kw)


def _run(cfg, tmp_path, dry=True, tag="run"):
    if dry:
        cfg = PC.dry_run_config(cfg)
    cfg = dataclasses.replace(cfg, save_dir=str(tmp_path / tag))
    assert not cfg.validate(), cfg.validate()
    path = tmp_path / f"{tag}.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    env = _env(PINNSTUDIO_SCRATCH=str(tmp_path / "scratch"), **({"PINNSTUDIO_DRY_RUN": "1"} if dry else {}))
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=900, env=env,
                       cwd=str(tmp_path))
    out = r.stdout + r.stderr
    if dry:
        res = PC.parse_dry_run(out)
        assert res.get("ok"), (res, out[-3000:])
        return res, out
    assert r.returncode == 0, out[-4000:]
    return None, out


def test_hooks_found_and_checked():
    code = PC.HOOK_TEMPLATES["points"] + "\n" + PC.HOOK_TEMPLATES["callbacks"]
    assert PC.hooks(code) == ["points", "callbacks"]
    assert PC.check_source(code, needs=()) == []
    bad = "def points(geom):\n    return geom.random_points(10)\n"
    assert any("points() takes 2 arguments" in e for e in PC.check_source(bad, needs=()))
    assert PC.uses_module(_diffusion(python_code=code))
    assert not PC.uses_module(_diffusion(python_code=""))
    assert PC.in_use(_diffusion(python_code=code)) == ["points", "callbacks"]
    for name, src in PC.HOOK_TEMPLATES.items():
        assert PC.hooks(src) == [name] and PC.check_source(src, needs=()) == [], name


def test_no_hooks_leave_the_scripts_as_they_were():
    cfg = _diffusion()
    s = CG.generate_script(cfg)
    assert "def _pde_hooks" not in s and "_PDE_SOURCE" not in s and "_pde_user_callbacks" not in s


def test_exact_initial_conditions_as_output_transform(tmp_path):
    cfg = _diffusion()
    code = PC.exact_initial_transform(cfg)
    res, _ = _run(dataclasses.replace(cfg, python_code=code), tmp_path)
    assert res["hooks"] == ["output_transform"]
    # the rows' initial condition (the last term) holds exactly
    assert res["losses"][-1] < 1e-10, res["losses"]
    plain, _ = _run(cfg, tmp_path, tag="plain")
    assert plain["losses"][-1] > 1e-4
    assert len(plain["losses"]) == len(res["losses"])


def test_output_transform_composes_with_a_scale_transform(tmp_path):
    cfg = _diffusion(output_transform_enabled=True, output_transform_mode="scale", output_transform_scale=[2.0],
                     output_transform_shift=[0.0])
    code = PC.exact_initial_transform(cfg)
    res, _ = _run(dataclasses.replace(cfg, python_code=code), tmp_path)
    assert res["losses"][-1] < 1e-10


def test_feature_transform_sizes_the_first_layer(tmp_path):
    code = ("def feature_transform(x):\n"
            "    return torch.cat([x, torch.sin(np.pi * x[:, 0:1]), torch.cos(np.pi * x[:, 0:1])], dim=1)\n")
    cfg = _diffusion(python_code=code)
    script = CG.generate_script(PC.dry_run_config(cfg))
    assert "_flat_layers[0] = int(_pde_hooks()" in script
    res, _ = _run(cfg, tmp_path)
    assert res["hooks"] == ["feature_transform"] and res["residuals"] == [[64, 1]]


def test_c_loss_adds_a_term_of_your_own(tmp_path):
    cfg = _burgers_python()
    before, _ = _run(cfg, tmp_path, tag="before")
    code = cfg.python_code.replace(
        "def conditions(c):",
        "def conditions(c):\n    c.loss(lambda x, y, d: torch.mean(y[d.rows, 0:1]) - 0.5, n=200, weight=2.0, name='mean')\n"
        "    c.loss(lambda x, y, d: d('u', 'x'), points='boundary', n=50, name='flat')", 1)
    res, _ = _run(dataclasses.replace(cfg, python_code=code), tmp_path, tag="after")
    assert res["conditions"][:2] == ["mean", "flat"]
    assert len(res["losses"]) == len(before["losses"]) + 2
    assert all(v >= 0 for v in res["losses"])


def test_points_and_callbacks_in_a_short_run(tmp_path):
    code = ('_SEEN = []\n\n\n'
            'def points(geom, n):\n'
            '    _SEEN.append(n)\n'
            '    print("POINTS HOOK", n, flush=True)\n'
            '    return geom.random_points(n)\n\n\n'
            'class _Hello(dde.callbacks.Callback):\n'
            '    def on_train_begin(self):\n'
            '        print("USER CALLBACK", len(self.model.data.train_x_all), flush=True)\n\n\n'
            'def callbacks():\n'
            '    return [_Hello()]\n')
    cfg = _diffusion(python_code=code, iterations=20, optimizer2="none", num_domain=120, num_boundary=20,
                     num_initial=20, num_test=50, train_device="cpu")
    _, out = _run(cfg, tmp_path, dry=False)
    assert "POINTS HOOK 120" in out, out[-3000:]
    assert "USER CALLBACK" in out, out[-3000:]
    n_all = int(out.split("USER CALLBACK")[1].split()[0])
    assert n_all >= 120


@pytest.mark.parametrize("settings, code, words", [
    (dict(output_transform_enabled=True, output_transform_mode="formula", output_transform_formula="u"),
     PC.HOOK_TEMPLATES["output_transform"], "output transform formula"),
    (dict(ic_hard="True"), PC.HOOK_TEMPLATES["output_transform"], "Exact initial conditions"),
    (dict(input_transform_enabled=True), PC.HOOK_TEMPLATES["feature_transform"], "input transform"),
    (dict(network_type="Designed"), PC.HOOK_TEMPLATES["feature_transform"], "designed network"),
])
def test_refused_combinations(settings, code, words):
    errs = _diffusion(python_code=code, **settings).validate()
    assert any(words in e for e in errs), errs


def test_restore_scripts_load_the_module_and_reattach_the_transforms():
    torch = pytest.importorskip("torch")
    dde = pytest.importorskip("deepxde")
    code = ("K = 3.0\n\n\n"
            "def output_transform(x, y):\n"
            "    return x[:, 0:1] * y * K * scale\n")
    cfg = {"python_code": code, "python_constants": {"scale": 2.0}, "num_outputs": 1, "output_names": "u",
           "problem_dim": "1D", "steady_state": True}
    src = CG._net_transforms_code(cfg)
    assert "def _pde_hooks():" in src
    ns = {"torch": torch, "np": __import__("numpy"), "dde": dde}
    exec(src, ns)
    net = ns["_apply_net_transforms"](dde.nn.FNN([1, 8, 1], "tanh", "Glorot normal"))
    x = torch.linspace(0, 1, 5).reshape(-1, 1)
    raw = net.linears[-1](torch.tanh(net.linears[0](x)))
    assert torch.allclose(net(x), x * raw * 6.0, atol=1e-6)


def test_hook_scripts_compile_and_export_is_the_solve_script():
    code = PC.HOOK_TEMPLATES["output_transform"] + "\n" + PC.HOOK_TEMPLATES["callbacks"]
    cfg = _diffusion(python_code=code)
    for script in (CG.generate_script(cfg), CG.generate_clean_script(cfg)):
        compile(script, "<script>", "exec")
        assert "_pde_user_callbacks()" in script and '"python_constants"' in script
    assert CG.generate_clean_script(cfg).startswith("# Exported from PINNStudio. This problem's Python code defines")
