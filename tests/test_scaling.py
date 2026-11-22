"""Scaling (Neural network page): the input transform's "domain" mode maps each input to [-1, 1] from the
domain's limits, in the Solve script, the export and the run record; suggested output scales come from the
reference formula, else the initial and Dirichlet boundary values, else 1."""
import dataclasses
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import codegen as CG  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core import pde_code as PC  # noqa: E402
from pinnstudio.core import scaling as S  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def test_domain_scaling_maps_the_limits_to_minus_one_and_one():
    cfg = dataclasses.replace(PINNConfig(), problem_dim="2D", steady_state=False, x_min=-2.0, x_max=6.0,
                              y_min=10.0, y_max=11.0, t_min=0.0, t_max=50.0)
    sc, sh = CG._domain_input_scaling(cfg)
    lo = np.array([-2.0, 10.0, 0.0])
    hi = np.array([6.0, 11.0, 50.0])
    assert np.allclose(lo * sc + sh, -1) and np.allclose(hi * sc + sh, 1)
    steady = dataclasses.replace(cfg, steady_state=True)
    assert len(CG._domain_input_scaling(steady)[0]) == 2


def test_domain_mode_in_the_scripts_and_the_run_record():
    cfg = dataclasses.replace(DX.build_config(DX.BY_NAME["Diffusion equation"], REF), input_transform_enabled=True,
                              input_transform_mode="domain", x_min=-1.0, x_max=3.0)
    src = CG._net_transforms_code(CG._effective_config(cfg))
    assert "in_scale = [0.5, 2.0]" in src and "in_shift = [-0.5, -1.0]" in src
    for script in (CG.generate_script(cfg), CG.generate_clean_script(cfg)):
        compile(script, "<script>", "exec")
        assert "each input to [-1, 1] from the domain's limits" in script
    rec = CG._transform_config_literal(CG._effective_config(cfg))
    assert "'input_transform_mode': 'scale'" in rec and "'input_transform_scale': [0.5, 2.0]" in rec
    # the restore scripts read the record: the same numbers in scale mode
    restored = {"input_transform_enabled": True, "input_transform_mode": "scale",
                "input_transform_scale": [0.5, 2.0], "input_transform_shift": [-0.5, -1.0],
                "num_outputs": 1, "output_names": "u", "problem_dim": "1D"}
    assert "in_scale = [0.5, 2.0]" in CG._net_transforms_code(restored)


def test_domain_mode_trains_the_same_problem_in_a_dry_run(tmp_path):
    cfg = dataclasses.replace(DX.build_config(DX.BY_NAME["Diffusion equation"], REF), input_transform_enabled=True,
                              input_transform_mode="domain")
    cfg = dataclasses.replace(PC.dry_run_config(cfg), save_dir=str(tmp_path / "run"))
    path = tmp_path / "run.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    env = dict(os.environ, DDE_BACKEND="pytorch", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg", PYTHONPATH=ROOT,
               PINNSTUDIO_DRY_RUN="1", OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=600, env=env,
                       cwd=str(tmp_path))
    res = PC.parse_dry_run(r.stdout + r.stderr)
    assert res.get("ok"), (res, (r.stdout + r.stderr)[-2000:])


def test_output_scales_from_the_reference_formula():
    cfg = dataclasses.replace(DX.build_config(DX.BY_NAME["Diffusion equation"], REF),
                              ea_ref_mode="formula", ea_ref_formula="40*sin(pi*x)*exp(-t)")
    s = S.suggest_output_scales(cfg)
    assert s["source"] == ["the reference formula"]
    assert 30 < s["scale"][0] <= 40 and s["shift"] == [0.0]


def test_output_scales_from_the_initial_and_boundary_values():
    cfg = dataclasses.replace(DX.build_config(DX.BY_NAME["Diffusion equation"], REF), ea_ref_mode="none",
                              ic_expressions="300 + 20*sin(pi*x)",
                              custom_bc_json='[{"type": "dirichlet", "location": "True", "value": "300", "component": 0}]')
    s = S.suggest_output_scales(cfg)
    assert s["source"] == ["the initial and boundary values"]
    assert 19 < s["scale"][0] <= 20 and 299 <= s["shift"][0] <= 301, s      # x in [-1, 1]: 280 to 320
    one = dataclasses.replace(cfg, ic_expressions="300", custom_bc_json="[]")
    assert S.suggest_output_scales(one)["scale"] == [300.0] and S.suggest_output_scales(one)["shift"] == [0.0]


def test_no_information_leaves_the_outputs_unchanged():
    cfg = dataclasses.replace(PINNConfig(), ea_ref_mode="none", ic_expressions="", custom_bc_json="[]",
                              steady_state=True, num_outputs=2, output_names="u, v")
    s = S.suggest_output_scales(cfg)
    assert s == {"scale": [1.0, 1.0], "shift": [0.0, 0.0], "source": ["none", "none"]}


def test_the_page_suggests_scales_and_keeps_the_domain_mode(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.main_window import MainWindow
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    try:
        cfg = dataclasses.replace(DX.build_config(DX.BY_NAME["Diffusion equation"], REF),
                                  ea_ref_mode="formula", ea_ref_formula="5*sin(pi*x)*exp(-t)",
                                  input_transform_enabled=True, input_transform_mode="domain")
        w._apply_config(cfg)
        app.processEvents()
        assert w.input_transform_mode_combo.currentData() == "domain"
        w._suggest_output_scales()
        r = w.output_transform_rows[0]
        assert 4 < r["scale"].value() <= 5 and "reference formula" in w.suggest_scales_note.text()
        out = w._build_config()
        assert out.input_transform_mode == "domain"
        assert 4 < out.output_transform_scale[0] <= 5
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()
