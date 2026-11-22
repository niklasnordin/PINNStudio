"""Automatic loss balancing (config.loss_balancing, codegen._loss_balance_defs): the methods' weights on
known loss sequences, the callback in the Solve script and the export, the settings' checks, and a short real
run whose weights change and are saved."""
import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import codegen as CG  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def _ns(method="relobralo", **kw):
    """The emitted definitions, run with a stand-in model."""
    torch = pytest.importorskip("torch")
    dde = pytest.importorskip("deepxde")
    ns = {"np": np, "torch": torch, "dde": dde, "os": os}
    exec(CG._loss_balance_defs(dataclasses.replace(PINNConfig(), loss_balancing=method, **kw)), ns)
    return ns


class _Model:
    def __init__(self, weights):
        self.loss_weights = weights
        self.opt_name = "adam"
        self.data = type("D", (), {"losses_train": lambda self, *a, **k: []})()
        self.train_state = type("S", (), {"step": 0})()


def _feed(ns, cb, model, seq):
    torch = ns["torch"]
    for step, L in enumerate(seq, 1):
        ns["_LB"]["last"] = [torch.tensor(v) for v in L]
        model.train_state.step = step
        cb.on_batch_end()


def test_relobralo_matches_the_formula():
    ns = _ns("relobralo", balance_rho=1.0)          # rho = 1: always the history term
    model = _Model([1.0, 10.0])
    cb = ns["_LossBalancer"](1000)
    cb.model = model
    cb.on_train_begin()
    seq = [[1.0, 1.0], [0.5, 1.0], [0.25, 1.0]]
    _feed(ns, cb, model, seq)
    lam = np.ones(2)
    bal = ns["_lb_bal"]
    L0 = np.array(seq[0])
    prev = L0
    for L in map(np.array, seq[1:]):
        lam = 0.95 * lam + 0.05 * bal(L, prev, 1.0)      # rho = 1 keeps lam in the history term
        prev = L
    assert np.allclose(ns["_LB"]["lam"], lam)
    assert np.allclose(model.loss_weights, [1.0 * lam[0], 10.0 * lam[1]])
    # the slower-falling term gets more weight
    assert lam[1] > lam[0]
    assert abs(sum(bal(np.array([0.5, 1.0]), np.array([1.0, 1.0]), 1.0)) - 2) < 1e-12


def test_softadapt_is_the_softmax_of_the_ratios():
    ns = _ns("softadapt", balance_temperature=0.5)
    model = _Model(None)
    cb = ns["_LossBalancer"](1000)
    cb.model = model
    cb.on_train_begin()
    _feed(ns, cb, model, [[2.0, 1.0, 4.0], [1.0, 1.0, 1.0]])
    r = np.array([0.5, 1.0, 0.25]) / 0.5
    want = 3 * np.exp(r - r.max()) / np.exp(r - r.max()).sum()
    assert np.allclose(model.loss_weights, want)


def test_weights_held_in_lbfgs_and_kept_across_a_recompile():
    ns = _ns("relobralo")
    model = _Model([1.0, 1.0])
    cb = ns["_LossBalancer"](1000)
    cb.model = model
    cb.on_train_begin()
    _feed(ns, cb, model, [[1.0, 1.0], [0.1, 1.0]])
    balanced = list(model.loss_weights)
    assert balanced != [1.0, 1.0]
    model.opt_name = "L-BFGS"
    model.loss_weights = [2.0, 2.0]                 # compiled again with new weights: the new base
    cb.on_train_begin()
    assert np.allclose(model.loss_weights, [2.0 * v for v in ns["_LB"]["lam"]])
    held = list(model.loss_weights)
    _feed(ns, cb, model, [[0.01, 1.0], [0.001, 1.0]])
    assert model.loss_weights == held, "L-BFGS changed the weights"
    # a different number of terms (the next time window) starts over
    model.opt_name = "adam"
    model.loss_weights = [1.0, 1.0, 1.0]
    cb.on_train_begin()
    assert ns["_LB"]["lam"] is None and model.loss_weights == [1.0, 1.0, 1.0]


def test_scripts_carry_the_callback_and_off_adds_nothing():
    off = PINNConfig()
    assert "_LossBalancer" not in CG.generate_script(off)
    for method in ("relobralo", "softadapt", "gradient"):
        c = dataclasses.replace(off, loss_balancing=method)
        for script in (CG.generate_script(c), CG.generate_clean_script(c)):
            compile(script, "<script>", "exec")
            assert "class _LossBalancer" in script and "_LossBalancer(1000)" in script
    assert '"every": 100' in CG.generate_script(dataclasses.replace(off, loss_balancing="gradient"))
    assert '"every": 7' in CG.generate_script(dataclasses.replace(off, loss_balancing="relobralo", balance_every=7))


@pytest.mark.parametrize("bad, field", [
    (dict(loss_balancing="magic"), "unknown method"),
    (dict(loss_balancing="relobralo", balance_temperature=0.0), "temperature"),
    (dict(loss_balancing="relobralo", balance_alpha=1.0), "alpha"),
    (dict(loss_balancing="relobralo", balance_rho=1.5), "rho"),
    (dict(loss_balancing="relobralo", balance_every=-1), "every"),
])
def test_bad_settings_are_refused(bad, field):
    errs = dataclasses.replace(PINNConfig(), **bad).validate()
    assert any(field in e for e in errs), errs


@pytest.mark.slow
@pytest.mark.parametrize("method", ["relobralo", "gradient"])
def test_a_short_run_changes_and_saves_the_weights(method, tmp_path):
    pytest.importorskip("deepxde")
    ex = DX.BY_NAME["Diffusion equation"]
    cfg = dataclasses.replace(DX.build_config(ex, REF), iterations=1000, optimizer2="none", loss_balancing=method,
                              balance_every=10 if method == "gradient" else 0, save_dir=str(tmp_path / "run"),
                              train_device="cpu", num_domain=200, num_boundary=40, num_initial=40, num_test=100,
                              time_adaptive=False, sweep_enabled=False, ic_pretrain=False)
    path = tmp_path / "run.py"
    path.write_text(CG.generate_script(cfg), encoding="utf-8")
    env = dict(os.environ, DDE_BACKEND="pytorch", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg", PYTHONPATH=ROOT,
               OMP_NUM_THREADS="2", PINNSTUDIO_SCRATCH=str(tmp_path / "scratch"))
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=900, env=env,
                       cwd=str(tmp_path))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(f"loss balancing ({method})")]
    assert lines, out[-3000:]
    ws = [float(v) for v in lines[-1].split("weights")[1].split(",")]
    assert len(ws) >= 2 and len(set(round(w, 6) for w in ws)) > 1, lines
    saved = tmp_path / "run" / "solution_results" / "loss_balance.txt"
    assert saved.is_file(), os.listdir(tmp_path / "run")
    rows = saved.read_text().splitlines()
    assert rows[0].startswith("# step") and rows[1].startswith("1000,")
