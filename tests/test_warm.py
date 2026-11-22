"""The Python kept ready for the next run (core/warm.py): a spare process imports PyTorch and DeepXDE, then
runs the one script it is sent, with that run's environment; a run whose before-import variables differ
starts a new Python; Stop (kill) ends it."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from pinnstudio.core import warm  # noqa: E402

SCRIPT = """
import os, sys
import deepxde, torch
print("RUN", os.environ.get("PINNSTUDIO_TEST_VAR"), os.getcwd() == os.environ.get("EXPECT_CWD"),
      sys.argv[0].endswith("job.py"), "deepxde" in sys.modules, torch.cuda.is_available(), flush=True)
"""


def _env(**kw):
    return warm.run_env(extra=dict(PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", **kw))


@pytest.fixture()
def pool():
    p = warm.WarmPool()
    p.enabled = True
    yield p
    p.close()


def _job(tmp_path):
    path = tmp_path / "job.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return str(path)


def test_spare_runs_the_script_with_the_runs_environment(pool, tmp_path):
    env = _env()
    pool.prestart(env)
    spare = pool._spare
    assert spare is not None
    assert spare.ready.wait(120) and spare.cuda is not None, "the spare never said it was ready"
    run = dict(env, PINNSTUDIO_TEST_VAR="hello", EXPECT_CWD=str(tmp_path))
    proc = pool.popen(_job(tmp_path), run, cwd=str(tmp_path))
    assert proc is spare.proc, "the ready spare was not used"
    out = proc.stdout.read()
    assert proc.wait(60) == 0, out
    assert "RUN hello True True True False" in out, out
    assert warm.READY not in out
    # a new spare is on its way for the next run
    for _ in range(50):
        if pool._spare is not None:
            break
        time.sleep(0.1)
    assert pool._spare is not None and pool._spare.proc is not proc


def test_a_different_before_import_variable_starts_a_new_python(pool, tmp_path):
    env = _env()
    pool.prestart(env)
    spare = pool._spare
    assert spare.ready.wait(120)
    other = dict(env, OMP_NUM_THREADS="1", PINNSTUDIO_TEST_VAR="cold", EXPECT_CWD=str(tmp_path))
    proc = pool.popen(_job(tmp_path), other, cwd=str(tmp_path))
    assert proc is not spare.proc
    out = proc.stdout.read()
    assert proc.wait(120) == 0, out
    assert "RUN cold True True True False" in out, out
    assert spare.proc.poll() is not None, "the spare that did not fit was left running"


def test_key_ignores_run_only_variables_and_the_gpu_choice_without_a_gpu():
    a = {"OMP_NUM_THREADS": "2", "PINNSTUDIO_SCRATCH": "/x", "CUDA_VISIBLE_DEVICES": "-1"}
    b = {"OMP_NUM_THREADS": "2", "PINNSTUDIO_SCRATCH": "/y"}
    assert warm._key(a, cuda=False) == warm._key(b, cuda=False)
    assert warm._key(a, cuda=True) != warm._key(b, cuda=True)


def test_stop_kills_a_run_in_the_spare(pool, tmp_path):
    env = _env()
    pool.prestart(env)
    assert pool._spare.ready.wait(120)
    path = tmp_path / "slow.py"
    path.write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    proc = pool.popen(str(path), env, cwd=str(tmp_path))
    assert proc.stdout.readline().strip() == "started"
    proc.kill()
    assert proc.wait(10) != 0


def test_disabled_pool_starts_python_as_before(tmp_path):
    p = warm.WarmPool()
    p.enabled = False
    proc = p.popen(_job(tmp_path), dict(_env(), PINNSTUDIO_TEST_VAR="off", EXPECT_CWD=str(tmp_path)),
                   cwd=str(tmp_path))
    out = proc.stdout.read()
    assert proc.wait(120) == 0, out
    assert "RUN off True True True False" in out
    assert p._spare is None


def test_cpu_runs_choose_the_cpu_before_import():
    from pinnstudio.core.config import PINNConfig
    assert warm.run_env(PINNConfig(train_device="cpu"))["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "CUDA_VISIBLE_DEVICES" not in warm.run_env(PINNConfig(train_device="auto")) or \
        os.environ.get("CUDA_VISIBLE_DEVICES") is not None
