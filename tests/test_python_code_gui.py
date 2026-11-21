"""The Python code page in the window: choosing Python on the Equations and Boundary conditions pages writes
the code from the expressions and rows, hides the expressions and disables the rows; the code round-trips
through save and open; Check reports a line-numbered error; code from a file asks to be trusted before
Solve; the controller's editing helpers."""
import dataclasses
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.core import pde_code as PC  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402
from pinnstudio.core.netdesign import trust as T  # noqa: E402
from pinnstudio.ui.controllers.prompter import Prompter  # noqa: E402
from pinnstudio.ui.controllers.python_code import PythonCodeController  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402
from pinnstudio.ui.state.python_code_state import PythonCodeState  # noqa: E402


class Answers(Prompter):
    def __init__(self):
        self.trust = False
        self.asked = 0
        self.yes = True
        self.open_path = ""
        self.save_path = ""
        self.opened = []

    def confirm_trust(self, items):
        self.asked += 1
        return self.trust

    def confirm(self, title, text):
        return self.yes

    def ask_open_path(self, *a):
        return self.open_path

    def ask_save_path(self, *a):
        return self.save_path

    def open_external(self, path):
        self.opened.append(path)
        return True


@pytest.fixture
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    for _ in range(3):
        app.processEvents()
    w._new_problem("1D")
    w.resize(1500, 950)
    w.show()
    app.processEvents()
    answers = Answers()
    w.python_code.prompter = answers
    yield w, answers
    w.close()
    w.deleteLater()
    app.processEvents()


def test_python_equations_and_conditions_from_the_pages(win):
    w, _a = win
    assert w.eq_mode_combo.currentData() == "expressions" and not w.python_state.code
    w._choose_pde_mode("python")
    app.processEvents()
    code = w.python_state.code
    assert "def pde(x, y, d):" in code and 'd("u", "x", "x")' in code
    assert all(box.isHidden() for box in w._pde_expr_boxes)
    w._choose_conditions_mode("python")
    app.processEvents()
    code = w.python_state.code
    assert "def conditions(c):" in code and "c.initial(" in code and "weight=100.0" in code
    assert not w.custom_bc_group.isEnabled() and not w.bc_group.isEnabled()
    cfg = w._build_config()
    assert cfg.pde_mode == "python" and cfg.conditions_mode == "python" and cfg.python_code == code
    assert not [e for e in cfg.validate() if "Python code" in e]
    w._choose_pde_mode("expressions")
    w._choose_conditions_mode("rows")
    app.processEvents()
    assert not any(box.isHidden() for box in w._pde_expr_boxes) and w.custom_bc_group.isEnabled()
    assert w.python_state.code == code                       # kept for later


def test_the_code_round_trips_through_save_and_open(win, tmp_path):
    w, _a = win
    w._choose_pde_mode("python")
    w.python_page.editor.setPlainText(w.python_state.code + "\n# my note\n")
    w.python_page.flush()
    cfg = w._build_config()
    assert "# my note" in cfg.python_code
    w._apply_config(dataclasses.replace(PINNConfig(), problem_dim="1D"))
    app.processEvents()
    assert w.python_state.pde_mode == "expressions" and not w.python_state.code
    w._apply_config(cfg)
    app.processEvents()
    assert w.python_state.pde_mode == "python" and "# my note" in w.python_state.code
    assert w.eq_mode_combo.currentData() == "python"
    assert "# my note" in w.python_page.editor.toPlainText()


def test_code_from_a_file_asks_before_solve(win):
    w, a = win
    code = PC.template(w._build_config()) + "\n# from someone else\n"
    cfg = dataclasses.replace(w._build_config(), pde_mode="python", python_code=code)
    w._apply_config(cfg)
    app.processEvents()
    assert w.python_state.from_outside
    a.trust = False
    assert not w.python_code.check_trust() and a.asked == 1
    a.trust = True
    assert w.python_code.check_trust() and a.asked == 2
    assert T.is_trusted(T.code_item(code)) and not w.python_state.from_outside
    w._apply_config(cfg)                    # the same code again: trusted, nothing asked
    assert w.python_code.check_trust() and a.asked == 2


def test_typed_code_is_the_users_own(win):
    w, a = win
    w._choose_pde_mode("python")
    w.python_page.editor.setPlainText(w.python_state.code + "\n# mine\n")
    w.python_page.flush()
    assert w.python_code.check_trust() and a.asked == 0


def test_check_reports_a_line_numbered_error(win):
    w, _a = win
    w._choose_pde_mode("python")
    bad = "def pde(x, y, d):\n    a = 1\n    return [d('u', 'q')]\n"
    w.python_page.editor.setPlainText(bad)
    w.python_page.flush()
    assert w.python_code.check()
    res = w.python_code.wait_for_check()
    assert not res["ok"] and res["line"] == 3 and "axis" in res["error"], res
    assert "Line 3" in w.python_page.result.text()
    w.python_page.editor.setPlainText(PC.from_expressions(w._build_config()))
    w.python_page.flush()
    assert w.python_code.check()
    res = w.python_code.wait_for_check()
    assert res["ok"] and res["residuals"] == [[64, 1]] and len(res["losses"]) == 2, res


def test_controller_files_and_editor(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    st = PythonCodeState()
    a = Answers()
    cfg = PINNConfig(custom_bc_json=json.dumps([{"type": "dirichlet", "location": "left or right", "value": "0"}]))
    ctl = PythonCodeController(st, a, lambda: dataclasses.replace(cfg, python_code=st.code,
                                                                   pde_mode=st.pde_mode,
                                                                   conditions_mode=st.conditions_mode))
    assert ctl.set_conditions_mode("python") and "def conditions(c):" in st.code
    a.save_path = str(tmp_path / "p.py")
    assert ctl.save_file() and open(a.save_path, encoding="utf-8").read() == st.code
    other = tmp_path / "other.py"
    other.write_text("def pde(x, y, d):\n    return [d.u]\n", encoding="utf-8")
    a.open_path = str(other)
    assert ctl.load_file() and st.from_outside and "return [d.u]" in st.code
    path = ctl.open_in_editor(str(tmp_path), "my problem")
    assert path.endswith("my_problem_python.py") and a.opened == [path]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("def pde(x, y, d):\n    return [2 * d.u]\n")
    ctl._file_changed(path)
    assert "2 * d.u" in st.code
    a.yes = False
    before = st.code
    assert not ctl.write_equations() and st.code == before     # replacing pde() was refused
    a.yes = True
    assert ctl.write_equations() and "2 * d.u" not in st.code


def test_rows_from_before_the_panel_are_refused():
    with pytest.raises(ValueError):
        PC.conditions_from_rows(PINNConfig(custom_bc_json=""))
    code = PC.conditions_from_rows(PINNConfig(custom_bc_json=json.dumps(
        [{"type": "dirichlet", "location": "left", "value": "0", "component": 0}])))
    assert 'c.dirichlet("u", where="left", value="0")' in code


def test_settings_undo_keeps_the_users_own_code_trusted(win):
    w, a = win
    w._choose_pde_mode("python")
    w.python_page.editor.setPlainText(w.python_state.code + "\n# mine\n")
    w.python_page.flush()
    cfg = w._build_config()
    w._apply_settings_snapshot(cfg)
    assert not w.python_state.from_outside
    assert w.python_code.check_trust() and a.asked == 0


def test_loading_another_problem_stops_watching_the_old_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    st = PythonCodeState()
    a = Answers()
    ctl = PythonCodeController(st, a, lambda: PINNConfig(python_code=st.code))
    ctl.load("python", "rows", "def pde(x, y, d):\n    return [d.u]\n")
    path = ctl.open_in_editor(str(tmp_path), "a")
    ctl.load("python", "rows", "def pde(x, y, d):\n    return [5 * d.u]\n")      # problem B
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("def pde(x, y, d):\n    return [7 * d.u]\n")
    ctl._file_changed(path)
    assert "5 * d.u" in st.code                                    # B's code untouched


def test_open_in_editor_asks_before_replacing_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    st = PythonCodeState()
    a = Answers()
    ctl = PythonCodeController(st, a, lambda: PINNConfig(python_code=st.code))
    ctl.edit("def pde(x, y, d):\n    return [d.u]\n")
    mine = tmp_path / "heat_python.py"
    mine.write_text("# my own file\n", encoding="utf-8")
    a.yes = False
    assert ctl.open_in_editor(str(tmp_path), "heat") == "" and mine.read_text(encoding="utf-8") == "# my own file\n"
    a.yes = True
    assert ctl.open_in_editor(str(tmp_path), "heat") == str(mine) and "return [d.u]" in mine.read_text(encoding="utf-8")


def test_undo_to_untrusted_code_asks_again(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    st = PythonCodeState()
    a = Answers()
    ctl = PythonCodeController(st, a, lambda: PINNConfig(python_code=st.code, custom_bc_json="[]",
                                                         pde_expressions="du_t - du_xx"))
    foreign = "import os\n\ndef pde(x, y, d):\n    return [d.u]\n"
    ctl.load("python", "rows", foreign)
    assert st.from_outside
    ctl.write_equations()                          # replaces pde(): Z'
    a.trust = True
    assert ctl.check_trust() and a.asked == 1      # Z' trusted
    ctl.follow("python", "rows", foreign)          # settings Undo back to Z
    assert st.from_outside
    a.trust = False
    assert not ctl.check_trust() and a.asked == 2


def test_a_shipped_example_is_not_asked_about(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    from pinnstudio.core import pinnstudio_examples as PX
    code = PX.BY_NAME["Burgers equation (Python code)"].settings["python_code"]
    st = PythonCodeState()
    a = Answers()
    ctl = PythonCodeController(st, a, lambda: PINNConfig(python_code=st.code))
    ctl.load("python", "python", code)
    assert not st.from_outside and ctl.check_trust() and a.asked == 0
