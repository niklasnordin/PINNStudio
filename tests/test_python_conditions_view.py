"""The Python conditions on the Sketch and Geometry views: with the conditions in Python, Check reports which
edges each condition applies to (the dry run evaluates each condition's place on the views' edge midpoints),
and the views colour the edges from it; after an edit the colours wait for the next Check."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.ui.main_window import MainWindow  # noqa: E402

CODE = '''def conditions(c):
    c.dirichlet("u", where="left", value="0", name="wall")
    c.neumann("u", where=lambda X, on: on and c.isclose(X[1], c.y_max), value=0, name="lid")
    c.initial("u", "0", name="start")
'''


@pytest.fixture
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w._new_problem("2D")
    w.resize(1500, 950)
    w.show()
    for _ in range(5):
        app.processEvents()
    w.python_code.prompter.confirm_trust = lambda items: True
    yield w
    w.close()
    w.deleteLater()
    app.processEvents()


def _edges_of(w):
    canvas = w.domain_view.canvas
    names = {b["index"]: b["label"] for b in w._sketch_bc_rows()}
    return {e["desc"]: [names.get(i, i) for i in lst] for e, lst in zip(canvas.edges(), canvas.edge_bcs())}


def test_check_colours_the_edges_of_the_python_conditions(win):
    w = win
    w._refresh_domain_view()
    assert w.domain_view.canvas.shapes, "no domain on the Geometry view"
    w._choose_conditions_mode("python")
    w.python_page.editor.setPlainText(w.python_state.code.split("def conditions")[0] + CODE)
    w.python_page.flush()
    assert w._sketch_bc_rows() == []                            # nothing until Check
    assert "press Check" in w._sketch_bc_note()
    assert w.python_code.check()
    res = w.python_code.wait_for_check()
    assert res.get("ok"), res
    assert res["conditions"] == ["wall", "lid", "start"] and res["edges"][2] is None, res
    on = _edges_of(w)
    walls = [d for d, names in on.items() if any(n.startswith("wall") for n in names)]
    lids = [d for d, names in on.items() if any(n.startswith("lid") for n in names)]
    assert len(walls) == 1 and walls[0].startswith("x = "), on
    assert len(lids) == 1 and lids[0].startswith("y = "), on
    assert sum(1 for names in on.values() if not names) == 2, on
    assert "from the last Check" in w._sketch_bc_note()
    # an edit: the colours wait for the next Check
    w.python_page.editor.setPlainText(w.python_state.code + "\n# changed\n")
    w.python_page.flush()
    assert w._sketch_bc_rows() == [] and all(not v for v in _edges_of(w).values())
    assert "press Check" in w._sketch_bc_note()
    # back to rows: the rows' colours
    w._choose_conditions_mode("rows")
    assert w._sketch_bc_note() == ""

