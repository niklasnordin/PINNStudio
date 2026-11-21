"""The Network tab on the whole screen: Full screen (or F11) leaves only the designer, Esc brings the window
back as it was, and Solve, another network type and closing leave full screen first."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.ui.main_window import MainWindow  # noqa: E402


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
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    yield w
    w._leave_full_screen()
    w.close()
    w.deleteLater()
    app.processEvents()


def _visible(w):
    return {"tree": w.main_splitter.widget(0).isVisible(), "settings": w.main_splitter.widget(1).isVisible(),
            "log": w.bottom_tabs.isVisible(), "tabbar": w.right_tabs.tabBar().isVisible(),
            "menu": w.menuBar().isVisible(), "panel": w.designer_panel.isVisible()}


def test_full_screen_shows_only_the_designer_and_esc_brings_the_window_back(win):
    w = win
    panel = w.designer_panel
    before = _visible(w)
    main, right = w.main_splitter.sizes(), w.right_splitter.sizes()
    assert before["panel"] and before["tree"] and before["log"]
    panel.full_btn.click()
    app.processEvents()
    assert w._full_screen is not None and panel.full_screen
    assert w.windowState() & Qt.WindowState.WindowFullScreen
    now = _visible(w)
    assert now["panel"] and not (now["tree"] or now["settings"] or now["log"] or now["tabbar"] or now["menu"])
    assert panel.full_btn.text().strip() == "Exit full screen"
    QTest.keyClick(panel.canvas, Qt.Key.Key_Escape)
    app.processEvents()
    assert w._full_screen is None and not panel.full_screen
    assert not (w.windowState() & Qt.WindowState.WindowFullScreen)
    assert _visible(w) == before
    assert w.main_splitter.sizes() == main and w.right_splitter.sizes() == right
    assert w.right_tabs.currentWidget() is panel
    assert panel.full_btn.text().strip() == "Full screen"


def test_f11_toggles_and_esc_does_nothing_in_the_window(win):
    w = win
    panel = w.designer_panel
    panel.canvas.setFocus()
    QTest.keyClick(panel.canvas, Qt.Key.Key_Escape)        # not in full screen: nothing happens
    app.processEvents()
    assert w._full_screen is None
    QTest.keyClick(panel.canvas, Qt.Key.Key_F11)
    app.processEvents()
    assert w._full_screen is not None
    QTest.keyClick(panel.canvas, Qt.Key.Key_F11)
    app.processEvents()
    assert w._full_screen is None


def test_another_network_type_and_closing_leave_full_screen(win):
    w = win
    before = _visible(w)
    w._set_network_full_screen(True)
    app.processEvents()
    w.network_type_combo.setCurrentText("FNN")
    app.processEvents()
    assert w._full_screen is None and _visible(w)["tree"] and _visible(w)["log"]
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    w._set_network_full_screen(True)
    saved = []
    w._save_layout = lambda: saved.append(_visible(w))
    w.close()
    app.processEvents()
    assert saved and saved[0]["tree"] and saved[0]["log"] == before["log"]
    assert w._full_screen is None


def test_solve_leaves_full_screen(win, monkeypatch):
    w = win
    w._set_network_full_screen(True)
    app.processEvents()
    monkeypatch.setattr(w, "_flush_designer", lambda: (_ for _ in ()).throw(RuntimeError("stop here")))
    with pytest.raises(RuntimeError):
        w._on_solve()
    assert w._full_screen is None
