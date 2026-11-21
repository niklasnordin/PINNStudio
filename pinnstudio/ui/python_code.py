"""The Python code page: the problem's Python module (pde() and conditions()), and its actions.

A view: it shows the state (`PythonCodeState`) and hands every action to the controller
(`PythonCodeController`) it is given. It builds neither (rule R8 of tests/test_mvc.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QTextCursor, QTextFormat
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTextEdit, QToolButton,
                               QMenu, QVBoxLayout, QWidget)

from pinnstudio.ui import theme as _theme

if TYPE_CHECKING:
    from pinnstudio.ui.controllers.python_code import PythonCodeController
    from pinnstudio.ui.state.python_code_state import PythonCodeState


class _Gutter(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, ev):
        self.editor.paint_gutter(ev)


class CodeEditor(QPlainTextEdit):
    """Python source: a monospace font, line numbers, Tab as four spaces, and one line marked (an error)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pythonCodeEditor")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabChangesFocus(False)
        f = QFont(_theme.mono_font_family())
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(f)
        self._highlighter = _theme.PythonHighlighter(self.document())
        self._gutter = _Gutter(self)
        self._marked = 0
        self.blockCountChanged.connect(lambda _n: self._update_margin())
        self.updateRequest.connect(self._update_gutter)
        self._update_margin()
        self.setAccessibleName("Python code")

    LINES = 26          # the height it asks for, in lines of code

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.fontMetrics().lineSpacing() * self.LINES + 12)

    def minimumSizeHint(self):
        return QSize(super().minimumSizeHint().width(), self.fontMetrics().lineSpacing() * 12 + 12)

    def gutter_width(self):
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 10 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margin(self):
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect, dy):
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, ev):
        p = QPainter(self._gutter)
        pal = self.palette()
        base = pal.base().color()
        p.fillRect(ev.rect(), base.lighter(125) if base.lightness() < 128 else base.darker(106))
        block = self.firstVisibleBlock()
        n = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        h = self.fontMetrics().height()
        muted = pal.text().color()
        muted.setAlpha(140)
        while block.isValid() and top <= ev.rect().bottom():
            if block.isVisible():
                p.setPen(QColor("#e03131") if n + 1 == self._marked else muted)
                p.drawText(0, top, self._gutter.width() - 5, h, Qt.AlignmentFlag.AlignRight, str(n + 1))
            block = block.next()
            top += round(self.blockBoundingRect(block).height()) if block.isValid() else h
            n += 1
        p.end()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Tab and not ev.modifiers():
            self.insertPlainText("    ")
            return
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not ev.modifiers():
            line = self.textCursor().block().text()
            indent = line[:len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(ev)
            self.insertPlainText(indent)
            return
        super().keyPressEvent(ev)

    def mark_line(self, line: int):
        """Mark line `line` (1-based; 0 clears) and show it."""
        self._marked = int(line or 0)
        sels = []
        if self._marked > 0:
            block = self.document().findBlockByNumber(self._marked - 1)
            if block.isValid():
                sel = QTextEdit.ExtraSelection()
                sel.format.setBackground(QColor(224, 49, 49, 48))
                sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                sel.cursor = QTextCursor(block)
                sels.append(sel)
                self.setTextCursor(QTextCursor(block))
                self.centerCursor()
        self.setExtraSelections(sels)
        self._gutter.update()


class PythonCodePage(QWidget):
    """The editor, the buttons above it, the names the code can use, and the last check's result."""

    TYPING_PAUSE_MS = 300

    def __init__(self, state: "PythonCodeState", controller: "PythonCodeController", parent=None):
        super().__init__(parent)
        self.setObjectName("pythonCodePage")
        self.state, self.ctl = state, controller
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.write_btn = QToolButton()
        self.write_btn.setText("Write from the page ▾")
        self.write_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.write_btn.setAccessibleName("Write the code from the pages")
        menu = QMenu(self.write_btn)
        menu.addAction("The equations, from the Equations page").triggered.connect(
            lambda _c=False: controller.write_equations())
        menu.addAction("The conditions, from the Boundary and Initial conditions rows").triggered.connect(
            lambda _c=False: controller.write_conditions())
        self.write_btn.setMenu(menu)
        self.check_btn = QPushButton("Check")
        self.check_btn.setToolTip("Run the Solve script once without training: the equations on a few points "
                                  "and the initial loss of every condition (a second process)")
        self.check_btn.clicked.connect(self._check)
        self.more_btn = QToolButton()
        self.more_btn.setText("File ▾")
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_btn.setAccessibleName("Python code file actions")
        fmenu = QMenu(self.more_btn)
        fmenu.addAction("Load .py…").triggered.connect(lambda _c=False: controller.load_file())
        fmenu.addAction("Save .py…").triggered.connect(lambda _c=False: controller.save_file())
        self._editor_folder = lambda: ""
        self._editor_name = lambda: "problem"
        fmenu.addAction("Open in your editor").triggered.connect(
            lambda _c=False: controller.open_in_editor(self._editor_folder(), self._editor_name()))
        self.more_btn.setMenu(fmenu)
        bar.addWidget(self.write_btn)
        bar.addWidget(self.more_btn)
        bar.addStretch(1)
        bar.addWidget(self.check_btn)
        lay.addLayout(bar)

        self.editor = CodeEditor()
        self.editor.setPlaceholderText("Choose Python on the Equations or Boundary conditions page, or "
                                       "Write from the page, to start")
        lay.addWidget(self.editor, 1)
        self.names = QLabel("")
        self.names.setObjectName("pageDesc")
        self.names.setWordWrap(True)
        self.names.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.names)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.result)

        self._typing = QTimer(self)
        self._typing.setSingleShot(True)
        self._typing.setInterval(self.TYPING_PAUSE_MS)
        self._typing.timeout.connect(self.flush)
        self.editor.textChanged.connect(self._typing.start)
        state.code_changed.connect(self._show_code)
        state.checked.connect(self._show_result)
        state.message.connect(self._show_message)
        self._show_code()

    def set_file_hints(self, folder_fn, name_fn):
        """Where Open in your editor writes the file: the problem's folder and name (the window knows them)."""
        self._editor_folder, self._editor_name = folder_fn, name_fn

    def flush(self):
        """The typed code reaches the state now (Solve, Save and Check call this first)."""
        self._typing.stop()
        if self.editor.toPlainText() != self.state.code:
            self.ctl.edit(self.editor.toPlainText())

    def refresh_names(self):
        self.names.setText("Names: " + self.ctl.names())

    def _check(self):
        self.flush()
        self.ctl.check()

    def _show_code(self):
        code = self.state.code
        if self.editor.toPlainText() != code:
            cur = self.editor.textCursor().position()
            self.editor.blockSignals(True)
            self.editor.setPlainText(code)
            self.editor.blockSignals(False)
            c = self.editor.textCursor()
            c.setPosition(min(cur, len(code)))
            self.editor.setTextCursor(c)
        self.editor.mark_line(0)

    def _show_result(self, res):
        if not res:
            self.result.setText("Checking…")
            self.check_btn.setEnabled(False)
            return
        self.check_btn.setEnabled(True)
        if res.get("ok"):
            r = res.get("residuals") or []
            conds = res.get("conditions") or []
            losses = res.get("losses") or []
            txt = (f"✓ Check passed: {len(r)} residual{'s' if len(r) != 1 else ''}"
                   f"{', '.join([''] + ['×'.join(str(v) for v in s) for s in r])}")
            if conds:
                txt += f"; conditions: {', '.join(conds)}"
            if losses:
                txt += f"; initial losses: {', '.join(f'{v:.3g}' for v in losses)}"
            self.result.setText(txt)
            self.editor.mark_line(0)
        else:
            line = int(res.get("line") or 0)
            self.result.setText(f"✗ {'Line %d: ' % line if line else ''}{res.get('error', '')}")
            self.editor.mark_line(line)

    def _show_message(self, text):
        if text and not text.startswith(("Check passed", "Check failed")):
            self.result.setText(text)
