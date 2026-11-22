"""tools/accuracy_report.py: the nightly accuracy table against the baseline (worse, better, new, missing and
failed examples), always exit code 0."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import accuracy_report as AR  # noqa: E402


def _write(path, examples, cap=1500):
    path.write_text(json.dumps({"cap": cap, "torch": "2.x", "python": "3.12", "examples": examples}))
    return str(path)


def _ex(name, l2, ok=True, error=""):
    return {"name": name, "ok": ok, "l2": l2, "seconds": 12.0, "error": error}


def test_the_table_marks_changes(tmp_path):
    base = _write(tmp_path / "base.json", [_ex("A", 0.010), _ex("B", 0.010), _ex("C", 0.010), _ex("Gone", 0.1)])
    now = _write(tmp_path / "now.json", [_ex("A", 0.0105), _ex("B", 0.020), _ex("C", 0.005),
                                         _ex("New", 0.3), _ex("Broken", None, ok=False, error="boom | x")])
    text = AR.report(now, base)
    lines = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("| ") and "Example" not in ln}
    assert "+5%" in lines["A"] and "worse" not in lines["A"] and "better" not in lines["A"]
    assert "+100%" in lines["B"] and "▲ worse" in lines["B"]
    assert "-50%" in lines["C"] and "▼ better" in lines["C"]
    assert "new" in lines["New"] and "missing" in lines["Gone"] and "failed" in lines["Broken"]
    assert "1 more than 10% worse, 1 more than 10% better" in text


def test_always_exit_zero_and_append_to_the_summary(tmp_path):
    now = _write(tmp_path / "now.json", [_ex("A", 0.01)])
    out = tmp_path / "summary.md"
    assert AR.main([now, "--baseline", str(tmp_path / "none.json"), "--out", str(out)]) == 0
    assert "no baseline yet" in out.read_text()
    (tmp_path / "bad.json").write_text("{not json")
    assert AR.main([str(tmp_path / "bad.json"), "--out", str(out)]) == 0
    assert "No report" in out.read_text()
    assert AR.main([now, "--baseline", str(tmp_path / "b" / "base.json"), "--write-baseline"]) == 0
    assert (tmp_path / "b" / "base.json").is_file()
