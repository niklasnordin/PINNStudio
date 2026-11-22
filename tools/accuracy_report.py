#!/usr/bin/env python3
"""Compare a night's example accuracy with the recorded baseline, as a Markdown table.

    python tools/accuracy_report.py RESULT.json [--baseline benchmarks/accuracy_baseline.json] [--out FILE]

RESULT.json and the baseline are written by tools/run_deepxde_examples.py --json (the same --cap). Each row is
an example's mean L2 relative error, now and in the baseline, with ▲ when it is more than 10 % worse and ▼ when
more than 10 % better. New, missing and failed examples are marked. The table goes to --out (the nightly job
passes $GITHUB_STEP_SUMMARY) or to stdout. It only reports: the exit code is always 0.

    python tools/accuracy_report.py RESULT.json --write-baseline    # make RESULT.json the baseline
"""
import argparse
import json
import os
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(_REPO, "benchmarks", "accuracy_baseline.json")
THRESHOLD = 0.10


def _load(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data, {e["name"]: e for e in data.get("examples", [])}


def _fmt(v):
    return "–" if v is None else f"{v:.3g}"


def report(result_path, baseline_path, threshold=THRESHOLD):
    now_meta, now = _load(result_path)
    if os.path.isfile(baseline_path):
        base_meta, base = _load(baseline_path)
    else:
        base_meta, base = {}, {}
    rows, worse, better = [], 0, 0
    for name in sorted(set(now) | set(base)):
        a, b = now.get(name), base.get(name)
        l2_now = a.get("l2") if a and a.get("ok") else None
        l2_base = b.get("l2") if b and b.get("ok") else None
        if a is None:
            mark, change = "missing", ""
        elif not a.get("ok"):
            mark, change = "failed", (a.get("error") or "")[:120].replace("|", "/")
        elif b is None:
            mark, change = "new", ""
        elif l2_now is None or l2_base is None or l2_base == 0:
            mark, change = "", ""
        else:
            r = l2_now / l2_base - 1.0
            change = f"{r:+.0%}"
            mark = "▲ worse" if r > threshold else "▼ better" if r < -threshold else ""
            worse += r > threshold
            better += r < -threshold
        secs = f"{a['seconds']:.0f}" if a and a.get("seconds") is not None else "–"
        rows.append(f"| {name} | {_fmt(l2_now)} | {_fmt(l2_base)} | {change} | {mark} | {secs} |")
    head = [f"## Example accuracy (Adam capped at {now_meta.get('cap') or 'full'} iterations)", "",
            f"{len(now)} examples run: {worse} more than {threshold:.0%} worse, {better} more than "
            f"{threshold:.0%} better than the baseline"
            + ("" if base else " (no baseline yet: tools/accuracy_report.py RESULT.json --write-baseline)") + ".",
            f"Now: PyTorch {now_meta.get('torch', '?')}, Python {now_meta.get('python', '?')}. Baseline: PyTorch "
            f"{base_meta.get('torch', '?')}, Python {base_meta.get('python', '?')}.", "",
            "| Example | L2 now | L2 baseline | Change | | Seconds |", "|---|---:|---:|---:|---|---:|"]
    return "\n".join(head + rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default="")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args(argv)
    if a.write_baseline:
        os.makedirs(os.path.dirname(a.baseline), exist_ok=True)
        shutil.copyfile(a.result, a.baseline)
        print(f"baseline written: {a.baseline}")
        return 0
    try:
        text = report(a.result, a.baseline, a.threshold)
    except (OSError, ValueError) as exc:
        text = f"## Example accuracy\n\nNo report: {exc}\n"
    if a.out:
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
