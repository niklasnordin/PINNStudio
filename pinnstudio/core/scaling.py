"""Scales for the network's outputs, so that it trains on values of order one.

A PINN trains best when its inputs and outputs are of order one. The inputs are mapped to [-1, 1] from the
domain by the input transform's "domain" mode (codegen._domain_input_scaling). For the outputs,
suggest_output_scales picks the output transform's scale and shift (y = scale * network + shift) from what
the problem already says about each output's size:
  1. the reference solution's formula (Error analysis), when there is one;
  2. otherwise the initial conditions and the Dirichlet boundary values;
  3. otherwise 1 and 0 (unchanged).
The values are sampled on random points of the domain's box. The equations stay in the problem's own units:
automatic differentiation goes through the transforms, so nothing in the residuals needs rewriting.
"""
from __future__ import annotations

import json
import math

import numpy as np


def _sig(v: float, digits: int = 3) -> float:
    if v == 0 or not math.isfinite(v):
        return 0.0
    return float(f"{v:.{digits}g}")


def _columns(config):
    dim = getattr(config, "problem_dim", "1D")
    if dim == "ODE":
        return [("t", config.t_min, config.t_max)]
    cols = [("x", config.x_min, config.x_max)]
    if dim in ("2D", "3D"):
        cols.append(("y", config.y_min, config.y_max))
    if dim == "3D":
        cols.append(("z", config.z_min, config.z_max))
    return cols


def _evaluate(expr: str, config, X: np.ndarray, t: np.ndarray, consts: dict):
    """One value per point of an expression written as on the pages (sin, pi, x, y, z, t, the constants)."""
    from pinnstudio.core import codegen as CG
    dim = getattr(config, "problem_dim", "1D")
    src = CG._simplify_expr(expr, dim == "2D", dim == "3D")
    ns = {"np": np, "x": X, "t": t, "pi": np.pi, "e": np.e, **consts}
    if dim == "ODE":
        ns["x"] = t.reshape(-1, 1)
    with np.errstate(all="ignore"):
        v = eval(src, {"__builtins__": {}}, ns)  # noqa: S307 -- the problem's own expressions
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        v = np.full(len(X), float(v))
    v = v.reshape(len(X), -1)[:, 0]
    return v[np.isfinite(v)]


def suggest_output_scales(config, n: int = 2000, seed: int = 0) -> dict:
    """{"scale": [...], "shift": [...], "source": [...]}, one entry per output: the scale is half the range of
    the output's values, the shift their middle (a constant value gives scale = |value| and no shift)."""
    from pinnstudio.core import codegen as CG
    rng = np.random.default_rng(seed)
    cols = _columns(config)
    X = np.column_stack([rng.uniform(lo, hi, n) for _n, lo, hi in cols]) if cols else np.zeros((n, 1))
    t = rng.uniform(config.t_min, config.t_max, n)
    t0 = np.full(n, float(config.t_min))
    try:
        consts = {name: val for name, _e, val in CG._constant_values(config)}
    except ValueError:
        consts = {}
    n_out = max(1, int(getattr(config, "num_outputs", 1) or 1))
    samples = [[] for _ in range(n_out)]
    source = ["none"] * n_out

    ref = (getattr(config, "ea_ref_formula", "") or "").split("|") \
        if getattr(config, "ea_ref_mode", "") == "formula" else []
    for i in range(n_out):
        if i < len(ref) and ref[i].strip():
            try:
                v = _evaluate(ref[i], config, X, t, consts)
                if len(v):
                    samples[i].append(v)
                    source[i] = "the reference formula"
            except Exception:
                pass

    ics = (getattr(config, "ic_expressions", "") or "").split("|")
    active = [a.strip() == "True" for a in (getattr(config, "ic_active", "") or "").split(",")]
    try:
        rows = json.loads(getattr(config, "custom_bc_json", "") or "[]") or []
    except (ValueError, TypeError):
        rows = []
    for i in range(n_out):
        if source[i] != "none":
            continue
        got = []
        if not getattr(config, "steady_state", False) and i < len(ics) and ics[i].strip() and \
                (i >= len(active) or active[i]):
            try:
                got.append(_evaluate(ics[i], config, X, t0, consts))
            except Exception:
                pass
        for r in rows:
            if isinstance(r, dict) and r.get("type") == "dirichlet" and int(r.get("component", 0) or 0) == i \
                    and str(r.get("value", "")).strip():
                try:
                    got.append(_evaluate(str(r["value"]), config, X, t, consts))
                except Exception:
                    pass
        got = [g for g in got if len(g)]
        if got:
            samples[i] = got
            source[i] = "the initial and boundary values"

    scale, shift = [], []
    for i in range(n_out):
        if not samples[i]:
            scale.append(1.0)
            shift.append(0.0)
            continue
        v = np.concatenate(samples[i])
        lo, hi = float(v.min()), float(v.max())
        half, mid = (hi - lo) / 2.0, (hi + lo) / 2.0
        if half <= 1e-12 * max(1.0, abs(mid)):          # one value: its size, no shift
            scale.append(_sig(abs(mid)) or 1.0)
            shift.append(0.0)
        else:
            scale.append(_sig(half))
            shift.append(_sig(mid) if abs(mid) > 0.05 * half else 0.0)   # a near-zero middle is noise
    return {"scale": scale, "shift": shift, "source": source}
