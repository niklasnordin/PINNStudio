"""Equations written in Python (Equations page › Equations as: Python).

The user writes one function, `pde(x, y, d)`, that returns the residuals, one per equation:

    def pde(x, y, d):
        u_t = d("u", "t")
        u_xx = d("u", "x", "x")
        return [u_t - 0.4 * u_xx]

`x` holds the inputs (one column per coordinate), `y` the outputs (one column per output), and `d` gives
derivatives of any order and mix by name, the outputs and coordinates as columns, and the problem's
constants and trainable variables. The Solve script and the exported script both embed the same helper
(`helper_code`), so the two train the same equations.

Domain code: no Qt. The GUI's Check runs `check_script(config)` in a second process, so the user's code
never runs inside the window.
"""
from __future__ import annotations

import ast
import json
import re

MODES = ("expressions", "python")
FILE_NAME = "<equations>"        # the name tracebacks show for the user's code

_DOC = '''    """The residuals: one per equation, each zero where its equation holds.

    x   the inputs, one column per coordinate     y   the outputs, one column per output
    d("u", "x")        du/dx; d("u", "x", "x") is d2u/dx2, any order and mix of axes
    d.diff(f, "x")     the derivative of any tensor f computed from x
    d.lap("u")         the Laplacian over the space axes; d.grad("u") the list of first derivatives
    d.u, d.x, d.t      an output or a coordinate as an (N, 1) column
    d.nu               a constant or a trainable variable, by name
    torch, np, dde     as usual
    """
'''


# ── the problem's names ───────────────────────────────────────────────
def axes(config) -> list:
    """The names of the input columns, in order: an ODE has t only; otherwise x[, y[, z]][, t]."""
    if getattr(config, "_ode", False) or getattr(config, "problem_dim", "1D") == "ODE":
        return ["t"]
    dim = getattr(config, "problem_dim", "1D")
    out = ["x", "y", "z"][:3 if dim == "3D" else 2 if dim == "2D" else 1]
    if not getattr(config, "steady_state", False):
        out.append("t")
    return out


def space_axes(config) -> list:
    return [a for a in axes(config) if a != "t"]


def outputs(config) -> list:
    n = max(1, int(getattr(config, "num_outputs", 1) or 1))
    names = [s.strip() for s in (getattr(config, "output_names", "") or "u").split(",") if s.strip()]
    while len(names) < n:
        names.append(f"u{len(names)}")
    return names[:n]


def inverse_names(config) -> list:
    if getattr(config, "problem_type", "Forward") != "Inverse":
        return []
    try:
        rows = json.loads(getattr(config, "inverse_variables_json", "") or "[]")
    except (ValueError, TypeError):
        rows = []
    out = []
    for i, v in enumerate(rows or []):
        name = str((v or {}).get("name") or "").strip()
        if name.isidentifier():
            out.append(name)
    if not out and getattr(config, "inverse_param_name", ""):
        out.append(str(config.inverse_param_name).strip())
    return out


def n_equations(config) -> int:
    return int(getattr(config, "num_equations", 0) or 0) or max(1, int(getattr(config, "num_outputs", 1) or 1))


# ── checks on the source ──────────────────────────────────────────────
_SIGNATURES = {"pde": (3, "pde(x, y, d), which returns the residuals"),
               "conditions": (1, "conditions(c), which adds the boundary and initial conditions")}


def check_source(code: str, needs=("pde",)) -> list:
    """What is wrong with the code, as messages with line numbers ([] when it can run). `needs`: the
    functions the problem takes from it ("pde", "conditions")."""
    if not (code or "").strip():
        return ["Python code: there is none (the Python code page writes a start from the expressions and rows)"]
    try:
        tree = ast.parse(code, filename=FILE_NAME)
    except SyntaxError as exc:
        return [f"Python code, line {exc.lineno}: {exc.msg}"]
    errors = []
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _SIGNATURES:
            found[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign,
                               ast.AnnAssign)):
            pass
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass                                     # a docstring or a note
        else:
            errors.append(f"Python code, line {node.lineno}: only imports, functions, classes and "
                          "assignments at the top level (they run as the script starts); put the rest in a function")
    for name in needs:
        n_args, what = _SIGNATURES[name]
        fn = found.get(name)
        if fn is None:
            errors.append(f"Python code: define the function {what}")
            continue
        if isinstance(fn, ast.AsyncFunctionDef):
            errors.append(f"Python code, line {fn.lineno}: {name}() cannot be async")
            continue
        a = fn.args
        if len(a.posonlyargs) + len(a.args) != n_args or a.vararg or a.kwonlyargs or a.kwarg:
            errors.append(f"Python code, line {fn.lineno}: {name}() takes {n_args} argument"
                          f"{'s' if n_args > 1 else ''}, {what.split(',')[0]}")
    return errors


# ── a start from the expressions ──────────────────────────────────────
def from_expressions(config) -> str:
    """The problem's expressions (Equations page) as Python code that gives the same residuals. Raises
    ValueError for what has no Python form here (integral(...) and frac(...))."""
    from pinnstudio.core import codegen as _cg
    n_eq = n_equations(config)
    raw = (getattr(config, "pde_expressions", "") or "").split("|")[:n_eq]
    while len(raw) < n_eq:
        raw.append("0")
    exprs = []
    for e in raw:
        if re.search(r"(?<![\w.])(integral|frac)\(", e or ""):
            raise ValueError("integral(...) and frac(...) have no Python form here: keep the expressions for "
                             "this problem")
        exprs.append(_cg._simplify_pde_expr(e or "0") or "0")
    outs = outputs(config)
    axs = axes(config)
    names = set()
    for e in exprs:
        try:
            names |= {n.id for n in ast.walk(ast.parse(e.strip() or "0", mode="eval")) if isinstance(n, ast.Name)}
        except SyntaxError as exc:
            raise ValueError(f"The expression {e!r} does not parse: {exc.msg}") from None
    taken = names | set(outs) | set(axs) | set(inverse_names(config)) | _constant_names(config)
    dn = next(c for c in ("d", "D", "der", "deriv", "_d") if c not in taken)   # the helper's name
    lines = []
    deriv = re.compile(r"d(" + "|".join(re.escape(o) for o in sorted(outs, key=len, reverse=True)) + r")_([xyzt]+)$")
    for n in sorted(names):
        m = deriv.match(n)
        if m and all(c in axs for c in m.group(2)):
            args = ", ".join(f'"{c}"' for c in m.group(2))
            lines.append(f'    {n} = {dn}("{m.group(1)}", {args})')
    for o in outs:
        if o in names:
            lines.append(f"    {o} = {dn}.{o}")
    for a in axs:
        if a in names:
            note = "   # the coordinate (the outputs are in d)" if a == "y" else ""
            lines.append(f"    {a} = {dn}.{a}{note}")
    for v in inverse_names(config):
        if v in names:
            lines.append(f"    {v} = {dn}.{v}            # a trainable variable")
    body = "\n".join(lines) + ("\n" if lines else "")
    if n_eq == 1:
        ret = f"    return [{exprs[0].strip()}]\n"
    else:
        ret = "    return [\n" + "".join(f"        {e.strip()},\n" for e in exprs) + "    ]\n"
    doc = _DOC if dn == "d" else _DOC.replace("d(", dn + "(").replace("d.", dn + ".") + (
        f"    # (the helper is called {dn} here: the problem has a name d)\n")
    return f"def pde(x, y, {dn}):\n" + doc + body + ret


def _constant_names(config) -> set:
    try:
        rows = json.loads(getattr(config, "constants", "") or "[]")
    except (ValueError, TypeError):
        return set()
    return {str(r.get("name", "")).strip() for r in rows if isinstance(r, dict)}


def template(config) -> str:
    """A start for a new problem's code: the expressions as Python, or one zero residual per equation."""
    try:
        return from_expressions(config)
    except ValueError:
        o = outputs(config)[0]
        res = ",\n".join(f"        0 * d.{o}" for _ in range(n_equations(config)))
        return "def pde(x, y, d):\n" + _DOC + f"    return [\n{res},\n    ]\n"


# ── editing the module ────────────────────────────────────────────────
def has_function(code: str, name: str) -> bool:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return re.search(rf"^def {re.escape(name)}\(", code or "", re.M) is not None
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body)


def replace_function(code: str, name: str, new_src: str) -> str:
    """`code` with its top-level function `name` replaced by `new_src` (added at the end when there is none,
    or when the code does not parse and the function cannot be found)."""
    code = code or ""
    new_src = new_src.rstrip("\n") + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                lines = code.splitlines(keepends=True)
                start = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
                end = n.end_lineno
                return "".join(lines[:start]) + new_src + "".join(lines[end:])
    if not code.strip():
        return new_src
    return code.rstrip("\n") + "\n\n\n" + new_src


def available_names(config) -> str:
    """One line for the Python code page: the names the code can use."""
    parts = [f"outputs {', '.join(outputs(config))}", f"axes {', '.join(axes(config))}"]
    consts = sorted(_constant_names(config) - {""})
    if consts:
        parts.append(f"constants {', '.join(consts)}")
    inv = inverse_names(config)
    if inv:
        parts.append(f"trainable {', '.join(inv)}")
    sides = ["left", "right"] if getattr(config, "problem_dim", "1D") != "ODE" else ["start", "end"]
    if getattr(config, "problem_dim", "") in ("2D", "3D"):
        sides += ["bottom", "top"]
    if getattr(config, "problem_dim", "") == "3D":
        sides += ["front", "back"]
    parts.append(f"sides {', '.join(sides)}")
    return " · ".join(parts)


# ── the conditions from the rows ──────────────────────────────────────
_ROW_CALLS = ("dirichlet", "neumann", "robin")


def _q(text) -> str:
    """A string literal for generated code, in double quotes."""
    return json.dumps(str(text), ensure_ascii=False)


def _num(v) -> str:
    return repr(float(v)) if float(v) != int(float(v)) else f"{float(v):.1f}"


def conditions_from_rows(config) -> str:
    """The problem's Boundary Conditions rows and initial conditions as conditions(c): one call per row
    and per active initial condition, in the loss order, each with the weight the Loss weights page gives
    it. Raises ValueError for a problem from before the Boundary Conditions panel (no rows)."""
    import dataclasses
    from pinnstudio.core import codegen as _cg
    raw = (getattr(config, "custom_bc_json", "") or "").strip()
    if not raw:
        raise ValueError("This problem's boundary conditions are in the old per-side fields: open its "
                         "Boundary conditions page once (it writes them as rows), then write them as Python")
    try:
        rows = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"The Boundary Conditions rows cannot be read: {exc}") from None
    outs = outputs(config)
    n_out = len(outs)
    ode = getattr(config, "problem_dim", "") == "ODE"
    steady = bool(getattr(config, "steady_state", False))
    n_eq = n_equations(config)
    try:
        w = [float(v) for v in (getattr(config, "loss_weights_multi", "") or "").split(",") if v.strip()]
    except ValueError:
        w = []
    wi = n_eq

    def weight():
        nonlocal wi
        v = w[wi] if wi < len(w) else 1.0
        wi += 1
        return "" if v == 1.0 else f", weight={_num(v)}"

    def out_name(r):
        i = int(r.get("component", 0) or 0)
        return _q(outs[i]) if 0 <= i < n_out else str(i)

    lines = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = r.get("type", "dirichlet")
        loc, val = r.get("location", "") or "", r.get("value", "0") or "0"
        where = "" if loc.strip() in ("",) else f", where={_q(loc)}"
        if t in _ROW_CALLS:
            lines.append(f"    c.{t}({out_name(r)}{where}, value={_q(val)}{weight()})")
        elif t == "periodic":
            order = int(r.get("deriv_order", 0) or 0)
            lines.append(f"    c.periodic({out_name(r)}, axis={_q(r.get('axis', 'x'))}{where}"
                         + (f", derivative_order={order}" if order else "") + f"{weight()})")
        elif t == "operator":
            at = ', at="initial"' if r.get("apply_at") == "initial" else ""
            lines.append(f"    c.operator({_q(val)}{where}{at}{weight()})")
        else:
            lines.append(f"    c.row({r!r}{weight()})")
    ic_lines = []
    if not steady:
        exprs = (getattr(config, "ic_expressions", "") or "").split("|")
        active = [a.strip() == "True" for a in (getattr(config, "ic_active", "") or "").split(",")]
        from_file = bool(getattr(config, "forward_ic_from_file", False)) and not ode
        rows_cfg = dataclasses.replace(config, conditions_mode="rows", pde_mode="expressions", constants="")
        conv = []
        if not ode:
            eff = _cg._effective_config(rows_cfg)
            is_2d, is_3d = eff.problem_dim == "2D", eff.problem_dim == "3D"
            conv = [_cg._simplify_expr(e, is_2d, is_3d) for e in (eff.ic_expressions or "").split("|")]
        for i in range(n_out):
            if from_file and i == 0:
                ic_lines.append(f"    c.initial_points({_q(getattr(config, 'forward_ic_file', ''))}{weight()})")
            elif i < len(active) and active[i] and i < len(exprs) and exprs[i].strip():
                if ode:
                    ic_lines.append(f"    c.initial({_q(outs[i])}, {_q(exprs[i].strip())}{weight()})")
                else:
                    body = (conv[i] if i < len(conv) else exprs[i]).strip() or "0"
                    ic_lines.append(f"    c.initial({_q(outs[i])}, lambda x: {body}{weight()})")
    doc = (
        '    """The boundary and initial conditions: each call adds one loss term, in order (weight= sets its weight).\n\n'
        '    where   a text as in the Boundary Conditions rows ("left", "x <= x_min", "on_boundary and y > 0.5"),\n'
        '            or a function (X, on_boundary) -> bool of one point X\n'
        '    value   a text as in the rows ("sin(pi*y)"), a number, or a function X -> (N, 1) of the points X\n'
        '    c.dirichlet, c.neumann, c.robin, c.periodic, c.operator, c.points, c.initial, c.initial_points\n'
        '    c.row({...})  a panel row as a dict;  c.add(condition)  any DeepXDE condition (c.geom is the domain)\n'
        '    c.x_min, c.left(X), c.isclose(a, b), c.d(x, y) (in an operator), np, torch, dde\n'
        '    """\n')
    body = "\n".join(lines + ic_lines) or "    pass"
    return "def conditions(c):\n" + doc + body + "\n"


# ── the code the scripts embed ────────────────────────────────────────
HELPER = '''
class _Deriv:
    """d in pde(x, y, d): derivatives by name, the outputs and coordinates as columns, and the problem's
    constants and trainable variables (PINNStudio, Equations page › Python)."""

    def __init__(self, x, y, names):
        self._x, self._y, self._names = x, y, names
        self.dde, self.np, self.torch = dde, np, torch

    def _axis(self, a):
        if a not in _PDE_AXES:
            raise NameError(f"d(...): no axis {a!r}; this problem's axes are {', '.join(_PDE_AXES)}")
        return _PDE_AXES[a]

    def __call__(self, name, *axes):
        if name not in _PDE_OUTPUTS:
            raise NameError(f"d({name!r}, ...): no output {name!r}; the outputs are {', '.join(_PDE_OUTPUTS)}")
        i = _PDE_OUTPUTS.index(name)
        if not axes:
            return self._y[:, i:i + 1]
        f = dde.grad.jacobian(self._y, self._x, i=i, j=self._axis(axes[0]))
        return self.diff(f, *axes[1:]) if len(axes) > 1 else f

    def diff(self, f, *axes):
        """The derivative of f, a column computed from x, along the axes in turn."""
        for a in axes:
            f = dde.grad.jacobian(f, self._x, i=0, j=self._axis(a))
        return f

    def grad(self, name):
        if not _PDE_SPACE:
            raise ValueError("d.grad: this problem has no space axes (an ODE has t only); use d(name, \\"t\\")")
        return [self(name, a) for a in _PDE_SPACE]

    def lap(self, name):
        if not _PDE_SPACE:
            raise ValueError("d.lap: this problem has no space axes (an ODE has t only); use d(name, \\"t\\", \\"t\\")")
        terms = [self(name, a, a) for a in _PDE_SPACE]
        out = terms[0]
        for t_ in terms[1:]:
            out = out + t_
        return out

    def __getattr__(self, n):
        if n.startswith("_"):
            raise AttributeError(n)
        if n in _PDE_OUTPUTS:
            i = _PDE_OUTPUTS.index(n)
            return self._y[:, i:i + 1]
        if n in _PDE_AXES:
            j = _PDE_AXES[n]
            return self._x[:, j:j + 1]
        if n in self._names:
            return self._names[n]
        known = list(_PDE_OUTPUTS) + list(_PDE_AXES) + list(self._names)
        raise AttributeError(f"d.{n}: no output, coordinate, constant or trainable variable {n!r} "
                             f"(there are {', '.join(known)})")


def _pde_residuals(res, n_eq, n_pts):
    res = list(res) if isinstance(res, (list, tuple)) else [res]
    if len(res) != n_eq:
        raise ValueError(f"pde() returned {len(res)} residual(s), but the problem has {n_eq} equation(s) "
                         "(Equations page › Equations)")
    for k, r in enumerate(res):
        if not torch.is_tensor(r) or r.ndim != 2 or r.shape[0] != n_pts or r.shape[1] != 1:
            got = tuple(r.shape) if torch.is_tensor(r) else type(r).__name__
            raise ValueError(f"pde(): residual {k + 1} is {got}, not an ({n_pts}, 1) tensor; build it from "
                             "the columns d gives (d.u, d(\\"u\\", \\"x\\"), ...)")
    return res[0] if n_eq == 1 else res
'''


def _literals(config) -> str:
    axs = axes(config)
    return (f"_PDE_AXES = {dict((a, j) for j, a in enumerate(axs))!r}\n"
            f"_PDE_SPACE = {[a for a in axs if a != 't']!r}\n"
            f"_PDE_OUTPUTS = {outputs(config)!r}\n")


COND_HELPER = """
def _bc_from_row(_be, _bi, _gt):
    # one Boundary Conditions panel row as a DeepXDE condition (the rows' own code, for c.row)
    _btype = _be.get('type', 'dirichlet')
    _bcomp = int(_be.get('component', 0) or 0)
    _bloc, _bval = _be.get('location', ''), _be.get('value', '0')
    _blabel = f"condition {_bi + 1} ({_btype})"
    _bcomponents_txt = (_be.get('components', '') or '').strip()
    _bcomponents = [int(_c.strip()) for _c in _bcomponents_txt.split(',') if _c.strip()] if _bcomponents_txt else None
    _bbatch = int(_be.get('batch_size', 0) or 0)
    _bkw = {"batch_size": _bbatch, "shuffle": bool(_be.get('shuffle', True))} if _bbatch > 0 else {}
    if _btype == 'neumann':
        return dde.icbc.NeumannBC(_gt, _bc_val_fn(_bval), _bc_loc_fn(_bloc), component=_bcomp)
    if _btype == 'robin':
        return dde.icbc.RobinBC(_gt, _bc_robin_fn(_bval, _bcomp), _bc_loc_fn(_bloc), component=_bcomp)
    if _btype == 'periodic':
        return dde.icbc.PeriodicBC(_gt, _axis_idx_map.get(_be.get('axis', 'x'), 0), _bc_loc_fn(_bloc),
                                   derivative_order=min(max(int(_be.get('deriv_order', 0) or 0), 0), 1),
                                   component=_bcomp)
    if _btype == 'pointset':
        _pts, _pvals = _bc_points(_be, _blabel, len(_bcomponents) if _bcomponents else 1, _gt)
        return dde.icbc.PointSetBC(_pts, _pvals, component=(_bcomponents if _bcomponents else _bcomp), **_bkw)
    if _btype == 'pointset_operator':
        _pts, _pvals = _bc_points(_be, _blabel, 1, _gt)
        return dde.icbc.PointSetOperatorBC(_pts, _pvals, _bc_operator_fn(_bval), **_bkw)
    if _btype == 'operator':
        _lf = _bc_init_loc_fn(_bloc) if _be.get('apply_at', 'boundary') == 'initial' else _bc_loc_fn(_bloc)
        return dde.icbc.OperatorBC(_gt, _bc_operator_fn(_bval), _lf)
    if _btype == 'interface2d':
        return dde.icbc.Interface2DBC(_Interface2DGeomAdapter(_gt, _is_steady), _bc_val_fn(_bval), _bc_loc_fn(_bloc),
                                      _bc_loc_fn(_be.get('location2', '')), direction=_be.get('direction', 'normal'))
    if _btype == 'dirichlet':
        return dde.icbc.DirichletBC(_gt, _bc_val_fn(_bval), _bc_loc_fn(_bloc), component=_bcomp)
    raise ValueError(f"{_blabel}: no condition type {_btype!r} (dirichlet, neumann, robin, periodic, pointset, "
                     "pointset_operator, operator, interface2d)")


def _cond_text(_s):
    # a text as the rows' texts are read: the constants by value; an ODE (solved in x = t) names t, start, end
    import re as _re_c
    _s = str(_s)
    for _n in sorted(_PDE_CONSTANTS, key=len, reverse=True):
        _s = _re_c.sub(r"(?<![\\w.])" + _re_c.escape(_n) + r"(?![\\w])", "(" + repr(_PDE_CONSTANTS[_n]) + ")", _s)
    if not _PDE_ODE:
        return _s
    _s = _re_c.sub(r"(?<![\\w.])(d[A-Za-z]\\w*?)_(t+)(?![\\w])", lambda m: m.group(1) + "_" + "x" * len(m.group(2)), _s)
    _s = _re_c.sub(r"(?<![\\w.])t_(min|max)(?![\\w])", r"x_\\1", _s)
    _s = _re_c.sub(r"(?<![\\w.])t(?![\\w])", "x", _s)
    _s = _re_c.sub(r"(?<![\\w.])start(?![\\w])", "left", _s)
    return _re_c.sub(r"(?<![\\w.])end(?![\\w])", "right", _s)


def _cond_column(_f):
    # a value function's result as an (N, 1) column (a number, a vector or a column)
    def _g(*_a):
        _X = _a[0]
        _r = _f(*_a)
        if torch.is_tensor(_r):
            return _r if _r.dim() == 2 else _r.reshape(-1, 1)
        _r = np.asarray(_r, dtype=float)
        return np.full((len(_X), 1), float(_r)) if _r.ndim == 0 else _r.reshape(-1, 1)
    return _g


class _Cond:
    # c in conditions(c): every call adds one loss term, in order (boundary conditions, then initial ones)

    def __init__(self, gt):
        self.geom = gt
        self.items = []              # [kind ("bc" or "ic"), condition, weight, name, output]
        self.np, self.torch, self.dde = np, torch, dde
        for _k, _v in _BC_DOMAIN.items():
            setattr(self, _k, _v)
        if _PDE_ODE and "x_min" in _BC_DOMAIN:
            self.t_min, self.t_max = _BC_DOMAIN["x_min"], _BC_DOMAIN["x_max"]

    def __getattr__(self, n):
        if n.startswith("_"):
            raise AttributeError(n)
        if n in _PDE_CONSTANTS:
            return _PDE_CONSTANTS[n]
        for _sn, _c, _v in _BC_SIDES:
            if _sn == n or (_PDE_ODE and {"start": "left", "end": "right"}.get(n) == _sn):
                return lambda X, _c=_c, _v=_v: bool(np.isclose(X[_c], _v))
        raise AttributeError(f"c.{n}: no such name (the calls are dirichlet, neumann, robin, periodic, operator, "
                             "points, initial, initial_points, row and add)")

    @staticmethod
    def isclose(a, b):
        return bool(np.isclose(a, b))

    def d(self, x, y):
        # the same helper pde(x, y, d) gets, for an operator condition's own derivatives
        return _Deriv(x, y, _pde_names())

    def _out(self, out):
        if isinstance(out, int):
            return out
        if out not in _PDE_OUTPUTS:
            raise NameError(f"no output {out!r}; the outputs are {', '.join(_PDE_OUTPUTS)}")
        return _PDE_OUTPUTS.index(out)

    def _where(self, where):
        if callable(where):
            return where
        return _bc_loc_fn(_cond_text("" if where is None else where))

    def _value(self, value):
        if callable(value):
            return _cond_column(value)
        if isinstance(value, (int, float)):
            _v = float(value)
            return lambda X: np.full((len(X), 1), _v)
        return _bc_val_fn(_cond_text(value))

    def _put(self, kind, cond, weight, name, out=None):
        self.items.append([kind, cond, float(weight), name or f"{kind} {len(self.items) + 1}", out])
        return cond

    def dirichlet(self, out, where="", value=0, weight=1.0, name=None):
        return self._put("bc", dde.icbc.DirichletBC(self.geom, self._value(value), self._where(where),
                                                    component=self._out(out)), weight, name or f"Dirichlet {out}")

    def neumann(self, out, where="", value=0, weight=1.0, name=None):
        return self._put("bc", dde.icbc.NeumannBC(self.geom, self._value(value), self._where(where),
                                                  component=self._out(out)), weight, name or f"Neumann {out}")

    def robin(self, out, where="", value=0, weight=1.0, name=None):
        _i = self._out(out)
        _f = _cond_column(value) if callable(value) else _bc_robin_fn(_cond_text(value), _i)
        return self._put("bc", dde.icbc.RobinBC(self.geom, _f, self._where(where), component=_i), weight,
                         name or f"Robin {out}")

    def periodic(self, out, axis="x", where="", derivative_order=0, weight=1.0, name=None):
        return self._put("bc", dde.icbc.PeriodicBC(self.geom, _axis_idx_map.get(axis, 0) if isinstance(axis, str) else int(axis),
                                                   self._where(where), derivative_order=int(derivative_order),
                                                   component=self._out(out)), weight, name or f"periodic {out}")

    def operator(self, fn, where="", at="boundary", weight=1.0, name=None):
        _f = fn if callable(fn) else _bc_operator_fn(_cond_text(fn))
        if at == "initial" and _PDE_ODE:
            where = where if callable(where) else ("start" if not str(where).strip() else f"start and ({where})")
            at = "boundary"
        if callable(where):
            _w = where
        else:
            _w = (_bc_init_loc_fn if at == "initial" else _bc_loc_fn)(_cond_text(where))
        return self._put("bc", dde.icbc.OperatorBC(self.geom, _f, _w), weight, name or "operator")

    def points(self, out, points=None, values=None, file=None, batch_size=0, shuffle=True, weight=1.0, name=None):
        _comp = [self._out(o) for o in out] if isinstance(out, (list, tuple)) else self._out(out)
        _n = len(_comp) if isinstance(_comp, list) else 1
        if file is not None:
            points, values = _bc_load_points_file(file, name or "points", _n)
        _kw = {"batch_size": int(batch_size), "shuffle": bool(shuffle)} if batch_size else {}
        return self._put("bc", dde.icbc.PointSetBC(np.asarray(points, dtype=float), np.asarray(values, dtype=float).reshape(len(points), -1),
                                                   component=_comp, **_kw), weight, name or f"points {out}")

    def initial(self, out, value=0, weight=1.0, name=None):
        _i = self._out(out)
        if _PDE_ODE:            # an ODE is solved in x = t: its start value is a condition at the start
            if isinstance(value, str):
                import re as _re_c
                value = _re_c.sub(r"x\\[:,\\s*0(:1)?\\]", "x", _cond_text(value))
            return self._put("ic", dde.icbc.DirichletBC(self.geom, self._value(value), _bc_loc_fn("left"),
                                                        component=_i), weight, name or f"initial {out}", _i)
        if _is_steady:
            raise ValueError("c.initial: a steady problem has no initial condition")
        if callable(value):
            _f0 = value
        else:                   # a number, or a text read as the rows' values are (sin, pi, x, y, constants)
            _f0 = self._value(value)

        def _ic_fn(x):
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return np.reshape(_f0(x), (-1, 1))
        return self._put("ic", dde.icbc.IC(self.geom, _ic_fn, lambda x, on_initial: on_initial, component=_i),
                         weight, name or f"initial {out}", _i)

    def initial_points(self, file, weight=1.0, name=None):
        _xyt, _vals = _load_ic_from_file(file)
        return self._put("ic", dde.icbc.PointSetBC(_xyt, _vals, component=0), weight, name or "initial (file)", 0)

    def row(self, d, weight=1.0, name=None):
        # a Boundary Conditions panel row, as a dict (the keys the problem file stores)
        d = dict(d)
        for _k in ("location", "value", "location2", "values_formula", "points_list"):
            if isinstance(d.get(_k), str):
                d[_k] = _cond_text(d[_k])
        if _PDE_ODE and d.get("type") == "operator" and d.get("apply_at") == "initial":
            _loc = (d.get("location") or "").strip()
            d["apply_at"], d["location"] = "boundary", "left" if _loc in ("", "True") else f"left and ({_loc})"
        return self._put("bc", _bc_from_row(d, len(self.items), self.geom), weight,
                         name or f"{d.get('type', 'dirichlet')} row")

    def add(self, cond, kind="bc", weight=1.0, name=None, output=None):
        if kind not in ("bc", "ic"):
            raise ValueError("c.add(..., kind=): 'bc' or 'ic'")
        return self._put(kind, cond, weight, name or type(cond).__name__, output)

    def objects(self, kind=None):
        _bc = [it[1] for it in self.items if it[0] == "bc"]
        _ic = [it[1] for it in self.items if it[0] == "ic"]
        return _bc if kind == "bc" else _ic if kind == "ic" else _bc + _ic

    def weights(self, kind=None):
        _bc = [it[2] for it in self.items if it[0] == "bc"]
        _ic = [it[2] for it in self.items if it[0] == "ic"]
        return _bc if kind == "bc" else _ic if kind == "ic" else _bc + _ic

    def names(self):
        return [it[3] for it in self.items if it[0] == "bc"] + [it[3] for it in self.items if it[0] == "ic"]


def _run_conditions(gt):
    _c = _Cond(gt)
    _PDE_NS["conditions"](_c)
    return _c
"""


def solve_code(config, constants: dict, equations: bool = True, conditions: bool = False) -> str:
    """The Solve script's Python module: the user's code compiled as <equations> (tracebacks show its own
    line numbers and text), the d helper, and with `equations` `pde` replaced by a function that calls the
    code's pde(); with `conditions` the c helper and _run_conditions(geometry). `config` is the generators'
    effective config; `constants` maps the problem's constant names to their values."""
    code = getattr(config, "python_code", "") or ""
    n_eq = n_equations(config)
    ode = bool(getattr(config, "_ode", False) or getattr(config, "problem_dim", "") == "ODE")
    out = f'''
# ── The problem in Python (the Python code page) ─────────────
{_literals(config)}_PDE_CONSTANTS = {dict(constants)!r}
_PDE_ODE = {ode!r}
_PDE_SOURCE = {code!r}
import linecache as _pde_linecache
_pde_linecache.cache[{FILE_NAME!r}] = (len(_PDE_SOURCE), None, _PDE_SOURCE.splitlines(True), {FILE_NAME!r})
_PDE_NS = {{"dde": dde, "np": np, "torch": torch, "__name__": "pinnstudio_equations", **_PDE_CONSTANTS}}
exec(compile(_PDE_SOURCE, {FILE_NAME!r}, "exec"), _PDE_NS)
{HELPER}

def _pde_names():
    _names = dict(_PDE_CONSTANTS)
    if _problem_type == "Inverse":
        _names.update(zip(_inv_var_names, _inv_vars))
    return _names
'''
    if equations:
        out += f'''
_pde_user = _PDE_NS["pde"]


def _pde_python(x, y):
    return _pde_residuals(_pde_user(x, y, _Deriv(x, y, _pde_names())), {n_eq}, x.shape[0])
pde = _pde_python
'''
    if conditions:
        out += COND_HELPER
    return out


def export_code(config, constants: dict) -> str:
    """The exported script's equations: the user's module as written (a readable string), run in a namespace of
    its own as Solve runs it, so a name in it (data, net, ...) and the script's own names never meet; then
    `pde(x, y)` for DeepXDE."""
    code = (getattr(config, "python_code", "") or "").rstrip() + "\n"
    n_eq = n_equations(config)
    inv = inverse_names(config)
    if "'''" in code:
        literal = repr(code)                       # (a triple-quoted string inside: one line, still exact)
    elif "\\" in code:
        literal = "r'''\n" + code + "'''"         # raw, so backslashes stay as written
    else:
        literal = "'''\\\n" + code + "'''"
    consts = "".join(f"{k} = {v!r}\n" for k, v in constants.items())
    inv_line = (f"    _names.update({{_n: globals()[_n] for _n in {inv!r}}})   # the trainable variables\n"
                if inv else "")
    return f'''# ---- The equations, in Python (PINNStudio's Python code page) --------
{consts}{_literals(config)}_PDE_CONSTANTS = {{{", ".join(f"{k!r}: {k}" for k in constants)}}}
{HELPER.strip()}


_PDE_SOURCE = {literal}
_PDE_NS = {{"dde": dde, "np": np, "torch": torch, "__name__": "pinnstudio_equations", **_PDE_CONSTANTS}}
exec(compile(_PDE_SOURCE, {FILE_NAME!r}, "exec"), _PDE_NS)   # the module, in a namespace of its own
_pde_user = _PDE_NS["pde"]


def pde(x, y):
    _names = dict(_PDE_CONSTANTS)
{inv_line}    return _pde_residuals(_pde_user(x, y, _Deriv(x, y, _names)), {n_eq}, x.shape[0])
'''


DRY_RUN_MARK = "PINNSTUDIO_DRY_RUN "


def dry_run_config(config):
    """The config Check runs as a dry run of the Solve script: one plain run (no time windows, no sweep, no
    pre-training), on the CPU, saving nothing, with at most a few thousand points (building them is most of a
    check's time after the imports, and the shapes and errors do not depend on how many there are)."""
    import dataclasses

    def cap(name, n):
        v = getattr(config, name, None)
        return min(int(v), n) if isinstance(v, (int, float)) and v else v
    return dataclasses.replace(config, time_adaptive=False, sweep_enabled=False, ic_pretrain=False,
                               train_device="cpu", iterations=1, optimizer2="none",
                               num_domain=cap("num_domain", CHECK_POINTS[0]),
                               num_boundary=cap("num_boundary", CHECK_POINTS[1]),
                               num_initial=cap("num_initial", CHECK_POINTS[1]),
                               num_test=cap("num_test", CHECK_POINTS[1]))


CHECK_POINTS = (2000, 500)      # Check: at most this many domain points, and boundary/initial/test points


def parse_dry_run(out: str) -> dict:
    """The dry run's result: {"ok": True, "residuals", "conditions", "losses"} or {"ok": False, "error",
    "line"} (the line in the Python code, 0 when the error is elsewhere). Also reads a traceback, when the
    script stopped before the dry run (the code itself failing to run, or conditions() raising)."""
    lines = (out or "").splitlines()
    for ln in reversed(lines):
        if ln.startswith(DRY_RUN_MARK):
            try:
                return json.loads(ln[len(DRY_RUN_MARK):])
            except ValueError:
                break
    line = 0
    for ln in lines:
        m = re.match(r'\s*File "' + re.escape(FILE_NAME) + r'", line (\d+)', ln)
        if m:
            line = int(m.group(1))
    err = ""
    for ln in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\b.*", ln.strip()) or ln.startswith("RuntimeError"):
            err = ln.strip()
            break
    if not err:
        tail = [ln for ln in lines if ln.strip()][-3:]
        err = " ".join(tail) or "the check printed nothing"
    return {"ok": False, "error": err, "line": line}
