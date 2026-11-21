# The problem in Python

The **Python code** page, under **Equations** in the model tree, holds the problem as a Python module. It is
for what the expression boxes and the rows cannot say:
- equations with intermediate quantities, derivatives of derived quantities (a flux, a divergence), any order
  and mix of derivatives, piecewise coefficients or PyTorch helpers;
- boundary and initial conditions of any kind, including any condition DeepXDE has.

The module can define two functions:

```python
def pde(x, y, d):
    """The residuals, one per equation."""
    u_t = d("u", "t")
    u_xx = d("u", "x", "x")
    return [u_t - nu * u_xx]


def conditions(c):
    """The boundary and initial conditions: each call adds one loss term."""
    c.dirichlet("u", where="left or right", value="0")
    c.initial("u", lambda x: -np.sin(np.pi * x[:, 0:1]))
```

Two choices say what the code is used for:
- **Equations as:** on the [Equations](equations.md) page, **Expressions** or **Python code**.
- **Conditions as:** on the [Boundary conditions](boundary-conditions.md) and
  [Initial conditions](initial-conditions.md) pages (one choice for both), **Rows** or **Python code**.

Each can be Python on its own; the other stays as it is. Choosing **Python code** when the module has no
`pde()` (or no `conditions()`) writes one from the page, so you start from exactly what the page had. The
expressions and rows are kept, and choosing them again brings them back.

## Start from the page

**Write from the page** writes:
- the equations, from the Equations page's expressions;
- the conditions, from the Boundary Conditions rows and the initial conditions.

It asks before replacing a function the module already has, and leaves the rest of the module alone. What it
writes trains the same problem: the tests run every shipped example both ways and compare the initial loss of
every term.

## The equations: `pde(x, y, d)`

`x` holds the inputs, one column per coordinate; `y` holds the outputs, one column per output. `pde()` returns a
list with one residual per equation (a single tensor when there is one), each an `(N, 1)` tensor that is
zero where its equation holds. The **Equations** count on the Equations page must match.

| `d` gives | |
|---|---|
| `d("u", "x")`, `d("u", "x", "x")`, `d("u", "x", "t")` | derivatives of an output by name, of any order and mix |
| `d.diff(f, "x")` | the derivative of any tensor `f` computed from `x`, such as a flux |
| `d.lap("u")`, `d.grad("u")` | the Laplacian over the space axes, and the list of first derivatives |
| `d.u`, `d.x`, `d.t` | an output or a coordinate as an `(N, 1)` column |
| `d.nu` | a named constant or a trainable variable (inverse problems), by name |

The axes are `x`, `y`, `z` and `t` as the problem has them; an ODE has `t` only. Named constants are also
plain names in the module, and `torch`, `np` and `dde` are there. When the problem has a constant called `d`,
the written code calls the helper something else.

## The conditions: `conditions(c)`

Each call adds one loss term, in order, and takes `weight=` (its loss weight, 1 by default) and `name=`
(shown by Check and in the Training Log). The boundary terms come first, then the initial ones. The weights of
the equations stay on the [Loss weights](loss-weights.md) page.

| Call | Condition |
|---|---|
| `c.dirichlet(out, where, value)` | the output's value |
| `c.neumann(out, where, value)` | its normal derivative |
| `c.robin(out, where, value)` | normal derivative = value, where the value may use the outputs (`"-k*u"`, or a function `(X, y)`) |
| `c.periodic(out, axis, where, derivative_order=0)` | periodicity along an axis |
| `c.operator(fn, where, at="boundary")` | `fn(x, y, X) = 0` on the points `where` picks; `at="initial"` for the initial time. In `fn`, `c.d(x, y)` is the same helper `pde()` gets |
| `c.points(out, points, values)`, `c.points(out, file=...)` | values at given points (a list of outputs for several columns) |
| `c.initial(out, value)` | the initial condition: a function of the points `x`, a text or a number |
| `c.initial_points(file)` | the initial condition from a points file |
| `c.row({...})` | a Boundary Conditions row, as the problem file stores it |
| `c.add(condition, kind="bc")` | any DeepXDE condition you build; `c.geom` is the domain |

`where` is a text written as in the rows (`"left"`, `"x <= x_min"`, `"on_boundary and y > 0.5"`), or a function
`(X, on_boundary) -> bool` of one point. `value` is a text as in the rows (`"sin(pi*y)"`), a number, or a
function of the points `X` that returns one value per point. `c` also gives the domain limits (`c.x_min`, …),
the sides as functions of a point (`c.left(X)`, …) and `c.isclose(a, b)`.

With time-adaptive training, `conditions()` runs for every time window, with `c.geom` that window's domain.
The first window takes the code's initial conditions; later windows hold each output to the previous
window's solution, as with the rows.

Exact initial conditions (Initial conditions page) are built from the rows' formulas, so they need
**Conditions as: Rows**. With conditions in Python, **Export script** writes the full script Solve runs.

## Check

**Check** runs the Solve script once, in a second process, without training. It evaluates the equations on a
few training points, builds every condition, and reports the residuals, the conditions and the initial loss
of each term. An error shows its line, marked in the editor.

## Files and your own editor

- **File › Load .py…** and **Save .py…** read and write the module.
- **File › Open in your editor** writes it beside the problem (`<problem>_python.py`) and opens it in your
  system's editor. Each save there updates the page.
- The module is saved in the problem file. A run records it in `model_config.json`.

## Trust

The code runs with your rights when the problem is solved or checked. Code you type or write from the page
is yours. Code that came in a problem file or a `.py` file asks once, the first time it would run, and the
decision is kept for exactly that code in `~/.pinnstudio/trusted.json`; a change asks again. On the command
line, `pinnstudio run` refuses untrusted code unless you pass `--trust-code`.
