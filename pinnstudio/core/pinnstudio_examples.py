"""PINNStudio templates defined by their settings.

The older templates are set up widget by widget in the main window; the
ones here are plain PINNConfig settings, loaded the same way as DeepXDE's
examples (deepxde_examples.build_config, then MainWindow._apply_config), so
a template is one dictionary. They are listed as PINNStudio templates in the
Examples lists (example_catalog), with the paper they follow.

Templates you keep for yourself go in pinnstudio/core/local_examples.py
(ignored by git), with their reference data under reference_data/local/:
the file defines EXAMPLES, CATALOG ({name: (shown name, paper)}) and PAPERS
({paper: link}) in the same form, and is added to the lists when present.
"""
import json as _json

from pinnstudio.core import deepxde_examples as _dx

PAPERS = {}
EXAMPLES = []
# shown name and paper of each, for the example catalogue
CATALOG = {}

try:                                    # local templates, if there are any
    from pinnstudio.core import local_examples as _local
except ImportError:
    _local = None
if _local is not None:
    PAPERS.update(getattr(_local, "PAPERS", {}))
    EXAMPLES.extend(getattr(_local, "EXAMPLES", []))
    CATALOG.update(getattr(_local, "CATALOG", {}))

# Poisson in a cube with a spherical cut-out: a 3D domain built from two primitives (a cube with a sphere
# centred on its top face subtracted), with a Dirichlet condition on the cube's faces and a Neumann condition
# on the spherical surface. The exact solution u = exp(-(x^2 + y^2 + z^2)) gives both, and Error Analysis
# compares against it directly. On the spherical surface the outward normal of the domain points into the
# sphere, n = -(p - c)/R with c = (0, 0, 1), so du/dn = 2 u (x^2 + y^2 + z (z - 1)) / R.
EXAMPLES.append(_dx._ex(
    "3D Poisson (Cube with spherical cut-out)", _dx.FORWARD, "",
    dict(problem_dim="3D", steady_state=True, geometry_type="Custom",
         geom_custom_shapes_json=_json.dumps([
             {"type": "Cuboid", "params": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0,
                                           "z_min": -1.0, "z_max": 1.0}},
             {"type": "Sphere", "op": "subtract", "params": {"cx": 0.0, "cy": 0.0, "cz": 1.0, "r": 0.7}}]),
         x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0, num_initial=0,
         pde_expressions="-du_xx - du_yy - du_zz - (6 - 4*(x**2 + y**2 + z**2))*exp(-(x**2 + y**2 + z**2))",
         custom_bc_json=_dx._bcs(
             _dx._bc("dirichlet", "left or right or bottom or top or front or back", "exp(-(x**2 + y**2 + z**2))"),
             _dx._bc("neumann", "np.isclose(x**2 + y**2 + (z - 1)**2, R**2)",
                     "2*exp(-(x**2 + y**2 + z**2))*(x**2 + y**2 + z*(z - 1))/R")),
         num_domain=8000, num_boundary=2000, num_test=5000, plot_resolution=60),
    _dx._consts(("R", "0.7")),
    _dx._net([3, 64, 64, 64, 64, 1]), _dx._adam(15000), dict(optimizer2="lbfgs", iterations2=15000),
    _dx._formula_ref(["exp(-(x**2 + y**2 + z**2))"]),
    note="A cube [-1, 1]^3 with a sphere of radius R = 0.7 centred on its top face subtracted (Geometry -> "
         "Custom); u = exp(-(x^2 + y^2 + z^2)) on the cube's faces (Dirichlet, the named sides) and its normal "
         "derivative on the spherical surface (Neumann, a coordinate expression); compared with the exact solution."))
CATALOG["3D Poisson (Cube with spherical cut-out)"] = ("Poisson 3D – cube with spherical cut-out", "")

# Burgers' equation written in Python (the Python code page): the equation in conservation form, with the
# flux's derivative taken by the d helper, and the conditions as functions. The same problem as the RAR
# example's expressions and rows, without RAR, so the two can be compared.
_BURGERS_PY = '"""Burgers\' equation, u_t + (u^2/2)_x = nu u_xx on [-1, 1] x [0, 1], written in Python."""\nNU = 0.01 / np.pi\n\n\ndef flux(u):\n    return u * u / 2\n\n\ndef pde(x, y, d):\n    """The residual of Burgers\' equation in conservation form: the flux\'s x-derivative is\n    d.diff(flux(u), "x"), which the expressions cannot write."""\n    return [d("u", "t") + d.diff(flux(d.u), "x") - NU * d("u", "x", "x")]\n\n\ndef conditions(c):\n    """u = 0 at both ends; u = -sin(pi x) at t = 0."""\n    c.dirichlet("u", where=lambda X, on: on and (c.isclose(X[0], c.x_min) or c.isclose(X[0], c.x_max)),\n                value=0, name="u = 0 at the ends")\n    c.initial("u", lambda x: -np.sin(np.pi * x[:, 0:1]), name="u at t = 0")\n'
_burgers = next(e for e in _dx.EXAMPLES if e.name == "Burgers equation with RAR")
EXAMPLES.append(_dx._ex(
    "Burgers equation (Python code)", _dx.FORWARD, "",
    dict(_burgers.settings),
    dict(adapt_method="None", pde_mode="python", conditions_mode="python", python_code=_BURGERS_PY),
    note="Burgers' equation on [-1, 1] with u(x, 0) = -sin(pi x), written on the Python code page: pde() in "
         "conservation form (the flux u^2/2 differentiated by d.diff) and conditions() with functions for "
         "where and the values. Compared with the reference solution at five times."))
CATALOG["Burgers equation (Python code)"] = ("Burgers 1D – Python code", "")

BY_NAME = {e.name: e for e in EXAMPLES}


def build_config(example, ref_root, keep=None, inverse=False):
    return _dx.build_config(example, ref_root, keep=keep, inverse=inverse)
