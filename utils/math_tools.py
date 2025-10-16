from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sympy import (
    Abs,
    E,
    Eq,
    I,
    Matrix,
    Symbol,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    diff,
    exp,
    lambdify,
    limit,
    log,
    pi,
    simplify,
    sin,
    sinh,
    solve,
    solveset,
    sqrt,
    symbols,
    tan,
    tanh,
)
from sympy.parsing.sympy_parser import (  # type: ignore
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ---------------------- SymPy helpers ----------------------

SAFE_NAMESPACE: Dict[str, object] = {
    # constants
    "pi": pi,
    "e": E,
    "E": E,
    "I": I,
    # functions
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
    "asinh": asinh,
    "acosh": acosh,
    "atanh": atanh,
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "abs": Abs,
}

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def build_symbols(names: Iterable[str]) -> Dict[str, Symbol]:
    return {n: symbols(n) for n in names}


def safe_sympify(expr: str, sym_names: Iterable[str]) -> Tuple[object, Dict[str, Symbol]]:
    syms = build_symbols(sym_names)
    local_dict = {**SAFE_NAMESPACE, **syms}
    parsed = parse_expr(expr, local_dict=local_dict, transformations=TRANSFORMATIONS, evaluate=True)
    return parsed, syms


def to_callable(expr_obj, vars_list: List[Symbol]) -> Callable:
    # vectorized numpy function
    return lambdify(vars_list, expr_obj, modules=[{"sin": np.sin, "cos": np.cos, "tan": np.tan,
                                                 "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                                                 "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
                                                 "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
                                                 "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                                                 "pi": np.pi, "e": np.e},
                                                "numpy"])


# ---------------------- Numeric grids ----------------------

def linspace(a: float, b: float, n: int) -> np.ndarray:
    if a == b:
        b = a + 1.0
    return np.linspace(a, b, n)


def meshgrid_xy(xmin: float, xmax: float, ymin: float, ymax: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    x = linspace(xmin, xmax, n)
    y = linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)
    return X, Y


# ---------------------- Rotation utilities ----------------------

def rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_points_xy(points: np.ndarray, theta_rad: float, origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    # points: shape (N, 2)
    R = rotation_matrix(theta_rad)
    o = np.array(origin)
    return ((points - o) @ R.T) + o


# ---------------------- Revolution surfaces ----------------------

def revolve_profile_xy(xs: np.ndarray, rs: np.ndarray, axis: str = "x", n_theta: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Given profile (x, r) with r >= 0, revolve around axis to build (X, Y, Z) grids
    theta = np.linspace(0, 2 * np.pi, n_theta)
    if axis == "x":
        X = np.tile(xs[:, None], (1, n_theta))
        Y = (rs[:, None] * np.cos(theta)[None, :])
        Z = (rs[:, None] * np.sin(theta)[None, :])
    elif axis == "y":
        Y = np.tile(xs[:, None], (1, n_theta))
        X = (rs[:, None] * np.cos(theta)[None, :])
        Z = (rs[:, None] * np.sin(theta)[None, :])
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return X, Y, Z


# ---------------------- Conics parametric forms ----------------------

def conic_circle(h: float, k: float, r: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    x = h + r * np.cos(t)
    y = k + r * np.sin(t)
    return np.c_[x, y]


def conic_ellipse(h: float, k: float, a: float, b: float, theta_rad: float, n: int = 600) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    x0 = a * np.cos(t)
    y0 = b * np.sin(t)
    pts = np.c_[x0, y0]
    pts = rotate_points_xy(pts, theta_rad, origin=(0.0, 0.0))
    pts[:, 0] += h
    pts[:, 1] += k
    return pts


def conic_parabola(h: float, k: float, p: float, theta_rad: float, tmin: float = -5, tmax: float = 5, n: int = 800) -> np.ndarray:
    # Standard y = (1/(4p)) x^2; parameterize with x = t, y = t^2/(4p)
    t = np.linspace(tmin, tmax, n)
    x0 = t
    y0 = (t ** 2) / (4.0 * p if p != 0 else 1e-6)
    pts = np.c_[x0, y0]
    pts = rotate_points_xy(pts, theta_rad, origin=(0.0, 0.0))
    pts[:, 0] += h
    pts[:, 1] += k
    return pts


def conic_hyperbola(h: float, k: float, a: float, b: float, theta_rad: float, umax: float = 2.0, n: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    # Param using hyperbolic cosh/sinh: x = ± a cosh(u), y = b sinh(u)
    u = np.linspace(-umax, umax, n)
    xR = a * np.cosh(u)
    yR = b * np.sinh(u)
    xL = -a * np.cosh(u)
    yL = b * np.sinh(u)
    R = np.c_[xR, yR]
    L = np.c_[xL, yL]
    R = rotate_points_xy(R, theta_rad, origin=(0.0, 0.0)) + np.array([h, k])
    L = rotate_points_xy(L, theta_rad, origin=(0.0, 0.0)) + np.array([h, k])
    return R, L
