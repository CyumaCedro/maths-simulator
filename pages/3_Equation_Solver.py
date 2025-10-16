import streamlit as st
import sympy as sp
from sympy import Eq

st.set_page_config(page_title="Equation Solver")

st.title("Equation & Calculus Tools")

st.caption("Use ** for powers (x**2), functions like sin, cos, exp, log, sqrt. Infinity: oo, -oo.")
with st.expander("Examples", expanded=False):
    st.markdown("""
    - Solve: x**2 - 5*x + 6 = 0; x + y = 3; x - y = 1 (vars: x, y)
    - Derivative: expr=sin(x)*exp(x), var=x, order=1
    - Integral: expr=x**2, var=x, bounds 0..3 (or leave blank for indefinite)
    - Limit: expr=(1+1/x)**x, var=x, point=oo, direction=both
    - Simplify/Factor/Expand: (x+1)**3; x**2-1; (x+2)*(x-2)
    """)

mode = st.selectbox("Mode", ["Solve equation(s)", "Derivative", "Integral", "Limit", "Simplify", "Factor", "Expand"])

expr_input = st.text_area("Expression(s)", value="x^2 - 5*x + 6 = 0\nx + y = 3\nx - y = 1")
vars_input = st.text_input("Variables (comma-separated)", value="x, y")

# Prepare variables
var_names = [v.strip() for v in vars_input.split(',') if v.strip()]
vars_syms = sp.symbols(var_names)

if mode == "Solve equation(s)":
    st.caption("Enter one or more equations, one per line. Use '=' for equality. E.g., x^2-5x+6=0")
    if st.button("Solve", use_container_width=True):
        try:
            eqs = []
            for line in expr_input.splitlines():
                line = line.strip()
                if not line:
                    continue
                if '=' in line:
                    lhs, rhs = line.split('=', 1)
                    eqs.append(Eq(sp.sympify(lhs), sp.sympify(rhs)))
                else:
                    eqs.append(sp.sympify(line))
            sol = sp.solve(eqs, vars_syms, dict=True)
            st.write("Solutions:")
            st.json([{str(k): sp.simplify(v) for k, v in s.items()} for s in sol])
        except Exception as e:
            st.error(f"Error: {e}")
else:
    if mode == "Derivative":
        order = st.number_input("order", min_value=1, max_value=5, value=1)
        var = st.selectbox("differentiate w.r.t.", var_names, index=0)
        if st.button("Differentiate", use_container_width=True):
            try:
                expr = sp.sympify(expr_input)
                d = sp.diff(expr, sp.Symbol(var), order)
                st.latex(sp.latex(d))
            except Exception as e:
                st.error(f"Error: {e}")
    elif mode == "Integral":
        var = st.selectbox("integrate w.r.t.", var_names, index=0)
        a = st.text_input("lower bound (blank for indefinite)", value="")
        b = st.text_input("upper bound (blank for indefinite)", value="")
        if st.button("Integrate", use_container_width=True):
            try:
                expr = sp.sympify(expr_input)
                if a.strip() and b.strip():
                    res = sp.integrate(expr, (sp.Symbol(var), sp.sympify(a), sp.sympify(b)))
                else:
                    res = sp.integrate(expr, sp.Symbol(var))
                st.latex(sp.latex(res))
            except Exception as e:
                st.error(f"Error: {e}")
    elif mode == "Limit":
        var = st.selectbox("variable", var_names, index=0)
        point = st.text_input("approach point (e.g., 0, oo, -oo)", value="0")
        dirn = st.selectbox("direction", ["+", "-", "both"], index=0)
        if st.button("Compute limit", use_container_width=True):
            try:
                expr = sp.sympify(expr_input)
                dir_kw = None if dirn == "both" else dirn
                res = sp.limit(expr, sp.Symbol(var), sp.sympify(point), dir=dir_kw)
                st.latex(sp.latex(res))
            except Exception as e:
                st.error(f"Error: {e}")
    elif mode == "Simplify":
        if st.button("Simplify", use_container_width=True):
            try:
                expr = sp.sympify(expr_input)
                st.latex(sp.latex(sp.simplify(expr)))
            except Exception as e:
                st.error(f"Error: {e}")
    elif mode == "Factor":
        if st.button("Factor", use_container_width=True):
            try:
                expr = sp.sympify(expr_input)
                st.latex(sp.latex(sp.factor(expr)))
            except Exception as e:
                st.error(f"Error: {e}")
    elif mode == "Expand":
        if st.button("Expand", use_container_width=True):
            try:
                expr = sp.sympify(expr_input)
                st.latex(sp.latex(sp.expand(expr)))
            except Exception as e:
                st.error(f"Error: {e}")
