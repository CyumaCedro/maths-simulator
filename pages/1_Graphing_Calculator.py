import numpy as np
import plotly.graph_objs as go
import streamlit as st
from utils.math_tools import safe_sympify, to_callable, meshgrid_xy, linspace

st.set_page_config(page_title="Graphing Calculator")

st.title("Graphing Calculator")

TAB2D, TABPAR, TABIMPL, TAB3D = st.tabs(["2D Function", "Parametric", "Implicit", "3D Surface"]) 

with TAB2D:
    st.subheader("y = f(x)")
    with st.expander("Examples", expanded=False):
        st.markdown("""
        - sin(x), cos(x)
        - exp(-x**2)
        - x**3 - 4*x
        - sin(x)/x
        """)
    expr = st.text_input("f(x) =", value="sin(x) + 0.2*x**2")
    xmin, xmax = st.columns(2)
    with xmin:
        x_min = st.number_input("x min", value=-10.0)
    with xmax:
        x_max = st.number_input("x max", value=10.0)
    n = st.slider("samples", min_value=200, max_value=4000, value=800, step=100)
    if st.button("Plot 2D", use_container_width=True):
        try:
            expr_obj, syms = safe_sympify(expr, ["x"])  # type: ignore
            f = to_callable(expr_obj, [syms["x"]])
            xs = linspace(x_min, x_max, n)
            ys = f(xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="y=f(x)"))
            fig.update_layout(height=500, template="plotly_dark", xaxis_title="x", yaxis_title="y")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

with TABPAR:
    st.subheader("Parametric: (x(t), y(t))")
    with st.expander("Examples", expanded=False):
        st.markdown("""
        - Circle: x=cos(t), y=sin(t)
        - Lissajous: x=sin(3*t), y=sin(4*t+pi/2)
        - Rose: x=cos(5*t)*cos(t), y=cos(5*t)*sin(t)
        """)
    col1, col2 = st.columns(2)
    with col1:
        x_expr = st.text_input("x(t) =", value="cos(3*t) * (1 + 0.5*cos(t))")
    with col2:
        y_expr = st.text_input("y(t) =", value="sin(3*t) * (1 + 0.5*cos(t))")
    tmin, tmax = st.columns(2)
    with tmin:
        t_min = st.number_input("t min", value=0.0)
    with tmax:
        t_max = st.number_input("t max", value=2*np.pi)
    n = st.slider("t samples", min_value=200, max_value=5000, value=1200, step=100, key="par_n")
    if st.button("Plot Parametric", use_container_width=True):
        try:
            x_obj, syms_x = safe_sympify(x_expr, ["t"])  # type: ignore
            y_obj, syms_y = safe_sympify(y_expr, ["t"])  # type: ignore
            tx = to_callable(x_obj, [syms_x["t"]])
            ty = to_callable(y_obj, [syms_y["t"]])
            ts = np.linspace(t_min, t_max, n)
            xs = tx(ts)
            ys = ty(ts)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="(x(t),y(t))"))
            fig.update_layout(height=500, template="plotly_dark", xaxis_title="x", yaxis_title="y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

with TABIMPL:
    st.subheader("Implicit: F(x,y) = 0")
    with st.expander("Examples", expanded=False):
        st.markdown("""
        - x**2 + y**2 - 4
        - (x**2 + y**2 - 1)**3 - x**2*y**3
        - x*y - 1
        """)
    impl_expr = st.text_input("F(x,y) =", value="x**3 - y**2 + x - 1")
    cols = st.columns(4)
    x_min = cols[0].number_input("x min", value=-4.0)
    x_max = cols[1].number_input("x max", value=4.0)
    y_min = cols[2].number_input("y min", value=-4.0)
    y_max = cols[3].number_input("y max", value=4.0)
    ngrid = st.slider("grid", min_value=100, max_value=800, value=300, step=50, key="impl_n")
    if st.button("Plot Implicit", use_container_width=True):
        try:
            F_obj, syms = safe_sympify(impl_expr, ["x", "y"])  # type: ignore
            F = to_callable(F_obj, [syms["x"], syms["y"]])
            X, Y = meshgrid_xy(x_min, x_max, y_min, y_max, n=ngrid)
            Z = F(X, Y)
            fig = go.Figure()
            # Show heatmap + contours including zero level
            fig.add_trace(go.Contour(x=X[0, :], y=Y[:, 0], z=Z, colorscale="Viridis", showscale=False,
                                     contours=dict(showlines=True, start=-2, end=2, size=0.2)))
            fig.add_hline(y=0, line_color="#555", line_dash="dot")
            fig.add_vline(x=0, line_color="#555", line_dash="dot")
            fig.update_layout(height=500, template="plotly_dark", xaxis_title="x", yaxis_title="y")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

with TAB3D:
    st.subheader("3D Surface: z = f(x,y)")
    with st.expander("Examples", expanded=False):
        st.markdown("""
        - sin(x)*cos(y)
        - sin(sqrt(x**2 + y**2)) / (sqrt(x**2 + y**2) + 1e-6)
        - x*y*exp(-(x**2 + y**2)/10)
        """)
    z_expr = st.text_input("f(x,y) =", value="sin(x)*cos(y)")
    cols = st.columns(4)
    x_min = cols[0].number_input("x min", value=-6.0)
    x_max = cols[1].number_input("x max", value=6.0)
    y_min = cols[2].number_input("y min", value=-6.0)
    y_max = cols[3].number_input("y max", value=6.0)
    ngrid = st.slider("grid", min_value=30, max_value=300, value=120, step=10, key="surf_n")
    if st.button("Plot 3D", use_container_width=True):
        try:
            Z_obj, syms = safe_sympify(z_expr, ["x", "y"])  # type: ignore
            F = to_callable(Z_obj, [syms["x"], syms["y"]])
            X, Y = meshgrid_xy(x_min, x_max, y_min, y_max, n=ngrid)
            Z = F(X, Y)
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
            fig.update_layout(height=600, template="plotly_dark", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
