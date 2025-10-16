import numpy as np
import plotly.graph_objs as go
import streamlit as st
from utils.math_tools import conic_circle, conic_ellipse, conic_parabola, conic_hyperbola

st.set_page_config(page_title="Conics Explorer")

st.title("Conics Explorer")

st.caption("Circle: (x-h)^2+(y-k)^2=r^2; Ellipse: ((x-h)/a)^2+((y-k)/b)^2=1; Parabola: y^2=4px (rotated); Hyperbola: (x/a)^2-(y/b)^2=1 (rotations supported).")
with st.expander("Examples", expanded=False):
    st.markdown("""
    - Circle: h=0, k=0, r=3
    - Ellipse: a=5, b=2, rotation=20°
    - Parabola: p=1, rotation=0°, t in [-5,5]
    - Hyperbola: a=3, b=1.5, rotation=0°, u range=2.0
    """)

conic = st.selectbox("Conic type", ["Circle", "Ellipse", "Parabola", "Hyperbola"], index=1)

if conic == "Circle":
    h = st.number_input("h", value=0.0)
    k = st.number_input("k", value=0.0)
    r = st.number_input("r", value=3.0, min_value=0.0, step=0.1)
    pts = conic_circle(h, k, r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="circle"))

elif conic == "Ellipse":
    h = st.number_input("h", value=0.0)
    k = st.number_input("k", value=0.0)
    a = st.number_input("a (semi-major)", value=5.0, min_value=0.0, step=0.1)
    b = st.number_input("b (semi-minor)", value=2.0, min_value=0.0, step=0.1)
    theta = st.slider("rotation (deg)", min_value=-180, max_value=180, value=20)
    pts = conic_ellipse(h, k, a, b, np.deg2rad(theta))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="ellipse"))

elif conic == "Parabola":
    h = st.number_input("h", value=0.0)
    k = st.number_input("k", value=0.0)
    p = st.number_input("p (focal length)", value=1.0, step=0.1)
    theta = st.slider("rotation (deg)", min_value=-180, max_value=180, value=0)
    tmin = st.number_input("t min", value=-5.0)
    tmax = st.number_input("t max", value=5.0)
    pts = conic_parabola(h, k, p, np.deg2rad(theta), tmin, tmax)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="parabola"))

else:  # Hyperbola
    h = st.number_input("h", value=0.0)
    k = st.number_input("k", value=0.0)
    a = st.number_input("a", value=3.0, min_value=0.0, step=0.1)
    b = st.number_input("b", value=1.5, min_value=0.0, step=0.1)
    theta = st.slider("rotation (deg)", min_value=-180, max_value=180, value=0)
    umax = st.slider("u range", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    R, L = conic_hyperbola(h, k, a, b, np.deg2rad(theta), umax)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=R[:, 0], y=R[:, 1], mode="lines", name="right branch"))
    fig.add_trace(go.Scatter(x=L[:, 0], y=L[:, 1], mode="lines", name="left branch"))

fig.update_layout(template="plotly_dark", height=600, xaxis_title="x", yaxis_title="y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
fig.add_hline(y=0, line_color="#555", line_dash="dot")
fig.add_vline(x=0, line_color="#555", line_dash="dot")
st.plotly_chart(fig, use_container_width=True)
