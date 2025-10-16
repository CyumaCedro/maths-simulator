import numpy as np
import plotly.graph_objs as go
import streamlit as st
from utils.math_tools import rotate_points_xy, revolve_profile_xy, safe_sympify, to_callable, linspace

st.set_page_config(page_title="3D Rotations")

st.title("Rotations & Surfaces of Revolution")

st.caption("Rotate 2D shapes in-plane, or revolve a profile y=f(x) around the x or y axis to create 3D surfaces.")
with st.expander("Examples", expanded=False):
    st.markdown("""
    - Rotate: Square by 45° about origin
    - Rotate: Custom polygon from CSV
    - Revolve: f(x)=exp(-0.1*x**2)*(1+0.2*cos(5*x)) about x-axis
    - Revolve points: use the default CSV to create a vase-like surface
    """)

mode = st.radio("Mode", ["Rotate 2D shape (in-plane)", "Surface of revolution (3D)"])

if mode == "Rotate 2D shape (in-plane)":
    shape = st.selectbox("Shape", ["Square", "Triangle", "Pentagon", "Circle (sampled)", "Custom points (x,y) CSV"], index=0)
    angle = st.slider("rotation angle (deg)", min_value=-360, max_value=360, value=45)
    origin_x, origin_y = st.columns(2)
    with origin_x:
        ox = st.number_input("origin x", value=0.0)
    with origin_y:
        oy = st.number_input("origin y", value=0.0)

    if shape == "Square":
        s = 2.0
        pts = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]], dtype=float)
    elif shape == "Triangle":
        pts = np.array([[0, 2], [-2, -2], [2, -2], [0, 2]], dtype=float)
    elif shape == "Pentagon":
        t = np.linspace(0, 2*np.pi, 6)
        pts = np.c_[np.cos(t), np.sin(t)]
    elif shape == "Circle (sampled)":
        t = np.linspace(0, 2*np.pi, 400)
        pts = np.c_[np.cos(t), np.sin(t)]
    else:
        csv = st.text_area("Points CSV (x,y per line)", value="-1,-1\n1,-1\n1,1\n-1,1\n-1,-1")
        arr = []
        for line in csv.splitlines():
            if not line.strip():
                continue
            x_str, y_str = line.split(',')
            arr.append([float(x_str), float(y_str)])
        pts = np.array(arr, dtype=float)

    rot = rotate_points_xy(pts, np.deg2rad(angle), origin=(ox, oy))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode="lines", name="original"))
    fig.add_trace(go.Scatter(x=rot[:,0], y=rot[:,1], mode="lines", name="rotated"))
    fig.update_layout(template="plotly_dark", height=600, xaxis_title="x", yaxis_title="y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

else:
    profile_mode = st.radio("Profile type", ["Function y=f(x)", "Points (x,y)"])
    axis = st.selectbox("Axis of revolution", ["x", "y"], index=0)
    if profile_mode == "Function y=f(x)":
        fx = st.text_input("f(x) =", value="exp(-0.1*x**2) * (1 + 0.2*cos(5*x))")
        x_min = st.number_input("x min", value=-4.0)
        x_max = st.number_input("x max", value=4.0)
        nsamp = st.slider("profile samples", min_value=50, max_value=1500, value=400, step=50)
        ntheta = st.slider("theta samples", min_value=20, max_value=400, value=120, step=10)
        if st.button("Generate surface", use_container_width=True):
            try:
                expr_obj, syms = safe_sympify(fx, ["x"])  # type: ignore
                f = to_callable(expr_obj, [syms["x"]])
                xs = linspace(x_min, x_max, nsamp)
                rs = np.maximum(0.0, np.array(f(xs), dtype=float))
                X, Y, Z = revolve_profile_xy(xs, rs, axis=axis, n_theta=ntheta)
                fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
                fig.update_layout(template="plotly_dark", height=700, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        csv = st.text_area("Points CSV (x,y per line)", value="-2,1\n-1,2\n0,2.5\n1,2\n2,1")
        nsamp = st.slider("resample profile to N points", min_value=20, max_value=2000, value=400, step=20)
        ntheta = st.slider("theta samples", min_value=20, max_value=400, value=120, step=10, key="rev_pts")
        if st.button("Generate surface from points", use_container_width=True):
            try:
                pts = []
                for line in csv.splitlines():
                    if not line.strip():
                        continue
                    xs, ys = line.split(',')
                    pts.append([float(xs), float(ys)])
                pts = np.array(pts, dtype=float)
                # simple linear resample
                t = np.linspace(0, 1, len(pts))
                tt = np.linspace(0, 1, nsamp)
                xs = np.interp(tt, t, pts[:,0])
                rs = np.maximum(0.0, np.interp(tt, t, np.abs(pts[:,1])))
                X, Y, Z = revolve_profile_xy(xs, rs, axis=axis, n_theta=ntheta)
                fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
                fig.update_layout(template="plotly_dark", height=700, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
