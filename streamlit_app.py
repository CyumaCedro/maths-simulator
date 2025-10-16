import streamlit as st

st.set_page_config(page_title="O'genius panda Simulator(Maths)", page_icon="🐼", layout="wide")

st.title("O'genius panda Simulator(Maths) 🐼")

st.markdown(
    """
Explore:
- Graphing Calculator: 2D/3D plots (functions, parametric, implicit).
- Conics Explorer: circles, ellipses, parabolas, hyperbolas with rotation.
- Equation Solver: solve equations/systems; derivatives, integrals, limits.
- 3D Rotations: rotate 2D shapes, generate surfaces of revolution.
- Matrix Calculator: determinants, inverses, rank, RREF, eigen, and Ax=b.

Use the sidebar to navigate between pages.
"""
)

st.success("Open the pages from the sidebar (top-left).")
