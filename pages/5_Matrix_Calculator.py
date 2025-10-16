import streamlit as st
import sympy as sp
import numpy as np

st.set_page_config(page_title="Matrix Calculator")

st.title("Matrix Calculator")

st.caption("Compute determinant, inverse, rank, transpose, trace, RREF, eigen, characteristic polynomial, and solve Ax=b.")

with st.expander("Examples", expanded=False):
    st.markdown("""
    - Identity 3x3
      1,0,0\n0,1,0\n0,0,1
    - Singular 3x3
      1,2,3\n2,4,6\n1,1,1
    - Rotation 2D 45°
      cos(pi/4), -sin(pi/4)\n sin(pi/4), cos(pi/4)
    - Symmetric 3x3
      2,1,0\n1,2,1\n0,1,2
    - Upper-triangular 3x3
      2,1,3\n0,1,-1\n0,0,4
    """)

mode = st.radio("Input mode", ["Grid", "CSV"], horizontal=True)

rows = st.number_input("rows", min_value=1, max_value=8, value=3, step=1)
cols = st.number_input("cols", min_value=1, max_value=8, value=3, step=1)

# Quick examples (applied after parsing inputs)
example = st.selectbox(
    "Quick example (overrides inputs for A)",
    ["None", "Identity 3x3", "Rotation 2D 45°", "Singular 3x3", "Symmetric 3x3", "Upper-triangular 3x3"],
    index=0,
)

if mode == "Grid":
    data = []
    for i in range(rows):
        cols_inputs = st.columns(int(cols))
        row = []
        for j, c in enumerate(cols_inputs):
            with c:
                row.append(st.number_input(f"a[{i+1},{j+1}]", value=float(1 if i==j else 0)))
        data.append(row)
else:
    csv = st.text_area("Enter CSV matrix", value="1,0,0\n0,1,0\n0,0,1", height=120)
    data = []
    for line in csv.splitlines():
        line = line.strip()
        if not line:
            continue
        row_vals = [float(sp.N(sp.sympify(x))) for x in line.split(',')]
        data.append(row_vals)
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0

if rows and cols:
    A = sp.Matrix(data)

    # Apply quick examples
    if example != "None":
        if example == "Identity 3x3":
            A = sp.eye(3)
        elif example == "Rotation 2D 45°":
            A = sp.Matrix([[sp.cos(sp.pi/4), -sp.sin(sp.pi/4)], [sp.sin(sp.pi/4), sp.cos(sp.pi/4)]])
        elif example == "Singular 3x3":
            A = sp.Matrix([[1,2,3],[2,4,6],[1,1,1]])
        elif example == "Symmetric 3x3":
            A = sp.Matrix([[2,1,0],[1,2,1],[0,1,2]])
        elif example == "Upper-triangular 3x3":
            A = sp.Matrix([[2,1,3],[0,1,-1],[0,0,4]])
        rows, cols = A.shape
        st.info("Using quick example matrix A (overrides inputs).")

    st.write("Matrix A:")
    st.dataframe(np.array(A.tolist(), dtype=float))

    st.subheader("Operations")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_det = st.button("Determinant")
        do_rank = st.button("Rank")
    with c2:
        do_inv = st.button("Inverse")
        do_rref = st.button("RREF")
    with c3:
        do_eigs = st.button("Eigenvalues/Vectors")
        do_charpoly = st.button("Characteristic Polynomial")

    c4, c5, c6 = st.columns(3)
    with c4:
        do_transpose = st.button("Transpose")
    with c5:
        do_trace = st.button("Trace")
    with c6:
        do_norm = st.button("Frobenius Norm")

    # Ax = b solver
    st.subheader("Solve Ax = b")
    b_default = ",".join(["1"] * int(rows))
    b_str = st.text_input("b (comma-separated, length = rows)", value=b_default)
    solve_btn = st.button("Solve Ax=b")

    try:
        if do_det:
            if rows != cols:
                st.error("Determinant requires a square matrix.")
            else:
                st.latex(r"\det(A) = " + sp.latex(A.det()))
        if do_rank:
            st.write("rank(A) =", int(A.rank()))
        if do_inv:
            if rows != cols:
                st.error("Inverse requires a square matrix.")
            else:
                if A.det() == 0:
                    st.warning("Matrix is singular; no inverse.")
                else:
                    st.write("A^{-1}:")
                    st.latex(sp.latex(A.inv()))
        if do_rref:
            R, piv = A.rref()
            st.write("RREF(A):")
            st.latex(sp.latex(R))
            st.write("Pivot columns (0-indexed):", piv)
        if do_eigs:
            if rows != cols:
                st.error("Eigen decomposition requires a square matrix.")
            else:
                evals = A.eigenvals()
                st.write("Eigenvalues (with algebraic multiplicity):")
                st.json({str(k): int(v) for k, v in evals.items()})
                st.write("Eigenvectors:")
                evecs = A.eigenvects()
                for val, mult, vecs in evecs:
                    st.write(f"λ = {sp.simplify(val)} (mult {mult})")
                    for idx, v in enumerate(vecs):
                        st.latex(sp.latex(v))
        if do_charpoly:
            if rows != cols:
                st.error("Characteristic polynomial requires a square matrix.")
            else:
                lam = sp.symbols('λ')
                cp = A.charpoly(lam)
                st.latex(r"p(\lambda) = " + sp.latex(cp.as_expr()))
        if do_transpose:
            st.write("A^T:")
            st.latex(sp.latex(A.T))
        if do_trace:
            if rows != cols:
                st.error("Trace requires a square matrix.")
            else:
                st.write("trace(A) =", sp.simplify(A.trace()))
        if do_norm:
            st.write("||A||_F =", float(A.norm()))

        if solve_btn:
            try:
                b_vals = [float(sp.N(sp.sympify(x))) for x in b_str.split(',') if x.strip()]
                if len(b_vals) != rows:
                    st.error(f"b must have length {rows}.")
                else:
                    b_vec = sp.Matrix(b_vals)
                    # Try exact solve (works for rectangular too)
                    sol = sp.linsolve((A, b_vec))
                    st.write("Solution set:")
                    st.latex(sp.latex(sol))
            except Exception as ex:
                st.error(f"Error parsing/solving Ax=b: {ex}")
    except Exception as e:
        st.error(f"Error: {e}")
