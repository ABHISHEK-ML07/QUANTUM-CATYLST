# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Quantum Catalyst — Demo Dashboard",
    layout="wide"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(".")
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
TABLES = RESULTS / "tables"

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("⚛️ Quantum Catalyst — Demo Dashboard")

st.markdown(
    "**Hybrid ML + Quantum pipeline for scalable green hydrogen catalyst discovery**"
)
st.markdown(
    "- ML pre-filters candidate active sites  \n"
    "- Quantum VQE + ZNE refines top candidates  \n"
    "- Fully reproducible, hardware-ready workflow"
)

st.divider()

# =================================================
# 🧬 INTERACTIVE 3D ACTIVE-SITE VIEW
# =================================================
st.header("🧬 Active-site 3D view")
st.markdown("Representative N-doped graphene active site used for Hamiltonian construction.")

np.random.seed(42)

# Carbon atoms
n_c = 60
x_c = np.random.uniform(0, 10, n_c)
y_c = np.random.uniform(0, 10, n_c)
z_c = np.random.normal(0, 0.15, n_c)

# Nitrogen dopant
x_n = [5.0]
y_n = [5.0]
z_n = [0.0]

fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=x_c,
    y=y_c,
    z=z_c,
    mode="markers",
    marker=dict(size=4, color="lightblue"),
    name="Carbon atoms"
))

fig_3d.add_trace(go.Scatter3d(
    x=x_n,
    y=y_n,
    z=z_n,
    mode="markers",
    marker=dict(size=10, color="red"),
    name="Nitrogen dopant (active site)"
))

fig_3d.update_layout(
    height=520,
    title="Interactive N-doped graphene active site",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# =================================================
# 🏆 LEADERBOARDS
# =================================================
st.header("🏆 Leaderboards")

final_lb = RESULTS / "final_leaderboard.csv"
st.subheader("Final leaderboard (Exact & VQE)")
if final_lb.exists():
    st.dataframe(pd.read_csv(final_lb), use_container_width=True)
else:
    st.warning("final_leaderboard.csv not found")

zne_lb = RESULTS / "zne_leaderboard.csv"
st.subheader("ZNE leaderboard (scales & extrapolated)")
if zne_lb.exists():
    st.dataframe(pd.read_csv(zne_lb), use_container_width=True)
else:
    st.info("ZNE leaderboard not found")

st.divider()

# =================================================
# 📊 RESULTS & PLOTS
# =================================================
st.header("📊 Results & Analysis")

def show_plot(title, filename):
    st.subheader(title)
    path = PLOTS / filename
    if path.exists():
        st.image(Image.open(path), use_column_width=True)
    else:
        st.warning(f"{filename} not found. Run src/plotter.py")

show_plot("Energy comparison — ML & Quantum pipeline", "energy_comparison_combined.png")
show_plot("ML predictions vs DFT", "ml_vs_true.png")
show_plot("Zero-Noise Extrapolation (ZNE)", "zne_curve.png")

st.divider()

# =================================================
# 🔁 ML → QUANTUM REFINED TABLE
# =================================================
st.header("🔁 ML → Quantum refined candidates")

qref = TABLES / "quantum_refined.csv"
if qref.exists():
    st.dataframe(pd.read_csv(qref), use_container_width=True)
else:
    st.info("quantum_refined.csv not found")

st.divider()

# =================================================
# ♻️ REPRODUCIBILITY
# =================================================
st.header("♻️ Reproducibility")

st.markdown("Run the full pipeline from project root:")
st.code(
    "conda activate quantum-catalyst\n"
    "python src/dft_loader.py\n"
    "python experiments/exp_01_ml_baseline.py\n"
    "python src/ml_quantum_bridge.py\n"
    "python src/vqe.py\n"
    "python src/zne.py\n"
    "python src/plotter.py\n"
    "streamlit run dashboard/app.py"
)

st.success("Dashboard ready for Phase-2 demo & judging ✅")