# src/plotter.py
"""
Polished plotting for Phase-2 deliverables.

Produces high-quality PNGs (PPT-ready) into results/plots/:
 - energy_bar_comparison.png
 - ml_vs_true.png
 - zne_curve.png
 - energy_comparison_combined.png

Designed to be robust to missing files.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Safe loaders
# ---------------------------------------------------------
def safe_csv(path):
    p = Path(path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return None

def safe_json(path):
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return None

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df_final = safe_csv("results/final_leaderboard.csv")
df_ml = safe_csv("results/tables/ml_predictions.csv")
df_qref = safe_csv("results/tables/quantum_refined.csv")
zne = safe_json("results/zne_results.json")

# ---------------------------------------------------------
# 1) Bar chart — Exact vs VQE
# ---------------------------------------------------------
def plot_energy_bar(df, outpath):
    if df is None:
        print("Skipping energy bar plot (missing final leaderboard).")
        return

    pivot = df.pivot_table(
        index="material",
        columns="method",
        values="energy",
        aggfunc="first"
    )

    materials = pivot.index.tolist()
    x = np.arange(len(materials))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "exact" in pivot.columns:
        ax.bar(x - width/2, pivot["exact"], width, label="Exact")
    if "vqe" in pivot.columns:
        ax.bar(x + width/2, pivot["vqe"], width, label="VQE")

    ax.set_xticks(x)
    ax.set_xticklabels(materials, rotation=15, ha="right")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Exact vs VQE energies")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath)

# ---------------------------------------------------------
# 2) ML vs DFT scatter
# ---------------------------------------------------------
def plot_ml_vs_true(df, outpath):
    if df is None:
        print("Skipping ML plot (missing ML predictions).")
        return

    y_true = df["true_energy"].values
    y_pred = df["predicted_energy"].values

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=70, alpha=0.9)

    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "--", color="red")

    ax.set_xlabel("DFT adsorption energy (eV)")
    ax.set_ylabel("ML predicted energy (eV)")
    ax.set_title(f"ML vs DFT — MAE={mae:.3f} eV | R²={r2:.3f}")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath)

# ---------------------------------------------------------
# 3) ZNE curve
# ---------------------------------------------------------
def plot_zne(zne_data, outpath):
    if zne_data is None:
        print("Skipping ZNE plot (missing ZNE results).")
        return

    scales = zne_data["scales"]
    exact_vals = zne_data["exact_values"]
    vqe_vals = zne_data["vqe_values"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scales, exact_vals, marker="o", label="Exact (scaled)")
    ax.plot(scales, vqe_vals, marker="s", label="VQE (scaled)")

    if "exact_extrapolated" in zne_data:
        ax.scatter(0, zne_data["exact_extrapolated"], marker="D", s=80, label="Exact extrap.")
    if "vqe_extrapolated" in zne_data:
        ax.scatter(0, zne_data["vqe_extrapolated"], marker="D", s=80, label="VQE extrap.")

    ax.set_xlabel("Noise scale")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Zero-Noise Extrapolation (ZNE)")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath)

# ---------------------------------------------------------
# 4) Combined figure (no tight_layout warning)
# ---------------------------------------------------------
def combined_figure(outpath):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.25)

    # Panel A — Energy bars
    ax0 = fig.add_subplot(gs[:, 0])
    if df_final is not None:
        pivot = df_final.pivot_table(index="material", columns="method", values="energy", aggfunc="first")
        x = np.arange(len(pivot))
        width = 0.35
        if "exact" in pivot.columns:
            ax0.bar(x - width/2, pivot["exact"], width, label="Exact")
        if "vqe" in pivot.columns:
            ax0.bar(x + width/2, pivot["vqe"], width, label="VQE")
        ax0.set_xticks(x)
        ax0.set_xticklabels(pivot.index, rotation=12, ha="right")
        ax0.legend()
    ax0.set_title("Exact vs VQE")
    ax0.set_ylabel("Energy (eV)")
    ax0.grid(axis="y", alpha=0.2)

    # Panel B — ML scatter
    ax1 = fig.add_subplot(gs[0, 1:])
    if df_ml is not None:
        y_true = df_ml["true_energy"].values
        y_pred = df_ml["predicted_energy"].values
        ax1.scatter(y_true, y_pred, s=50)
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax1.plot([mn, mx], [mn, mx], "--", color="red")
        ax1.set_title("ML vs DFT")
        ax1.set_xlabel("DFT (eV)")
        ax1.set_ylabel("ML (eV)")
    else:
        ax1.text(0.5, 0.5, "ML data missing", ha="center")

    # Panel C — ZNE
    ax2 = fig.add_subplot(gs[1, 1:])
    if zne is not None:
        ax2.plot(zne["scales"], zne["exact_values"], marker="o", label="Exact")
        ax2.plot(zne["scales"], zne["vqe_values"], marker="s", label="VQE")
        ax2.legend()
    ax2.set_title("ZNE")

    fig.suptitle("Energy comparison — ML & Quantum pipeline", fontsize=14)
    fig.subplots_adjust(top=0.9)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined figure:", outpath)

# ---------------------------------------------------------
# Run all plots
# ---------------------------------------------------------
plot_energy_bar(df_final, OUT / "energy_bar_comparison.png")
plot_ml_vs_true(df_ml, OUT / "ml_vs_true.png")
plot_zne(zne, OUT / "zne_curve.png")
combined_figure(OUT / "energy_comparison_combined.png")

print("All plots generated in", OUT.resolve())