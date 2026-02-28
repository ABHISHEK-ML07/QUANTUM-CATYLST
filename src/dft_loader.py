# src/dft_loader.py
"""
DFT Dataset Loader
Loads DFT-derived adsorption energy data (OCP / MP style)
"""

import pandas as pd
from pathlib import Path

DFT_DATA_PATH = Path("data/dft/ocp_graphene_sample.csv")

def load_dft_data():
    if not DFT_DATA_PATH.exists():
        raise FileNotFoundError(f"DFT dataset not found at {DFT_DATA_PATH}")

    df = pd.read_csv(DFT_DATA_PATH)

    required_cols = [
        "material",
        "surface",
        "adsorbate",
        "dopant",
        "coordination",
        "adsorption_energy_ev"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in DFT data: {col}")

    return df

def summarize_dft(df):
    summary = df.groupby(["material", "dopant", "adsorbate"])["adsorption_energy_ev"].mean()
    return summary.reset_index()

if __name__ == "__main__":
    df = load_dft_data()
    print("DFT dataset loaded successfully")
    print(df.head())

    summary = summarize_dft(df)
    print("\nDFT Summary (mean adsorption energies):")
    print(summary)