# src/ml_features.py
"""
Simple feature engineering for catalyst ML:
- Loads DFT CSV (data/dft/ocp_graphene_sample.csv)
- Adds one-hot encoding for dopant and adsorbate
- Adds coordination as numeric feature
- If matching structure file exists under structures/doped, compute mean bond distance for cluster
"""

from pathlib import Path
import pandas as pd
import numpy as np

try:
    from ase.io import read
    from scipy.spatial.distance import pdist
    ASE_AVAILABLE = True
except Exception:
    ASE_AVAILABLE = False

DATA_PATH = Path("data/dft/ocp_graphene_sample.csv")
STRUCTURE_DIR = Path("structures/doped")

def load_dft_table():
    df = pd.read_csv(DATA_PATH)
    return df

def compute_structural_feature_for_row(row):
    """
    Try to find a matching structure file for the row and compute a simple structural feature:
    mean pairwise distance among first 6 atoms (as a proxy). If no structure, return NaN.
    Matching heuristic:
      - For graphene + dopant 'N' -> structures/doped/graphene_N_doped.xyz
      - For graphene + dopant 'S' -> structures/doped/graphene_S_doped.xyz
    """
    if not ASE_AVAILABLE:
        return np.nan
    material = row.get("material", "")
    dopant = row.get("dopant", "")
    # build candidate filename
    if pd.isna(dopant) or dopant in ("None", ""):
        fname = None
    else:
        fname = f"{material}_{dopant}_doped.xyz"
    if fname:
        path = STRUCTURE_DIR / fname
        if path.exists():
            try:
                atoms = read(str(path))
                pos = atoms.get_positions()
                n = min(len(pos), 6)
                if n < 2:
                    return np.nan
                # use first n positions as proxy cluster
                pair_dists = pdist(pos[:n])
                return float(pair_dists.mean())
            except Exception:
                return np.nan
    return np.nan

def build_features(df):
    df = df.copy()
    # fill dopant None -> "None"
    df["dopant"] = df["dopant"].fillna("None").astype(str)
    df["adsorbate"] = df["adsorbate"].astype(str)
    # basic numeric features
    df["coordination"] = pd.to_numeric(df["coordination"], errors="coerce").fillna(0).astype(float)
    # structural feature
    df["mean_pair_dist"] = df.apply(compute_structural_feature_for_row, axis=1)
    # one-hot encode dopant & adsorbate (small)
    dopant_dummies = pd.get_dummies(df["dopant"], prefix="dop")
    ads_dummies = pd.get_dummies(df["adsorbate"], prefix="ads")
    X = pd.concat([df[["coordination", "mean_pair_dist"]], dopant_dummies, ads_dummies], axis=1).fillna(0.0)
    y = df["adsorption_energy_ev"].astype(float)
    return X, y, df

if __name__ == "__main__":
    df = load_dft_table()
    X, y, dffull = build_features(df)
    print("Features shape:", X.shape)
    print("Feature columns:", X.columns.tolist())
    print(dffull.head())