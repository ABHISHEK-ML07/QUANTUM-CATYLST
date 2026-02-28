# src/ml_quantum_bridge.py
"""
ML -> Quantum bridge:
- Loads trained ML regressor (results/models/ml_regressor.pkl)
- Loads DFT table (data/dft/ocp_graphene_sample.csv)
- Builds features via src.ml_features.build_features
- Selects top_k candidates by ML predicted energy (most negative = best)
- For each selected candidate:
    - map to a structure file under structures/
    - build a distance-based Pauli ZZ + X Hamiltonian for a small cluster
    - run exact diagonalization and VQE (importing functions from src.vqe)
- Save quantum-refined CSV to results/tables/quantum_refined.csv
"""
import sys, pathlib, json
project_root = str(pathlib.Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from ase.io import read

from src.ml_features import load_dft_table, build_features
from src.vqe import exact_diagonalization, vqe_with_scipy, vqe_random_search

# local helper: build pauli terms & full operator (copied from hamiltonian)
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def kron_n(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def build_full_operator_from_pauli(pauli_list):
    n = len(pauli_list[0][1])
    H = np.zeros((2**n, 2**n), dtype=complex)
    for coef, pstr in pauli_list:
        mats = [I if c=='I' else X if c=='X' else Z for c in pstr]
        H += coef * kron_n(mats)
    return H

def build_pauli_from_structure(struct_path, cluster_size=6, H_X=0.15, SCALE_ZZ=-1.0):
    atoms = read(str(struct_path))
    pos = atoms.get_positions()
    N = min(cluster_size, len(atoms))
    # choose center = first atom that is dopant if possible, else center
    symbols = atoms.get_chemical_symbols()
    dop_idx = None
    for i,s in enumerate(symbols):
        if s.upper() in ("N","S"):
            dop_idx = i; break
    if dop_idx is None:
        dop_idx = N//2
    # compute distances to dop_idx
    dists = np.linalg.norm(pos - pos[dop_idx], axis=1)
    idx_sorted = np.argsort(dists)[:N]
    selected_pos = pos[idx_sorted]
    pauli_terms = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(selected_pos[i]-selected_pos[j])
            coef = SCALE_ZZ * (1.0 / float(max(dist, 1e-6)))
            pstr = ['I'] * N
            pstr[i] = 'Z'; pstr[j] = 'Z'
            pauli_terms.append((float(coef), pstr))
    for i in range(N):
        pstr = ['I'] * N
        pstr[i] = 'X'
        pauli_terms.append((float(H_X), pstr))
    return pauli_terms, N

def map_row_to_structure(row):
    """Heuristic mapping from row to structure file path"""
    material = row["material"]
    dopant = row.get("dopant", "None")
    base = None
    if material.lower().startswith("graph"):
        if dopant and str(dopant) not in ("None","nan"):
            fname = f"{material}_{dopant}_doped.xyz"
        else:
            fname = f"{material}.xyz"
        # two possible locations
        cand1 = Path("structures/doped") / fname
        cand2 = Path("structures/pristine") / fname
        if cand1.exists(): return cand1
        if cand2.exists(): return cand2
    # fallback platinum
    if material.lower().startswith("plati") or material.lower().startswith("pt"):
        return Path("structures/platinum/pt_cluster.xyz")
    # ultimate fallback: try pristine graphene
    return Path("structures/pristine/graphene.xyz")

def main(top_k=2, cluster_size_graphene=6, cluster_size_pt=4):
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    # 1) load model
    model_path = Path("results/models/ml_regressor.pkl")
    if not model_path.exists():
        raise FileNotFoundError("ML model not found. Run experiments/exp_01_ml_baseline.py first.")
    model = joblib.load(str(model_path))

    # 2) load DFT table and compute features
    df = load_dft_table()
    X, y, df_full = build_features(df)
    # predict
    preds = model.predict(X)
    df_full = df_full.copy()
    df_full["ml_predicted_energy"] = preds

    # sort by predicted energy (most negative best)
    df_sorted = df_full.sort_values("ml_predicted_energy").reset_index(drop=True)
    selected = df_sorted.head(top_k)

    results = []
    for idx, row in selected.iterrows():
        struct_path = map_row_to_structure(row)
        # choose cluster_size depending on material
        cluster_size = cluster_size_pt if row["material"].lower().startswith("pt") or row["material"].lower().startswith("plati") else cluster_size_graphene
        pauli_terms, N = build_pauli_from_structure(struct_path, cluster_size=cluster_size)
        H = build_full_operator_from_pauli(pauli_terms)
        # exact
        eg, evec = exact_diagonalization(H)
        # vqe
        try:
            vqe_val, angles, _ = vqe_with_scipy(H, N)
            method = "scipy_nelder_mead"
        except Exception:
            vqe_val, angles = vqe_random_search(H, N)
            method = "random_search"
        results.append({
            "index": int(idx),
            "material": row["material"],
            "dopant": row["dopant"],
            "adsorbate": row["adsorbate"],
            "ml_predicted": float(row["ml_predicted_energy"]),
            "exact_energy": float(eg),
            "vqe_energy": float(vqe_val),
            "n_qubits": int(N),
            "structure_used": str(struct_path),
            "method": method
        })
    # write results CSV
    out = Path("results/tables/quantum_refined.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print("Saved quantum_refined results to", out)
    return results

if __name__ == "__main__":
    res = main(top_k=2)
    print(res)