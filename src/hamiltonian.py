# src/hamiltonian.py
import os
import json
import numpy as np
from ase.io import read
from scipy.spatial import distance_matrix

os.makedirs("results", exist_ok=True)

# Parameters: tuneable
STRUCTURE_PATH = "structures/doped/graphene_N_doped.xyz"  # change to the file to use
CLUSTER_SIZE = 6   # number of atoms to include around dopant (recommend 4-8 for 2^n feasiblity)
H_X = 0.15         # coefficient for single-qubit X terms (transverse field)
SCALE_ZZ = -1.0    # prefactor scale for pairwise ZZ terms (negative -> attractive)

# Pauli matrices (numpy)
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def kron_n(matrices):
    """Kronecker product of list of matrices in order."""
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out

def build_full_operator(pauli_list):
    """Build full 2^n matrix from list of (coef, pauli_string) where pauli_string is list of 'I','X','Z'."""
    n = len(pauli_list[0][1])
    H = np.zeros((2**n, 2**n), dtype=complex)
    for coef, pstr in pauli_list:
        mats = [I if c=='I' else X if c=='X' else Z for c in pstr]
        H += coef * kron_n(mats)
    return H

def node_cluster(atoms, center_idx, k):
    # compute distances to center
    pos = atoms.get_positions()
    dists = np.linalg.norm(pos - pos[center_idx], axis=1)
    idx_sorted = np.argsort(dists)
    selected = idx_sorted[:k]
    return atoms[selected]

def main():
    # 1) load structure
    if not os.path.exists(STRUCTURE_PATH):
        print(f"ERROR: structure not found: {STRUCTURE_PATH}")
        return
    atoms = read(STRUCTURE_PATH)
    symbols = atoms.get_chemical_symbols()

    # 2) find dopant index (N or S) otherwise pick center
    dopant_idx = None
    for i, s in enumerate(symbols):
        if s.upper() in ("N", "S"):
            dopant_idx = i
            break
    if dopant_idx is None:
        # fallback: pick central atom by box center
        dopant_idx = len(atoms)//2
        print("Warning: no N/S dopant found, using center index", dopant_idx)

    # 3) extract local cluster
    cluster_atoms = node_cluster(atoms, dopant_idx, CLUSTER_SIZE)
    cluster_symbols = cluster_atoms.get_chemical_symbols()
    print("Selected cluster symbols:", cluster_symbols)

    # 4) compute pairwise distances and create Pauli representation
    pos = cluster_atoms.get_positions()
    D = distance_matrix(pos, pos) + np.eye(CLUSTER_SIZE)*1e9  # self-dist large
    # Build Pauli list: ZZ terms for each pair with strength SCALE_ZZ*(1/dist)
    pauli_terms = []
    for i in range(CLUSTER_SIZE):
        for j in range(i+1, CLUSTER_SIZE):
            coef = SCALE_ZZ * (1.0 / float(np.linalg.norm(pos[i]-pos[j])))
            # pauli string with Z at i and j, I elsewhere
            pstr = ['I'] * CLUSTER_SIZE
            pstr[i] = 'Z'
            pstr[j] = 'Z'
            pauli_terms.append((float(coef), pstr))

    # single-qubit X fields
    for i in range(CLUSTER_SIZE):
        pstr = ['I'] * CLUSTER_SIZE
        pstr[i] = 'X'
        pauli_terms.append((float(H_X), pstr))

    # 5) build full matrix (2^n x 2^n)
    print(f"Building full Hamiltonian matrix for n={CLUSTER_SIZE} qubits (2^{CLUSTER_SIZE} = {2**CLUSTER_SIZE} states)")
    H = build_full_operator(pauli_terms)

    # 6) save results
    np.save("results/hamiltonian.npy", H)
    info = {
        "n_qubits": CLUSTER_SIZE,
        "structure_used": STRUCTURE_PATH,
        "cluster_symbols": cluster_symbols,
        "num_pauli_terms": len(pauli_terms),
        "pauli_terms_sample": [
            {"coef": pauli_terms[i][0], "pstr": "".join(pauli_terms[i][1])}
            for i in range(min(6, len(pauli_terms)))
        ]
    }
    with open("results/hamiltonian_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # also write a textual pauli list for later conversion
    with open("results/pauli_terms.txt", "w") as f:
        for coef, p in pauli_terms:
            f.write(f"{coef:.6e} {' '.join(p)}\n")

    print("Hamiltonian matrix saved to results/hamiltonian.npy")
    print("Metadata saved to results/hamiltonian_info.json")
    print("Sample pauli terms written to results/pauli_terms.txt")
    print("Done.")

if __name__ == "__main__":
    main()