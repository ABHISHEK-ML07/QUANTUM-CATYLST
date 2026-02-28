# src/ranker.py
import os, json, csv, numpy as np
from time import time
from ase.io import read

# reuse some logic from hamiltonian.py and vqe.py (kept simple & self-contained)
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def kron_n(matrices):
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out

def build_full_operator(pauli_list):
    n = len(pauli_list[0][1])
    H = np.zeros((2**n, 2**n), dtype=complex)
    for coef, pstr in pauli_list:
        mats = [I if c=='I' else X if c=='X' else Z for c in pstr]
        H += coef * kron_n(mats)
    return H

def build_pauli_terms_from_structure(path, cluster_size=4, H_X=0.15, SCALE_ZZ=-1.0):
    atoms = read(path)
    pos = atoms.get_positions()
    # if cluster_size > number of atoms use all
    N = min(cluster_size, len(atoms))
    selected_pos = pos[:N]
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

def exact_diagonalization(H):
    vals, vecs = np.linalg.eigh(H)
    idx = np.argmin(vals)
    return float(vals[idx]), vecs[:, idx]

def product_state_from_angles(angles):
    states = []
    for th in angles:
        a = np.cos(th/2.0)
        b = np.sin(th/2.0)
        states.append(np.array([a, b], dtype=complex))
    psi = states[0]
    for s in states[1:]:
        psi = np.kron(psi, s)
    psi = psi / np.linalg.norm(psi)
    return psi

def expectation_from_angles(H, angles):
    psi = product_state_from_angles(angles)
    e = np.vdot(psi, H.dot(psi)).real
    return float(e)

def vqe_random_search(H, n_qubits, n_iters=2000, step_scale=0.5):
    best_angles = np.zeros(n_qubits)
    best_val = expectation_from_angles(H, best_angles)
    for i in range(n_iters):
        cand = best_angles + np.random.normal(scale=step_scale, size=n_qubits)
        val = expectation_from_angles(H, cand)
        if val < best_val:
            best_val = val
            best_angles = cand
    return best_val, best_angles

def vqe_with_scipy(H, n_qubits):
    try:
        from scipy.optimize import minimize
    except Exception:
        return vqe_random_search(H, n_qubits)
    def obj(x):
        return expectation_from_angles(H, x)
    x0 = np.zeros(n_qubits)
    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter":1000, "disp": False})
    return float(res.fun), res.x

def process_material(name, struct_path, cluster_size=6):
    pauli_terms, n = build_pauli_terms_from_structure(struct_path, cluster_size=cluster_size)
    H = build_full_operator(pauli_terms)
    eg, _ = exact_diagonalization(H)
    # VQE
    try:
        import scipy
        use_scipy = True
    except:
        use_scipy = False
    if use_scipy:
        v, ang = vqe_with_scipy(H, n)
        method = "scipy_nelder_mead"
    else:
        v, ang = vqe_random_search(H, n, n_iters=2000, step_scale=0.6)
        method = "random_search"
    return {"material": name, "n_qubits": n, "exact": eg, "vqe": v, "method": method}

def main():
    os.makedirs("results", exist_ok=True)
    # 1) Load existing graphene result if exists
    leaderboard = []
    # graphene entry: use existing exact if present
    if os.path.exists("results/exact_ground.npy"):
        eg = float(np.load("results/exact_ground.npy")[0])
        vq = float(np.load("results/vqe_energy.npy")[0])
        leaderboard.append({"material":"graphene_N_doped_cluster","method":"exact","energy":eg})
        leaderboard.append({"material":"graphene_N_doped_cluster","method":"vqe","energy":vq})
    else:
        print("Warning: graphene exact not found, will compute from structure.")
        g = process_material("graphene_N_doped_cluster","structures/doped/graphene_N_doped.xyz", cluster_size=6)
        leaderboard.append({"material":g["material"],"method":"exact","energy":g["exact"]})
        leaderboard.append({"material":g["material"],"method":"vqe","energy":g["vqe"]})

    # 2) Process Pt cluster
    pt_result = process_material("platinum_cluster","structures/platinum/pt_cluster.xyz", cluster_size=4)
    leaderboard.append({"material":"platinum_cluster","method":"exact","energy":pt_result["exact"]})
    leaderboard.append({"material":"platinum_cluster","method":"vqe","energy":pt_result["vqe"]})

    # 3) Save final leaderboard CSV
    outcsv = "results/final_leaderboard.csv"
    with open(outcsv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["material","method","energy"])
        for r in leaderboard:
            writer.writerow([r["material"], r["method"], f"{r['energy']:.8f}"])
    print(f"Final leaderboard saved to {outcsv}")
    # print contents for quick check
    for r in leaderboard:
        print(r)
    print("Done.")

if __name__ == '__main__':
    main()